"""CPU-only plumbing smoke test for the MPK-Kuramoto Qwen vision head.

Proves the end-to-end path without any download or GPU:
  1. Field invariants: gains in (0, max_gain), coherence in [0, 1], finite readout.
  2. Vision tokens actually replace the image-placeholder embeddings.
  3. With the LLM FROZEN (the real fine-tune regime), a backward pass puts finite,
     nonzero gradients on the M/P/K coupling matrices and the projector.
  4. Head-only training drives the caption cross-entropy down on a fixed batch.

Run (CPU forced, offline):
    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python -m kuramoto_vlm.smoke_test
"""

from __future__ import annotations

import os

import torch

# This host exposes a virtual GPU that NVML reports as "available" while the CUDA
# runtime has zero usable devices; that mismatch crashes torchao's Triton probe
# when transformers imports its quantizer registry. Force CPU for the smoke test
# (real GPUs on RunPod set KVLM_FORCE_CPU=0).
if os.environ.get("KVLM_FORCE_CPU", "1") == "1":
    torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    torch.cuda.device_count = lambda: 0  # type: ignore[assignment]

from .kuramoto import MPKKuramotoField, order_parameter
from .qwen_glue import VLMWithKuramotoHead, build_smoke_llm
from .vision_head import KuramotoVisionHead

HIDDEN = 2048  # matches Qwen3.5-2B; trivial cost at this batch/token scale
IMAGE_SIZE = 32
PATCH = 8  # -> 16 vision tokens
BATCH = 2
VOCAB = 512
PROMPT_LEN = 3
CAPTION_LEN = 5


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise SystemExit(1)


def _check(cond: bool, msg: str) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        raise SystemExit(1)


def build_batch(head: KuramotoVisionHead, image_token_id: int, generator: torch.Generator):
    """Synthetic (image, prompt+image+caption, labels) batch."""
    n_img = head.num_tokens
    pixel_values = torch.rand(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)

    seq_len = PROMPT_LEN + n_img + CAPTION_LEN
    input_ids = torch.zeros(BATCH, seq_len, dtype=torch.long)
    labels = torch.full((BATCH, seq_len), -100, dtype=torch.long)
    # text ids avoid the reserved image placeholder id.
    text = torch.randint(0, image_token_id, (BATCH, PROMPT_LEN + CAPTION_LEN), generator=generator)
    input_ids[:, :PROMPT_LEN] = text[:, :PROMPT_LEN]
    input_ids[:, PROMPT_LEN : PROMPT_LEN + n_img] = image_token_id
    input_ids[:, PROMPT_LEN + n_img :] = text[:, PROMPT_LEN:]
    # supervise only the caption span (predict-next handled by the LM's shift).
    labels[:, PROMPT_LEN + n_img :] = input_ids[:, PROMPT_LEN + n_img :]
    attention_mask = torch.ones(BATCH, seq_len, dtype=torch.long)
    return pixel_values, input_ids, attention_mask, labels


def test_field_invariants(generator: torch.Generator) -> None:
    print("[1] field invariants")
    field = MPKKuramotoField(
        n_m=32, n_p=32, n_k=24, drive_dim_m=4, drive_dim_p=12, drive_dim_k=4, num_steps=6
    )
    b = 7
    out = field(
        torch.randn(b, 4, generator=generator),
        torch.randn(b, 12, generator=generator),
        torch.randn(b, 4, generator=generator),
    )
    _check(out.readout.shape == (b, field.readout_dim), f"readout shape {tuple(out.readout.shape)}")
    _check(out.readout.shape[1] == 2 * (32 + 32 + 24), "readout includes K (encoding)")
    _check(torch.isfinite(out.readout).all().item(), "readout finite")
    _check(
        bool(((out.mod_gain > 0) & (out.mod_gain < field.max_gain)).all()),
        f"mod_gain in (0, {field.max_gain})",
    )
    _check(bool(((out.coherence_k >= 0) & (out.coherence_k <= 1)).all()), "coherence_k in [0, 1]")
    # coherence of identical phases == 1, of a full spread ~ 0
    _check(abs(order_parameter(torch.zeros(1, 64)).item() - 1.0) < 1e-5, "coherence(uniform)=1")


def build_model(image_token_id_holder: list[int]):
    head = KuramotoVisionHead(
        hidden_size=HIDDEN, image_size=IMAGE_SIZE, patch_size=PATCH, n_m=48, n_p=48, n_k=32, num_steps=6
    )
    llm, image_token_id = build_smoke_llm(hidden_size=HIDDEN, vocab_size=VOCAB)
    image_token_id_holder.append(image_token_id)
    model = VLMWithKuramotoHead(head, llm, image_token_id=image_token_id, freeze_llm=True)
    return model


def test_injection_and_grads(generator: torch.Generator) -> VLMWithKuramotoHead:
    print("[2] vision-token injection + frozen-LLM gradients")
    holder: list[int] = []
    model = build_model(holder)
    image_token_id = holder[0]
    pixel_values, input_ids, attention_mask, labels = build_batch(
        model.vision_head, image_token_id, generator
    )

    # injection actually changes the image-slot embeddings
    with torch.no_grad():
        base = model.llm.get_input_embeddings()(input_ids)
        vis = model.vision_head(pixel_values)
        injected = model.embed_with_vision(input_ids, vis)
    _check(tuple(vis.shape) == (BATCH, model.vision_head.num_tokens, HIDDEN), f"vision {tuple(vis.shape)}")
    slot = input_ids == image_token_id
    changed = not torch.allclose(base[slot], injected[slot])
    unchanged = torch.allclose(base[~slot], injected[~slot])
    _check(changed, "image slots replaced by vision tokens")
    _check(unchanged, "text slots untouched")

    out = model(pixel_values, input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss
    _check(torch.isfinite(loss).item(), f"loss finite ({loss.item():.4f})")
    loss.backward()

    watched = {
        "field.magno.coupling": model.vision_head.field.magno.coupling,
        "field.parvo.coupling": model.vision_head.field.parvo.coupling,
        "field.konio.coupling": model.vision_head.field.konio.coupling,
        "field.mod.weight": model.vision_head.field.mod.weight,
    }
    proj_last = [m for m in model.vision_head.projector if isinstance(m, torch.nn.Linear)][-1]
    watched["projector.final.weight"] = proj_last.weight
    for name, param in watched.items():
        g = param.grad
        ok = g is not None and torch.isfinite(g).all().item() and g.abs().sum().item() > 0
        _check(ok, f"grad on {name} present/finite/nonzero")

    llm_frozen = all(not p.requires_grad for p in model.llm.parameters())
    llm_no_grad = all(p.grad is None for p in model.llm.parameters())
    _check(llm_frozen and llm_no_grad, "LLM frozen and received no gradient")
    return model


def test_overfit(generator: torch.Generator) -> None:
    print("[3] head-only training reduces caption loss on a fixed batch")
    holder: list[int] = []
    model = build_model(holder)
    image_token_id = holder[0]
    batch = build_batch(model.vision_head, image_token_id, generator)
    pixel_values, input_ids, attention_mask, labels = batch

    params = model.trainable_parameters()
    llm_params = {id(p) for p in model.llm.parameters()}
    _check(all(id(p) not in llm_params for p in params), "only head params are trainable")
    opt = torch.optim.Adam(params, lr=1e-2)

    losses = []
    for _ in range(60):
        opt.zero_grad()
        out = model(pixel_values, input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    first, last = losses[0], min(losses[-5:])
    print(f"      loss {first:.4f} -> {last:.4f} over {len(losses)} steps")
    _check(all(torch.isfinite(torch.tensor(l)) for l in losses), "all losses finite")
    _check(last < 0.7 * first, f"caption loss dropped >30% ({first:.3f} -> {last:.3f})")


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(4)
    gen = torch.Generator().manual_seed(0)
    print(f"torch {torch.__version__} | device cpu | cuda_available={torch.cuda.is_available()}")
    test_field_invariants(gen)
    test_injection_and_grads(gen)
    test_overfit(gen)
    print("\nSMOKE OK: image -> MPK-Kuramoto oscillators -> Qwen-style tokens -> LM loss -> trains.")


if __name__ == "__main__":
    main()
