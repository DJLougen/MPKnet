"""Train the MPK-Kuramoto vision head as a frozen-Qwen vision connector.

Classification framed as vision-language: an image becomes Kuramoto vision
tokens, spliced into the prompt at the image-placeholder slots; the frozen LLM
must emit the class name. Only the head (retina front is parameter-free; M/P/K
oscillator fields + projector) is trained, by next-token cross-entropy on the
answer span. This is the standard connector-tuning setup, with our oscillator
head standing in for a ViT.

Examples:
    # validate the whole stack on GPU (downloads the LLM, one synthetic step):
    python -m kuramoto_vlm.train_classify --dataset cifar100 --dry-run

    # real CIFAR-100 run:
    python -m kuramoto_vlm.train_classify --dataset cifar100 \
        --data-root ~/data --epochs 3 --batch-size 64 --out runs/cifar100
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .backbone import load_backbone
from .data import get_dataset
from .qwen_glue import VLMWithKuramotoHead
from .vision_head import KuramotoVisionHead

PROMPT_PREFIX = "Picture: "
PROMPT_QUESTION = "\nWhat is the main object in the picture? Answer with one word:"


def build_text_batch(labels, *, tokenizer, image_token_id, num_tokens, class_names, device):
    """Assemble ``input_ids / attention_mask / label_ids`` for a label batch.

    Layout per row: ``[prefix] [IMG]*num_tokens [question] [ answer <eos>]``.
    Only the answer span is supervised (everything else is ``-100``).
    """
    prefix = tokenizer(PROMPT_PREFIX, add_special_tokens=False)["input_ids"]
    question = tokenizer(PROMPT_QUESTION, add_special_tokens=False)["input_ids"]
    img = [image_token_id] * num_tokens
    eos = tokenizer.eos_token_id

    seqs, labs = [], []
    for label in labels.tolist():
        answer = tokenizer(" " + class_names[label], add_special_tokens=False)["input_ids"] + [eos]
        ids = prefix + img + question + answer
        mask = [-100] * (len(prefix) + len(img) + len(question)) + answer
        seqs.append(ids)
        labs.append(mask)

    max_len = max(len(s) for s in seqs)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(seqs), max_len), pad, dtype=torch.long)
    label_ids = torch.full((len(seqs), max_len), -100, dtype=torch.long)
    attention = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, (s, m) in enumerate(zip(seqs, labs)):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        label_ids[i, : len(m)] = torch.tensor(m, dtype=torch.long)
        attention[i, : len(s)] = 1
    return input_ids.to(device), attention.to(device), label_ids.to(device)


@torch.no_grad()
def evaluate(model, loader, *, cfg, class_names, device, max_batches):
    """Teacher-forced answer-token accuracy and loss over a few val batches."""
    model.eval()
    tok = model._tok  # attached in main
    tot_loss = correct = total = seq_correct = seq_total = 0.0
    n = 0
    for step, (images, labels) in enumerate(loader):
        if step >= max_batches:
            break
        pixel_values = images.to(device)
        input_ids, attn, label_ids = build_text_batch(
            labels, tokenizer=tok, image_token_id=model.image_token_id,
            num_tokens=model.vision_head.num_tokens, class_names=class_names, device=device,
        )
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            out = model(pixel_values, input_ids, attention_mask=attn, labels=label_ids)
        tot_loss += float(out.loss)
        n += 1
        shift_logits = out.logits[:, :-1].float()
        shift_labels = label_ids[:, 1:]
        mask = shift_labels != -100
        pred = shift_logits.argmax(-1)
        correct += float((pred[mask] == shift_labels[mask]).sum())
        total += float(mask.sum())
        row_ok = ((pred == shift_labels) | ~mask).all(dim=1) & mask.any(dim=1)
        seq_correct += float(row_ok.sum())
        seq_total += float(mask.any(dim=1).sum())
    model.train()
    return {
        "val_loss": tot_loss / max(n, 1),
        "token_acc": correct / max(total, 1),
        "exact_acc": seq_correct / max(seq_total, 1),
    }


def build_model(args, device):
    backbone = load_backbone(args.llm, dtype=args.dtype, device=device)
    head = KuramotoVisionHead(
        hidden_size=backbone.hidden_size,
        image_size=args.image_size,
        patch_size=args.patch,
        n_m=args.n_m,
        n_p=args.n_p,
        n_k=args.n_k,
        num_steps=args.num_steps,
        k_encode=args.k_encode,
        k_modulate=args.k_modulate,
        freeze_coupling=args.freeze_coupling,
    )
    model = VLMWithKuramotoHead(
        head, backbone.model, image_token_id=backbone.image_token_id, freeze_llm=True
    )
    head.to(device)
    if hasattr(backbone.model, "config"):
        backbone.model.config.use_cache = False
    model._tok = backbone.tokenizer  # stash for eval/build
    model.image_token_id = backbone.image_token_id
    return model, backbone


def dry_run(args, device):
    print(f"[dry-run] loading {args.llm} on {device} ...")
    model, backbone = build_model(args, device)
    head = model.vision_head
    n_classes = 20
    class_names = [f"class{i}" for i in range(n_classes)]
    images = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=device)
    labels = torch.arange(args.batch_size) % n_classes
    input_ids, attn, label_ids = build_text_batch(
        labels, tokenizer=backbone.tokenizer, image_token_id=model.image_token_id,
        num_tokens=head.num_tokens, class_names=class_names, device=device,
    )
    print(f"[dry-run] hidden={backbone.hidden_size} image_token_id={model.image_token_id} "
          f"vision_tokens={head.num_tokens} seq_len={input_ids.shape[1]}")
    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
        out = model(images, input_ids, attention_mask=attn, labels=label_ids)
    out.loss.backward()
    gnorm = sum(
        float(p.grad.norm()) for p in model.trainable_parameters() if p.grad is not None
    )
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"[dry-run] loss={float(out.loss):.4f} head_params={n_train/1e6:.2f}M grad_norm={gnorm:.3e}")
    assert torch.isfinite(out.loss).item() and gnorm > 0, "dry-run failed: no finite loss/grad"
    print("[dry-run] OK — image -> Kuramoto tokens -> Qwen -> loss -> grad reaches head.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llm", default="Qwen/Qwen3.5-2B")
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--n-m", type=int, default=256)
    p.add_argument("--n-p", type=int, default=256)
    p.add_argument("--n-k", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--k-encode", action=argparse.BooleanOptionalAction, default=True,
                   help="K contributes its phases to the readout (encoding role)")
    p.add_argument("--k-modulate", action=argparse.BooleanOptionalAction, default=True,
                   help="K injects a diffuse gain on the integrated DV (modulatory role)")
    p.add_argument("--freeze-coupling", action="store_true",
                   help="reservoir ablation: freeze M/P/K dynamics at init, train readout only")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--limit-train", type=int, default=0, help="cap train examples (0 = all)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    torch.manual_seed(0)
    if args.dry_run:
        dry_run(args, device)
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, backbone = build_model(args, device)
    tok = backbone.tokenizer
    train_ds, class_names = get_dataset(
        args.dataset, root=args.data_root, image_size=args.image_size, train=True
    )
    val_ds, _ = get_dataset(
        args.dataset, root=args.data_root, image_size=args.image_size, train=False
    )
    if args.limit_train:
        train_ds = torch.utils.data.Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"dataset={args.dataset} classes={len(class_names)} "
          f"train={len(train_ds)} val={len(val_ds)} hidden={backbone.hidden_size} "
          f"vision_tokens={model.vision_head.num_tokens} "
          f"trainable={sum(p.numel() for p in model.trainable_parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.max_steps if args.max_steps else args.epochs * len(train_loader)

    def lr_at(step: int) -> float:
        warm = max(1, args.warmup)
        if step < warm:
            return args.lr * (step + 1) / warm
        progress = (step - warm) / max(1, total_steps - warm)
        return 0.5 * args.lr * (1.0 + math.cos(math.pi * min(1.0, progress)))

    history: list[dict] = []
    metrics_path = out_dir / "metrics.json"
    step = 0
    t0 = time.time()
    model.train()
    for epoch in range(args.epochs):
        for images, labels in train_loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            pixel_values = images.to(device, non_blocking=True)
            input_ids, attn, label_ids = build_text_batch(
                labels, tokenizer=tok, image_token_id=model.image_token_id,
                num_tokens=model.vision_head.num_tokens, class_names=class_names, device=device,
            )
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
                out = model(pixel_values, input_ids, attention_mask=attn, labels=label_ids)
            loss = out.loss
            loss.backward()
            loss_item = float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.grad_clip)
            opt.step()
            step += 1

            if step % args.log_every == 0:
                rate = step * args.batch_size / (time.time() - t0)
                print(f"epoch {epoch} step {step} loss {loss_item:.4f} "
                      f"lr {lr_at(step):.2e} {rate:.0f} img/s", flush=True)
            if step % args.eval_every == 0:
                ev = evaluate(model, val_loader, cfg=args, class_names=class_names,
                              device=device, max_batches=args.eval_batches)
                ev.update({"step": step, "epoch": epoch, "train_loss": loss_item})
                history.append(ev)
                metrics_path.write_text(json.dumps(history, indent=2))
                print(f"  [eval] step {step} val_loss {ev['val_loss']:.4f} "
                      f"token_acc {ev['token_acc']:.4f} exact_acc {ev['exact_acc']:.4f}", flush=True)
                torch.save(model.vision_head.state_dict(), out_dir / "head.pt")
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    ev = evaluate(model, val_loader, cfg=args, class_names=class_names,
                  device=device, max_batches=args.eval_batches)
    ev.update({"step": step, "epoch": args.epochs, "final": True})
    history.append(ev)
    metrics_path.write_text(json.dumps(history, indent=2))
    torch.save(model.vision_head.state_dict(), out_dir / "head.pt")
    print(f"DONE step {step} final val_loss {ev['val_loss']:.4f} "
          f"token_acc {ev['token_acc']:.4f} exact_acc {ev['exact_acc']:.4f}")


if __name__ == "__main__":
    main()
