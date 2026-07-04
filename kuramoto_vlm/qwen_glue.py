"""Wire the Kuramoto vision head into a causal LLM as a vision connector.

The head emits ``(B, num_tokens, hidden)`` vision tokens; we splice them into the
LLM's input-embedding stream at image-placeholder positions and let the LLM do
the rest. This is the LLaVA / Qwen-VL connector contract, so the same wrapper
works for a tiny stand-in LLM (smoke test, CPU) and for real Qwen3.5-2B: only
``build_*`` changes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def build_smoke_llm(
    *,
    hidden_size: int = 2048,
    vocab_size: int = 512,
    num_layers: int = 2,
    num_heads: int = 8,
    intermediate_size: int = 512,
    image_token_id: int | None = None,
):
    """Build a tiny real ``LlamaForCausalLM`` for CPU plumbing tests.

    Same ``inputs_embeds`` + ``labels`` contract as Qwen3.5, but a few layers and
    a small vocab so forward/backward/step run in milliseconds on CPU. ``hidden``
    matches the head's projector so the injection path is identical to prod.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    if image_token_id is None:
        image_token_id = vocab_size - 1
    if not 0 <= image_token_id < vocab_size:
        raise ValueError(f"image_token_id {image_token_id} out of range for vocab {vocab_size}.")
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    return LlamaForCausalLM(config), image_token_id


def load_qwen(model_name: str = "Qwen/Qwen3.5-2B", *, dtype: str = "bfloat16", **kwargs):
    """Load the real Qwen LLM for production training (not used in the smoke test).

    Returns the causal-LM module. The image placeholder id must come from the
    model's own processor/config (``processor.tokenizer`` special tokens) at the
    call site — do not guess it.
    """
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype)
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype, **kwargs)


class VLMWithKuramotoHead(nn.Module):
    """A causal LLM whose vision comes from a :class:`KuramotoVisionHead`.

    Args:
        vision_head: Produces ``(B, num_tokens, hidden)`` tokens.
        llm: Any causal LM taking ``inputs_embeds``/``labels`` (Llama, Qwen, ...).
        image_token_id: The placeholder token id occupying image positions in
            ``input_ids``; each image contributes ``vision_head.num_tokens`` of
            them.
        freeze_llm: If set, LLM params are frozen (train only the head/projector),
            the intended fine-tune regime.
    """

    def __init__(
        self,
        vision_head: nn.Module,
        llm: nn.Module,
        *,
        image_token_id: int,
        freeze_llm: bool = True,
    ) -> None:
        super().__init__()
        self.vision_head = vision_head
        self.llm = llm
        self.image_token_id = int(image_token_id)
        self.freeze_llm = bool(freeze_llm)
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad_(False)

    def embed_with_vision(
        self,
        input_ids: Tensor,
        vision_tokens: Tensor,
    ) -> Tensor:
        """Splice vision tokens into the text embeddings at placeholder slots.

        Args:
            input_ids: ``(B, T)`` with exactly ``B * num_tokens`` image
                placeholders, in reading order.
            vision_tokens: ``(B, num_tokens, hidden)``.

        Returns:
            ``inputs_embeds`` ``(B, T, hidden)`` with image slots replaced.
        """
        embeds = self.llm.get_input_embeddings()(input_ids)
        mask = input_ids == self.image_token_id
        n_slots = int(mask.sum())
        n_vision = vision_tokens.shape[0] * vision_tokens.shape[1]
        if n_slots != n_vision:
            raise ValueError(
                f"{n_slots} image placeholders but {n_vision} vision tokens; "
                "each image needs exactly vision_head.num_tokens placeholders."
            )
        source = vision_tokens.reshape(-1, vision_tokens.shape[-1]).to(embeds.dtype)
        return embeds.masked_scatter(mask.unsqueeze(-1), source)

    def forward(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ):
        """Run the full VLM step; returns the LLM output (``.loss`` when labels)."""
        vision_tokens = self.vision_head(pixel_values)
        inputs_embeds = self.embed_with_vision(input_ids, vision_tokens)
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters with grad enabled (head + projector when the LLM is frozen)."""
        return [p for p in self.parameters() if p.requires_grad]
