"""Load a (possibly multimodal) Qwen backbone as a frozen text decoder.

We only use the language side: the Kuramoto head supplies vision tokens that we
splice into the input-embedding stream at the model's image-placeholder id, and
we drive the model with ``inputs_embeds`` (no ``pixel_values``), which bypasses
its native vision tower entirely. This loader resolves, robustly across text and
VL configs: the causal-LM module, the tokenizer, the image placeholder id, and
the embedding hidden size (read off the embedding weight, not guessed).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Backbone:
    """A loaded frozen LLM backbone and the handles the trainer needs."""

    model: nn.Module
    tokenizer: object
    image_token_id: int
    hidden_size: int
    name: str


def resolve_image_token_id(config, tokenizer) -> int:
    """Find a usable image-placeholder token id from config or tokenizer."""
    candidates = ("image_token_id", "image_token_index", "image_placeholder_token_id")
    configs = [config]
    for sub in ("text_config", "vision_config", "thinker_config"):
        if getattr(config, sub, None) is not None:
            configs.append(getattr(config, sub))
    for cfg in configs:
        for attr in candidates:
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value >= 0:
                return value
    unk = getattr(tokenizer, "unk_token_id", None)
    for token in ("<|image_pad|>", "<|image|>", "<image>", "<|vision_pad|>", "<|imgpad|>"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            return tid
    raise ValueError(
        "Could not resolve an image placeholder token id from config or tokenizer; "
        "pass one explicitly."
    )


def load_backbone(
    name: str,
    *,
    dtype: str = "bfloat16",
    device: str = "cuda",
    attn_implementation: str | None = None,
) -> Backbone:
    """Load ``name`` as a frozen causal LM, trying VL then text auto-classes."""
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(name)
    torch_dtype = getattr(torch, dtype)
    load_kwargs: dict = {"dtype": torch_dtype}
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation

    errors: list[str] = []
    model = None
    for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        loader = getattr(transformers, loader_name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(name, **load_kwargs)
            break
        except Exception as exc:  # noqa: BLE001 — report every failure path
            errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")
    if model is None:
        raise RuntimeError("Could not load backbone:\n" + "\n".join(errors))

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    hidden_size = int(model.get_input_embeddings().weight.shape[1])
    image_token_id = resolve_image_token_id(config, tokenizer)
    return Backbone(
        model=model,
        tokenizer=tokenizer,
        image_token_id=image_token_id,
        hidden_size=hidden_size,
        name=name,
    )
