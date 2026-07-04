"""MPK-Kuramoto vision head for a Qwen LLM.

An image-driven coupled-oscillator vision encoder: the Un-0 Kuramoto primitive
(unconv-ai/Un-0) inverted from a class-conditioned *generator* into an
image-conditioned *encoder*, split into magno/parvo/konio pathways per the
Koniocellular Circuit-Family manuscript (M/P encode, K modulates). Outputs vision
tokens that plug into a causal LLM as a connector.
"""

from __future__ import annotations

from .kuramoto import (
    MPKFieldOutput,
    MPKKuramotoField,
    PathwayField,
    kuramoto_velocity,
    order_parameter,
)
from .qwen_glue import VLMWithKuramotoHead, build_smoke_llm, load_qwen
from .vision_head import KuramotoVisionHead, RetinaPatchFront

__all__ = [
    "KuramotoVisionHead",
    "RetinaPatchFront",
    "MPKKuramotoField",
    "MPKFieldOutput",
    "PathwayField",
    "kuramoto_velocity",
    "order_parameter",
    "VLMWithKuramotoHead",
    "build_smoke_llm",
    "load_qwen",
]
