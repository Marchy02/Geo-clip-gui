"""LLM reasoning layer: one arbiter, several interchangeable providers."""

from .common import build_context, parse_verdict, verdict_from_dict
from .gemini import analyze
from .providers import KINDS, TEMPLATES, ProviderConfig

__all__ = ["KINDS", "TEMPLATES", "ProviderConfig", "analyze", "build_context", "parse_verdict", "verdict_from_dict"]
