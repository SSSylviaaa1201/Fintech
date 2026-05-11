"""Detect available capabilities for graceful degradation when API keys are missing."""

import os
from dataclasses import dataclass
from enum import Enum


class CapabilityLevel(Enum):
    FULL = "full"
    ENHANCED = "enhanced"
    BASIC = "basic"
    DEMO = "demo"


@dataclass
class CapabilityReport:
    level: CapabilityLevel
    sentiment_methods: list[str]
    data_sources: list[str]
    llm_enabled: bool
    rag_enabled: bool
    chat_enabled: bool
    warnings: list[str]


def detect_capabilities() -> CapabilityReport:
    """Check environment and report what's available."""
    methods = ["vader", "lr"]
    sources = ["synthetic"]
    warnings = []

    # Check FinBERT model cache
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("ProsusAI/finbert")
        methods.append("finbert")
    except Exception:
        warnings.append("FinBERT model not cached. Run pipeline once to download it.")

    # Check data source APIs
    if os.getenv("ALPHA_VANTAGE_KEY"):
        sources.append("alpha_vantage")
    if os.getenv("NEWSAPI_KEY"):
        sources.append("newsapi")
    sources.append("rss")  # Always available (free, no key)

    # Determine level
    if "finbert" in methods and ("newsapi" in sources or "rss" in sources):
        level = CapabilityLevel.ENHANCED
    elif len(methods) >= 2:
        level = CapabilityLevel.BASIC
    else:
        level = CapabilityLevel.DEMO

    return CapabilityReport(
        level=level,
        sentiment_methods=methods,
        data_sources=sources,
        llm_enabled=False,
        rag_enabled=False,
        chat_enabled=False,
        warnings=warnings,
    )
