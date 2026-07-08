"""
config.py — v2
=========
Centralized, env-driven configuration.

WHY: pehle constants (MAX_CHUNKS_TO_WRITER, RETRY_LIMIT, GROQ_MODELS,
MAX_REACT_ITERATIONS, etc.) alag-alag files mein hardcoded the. Prod mein
har environment (dev/staging/prod) ke liye alag values chahiye hoti hain
bina code change kiye — isliye sab kuch yahan laaya gaya hai, .env se
override ho sakta hai.

v2 CHANGES (this round — fixes real gaps from v1):
- SAFE PARSING: v1 mein `int(os.getenv(...))` / `float(os.getenv(...))`
  direct call the — agar .env mein koi bhi typo ho jaaye
  (e.g. MAX_REACT_ITERATIONS=abc), poora app import-time pe crash ho
  jaata tha with a raw ValueError traceback, kahin log bhi nahi hota
  ki KIS variable ki wajah se crash hua. FIX: `_get_int()` / `_get_float()`
  / `_get_bool()` helpers ab malformed values ko catch karte hain, ek
  clear WARNING log karte hain (variable name + bad value + fallback
  value), aur default pe safely fall back karte hain — app kabhi is
  wajah se nahi girega.
- RANGE VALIDATION: purely-parseable but nonsensical values (negative
  retry counts, temperature > 2, a 0.35 top_fraction becoming 3.5 by
  typo, etc.) ab clamp hote hain to a safe range with a warning, instead
  of silently propagating into runtime behavior nobody asked for.
- GROQ_MODELS SCHEMA VALIDATION: pehle sirf `json.loads()` fail hone pe
  hi fallback hota tha. Agar JSON valid tha but shape galat (e.g. missing
  "name"/"max_context_chars" keys, or not a list of dicts), yeh silently
  aage jaake `model_cfg["name"]` pe KeyError se crash karta writer_node
  mein — bohot door jaake, debug karna mushkil. Ab schema explicitly
  validate hoti hai HERE, config load ke waqt hi, saaf warning ke saath.
- CENTRALIZED API KEYS: GROQ_API_KEY, TAVILY_API_KEY, LANGCHAIN_API_KEY /
  LANGCHAIN_PROJECT ab yahan bhi expose hote hain (settings.GROQ_API_KEY
  etc.) instead of sirf scattered `os.getenv()` calls in agent.py/tools.py.
  Existing `os.getenv()` call sites still work (backward compatible),
  but new code / other files should prefer `settings.*` going forward.
- CENTRALIZED PATHS: KNOWLEDGE_BASE_DIR, PERMANENT_STORE_DIR, CACHE_DIR,
  EMBEDDING_MODEL — pehle knowledge_base.py aur retriever.py mein
  hardcoded the. Ab settings se aate hain (with the exact same defaults,
  so nothing breaks), env-tunable without touching those files' code.
- STARTUP VALIDATION: `validate_startup()` — ek explicit check jo missing
  /misconfigured critical keys ko LOUDLY warn karta hai app start hote hi
  (e.g. GROQ_API_KEY missing, ya LANGSMITH_ENABLED=true but no
  LANGCHAIN_API_KEY set — yeh exactly woh silent failure mode hai jo
  agent.py mein "no run_id at top of run_agent" warning ke roop mein
  dikhta tha, root cause ab startup pe hi pakड़ में aa jaata hai instead
  of discovering it query-by-query in prod logs).
- `settings.as_dict()` — poora effective config ek jagah dump karne ke
  liye (startup log / debug ke liye), secrets ko masked karke.

Usage: `from utils.config import settings` and access `settings.MAX_CHUNKS_TO_WRITER`
"""

import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ── Logging setup (call once, at app entrypoint) ─────────────────────────────

def setup_logging():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if level is None:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        logging.getLogger(__name__).warning(
            f"LOG_LEVEL='{level_name}' is not a valid logging level, falling back to INFO"
        )
        level = logging.INFO
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )
    # Quiet down noisy third-party libs unless explicitly debugging
    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


# ── Safe env-parsing helpers ──────────────────────────────────────────────────
# WHY: raw int()/float() on a user-edited .env value crashes the WHOLE app
# at import time on a single typo, with no indication of which variable
# was the culprit. These helpers parse defensively, clamp to a sane range
# if given, and always log a clear warning instead of raising.

def _get_int(name: str, default: int, min_val: int = None, max_val: int = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = int(raw)
    except ValueError:
        logger.warning(f"[config] {name}='{raw}' is not a valid int — using default {default}")
        return default
    if min_val is not None and val < min_val:
        logger.warning(f"[config] {name}={val} below minimum {min_val} — clamping to {min_val}")
        val = min_val
    if max_val is not None and val > max_val:
        logger.warning(f"[config] {name}={val} above maximum {max_val} — clamping to {max_val}")
        val = max_val
    return val


def _get_float(name: str, default: float, min_val: float = None, max_val: float = None) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        val = float(raw)
    except ValueError:
        logger.warning(f"[config] {name}='{raw}' is not a valid float — using default {default}")
        return default
    if min_val is not None and val < min_val:
        logger.warning(f"[config] {name}={val} below minimum {min_val} — clamping to {min_val}")
        val = min_val
    if max_val is not None and val > max_val:
        logger.warning(f"[config] {name}={val} above maximum {max_val} — clamping to {max_val}")
        val = max_val
    return val


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in ("true", "1", "yes", "on"):
        return True
    if normalized in ("false", "0", "no", "off"):
        return False
    logger.warning(f"[config] {name}='{raw}' is not a valid bool — using default {default}")
    return default


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw is not None and raw.strip() != "" else default


# ── GROQ_MODELS schema validation ────────────────────────────────────────────

def _validate_groq_models(models, source_label: str) -> list | None:
    """
    Returns the models list if it passes schema validation, else None
    (caller falls back to default). Checked once at load time so a bad
    GROQ_MODELS_JSON fails loudly HERE with a clear reason, instead of
    surfacing as a random KeyError deep inside writer_node in prod.
    """
    if not isinstance(models, list) or not models:
        logger.warning(f"[config] {source_label} must be a non-empty list — ignoring")
        return None
    for i, m in enumerate(models):
        if not isinstance(m, dict):
            logger.warning(f"[config] {source_label}[{i}] is not an object — ignoring whole list")
            return None
        if "name" not in m or not isinstance(m["name"], str) or not m["name"].strip():
            logger.warning(f"[config] {source_label}[{i}] missing/invalid 'name' — ignoring whole list")
            return None
        if "max_context_chars" not in m or not isinstance(m["max_context_chars"], (int, float)) or m["max_context_chars"] <= 0:
            logger.warning(f"[config] {source_label}[{i}] missing/invalid 'max_context_chars' — ignoring whole list")
            return None
    return models


class Settings:
    # ── Agent / ReAct loop ────────────────────────────────────────────────
    MAX_REACT_ITERATIONS: int = _get_int("MAX_REACT_ITERATIONS", 3, min_val=0, max_val=10)
    RETRY_LIMIT: int = _get_int("RETRY_LIMIT", 2, min_val=0, max_val=10)
    RETRY_WAIT_SEC: int = _get_int("RETRY_WAIT_SEC", 5, min_val=0, max_val=120)
    WRITER_MAX_TOKENS: int = _get_int("WRITER_MAX_TOKENS", 1000, min_val=50, max_val=8000)
    REFINE_MAX_TOKENS: int = _get_int("REFINE_MAX_TOKENS", 50, min_val=10, max_val=500)
    WRITER_TEMPERATURE: float = _get_float("WRITER_TEMPERATURE", 0.1, min_val=0.0, max_val=2.0)

    # ── Retrieval / chunking ──────────────────────────────────────────────
    MAX_CHUNKS_TO_WRITER: int = _get_int("MAX_CHUNKS_TO_WRITER", 12, min_val=1, max_val=200)
    MAX_CHUNKS_FOR_FULL_SECTION: int = _get_int("MAX_CHUNKS_FOR_FULL_SECTION", 35, min_val=1, max_val=500)
    HYBRID_SEARCH_K: int = _get_int("HYBRID_SEARCH_K", 20, min_val=1, max_val=200)
    MIN_PER_SOURCE: int = _get_int("MIN_PER_SOURCE", 2, min_val=0, max_val=50)

    # ── PDF indexing ──────────────────────────────────────────────────────
    PDF_MAX_WORKERS: int = _get_int("PDF_MAX_WORKERS", 4, min_val=1, max_val=32)
    HEADING_MIN_SIZE: float = _get_float("HEADING_MIN_SIZE", 15.0, min_val=1.0, max_val=200.0)
    HEADING_TOP_FRACTION: float = _get_float("HEADING_TOP_FRACTION", 0.35, min_val=0.0, max_val=1.0)

    # ── Chat history ──────────────────────────────────────────────────────
    CHAT_HISTORY_TURNS: int = _get_int("CHAT_HISTORY_TURNS", 5, min_val=0, max_val=100)

    # ── LangSmith ─────────────────────────────────────────────────────────
    LANGSMITH_ENABLED: bool = _get_bool("LANGCHAIN_TRACING_V2", False)
    LANGCHAIN_API_KEY: str = _get_str("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = _get_str("LANGCHAIN_PROJECT", "default")

    # ── External API keys (centralized — was scattered os.getenv() calls) ──
    GROQ_API_KEY: str = _get_str("GROQ_API_KEY", "")
    TAVILY_API_KEY: str = _get_str("TAVILY_API_KEY", "")

    # ── Paths (centralized — was hardcoded in knowledge_base.py/retriever.py)
    KNOWLEDGE_BASE_DIR: str = _get_str("KNOWLEDGE_BASE_DIR", "knowledge_base")
    PERMANENT_STORE_DIR: str = _get_str("PERMANENT_STORE_DIR", "permanent_store")
    VECTORSTORE_CACHE_DIR: str = _get_str("VECTORSTORE_CACHE_DIR", "vectorstore_cache")
    EMBEDDING_MODEL: str = _get_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # ── Groq models — configurable via env as a JSON string, else default ──
    _default_groq_models = [
        {"name": "openai/gpt-oss-120b", "max_context_chars": 25000},
        {"name": "llama-3.3-70b-versatile", "max_context_chars": 20000},
        {"name": "openai/gpt-oss-20b", "max_context_chars": 12000},
    ]

    @property
    def GROQ_MODELS(self):
        raw = os.getenv("GROQ_MODELS_JSON")
        if raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"[config] GROQ_MODELS_JSON is invalid JSON ({e}) — falling back to defaults")
                return self._default_groq_models
            validated = _validate_groq_models(parsed, "GROQ_MODELS_JSON")
            if validated is not None:
                return validated
            return self._default_groq_models
        return self._default_groq_models

    # ── Cost tracking — $ per 1M tokens, per model ──────────────────────────
    # WHY: agent.py was only logging raw token COUNTS (prompt_tokens /
    # completion_tokens) — useful but not the actual thing a manager asks
    # ("kitna kharcha aa raha hai"). These are APPROXIMATE Groq list-price
    # numbers as placeholders — update via GROQ_PRICING_JSON env var (or
    # edit _default_groq_pricing) to match your actual current Groq pricing,
    # since provider pricing changes over time and this file can't verify
    # it live.
    _default_groq_pricing = {
        "openai/gpt-oss-120b":       {"input_per_1m": 0.15, "output_per_1m": 0.75},
        "llama-3.3-70b-versatile":   {"input_per_1m": 0.59, "output_per_1m": 0.79},
        "openai/gpt-oss-20b":        {"input_per_1m": 0.10, "output_per_1m": 0.50},
    }

    @property
    def GROQ_PRICING(self):
        raw = os.getenv("GROQ_PRICING_JSON")
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and all(
                    isinstance(v, dict) and "input_per_1m" in v and "output_per_1m" in v
                    for v in parsed.values()
                ):
                    return parsed
                logger.warning("[config] GROQ_PRICING_JSON has invalid shape — falling back to defaults")
            except json.JSONDecodeError as e:
                logger.warning(f"[config] GROQ_PRICING_JSON is invalid JSON ({e}) — falling back to defaults")
        return self._default_groq_pricing

    def estimate_cost_usd(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Best-effort $ cost estimate for one LLM call. Returns 0.0 if model pricing unknown."""
        pricing = self.GROQ_PRICING.get(model_name)
        if not pricing:
            return 0.0
        cost = (prompt_tokens / 1_000_000) * pricing["input_per_1m"] + \
               (completion_tokens / 1_000_000) * pricing["output_per_1m"]
        return round(cost, 6)


    # ── Startup diagnostics ───────────────────────────────────────────────

    def validate_startup(self) -> list[str]:
        """
        Call once at app entrypoint (after setup_logging()). Loudly warns
        about missing/misconfigured critical config instead of letting
        problems surface as confusing errors deep in a query path later.
        Returns the list of warning strings (also logged).
        """
        warnings = []

        if not self.GROQ_API_KEY:
            warnings.append("GROQ_API_KEY is not set — all LLM calls will fail.")

        if self.LANGSMITH_ENABLED and not self.LANGCHAIN_API_KEY:
            warnings.append(
                "LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is missing — "
                "traces/feedback will silently NOT be sent to LangSmith "
                "(this was the exact root cause behind the 'no run_id' bug)."
            )

        if not self.TAVILY_API_KEY:
            warnings.append(
                "TAVILY_API_KEY is not set — web search will fall back to DuckDuckGo."
            )

        for w in warnings:
            logger.warning(f"[config] STARTUP CHECK: {w}")

        if not warnings:
            logger.info("[config] Startup validation passed — no critical config issues found.")

        return warnings

    def as_dict(self, mask_secrets: bool = True) -> dict:
        """Full effective config snapshot — useful for a startup log line or a debug endpoint."""
        def mask(v):
            if not v:
                return v
            return v[:4] + "…" if mask_secrets else v

        return {
            "MAX_REACT_ITERATIONS": self.MAX_REACT_ITERATIONS,
            "RETRY_LIMIT": self.RETRY_LIMIT,
            "RETRY_WAIT_SEC": self.RETRY_WAIT_SEC,
            "WRITER_MAX_TOKENS": self.WRITER_MAX_TOKENS,
            "REFINE_MAX_TOKENS": self.REFINE_MAX_TOKENS,
            "WRITER_TEMPERATURE": self.WRITER_TEMPERATURE,
            "MAX_CHUNKS_TO_WRITER": self.MAX_CHUNKS_TO_WRITER,
            "MAX_CHUNKS_FOR_FULL_SECTION": self.MAX_CHUNKS_FOR_FULL_SECTION,
            "HYBRID_SEARCH_K": self.HYBRID_SEARCH_K,
            "MIN_PER_SOURCE": self.MIN_PER_SOURCE,
            "PDF_MAX_WORKERS": self.PDF_MAX_WORKERS,
            "HEADING_MIN_SIZE": self.HEADING_MIN_SIZE,
            "HEADING_TOP_FRACTION": self.HEADING_TOP_FRACTION,
            "CHAT_HISTORY_TURNS": self.CHAT_HISTORY_TURNS,
            "LANGSMITH_ENABLED": self.LANGSMITH_ENABLED,
            "LANGCHAIN_PROJECT": self.LANGCHAIN_PROJECT,
            "GROQ_API_KEY": mask(self.GROQ_API_KEY),
            "TAVILY_API_KEY": mask(self.TAVILY_API_KEY),
            "LANGCHAIN_API_KEY": mask(self.LANGCHAIN_API_KEY),
            "KNOWLEDGE_BASE_DIR": self.KNOWLEDGE_BASE_DIR,
            "PERMANENT_STORE_DIR": self.PERMANENT_STORE_DIR,
            "VECTORSTORE_CACHE_DIR": self.VECTORSTORE_CACHE_DIR,
            "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
            "GROQ_MODELS": self.GROQ_MODELS,
        }


settings = Settings()