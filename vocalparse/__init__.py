# coding=utf-8
# VocalParse Package
#
# Unified singing voice transcription with Large Audio Language Models.
#
# Pure-Python helpers (prompts / tokens / evaluation) are eager-imported so
# downstream consumers that only want AST parsing or metric scoring can
# `import vocalparse.evaluation` without pulling in transformers / torch /
# qwen-asr. Heavy symbols (model loading, the batch transcribe API, the
# single-sample demo) are exposed via PEP 562 ``__getattr__`` and only
# resolved on first access — that's what lets external programs install
# the package with ``pip install --no-deps .`` and use the parsing layer
# without an ML stack.

# ── Eager: pure-Python helpers (zero heavy deps) ─────────────────────
from vocalparse.tokens import get_token_maps
from vocalparse.prompts import (
    convert_annotation_to_syllables,
    expand_syllables,
    extract_lyrics_text,
    filter_lyrics_syllables,
    build_interleaved_text,
    build_prefix_text,
)
from vocalparse.evaluation import (
    parse_transcription_text,
    aggregate_to_words,
    ASTWord,
    SILENCE_WORDS,
    compute_metrics,
    aggregate_metrics,
)


# ── Lazy: heavy modules resolved on first attribute access ───────────
_LAZY = {
    "VocalParseTranscriber":             ("vocalparse.api",        "VocalParseTranscriber"),
    "transcribe_one":                    ("vocalparse.demo",       "transcribe_one"),
    "load_model":                        ("vocalparse.model",      "load_model"),
    "load_audio":                        ("vocalparse.model",      "load_audio"),
    "patch_outer_forward":               ("vocalparse.model",      "patch_outer_forward"),
    "register_vocalparse_tokens":        ("vocalparse.model",      "register_vocalparse_tokens"),
    "_get_encoder_output_length":        ("vocalparse.model",      "_get_encoder_output_length"),
    "find_latest_checkpoint":            ("vocalparse.checkpoint", "find_latest_checkpoint"),
    "copy_required_hf_files":            ("vocalparse.checkpoint", "copy_required_hf_files"),
    "MakeEveryCheckpointInferableCallback":
        ("vocalparse.checkpoint", "MakeEveryCheckpointInferableCallback"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod_path, attr = _LAZY[name]
        value = getattr(importlib.import_module(mod_path), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'vocalparse' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    # Pure-Python (eager)
    "get_token_maps",
    "convert_annotation_to_syllables", "expand_syllables",
    "extract_lyrics_text", "filter_lyrics_syllables",
    "build_interleaved_text", "build_prefix_text",
    "parse_transcription_text", "aggregate_to_words",
    "ASTWord", "SILENCE_WORDS",
    "compute_metrics", "aggregate_metrics",
    # Lazy (require torch / transformers / qwen-asr)
    *_LAZY.keys(),
]
