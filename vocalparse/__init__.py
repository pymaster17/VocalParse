# coding=utf-8
# VocalParse Package
#
# Unified singing voice transcription with Large Audio Language Models.

from vocalparse.tokens import get_token_maps
from vocalparse.prompts import (
    convert_annotation_to_syllables,
    expand_syllables,
    extract_lyrics_text,
    filter_lyrics_syllables,
    build_interleaved_text,
    build_prefix_text,
)
from vocalparse.checkpoint import (
    find_latest_checkpoint,
    copy_required_hf_files,
    MakeEveryCheckpointInferableCallback,
)
from vocalparse.model import (
    patch_outer_forward,
    _get_encoder_output_length,
    load_audio,
    register_vocalparse_tokens,
)
