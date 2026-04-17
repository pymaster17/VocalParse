# coding=utf-8
# VocalParse Model Patching and Audio Utilities

import torch

from vocalparse.tokens import get_token_maps


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


def _get_encoder_output_length(mel_frames: int) -> int:
    """Compute Whisper encoder output token count from mel frame count."""
    input_lengths_leave = mel_frames % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (mel_frames // 100) * 13


def load_audio(path: str, sr: int = 16000):
    import librosa
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def register_vocalparse_tokens(processor, model) -> int:
    """Add AST special tokens to the tokenizer and resize model embeddings."""
    pitch_tokens, note_tokens, bpm_tokens, _ = get_token_maps()
    new_tokens = pitch_tokens + note_tokens + bpm_tokens
    num_added = processor.tokenizer.add_tokens(new_tokens)
    if num_added > 0:
        new_vocab_size = len(processor.tokenizer)
        model.thinker.resize_token_embeddings(new_vocab_size)
        model.thinker.config.text_config.vocab_size = new_vocab_size
        model.thinker.vocab_size = new_vocab_size
        print(f"Added {num_added} AST tokens, vocab size = {new_vocab_size}")
    return num_added
