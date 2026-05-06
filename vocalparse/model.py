# coding=utf-8
# VocalParse Model Patching and Audio Utilities

import os

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


def _vocalparse_tokens(include_legacy: bool = False):
    pitch_tokens, note_tokens, bpm_tokens, _ = get_token_maps()
    tokens = pitch_tokens + note_tokens + bpm_tokens
    if include_legacy:
        # Legacy compat: pre-e2b4c13 checkpoints carry 1000 <dur_*> and 2
        # <SLUR>/<SVS_MASK> rows in their embedding. Downstream prompts/eval
        # code does not use them, but inference must recreate the old vocab.
        legacy_special_tokens = ["<SLUR>", "<SVS_MASK>"]
        legacy_dur_tokens = [f"<dur_{i / 100:.2f}>" for i in range(1, 1001)]
        tokens += legacy_special_tokens + legacy_dur_tokens
    return tokens


def register_vocalparse_tokens(
    processor,
    model,
    include_legacy: bool = False,
    target_vocab_size: int | None = None,
) -> int:
    """Add AST special tokens to the tokenizer and resize model embeddings."""
    new_tokens = _vocalparse_tokens(include_legacy=include_legacy)
    num_added = processor.tokenizer.add_tokens(new_tokens)
    if target_vocab_size is not None and len(processor.tokenizer) != target_vocab_size:
        raise ValueError(
            "Checkpoint vocab size does not match VocalParse token registration: "
            f"checkpoint={target_vocab_size}, tokenizer={len(processor.tokenizer)}"
        )
    if num_added > 0:
        new_vocab_size = len(processor.tokenizer)
        model.thinker.resize_token_embeddings(new_vocab_size)
        model.thinker.config.text_config.vocab_size = new_vocab_size
        model.thinker.vocab_size = new_vocab_size
        print(f"Added {num_added} AST tokens, vocab size = {new_vocab_size}")
    return num_added


# ══════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════

def _detect_base_model_path(checkpoint_dir: str) -> str:
    """Try to find the original base model path from training artifacts.

    Checks (in order):
    1. training_args.bin → _model_path field (set by our training script)
    2. Falls back to well-known Qwen3-ASR model names by inspecting config.json
    """
    args_path = os.path.join(checkpoint_dir, "training_args.bin")
    if os.path.exists(args_path):
        try:
            training_args = torch.load(args_path, map_location="cpu")
            if hasattr(training_args, "_name_or_path"):
                return training_args._name_or_path
        except Exception:
            pass

    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        import json as _json
        with open(config_path, "r") as f:
            cfg_data = _json.load(f)
        # Qwen3-ASR config nests under thinker_config.text_config; use
        # hidden_size to distinguish 0.6B vs 1.7B.
        thinker_cfg = cfg_data.get("thinker_config", {})
        text_cfg = thinker_cfg.get("text_config", cfg_data.get("text_config", {}))
        hidden_size = text_cfg.get("hidden_size", 0)
        if hidden_size <= 1024:
            return "Qwen/Qwen3-ASR-0.6B"
        else:
            return "Qwen/Qwen3-ASR-1.7B"

    return "Qwen/Qwen3-ASR-1.7B"


def _infer_checkpoint_vocab_size(checkpoint_dir: str) -> int | None:
    """Read the saved embedding/lm-head row count without loading all weights."""
    ckpt_file = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(ckpt_file):
        return None

    try:
        from safetensors import safe_open

        sizes = []
        with safe_open(ckpt_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not (
                    key.endswith("embed_tokens.weight")
                    or key.endswith("lm_head.weight")
                ):
                    continue
                try:
                    shape = f.get_slice(key).get_shape()
                except AttributeError:
                    shape = f.get_tensor(key).shape
                if len(shape) == 2:
                    sizes.append(int(shape[0]))
        return max(sizes) if sizes else None
    except Exception:
        return None


def load_model(cfg):
    """Load model and processor from a single checkpoint path.

    Since training checkpoints don't contain preprocessor_config.json
    (needed by the feature extractor), we load the processor from the
    original base model and then load fine-tuned weights from the checkpoint.
    """
    from qwen_asr import Qwen3ASRModel
    from transformers import GenerationConfig
    from safetensors.torch import load_file
    from vocalparse.checkpoint import find_latest_checkpoint

    checkpoint = cfg["checkpoint"]

    # If checkpoint points to a training output_dir (not a checkpoint-XXXX),
    # auto-detect the latest checkpoint inside it.
    if not os.path.exists(os.path.join(checkpoint, "config.json")):
        latest = find_latest_checkpoint(checkpoint)
        if latest:
            print(f"Auto-detected latest checkpoint: {latest}")
            checkpoint = latest
        else:
            raise FileNotFoundError(
                f"No config.json or checkpoint-* found in '{checkpoint}'. "
                f"Provide a valid checkpoint directory."
            )

    base_model = _detect_base_model_path(checkpoint)
    print(f"Loading processor from base model: {base_model}")
    print(f"Loading weights from checkpoint: {checkpoint}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    attn_impl = cfg.get("attn_implementation", "flash_attention_2")

    try:
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            base_model, dtype=dtype, device_map=None,
            attn_implementation=attn_impl,
        )
        print(f"  Attention implementation: {attn_impl}")
    except Exception as e:
        print(f"  Warning: {attn_impl} not available ({e}), falling back to default")
        asr_wrapper = Qwen3ASRModel.from_pretrained(
            base_model, dtype=dtype, device_map=None,
        )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    checkpoint_vocab_size = _infer_checkpoint_vocab_size(checkpoint)
    include_legacy_tokens = False
    if checkpoint_vocab_size is not None:
        base_vocab_size = len(processor.tokenizer)
        standard_vocab_size = base_vocab_size + len(_vocalparse_tokens(False))
        legacy_vocab_size = base_vocab_size + len(_vocalparse_tokens(True))
        include_legacy_tokens = checkpoint_vocab_size == legacy_vocab_size
        if checkpoint_vocab_size not in (standard_vocab_size, legacy_vocab_size):
            raise ValueError(
                "Unsupported VocalParse checkpoint vocab size: "
                f"checkpoint={checkpoint_vocab_size}, base={base_vocab_size}, "
                f"standard={standard_vocab_size}, legacy={legacy_vocab_size}"
            )

    register_vocalparse_tokens(
        processor,
        model,
        include_legacy=include_legacy_tokens,
        target_vocab_size=checkpoint_vocab_size,
    )

    ckpt_file = os.path.join(checkpoint, "model.safetensors")
    if os.path.exists(ckpt_file):
        state_dict = load_file(ckpt_file)
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded fine-tuned weights from {ckpt_file}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, processor, device
