# coding=utf-8
# VocalParse Inference Script
#
# Supports both preprocessed Arrow datasets and raw audio JSON lists.
#
# Usage (single GPU):
#   python -m vocalparse.inference --config configs/inference.yaml
#
# Usage (multi GPU):
#   torchrun --nproc_per_node=2 -m vocalparse.inference --config configs/inference.yaml

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from vocalparse.distributed import (
    init_distributed,
    cleanup_distributed,
    gather_results_via_shm,
    pre_encode_audio_features,
    pack_batches,
    left_pad_input_ids,
)





# ══════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════

def _detect_base_model_path(checkpoint_dir: str) -> str:
    """Try to find the original base model path from training artifacts.

    Checks (in order):
    1. training_args.bin → _model_path field (set by our training script)
    2. Falls back to well-known Qwen3-ASR model names by inspecting config.json
    """
    # Try training_args.bin
    args_path = os.path.join(checkpoint_dir, "training_args.bin")
    if os.path.exists(args_path):
        try:
            training_args = torch.load(args_path, map_location="cpu")
            # Our MakeEveryCheckpointInferableCallback doesn't store this,
            # but TrainingArguments has _name_or_path from the model config
            if hasattr(training_args, "_name_or_path"):
                return training_args._name_or_path
        except Exception:
            pass

    # Inspect config.json for model type hints
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        import json as _json
        with open(config_path, "r") as f:
            cfg_data = _json.load(f)
        # Check hidden_size to distinguish 0.6B vs 1.7B
        # Qwen3-ASR config nests under thinker_config.text_config
        thinker_cfg = cfg_data.get("thinker_config", {})
        text_cfg = thinker_cfg.get("text_config", cfg_data.get("text_config", {}))
        hidden_size = text_cfg.get("hidden_size", 0)
        if hidden_size <= 1024:
            return "Qwen/Qwen3-ASR-0.6B"
        else:
            return "Qwen/Qwen3-ASR-1.7B"

    return "Qwen/Qwen3-ASR-1.7B"


def load_model(cfg):
    """Load model and processor from a single checkpoint path.

    Since training checkpoints don't contain preprocessor_config.json
    (needed by the feature extractor), we load the processor from the
    original base model and then load fine-tuned weights from the checkpoint.
    """
    from qwen_asr import Qwen3ASRModel
    from transformers import GenerationConfig
    from safetensors.torch import load_file
    from vocalparse.model import patch_outer_forward, register_vocalparse_tokens
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

    # Detect base model for loading processor (has preprocessor_config.json)
    base_model = _detect_base_model_path(checkpoint)
    print(f"Loading processor from base model: {base_model}")
    print(f"Loading weights from checkpoint: {checkpoint}")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Prefer flash_attention_2 (padding-free, faster) → sdpa → eager fallback
    attn_impl = cfg.get("attn_implementation", "flash_attention_2")

    # Load base model (architecture + processor)
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

    # Register AST tokens
    register_vocalparse_tokens(processor, model)

    # Load fine-tuned weights from checkpoint
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


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════

def load_data(cfg):
    """Load validation dataset from preprocessed data."""
    from vocalparse.data import load_from_preprocessed

    preprocessed_dir = cfg["preprocessed_dir"]
    val_datasets = cfg.get("val_datasets", [])
    num_samples = int(cfg.get("num_samples", -1))

    print(f"\nLoading data from: {preprocessed_dir}")
    print(f"Validation datasets: {val_datasets}")

    datasets = load_from_preprocessed(
        preprocessed_dir=preprocessed_dir,
        processor=None,
        val_datasets=val_datasets,
        eval_file="",
    )

    val_ds = datasets.get("validation", None)
    if val_ds is None:
        raise ValueError(
            f"No validation data found. Check val_datasets={val_datasets}."
        )

    if num_samples > 0:
        val_ds = val_ds.select(range(min(num_samples, len(val_ds))))

    print(f"Validation samples: {len(val_ds)}")
    return val_ds


def load_raw_audio_data(cfg):
    """Load audio file list from a JSON file.

    The JSON file can be:
    - A flat list of audio paths:  ["/path/a.wav", "/path/b.flac", ...]
    - A list of objects:           [{"audio_path": "/path/a.wav", ...}, ...]

    An optional ``audio_root`` in cfg is prepended to relative paths.
    """
    audio_json = cfg["audio_json"]
    audio_root = cfg.get("audio_root", "")
    num_samples = int(cfg.get("num_samples", -1))

    with open(audio_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    samples = []
    for item in items:
        if isinstance(item, str):
            audio_path = item
            gt_text = None
        elif isinstance(item, dict):
            audio_path = item.get("audio_path") or item.get("audio", "")
            gt_text = item.get("text") or item.get("lyrics") or None
        else:
            continue

        if not audio_path:
            continue

        # Resolve relative paths
        if audio_root and not os.path.isabs(audio_path):
            audio_path = os.path.join(audio_root, audio_path)

        if not os.path.isfile(audio_path):
            print(f"  WARNING: audio file not found, skipping: {audio_path}")
            continue

        entry = {"audio_path": audio_path}
        if gt_text:
            entry["gt_text"] = gt_text
        samples.append(entry)

    if num_samples > 0:
        samples = samples[:num_samples]

    print(f"\nLoaded {len(samples)} audio files from: {audio_json}")
    return samples


# ══════════════════════════════════════════════════════════════════════
# Batched generation
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_batch(model, tokenizer, samples, device, cfg):
    """Run batched generation on a list of samples."""
    from vocalparse.model import _get_encoder_output_length
    from vocalparse.prompts import build_interleaved_text

    bpm_position = cfg.get("bpm_position", "last")
    include_dur = cfg.get("include_dur", False)
    max_new_tokens = int(cfg.get("max_new_tokens", 512))
    prefix_text = cfg["_prefix_text"]

    base_prefix_ids = tokenizer(
        prefix_text, return_tensors="np", add_special_tokens=False,
    )["input_ids"][0].astype(np.int64)
    audio_token_id = 151676

    batch_mels, batch_mel_lens = [], []
    batch_ids, batch_prefix_lens = [], []
    gt_texts = []

    for sample in samples:
        mel_frames = int(sample["mel_frames"])
        mel_bins = int(sample["mel_bins"])

        mel_np = np.array(sample["input_features"], dtype=np.float16, copy=True)
        mel = torch.from_numpy(mel_np.reshape(mel_frames, mel_bins)).T
        batch_mels.append(mel)
        batch_mel_lens.append(mel_frames)

        syllables = json.loads(sample["syllables_json"])
        bpm = int(sample["bpm"])

        prompt_syllables = syllables

        ast_text = build_interleaved_text(
            syllables=prompt_syllables, bpm=bpm,
            bpm_position=bpm_position,
            include_dur=include_dur,
        )
        gt_texts.append(f"language Chinese<asr_text>{ast_text}")

        # Build prefix — per-sample if audio-lyric mode
        inference_mode = cfg.get("inference_mode", "audio-only")
        if inference_mode == "audio-lyric":
            from vocalparse.prompts import extract_lyrics_text
            lyrics = extract_lyrics_text(syllables)
            sample_prefix = prefix_text + f"language Chinese<asr_text>{lyrics}<|file_sep|>"
        else:
            sample_prefix = prefix_text

        prefix_ids_np = tokenizer(
            sample_prefix, return_tensors="np", add_special_tokens=False,
        )["input_ids"][0].astype(np.int64)
        encoder_len = _get_encoder_output_length(mel_frames)
        audio_positions = np.where(prefix_ids_np == audio_token_id)[0]
        prefix_len = len(prefix_ids_np)
        if len(audio_positions) == 1 and encoder_len > 1:
            pos = audio_positions[0]
            prefix_ids_np = np.concatenate([
                prefix_ids_np[:pos],
                np.full(encoder_len, audio_token_id, dtype=np.int64),
                prefix_ids_np[pos + 1:],
            ])
            prefix_len += (encoder_len - 1)

        batch_ids.append(torch.from_numpy(prefix_ids_np))
        batch_prefix_lens.append(prefix_len)

    # Left-pad input_ids
    pad_token_id = tokenizer.pad_token_id or 151643
    padded_ids, attention_mask, pad_offsets = left_pad_input_ids(batch_ids, pad_token_id)

    padded_ids = padded_ids.to(device)
    attention_mask = attention_mask.to(device)

    # ── Per-sample audio encoding (precision-safe) ────────────
    inputs_embeds = pre_encode_audio_features(
        model, padded_ids, batch_mels, batch_mel_lens, device,
    )

    # ── Generate ──────────────────────────────────────────────
    outputs = model.generate(
        input_ids=padded_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
    )
    gen_seqs = outputs.sequences if hasattr(outputs, "sequences") else outputs

    results = []
    for i in range(len(samples)):
        actual_start = pad_offsets[i] + batch_prefix_lens[i]
        pred_tokens = gen_seqs[i][actual_start:]
        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=False)
        results.append((pred_text, gt_texts[i]))
    return results



@torch.no_grad()
def generate_batch_raw(model, processor, samples, device, cfg):
    """Run generation on a batch of raw audio files.

    Uses per-sample audio encoding via pre_encode_audio_features() to avoid
    cross-sample mel padding precision loss in Conv2d layers, matching the
    preprocessed mode's approach.
    """
    from vocalparse.model import load_audio

    max_new_tokens = int(cfg.get("max_new_tokens", 512))
    tokenizer = processor.tokenizer

    # Process each sample individually to avoid batched mel padding
    batch_mels = []
    batch_mel_lens = []
    batch_ids = []
    batch_prefix_lens = []

    for sample in samples:
        audio_path = sample["audio_path"]
        audio = load_audio(audio_path, sr=16000)

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": None}]},
        ]
        text = processor.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=False
        )[0]

        # Audio-lyric mode: prepend GT lyrics as part of the generation prefix
        inference_mode = cfg.get("inference_mode", "audio-only")
        if inference_mode == "audio-lyric":
            gt_text = sample.get("gt_text", "")
            text = text + f"language Chinese<asr_text>{gt_text}<|file_sep|>"

        # Process single sample to get mel features and input_ids
        single_inputs = processor(
            text=[text],
            audio=[audio],
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        # Extract mel: (1, n_mels, mel_frames) -> (n_mels, mel_frames)
        mel = single_inputs["input_features"][0]
        mel_frames = mel.shape[1]
        batch_mels.append(mel)
        batch_mel_lens.append(mel_frames)

        # Extract input_ids: (1, seq_len) -> (seq_len,)
        ids = single_inputs["input_ids"][0]
        batch_ids.append(ids)
        batch_prefix_lens.append(len(ids))

    # Left-pad input_ids
    pad_token_id = tokenizer.pad_token_id or 151643
    padded_ids, attention_mask, pad_offsets = left_pad_input_ids(
        batch_ids, pad_token_id,
    )
    padded_ids = padded_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Per-sample audio encoding (precision-safe, same as preprocessed mode)
    inputs_embeds = pre_encode_audio_features(
        model, padded_ids, batch_mels, batch_mel_lens, device,
    )

    # Generate
    outputs = model.generate(
        input_ids=padded_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
    )
    gen_seqs = outputs.sequences if hasattr(outputs, "sequences") else outputs
    results = []
    for i in range(len(samples)):
        actual_start = pad_offsets[i] + batch_prefix_lens[i]
        pred_tokens = gen_seqs[i][actual_start:]
        pred_text = tokenizer.decode(
            pred_tokens, skip_special_tokens=False
        )
        results.append(pred_text)
    return results


# ══════════════════════════════════════════════════════════════════════
# Prefetched batch pipeline
# ══════════════════════════════════════════════════════════════════════

def _prepare_batch_tensors(tokenizer, samples, cfg):
    """Pre-compute batch tensors on CPU.

    This function performs all CPU-bound work (JSON parsing, tokenization,
    mel reconstruction, padding) and returns pinned CPU tensors ready for
    fast asynchronous GPU transfer.  It is designed to run in a background
    thread so that CPU preparation of batch N+1 overlaps with GPU
    generation of batch N.
    """
    from vocalparse.model import _get_encoder_output_length
    from vocalparse.prompts import build_interleaved_text

    bpm_position = cfg.get("bpm_position", "last")
    include_dur = cfg.get("include_dur", False)
    prefix_text = cfg["_prefix_text"]

    base_prefix_ids = tokenizer(
        prefix_text, return_tensors="np", add_special_tokens=False,
    )["input_ids"][0].astype(np.int64)
    audio_token_id = 151676

    batch_mels, batch_mel_lens = [], []
    batch_ids, batch_prefix_lens = [], []
    gt_texts = []

    for sample in samples:
        mel_frames = int(sample["mel_frames"])
        mel_bins = int(sample["mel_bins"])

        mel_np = np.array(sample["input_features"], dtype=np.float16, copy=True)
        mel = torch.from_numpy(mel_np.reshape(mel_frames, mel_bins)).T
        batch_mels.append(mel)
        batch_mel_lens.append(mel_frames)

        syllables = json.loads(sample["syllables_json"])
        bpm = int(sample["bpm"])

        prompt_syllables = syllables

        ast_text = build_interleaved_text(
            syllables=prompt_syllables, bpm=bpm,
            bpm_position=bpm_position,
            include_dur=include_dur,
        )
        gt_texts.append(f"language Chinese<asr_text>{ast_text}")

        # Build prefix — per-sample if audio-lyric mode
        inference_mode = cfg.get("inference_mode", "audio-only")
        if inference_mode == "audio-lyric":
            from vocalparse.prompts import extract_lyrics_text
            lyrics = extract_lyrics_text(syllables)
            sample_prefix = prefix_text + f"language Chinese<asr_text>{lyrics}<|file_sep|>"
        else:
            sample_prefix = prefix_text

        prefix_ids_np = tokenizer(
            sample_prefix, return_tensors="np", add_special_tokens=False,
        )["input_ids"][0].astype(np.int64)
        encoder_len = _get_encoder_output_length(mel_frames)
        audio_positions = np.where(prefix_ids_np == audio_token_id)[0]
        prefix_len = len(prefix_ids_np)
        if len(audio_positions) == 1 and encoder_len > 1:
            pos = audio_positions[0]
            prefix_ids_np = np.concatenate([
                prefix_ids_np[:pos],
                np.full(encoder_len, audio_token_id, dtype=np.int64),
                prefix_ids_np[pos + 1:],
            ])
            prefix_len += (encoder_len - 1)

        batch_ids.append(torch.from_numpy(prefix_ids_np))
        batch_prefix_lens.append(prefix_len)

    # ── Pad mels ────────────────────────────────────────────────
    n_mels = batch_mels[0].shape[0]
    max_mel_frames = max(batch_mel_lens)
    padded_mels = torch.zeros(len(batch_mels), n_mels, max_mel_frames)
    for i, (mel, ml) in enumerate(zip(batch_mels, batch_mel_lens)):
        padded_mels[i, :, :ml] = mel

    mel_lens_t = torch.tensor(batch_mel_lens)
    feature_attention_mask = (
        torch.arange(max_mel_frames).unsqueeze(0) < mel_lens_t.unsqueeze(1)
    ).long()

    # ── Left-pad input_ids ──────────────────────────────────────
    pad_token_id = tokenizer.pad_token_id or 151643
    padded_ids, attention_mask, pad_offsets = left_pad_input_ids(batch_ids, pad_token_id)

    actual_prefix_lens = [
        pad_offsets[i] + batch_prefix_lens[i] for i in range(len(samples))
    ]

    # ── Pin memory for async H2D transfer ───────────────────────
    if torch.cuda.is_available():
        padded_mels = padded_mels.pin_memory()
        padded_ids = padded_ids.pin_memory()
        attention_mask = attention_mask.pin_memory()
        feature_attention_mask = feature_attention_mask.pin_memory()

    return {
        "padded_mels": padded_mels,
        "padded_ids": padded_ids,
        "attention_mask": attention_mask,
        "feature_attention_mask": feature_attention_mask,
        "actual_prefix_lens": actual_prefix_lens,
        "gt_texts": gt_texts,
        "pad_offsets": pad_offsets,
        "batch_prefix_lens": batch_prefix_lens,
        # Per-sample unpadded mels for precision-safe encoding
        "batch_mels": batch_mels,
        "batch_mel_lens": batch_mel_lens,
    }


# _pre_encode_audio_features moved to inference_utils.pre_encode_audio_features


@torch.no_grad()
def _generate_from_prepared(model, tokenizer, prepared, device, cfg):
    """Run model generation on pre-prepared CPU tensors.

    Uses per-sample audio encoding to avoid padding-induced precision
    errors in the audio encoder's Conv2d layers.  Decoder generation
    remains fully batched for high throughput.
    """
    max_new_tokens = int(cfg.get("max_new_tokens", 512))

    # Non-blocking H2D (effective because tensors are pinned)
    padded_ids = prepared["padded_ids"].to(device, non_blocking=True)
    attention_mask = prepared["attention_mask"].to(device, non_blocking=True)

    gt_texts = prepared["gt_texts"]
    pad_offsets = prepared["pad_offsets"]
    batch_prefix_lens = prepared["batch_prefix_lens"]
    n_samples = len(gt_texts)

    # ── Per-sample audio encoding (precision-safe) ──────────────
    batch_mels = prepared["batch_mels"]
    batch_mel_lens = prepared["batch_mel_lens"]
    inputs_embeds = pre_encode_audio_features(
        model, padded_ids, batch_mels, batch_mel_lens, device,
    )

    # ── Generate (with pre-built inputs_embeds, no input_features) ──
    outputs = model.generate(
        input_ids=padded_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
    )
    gen_seqs = outputs.sequences if hasattr(outputs, "sequences") else outputs

    results = []
    for i in range(n_samples):
        actual_start = pad_offsets[i] + batch_prefix_lens[i]
        pred_tokens = gen_seqs[i][actual_start:]
        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=False)
        results.append((pred_text, gt_texts[i]))
    return results


# ══════════════════════════════════════════════════════════════════════
# Generation pipelines (input-type dependent)
# ══════════════════════════════════════════════════════════════════════

def _generate_raw_audio(cfg, model, processor, device, rank, world_size):
    """Run inference from raw audio files.

    Returns list of result dicts on rank 0, None on other ranks.
    Each result dict: ``{"idx", "pred", "gt_lyrics", "audio_path"}``.
    """
    import soundfile as sf

    batch_mel_tokens = int(cfg.get("batch_mel_tokens", 10000))
    batch_size = int(cfg.get("batch_size", 32))
    inference_mode = cfg.get("inference_mode", "audio-only")

    # ── Load audio list ─────────────────────────────────────────
    samples = load_raw_audio_data(cfg)
    if not samples:
        print("No audio files to process.")
        return [] if rank == 0 else None

    # ── Validate audio-lyric mode has GT text ───────────────────
    if inference_mode == "audio-lyric":
        missing = [s["audio_path"] for s in samples if not s.get("gt_text")]
        if missing:
            raise ValueError(
                f"inference_mode='audio-lyric' requires GT text for all samples, "
                f"but {len(missing)} samples lack 'text'/'lyrics'. "
                f"First missing: {missing[0]}"
            )

    # ── Pre-compute mel_frames ──────────────────────────────────
    MEL_FRAMES_PER_SEC = 100
    for sample in samples:
        try:
            info = sf.info(sample["audio_path"])
            sample["mel_frames"] = int(info.duration * MEL_FRAMES_PER_SEC)
        except Exception as e:
            print(f"  WARNING: cannot read duration for {sample['audio_path']}: {e}")
            sample["mel_frames"] = 0

    # ── Sort and pack batches ───────────────────────────────────
    indexed_samples = list(enumerate(samples))
    batches = pack_batches(indexed_samples, batch_mel_tokens, batch_size)

    if rank == 0:
        print(f"\nPacked {len(samples)} samples into {len(batches)} batches "
              f"(sizes: {[len(b) for b in batches]}), {world_size} GPU(s)")

    my_batches = batches[rank::world_size]

    # ── Generate ────────────────────────────────────────────────
    my_preds = []
    t_start = time.perf_counter()

    for bi, batch in enumerate(my_batches):
        batch_samples = [s for _, s in batch]
        batch_indices = [idx for idx, _ in batch]

        t0 = time.perf_counter()
        try:
            preds = generate_batch_raw(
                model, processor, batch_samples, device, cfg,
            )
            elapsed = time.perf_counter() - t0
            print(f"  [Rank {rank}] Batch {bi+1}/{len(my_batches)}: "
                  f"{len(batch_samples)} samples in {elapsed:.1f}s")

            for idx, pred in zip(batch_indices, preds):
                my_preds.append((idx, pred, samples[idx]["audio_path"]))
        except Exception as e:
            import traceback
            print(f"  [Rank {rank}] Batch {bi+1} failed: {e}")
            traceback.print_exc()

    elapsed_rank = time.perf_counter() - t_start
    print(f"  [Rank {rank}] Generated {len(my_preds)} samples in {elapsed_rank:.1f}s")

    # ── Gather results to rank 0 ────────────────────────────────
    is_distributed = world_size > 1
    if is_distributed:
        gathered = gather_results_via_shm(my_preds, rank, world_size, tag="rawaudio")
        if rank != 0:
            return None
        all_preds = gathered
        print(f"  Gathered {len(all_preds)} results from {world_size} GPUs")
    else:
        my_preds.sort(key=lambda x: x[0])
        all_preds = my_preds

    total_time = time.perf_counter() - t_start
    print(f"\nGenerated {len(all_preds)} samples in {total_time:.1f}s "
          f"({len(all_preds)/max(total_time, 0.001):.1f} samples/s)")

    # ── Normalize to common result format ───────────────────────
    results = []
    for idx, pred, audio_path in all_preds:
        results.append({
            "idx": idx,
            "pred": pred,
            "gt_ast": None,
            "gt_lyrics": samples[idx].get("gt_text"),
            "audio_path": audio_path,
        })
    return results


def _generate_preprocessed(cfg, model, processor, device, rank, world_size):
    """Run inference using preprocessed Arrow dataset.

    Returns list of result dicts on rank 0, None on other ranks.
    Each result dict: ``{"idx", "pred", "gt_ast", "gt_lyrics", "audio_path"}``.
    """
    tokenizer = processor.tokenizer
    batch_mel_tokens = int(cfg.get("batch_mel_tokens", 24000))
    batch_size = int(cfg.get("batch_size", 32))

    # ── Load data ───────────────────────────────────────────────
    val_ds = load_data(cfg)

    # ── Build prefix text (cached in cfg) ───────────────────────
    from vocalparse.prompts import build_prefix_text
    cfg["_prefix_text"] = build_prefix_text(processor)

    # ── Sort and pack batches ───────────────────────────────────
    raw_samples = [(i, val_ds[i]) for i in range(len(val_ds))]
    batches = pack_batches(raw_samples, batch_mel_tokens, batch_size)

    if rank == 0:
        print(f"\nPacked {len(val_ds)} samples into {len(batches)} batches "
              f"(sizes: {[len(b) for b in batches]}), {world_size} GPU(s)")

    my_batches = batches[rank::world_size]

    # ── Generate (with prefetched batch pipeline) ───────────────
    from concurrent.futures import ThreadPoolExecutor

    my_results = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch") as pool:
        next_future = None
        if my_batches:
            first_samples = [s for _, s in my_batches[0]]
            next_future = pool.submit(
                _prepare_batch_tensors, tokenizer, first_samples, cfg,
            )

        for bi, batch in enumerate(my_batches):
            chunk_indices = [idx for idx, _ in batch]

            prepared = next_future.result()

            if bi + 1 < len(my_batches):
                next_samples = [s for _, s in my_batches[bi + 1]]
                next_future = pool.submit(
                    _prepare_batch_tensors, tokenizer, next_samples, cfg,
                )

            t0 = time.perf_counter()
            try:
                chunk_results = _generate_from_prepared(
                    model, tokenizer, prepared, device, cfg,
                )
                elapsed = time.perf_counter() - t0
                print(f"  [Rank {rank}] Batch {bi+1}/{len(my_batches)}: "
                      f"{len(batch)} samples in {elapsed:.1f}s")

                for idx, (pred, gt) in zip(chunk_indices, chunk_results):
                    my_results.append((idx, pred, gt))
            except Exception as e:
                import traceback
                print(f"  [Rank {rank}] Batch {bi+1} failed: {e}")
                traceback.print_exc()

    elapsed_rank = time.perf_counter() - t_start
    print(f"  [Rank {rank}] Generated {len(my_results)} samples in {elapsed_rank:.1f}s")

    # ── Gather results to rank 0 ────────────────────────────────
    is_distributed = world_size > 1
    if is_distributed:
        all_results = gather_results_via_shm(my_results, rank, world_size, tag="preproc")
        if rank != 0:
            return None
        print(f"  Gathered {len(all_results)} results from {world_size} GPUs")
    else:
        my_results.sort(key=lambda x: x[0])
        all_results = my_results

    total_time = time.perf_counter() - t_start
    print(f"\nGenerated {len(all_results)} samples in {total_time:.1f}s "
          f"({len(all_results)/max(total_time, 0.001):.1f} samples/s)")

    # ── Normalize to common result format ───────────────────────
    results = []
    for idx, pred, gt in all_results:
        results.append({
            "idx": idx,
            "pred": pred,
            "gt_ast": gt,
            "gt_lyrics": None,
            "audio_path": None,
        })
    return results



from vocalparse.output import _output_test_weak, _output_test_full, _output_annotation


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def _resolve_mode(cfg, input_type):
    """Resolve the ``mode`` from config.

    Returns the explicit ``mode`` config value, or a sensible default:
      - ``"test_full"`` for preprocessed input
      - ``"test_weak"`` for raw_audio input
    """
    mode = cfg.get("mode")
    if mode:
        return mode
    return "test_full" if input_type == "preprocessed" else "test_weak"


def main():
    # ── Init distributed (no-op if launched with plain python) ──
    rank, world_size = init_distributed()

    parser = argparse.ArgumentParser(description="VocalParse Inference")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # ── Validate required fields ────────────────────────────────
    if not cfg.get("checkpoint"):
        raise ValueError("'checkpoint' is required in config")

    has_preprocessed = bool(cfg.get("preprocessed_dir"))
    has_audio_json = bool(cfg.get("audio_json"))

    if not has_preprocessed and not has_audio_json:
        raise ValueError(
            "Either 'preprocessed_dir' (preprocessed mode) or "
            "'audio_json' (raw audio mode) is required in config."
        )

    # audio_json takes priority when both are present
    input_type = "raw_audio" if has_audio_json else "preprocessed"
    mode = _resolve_mode(cfg, input_type)
    inference_mode = cfg.get("inference_mode", "audio-only")

    # ── Validate mode × input_type ──────────────────────────────
    if mode not in ("test_weak", "test_full", "annotation"):
        raise ValueError(
            f"Invalid mode='{mode}'. "
            f"Must be 'test_weak', 'test_full', or 'annotation'."
        )

    if mode == "test_full" and input_type == "raw_audio":
        raise ValueError(
            "mode='test_full' requires preprocessed input (preprocessed_dir). "
            "Raw audio input only has GT lyrics, not full GT AST "
            "(pitch/note/BPM). Use mode='test_weak' for CER-only evaluation."
        )

    if inference_mode not in ("audio-only", "audio-lyric"):
        raise ValueError(
            f"Invalid inference_mode='{inference_mode}'. "
            f"Must be 'audio-only' or 'audio-lyric'."
        )

    # ── Load model (each rank loads to its own GPU) ────────────
    model, processor, device = load_model(cfg)

    if rank == 0:
        print(f"\nUsing {world_size} GPU(s) for inference")
        print(f"  Input type: {input_type}")
        print(f"  Mode: {mode}")
        print(f"  Inference mode: {inference_mode}")

    # ── Run generation (input-type dependent) ───────────────────
    try:
        if input_type == "raw_audio":
            if rank == 0:
                print("\n" + "=" * 70)
                print("Input: Raw Audio (from JSON file list)")
                print("=" * 70)
            results = _generate_raw_audio(
                cfg, model, processor, device, rank, world_size,
            )
        else:
            if rank == 0:
                print("\n" + "=" * 70)
                print("Input: Preprocessed Dataset")
                print("=" * 70)
            results = _generate_preprocessed(
                cfg, model, processor, device, rank, world_size,
            )

        # Only rank 0 processes output
        if results is None:
            return

        # ── Run output (mode dependent) ─────────────────────────
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Output mode: {mode}")
            print(f"{'='*70}")

        if mode == "test_weak":
            _output_test_weak(results, cfg)
        elif mode == "test_full":
            _output_test_full(results, cfg)
        elif mode == "annotation":
            _output_annotation(results, cfg)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
