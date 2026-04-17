#!/usr/bin/env python3
# coding=utf-8
"""
VocalParse Data Preprocessor

Precomputes mel spectrograms and tokenized text, saving to Arrow format
for fast loading during training. Eliminates NFS audio reads and online
mel computation at training time.

Key optimizations:
  - Only stores ACTUAL mel frames (not 30s-padded), reducing storage ~5-10x
  - Threading-based parallelism (avoids fork deadlocks)
  - Incremental shard writing

Usage:
    python preprocess.py --config preprocess.yaml
    python preprocess.py --config preprocess.yaml --num_workers 16
"""

import argparse
import json
import os
from pathlib import Path

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import librosa
import numpy as np
import yaml
from tqdm import tqdm

# ── Reuse data loading from training script ────────────────────
from vocalparse.data import load_all_datasets


def _process_one(sample: Dict, feature_extractor, sr: int) -> dict:
    """Process a single sample: extract mel + serialize metadata (called in a thread).

    Stores only raw metadata (syllables, bpm) — no tokenization.
    Prompt construction is deferred to training-time collator.
    """
    audio_path = sample["audio_path"]
    dataset_name = sample.get("dataset_name", "unknown")

    if not os.path.isfile(audio_path):
        return {"status": "missing"}

    try:
        # 1. Load audio and compute mel spectrogram
        wav, _ = librosa.load(audio_path, sr=sr, mono=True)
        audio_inputs = feature_extractor(
            wav, sampling_rate=sr, return_tensors="np",
            return_attention_mask=True,
        )
        mel = audio_inputs["input_features"][0]
        mel_mask = audio_inputs["attention_mask"][0]

        # Only store actual frames, not 30s padding
        actual_frames = int(mel_mask.sum())
        if actual_frames == 0:
            actual_frames = mel.shape[-1]
        mel_trimmed = mel[:, :actual_frames].T  # (actual_frames, n_mels)

        # 2. Serialize syllables as JSON string
        import json as json_lib
        syllables_json = json_lib.dumps(sample["syllables"], ensure_ascii=False)

        return {
            "status": "ok",
            "input_features": mel_trimmed.astype(np.float16).flatten(),
            "mel_frames": actual_frames,
            "mel_bins": mel.shape[0],
            "syllables_json": syllables_json,
            "bpm": int(sample["bpm"]),
            "dataset_name": dataset_name,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def preprocess_and_save(
    samples: List[Dict],
    feature_extractor,
    output_dir: Path,
    shard_size: int = 50000,
    sr: int = 16000,
    num_workers: int = 8,
):
    """Extract mel spectrograms and serialize metadata to Arrow.

    Stores only raw metadata (syllables, bpm) — no tokenization or prompt
    construction.  Prompt format is decided at training time.

    Only stores actual mel frames (not 30s-padded), reducing storage ~5-10x.
    Uses ThreadPoolExecutor for safe parallelism.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc
    from datasets import Features, Sequence, Value

    output_dir.mkdir(parents=True, exist_ok=True)

    fe = feature_extractor

    # HuggingFace features definition (for dataset_info.json)
    hf_features = Features({
        "input_features": Sequence(Value("float16")),  # flat mel array (frames * bins), half-precision to save space
        "mel_frames": Value("int32"),
        "mel_bins": Value("int32"),
        "syllables_json": Value("string"),             # JSON-serialized syllables
        "bpm": Value("int32"),
        "dataset_name": Value("string"),
    })
    arrow_schema = hf_features.arrow_schema

    # ── Incremental ShardWriter ────────────────────────────────────
    class ShardWriter:
        """Writes Arrow RecordBatches to IPC stream files with shard rotation."""
        def __init__(self):
            self.shard_idx = 0
            self.shard_sample_count = 0
            self.shard_paths: List[str] = []
            self._file = None
            self._writer = None
            self._open_new_shard()

        def _open_new_shard(self):
            shard_path = output_dir / f"data-{self.shard_idx:05d}.arrow"
            self.shard_paths.append(str(shard_path))
            self._file = pa.OSFile(str(shard_path), 'wb')
            self._writer = ipc.new_stream(self._file, arrow_schema)

        def write_batch(self, batch_dict: Dict, n_samples: int):
            if n_samples == 0:
                return
            # Rotate shard if needed
            if self.shard_sample_count > 0 and self.shard_sample_count + n_samples > shard_size:
                self._close_current_shard()
                self.shard_idx += 1
                self._open_new_shard()
            # Write RecordBatch
            arrays = [
                pa.array(batch_dict[field.name], type=field.type)
                for field in arrow_schema
            ]
            rb = pa.RecordBatch.from_arrays(arrays, schema=arrow_schema)
            self._writer.write_batch(rb)
            self.shard_sample_count += n_samples

        def _close_current_shard(self):
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            if self._file is not None:
                self._file.close()
                self._file = None
            print(f"  Wrote shard {self.shard_idx}: {self.shard_sample_count} samples")
            self.shard_sample_count = 0

        def close(self):
            self._close_current_shard()

    shard_writer = ShardWriter()
    errors = 0
    skipped_missing = 0
    total_ok = 0
    total_mel_frames = 0

    print(f"\nPreprocessing {len(samples)} samples with {num_workers} threads...")
    print(f"(Only storing actual mel frames, not 30s-padded)")
    print(f"Incremental shard writing, shard_size={shard_size}")
    t0 = time.time()

    SUBMIT_BATCH = 10000
    pbar = tqdm(total=len(samples), desc="Mel extraction", unit="samples")

    # Accumulate a small batch, then flush to ShardWriter
    batch_buf = {col: [] for col in arrow_schema.names}

    def _flush_batch():
        nonlocal total_ok
        n = len(batch_buf["mel_frames"])
        if n > 0:
            shard_writer.write_batch(batch_buf, n)
            total_ok += n
            for col in batch_buf:
                batch_buf[col].clear()

    i = 0
    while i < len(samples):
        batch_end = min(i + SUBMIT_BATCH, len(samples))
        futures = {}
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for j in range(i, batch_end):
                fut = executor.submit(
                    _process_one, samples[j], fe, sr,
                )
                futures[fut] = j

            for fut in as_completed(futures):
                result = fut.result()
                pbar.update(1)

                if result["status"] == "missing":
                    skipped_missing += 1
                elif result["status"] == "error":
                    errors += 1
                    if errors <= 10:
                        print(f"\n  Warning: sample {futures[fut]} failed: {result['error']}")
                else:
                    for col in arrow_schema.names:
                        batch_buf[col].append(result[col])
                    total_mel_frames += result["mel_frames"]

        # Flush after each SUBMIT_BATCH round
        _flush_batch()
        i = batch_end

    pbar.close()

    # Close shard writer (flushes last shard)
    shard_writer.close()

    elapsed = time.time() - t0
    num_shards = shard_writer.shard_idx + 1
    shard_paths = shard_writer.shard_paths

    # ── Generate HuggingFace dataset metadata ─────────────────────────
    # dataset_info.json
    dataset_info = {
        "description": "VocalParse preprocessed dataset",
        "features": hf_features.to_dict(),
        "num_rows": total_ok,
        "num_shards": num_shards,
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    # state.json (for datasets library load_from_disk compatibility)
    state = {
        "_data_files": [{"filename": Path(p).name} for p in shard_paths],
        "_fingerprint": None,
        "_format_columns": None,
        "_format_kwargs": {},
        "_format_type": None,
        "_output_all_columns": False,
        "_split": None,
    }
    with open(output_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)

    # Compute storage stats
    total_bytes = sum(
        f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
    )
    avg_frames = total_mel_frames / max(total_ok, 1)

    metadata = {
        "num_samples": total_ok,
        "num_shards": num_shards,
        "errors": errors,
        "skipped_missing": skipped_missing,
        "elapsed_seconds": round(elapsed, 1),
        "format": "hf_dataset",
        "sample_rate": sr,
        "num_workers": num_workers,
        "total_bytes": total_bytes,
        "avg_mel_frames": round(avg_frames, 1),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    speed = total_ok / max(elapsed, 1)
    print(f"\nDone in {elapsed:.1f}s ({speed:.0f} samples/s)")
    print(f"  Total OK: {total_ok}")
    print(f"  Errors: {errors}, Missing audio: {skipped_missing}")
    print(f"  Shards: {num_shards}")
    print(f"  Total size: {total_bytes / 1024**3:.2f} GB")
    print(f"  Avg mel frames/sample: {avg_frames:.0f} (vs 3000 if 30s-padded)")
    print(f"  Output: {output_dir} (HuggingFace Dataset, memory-mappable)")


def main():
    parser = argparse.ArgumentParser("VocalParse Data Preprocessor")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file (e.g. preprocess.yaml)")
    parser.add_argument("--output_dir", type=str, default="",
                        help="Output directory for Arrow shards")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples to process (-1 for all)")
    parser.add_argument("--shard_size", type=int, default=5000,
                        help="Samples per Arrow shard file")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Audio sample rate")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel threads for mel extraction")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    datasets_config = config.get("datasets", [])
    if not datasets_config:
        raise ValueError("No datasets in config.")

    max_samples = args.max_samples if args.max_samples != -1 else config.get("max_samples", -1)

    output_dir = args.output_dir or config.get("output_dir", "")
    if not output_dir:
        output_dir = str(Path(args.config).parent / "vocalparse-preprocessed")
    output_dir = Path(output_dir)

    model_path = config.get("model_path", "Qwen/Qwen3-ASR-0.6B")
    print("=== VocalParse Data Preprocessor ===")
    print(f"Config: {args.config}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Max samples: {max_samples}")
    print(f"Workers: {args.num_workers}")
    print()

    from qwen_asr import Qwen3ASRModel
    print("Loading feature extractor...")
    asr_wrapper = Qwen3ASRModel.from_pretrained(model_path, device_map=None)
    feature_extractor = asr_wrapper.processor.feature_extractor

    # Load all datasets
    all_samples = load_all_datasets(datasets_config, max_samples=max_samples)
    if not all_samples:
        raise ValueError("No samples loaded.")

    # Process and save
    preprocess_and_save(
        samples=all_samples,
        feature_extractor=feature_extractor,
        output_dir=output_dir,
        shard_size=args.shard_size,
        sr=args.sr,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
