# coding=utf-8
# AST Data Loading, Dataset Building, and DataCollators
#
# Handles all data I/O for AST fine-tuning:
# - Loading raw samples from folder-based or JSON annotation formats
# - Building HuggingFace Datasets from raw samples
# - Loading preprocessed Arrow datasets (fast path)
# - DataCollators for both raw audio and precomputed mel modes

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset

from vocalparse.prompts import (
    build_interleaved_text,
    convert_annotation_to_syllables,
    extract_lyrics_text,
    build_prefix_text,
)
from vocalparse.model import load_audio, _get_encoder_output_length


# ══════════════════════════════════════════════════════════════════════
# AST data loading
# ══════════════════════════════════════════════════════════════════════


def _process_one_song_folder(args):
    """Process a single song folder — used by multiprocessing pool.

    Returns list of sample dicts for this folder.
    """
    song_folder_str, audio_exts = args
    song_folder = song_folder_str  # keep as string for os.path ops
    samples = []

    # Read BPM once per song folder
    metadata_path = os.path.join(song_folder, "metadata.json")
    bpm = 120
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                bpm = json.load(f).get("bpm", 120)
        except Exception:
            pass

    # Scan audio files with os.scandir (much faster than Path.iterdir)
    try:
        entries = list(os.scandir(song_folder))
    except OSError:
        return samples

    # Build a set of basenames (without extension) that have json annotations
    json_basenames = set()
    audio_entries = []
    for entry in entries:
        if not entry.is_file(follow_symlinks=False):
            continue
        name = entry.name
        _, ext = os.path.splitext(name)
        ext_lower = ext.lower()
        if ext_lower == ".json" and name != "metadata.json":
            json_basenames.add(os.path.splitext(name)[0])
        elif ext_lower in audio_exts:
            audio_entries.append(entry)

    # Only process audio files that have matching .json
    for audio_entry in audio_entries:
        basename = os.path.splitext(audio_entry.name)[0]
        if basename not in json_basenames:
            continue

        json_path = os.path.join(song_folder, basename + ".json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                annotation = json.load(f)
        except Exception:
            continue

        words = annotation.get("word", [])
        if not words:
            continue

        pitches = annotation.get("pitch", [])
        notes = annotation.get("note", [])
        pitch2word = annotation.get("pitch2word", [])

        if not (pitches and notes and pitch2word):
            continue  # skip samples without score annotations

        syllables = convert_annotation_to_syllables(
            words=words, pitches=pitches, notes=notes, pitch2word=pitch2word)

        samples.append({
            "audio_path": audio_entry.path,
            "bpm": bpm,
            "syllables": syllables,
        })

    return samples


def load_samples_from_folder(
    dataset_name: str,
    dataset_root: str,
    max_samples: int = -1,
    audio_extensions: Tuple[str, ...] = (".flac", ".wav", ".mp3"),
    num_workers: int = 8,
) -> List[Dict]:
    """Load raw samples from folder-based structure.

    Optimized with:
    - os.scandir() for fast directory listing
    - Multiprocessing per song folder
    - Pre-indexed JSON basenames to skip stat() calls
    """
    from multiprocessing import Pool
    from tqdm import tqdm

    dataset_root = str(dataset_root)
    audio_exts = set(audio_extensions)

    # Discover song folders
    song_folders = []
    try:
        for entry in os.scandir(dataset_root):
            if entry.is_dir(follow_symlinks=False):
                song_folders.append(entry.path)
    except OSError as e:
        print(f"[{dataset_name}] Error scanning {dataset_root}: {e}")
        return []

    print(f"[{dataset_name}] Scanning {len(song_folders)} song folders...")

    # Process in parallel
    args_list = [(sf, audio_exts) for sf in song_folders]
    samples = []

    effective_workers = min(num_workers, len(song_folders))
    if effective_workers > 1:
        with Pool(effective_workers) as pool:
            for folder_samples in tqdm(
                pool.imap_unordered(_process_one_song_folder, args_list, chunksize=64),
                total=len(song_folders),
                desc=f"  [{dataset_name}]",
                unit="folders",
            ):
                samples.extend(folder_samples)
                if 0 < max_samples <= len(samples):
                    pool.terminate()
                    break
    else:
        for args in tqdm(args_list, desc=f"  [{dataset_name}]", unit="folders"):
            folder_samples = _process_one_song_folder(args)
            samples.extend(folder_samples)
            if 0 < max_samples <= len(samples):
                break

    if 0 < max_samples < len(samples):
        samples = samples[:max_samples]

    print(f"[{dataset_name}] Loaded {len(samples)} samples from folder structure")
    return samples


def load_samples_from_json_file(
    dataset_name: str,
    json_path: str,
    audio_root: str,
    song_id_indices: Optional[List[int]] = None,
    song_id_slice: Optional[List[int]] = None,
    max_samples: int = -1,
) -> List[Dict]:
    """Load raw samples from a single JSON annotation file."""
    from tqdm import tqdm

    json_path = Path(json_path)
    audio_root_str = str(audio_root)

    if not json_path.is_absolute():
        json_path = Path.cwd() / json_path

    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    samples = []
    for item in tqdm(items, desc=f"  [{dataset_name}]", unit="items"):
        if 0 < max_samples <= len(samples):
            break

        words = item.get("word", [])
        pitches = item.get("pitch", [])
        notes = item.get("note", [])
        pitch2word = item.get("pitch2word", [])
        bpm = item.get("bpm", 120)
        wav_fn = item.get("wav_fn", "")

        if not words or not wav_fn:
            continue

        audio_path = os.path.join(audio_root_str, wav_fn)
        if not os.path.isfile(audio_path):
            continue

        if not (pitches and notes and pitch2word):
            continue  # skip samples without score annotations

        syllables = convert_annotation_to_syllables(
            words=words, pitches=pitches, notes=notes, pitch2word=pitch2word)

        samples.append({
            "audio_path": audio_path,
            "bpm": bpm,
            "syllables": syllables,
        })

    print(f"[{dataset_name}] Loaded {len(samples)} samples from JSON file")
    return samples


def load_all_datasets(datasets_config: List[Dict], max_samples: int = -1) -> List[Dict]:
    """Load samples from multiple datasets."""
    all_samples = []

    for ds_config in datasets_config:
        ds_name = ds_config.get("name", "unknown")
        ds_type = ds_config.get("type", "folder_based")

        remaining = max_samples - len(all_samples) if max_samples > 0 else -1
        if max_samples > 0 and remaining <= 0:
            break

        if ds_type == "folder_based":
            samples = load_samples_from_folder(
                dataset_name=ds_name,
                dataset_root=ds_config["dataset_root"],
                max_samples=remaining,
            )
        elif ds_type == "json_file":
            samples = load_samples_from_json_file(
                dataset_name=ds_name,
                json_path=ds_config["json_path"],
                audio_root=ds_config["audio_root"],
                song_id_indices=ds_config.get("song_id_indices"),
                song_id_slice=ds_config.get("song_id_slice"),
                max_samples=remaining,
            )

        else:
            print(f"[{ds_name}] Unknown dataset type '{ds_type}', skipping")
            continue

        # Tag every sample with its dataset name for train/val splitting
        for s in samples:
            s["dataset_name"] = ds_name
        all_samples.extend(samples)

    print(f"\nTotal samples loaded: {len(all_samples)}")
    return all_samples


# ══════════════════════════════════════════════════════════════════════
# Dataset building (raw scan path)
# ══════════════════════════════════════════════════════════════════════

def build_dataset(
    samples: List[Dict],
    processor,
    eval_file: str = "",
    bpm_position: str = "last",
) -> Dict[str, Dataset]:
    """Build HuggingFace DatasetDict from AST sample list (raw scan path)."""
    records = []
    for sample in samples:
        ast_text = build_interleaved_text(
            syllables=sample["syllables"],
            bpm=sample["bpm"],
            bpm_position=bpm_position,
        )
        target = f"language Chinese<asr_text>{ast_text}"
        records.append({
            "audio": sample["audio_path"],
            "text": target,
        })

    prefix_text = build_prefix_text(processor)
    data_dict = {
        "audio": [r["audio"] for r in records],
        "text": [r["text"] for r in records],
        "prefix_text": [prefix_text] * len(records),
    }
    ds = Dataset.from_dict(data_dict)
    result = {"train": ds}

    if eval_file and os.path.isfile(eval_file):
        eval_ds = load_dataset("json", data_files=eval_file, split="train")
        eval_ds = eval_ds.map(lambda _: {"prefix_text": prefix_text}, batched=False)
        result["validation"] = eval_ds

    return result


# ══════════════════════════════════════════════════════════════════════
# Dataset loading from preprocessed Arrow (fast path — precomputed mel)
# ══════════════════════════════════════════════════════════════════════

def split_train_val(
    ds: Dataset,
    val_datasets: Optional[List[str]] = None,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Split a HuggingFace Dataset into train/val by dataset_name column.

    Args:
        ds: Full dataset with a ``dataset_name`` column.
        val_datasets: List of dataset names to use entirely as validation.

    Returns:
        (train_ds, val_ds) — val_ds is None if no split is performed.
    """
    if not val_datasets:
        return ds, None

    if "dataset_name" not in ds.column_names:
        print("Warning: dataset_name column not found in preprocessed data. "
              "Please re-run preprocess.py. Skipping val split.")
        return ds, None

    val_datasets_set = set(val_datasets)
    all_dataset_names = ds["dataset_name"]

    val_indices = []
    train_indices = []
    for i, dn in enumerate(all_dataset_names):
        if dn in val_datasets_set:
            val_indices.append(i)
        else:
            train_indices.append(i)

    train_ds = ds.select(train_indices) if train_indices else ds.select([])
    val_ds = ds.select(val_indices) if val_indices else None

    # Report
    if val_datasets:
        found = set(all_dataset_names[i] for i in val_indices) & val_datasets_set
        missing = val_datasets_set - found
        if missing:
            print(f"Warning: val datasets not found in data: {missing}")
        val_by_ds = {}
        for i in val_indices:
            dn = all_dataset_names[i]
            if dn in val_datasets_set:
                val_by_ds[dn] = val_by_ds.get(dn, 0) + 1
        print(f"Validation datasets: {val_by_ds}")

    print(f"Split: {len(train_indices)} train, {len(val_indices)} val")
    return train_ds, val_ds


def load_from_preprocessed(
    preprocessed_dir: str,
    processor,
    val_datasets: Optional[List[str]] = None,
    eval_file: str = "",
) -> Dict[str, Dataset]:
    """Load training data from preprocessed HuggingFace Dataset.

    Uses load_from_disk() for memory-mapped access — data stays on SSD
    and is paged into RAM on demand by the OS, avoiding full-memory copy.

    If val_datasets is provided, splits by dataset_name column.
    """
    from datasets import load_from_disk as hf_load_from_disk

    preprocessed_dir = Path(preprocessed_dir)

    ds = hf_load_from_disk(str(preprocessed_dir))
    print(f"Loaded HuggingFace Dataset: {len(ds)} samples (memory-mapped)")

    # Split train/val by dataset name
    train_ds, val_ds = split_train_val(ds, val_datasets=val_datasets)

    # Return array columns as numpy for zero-copy torch.from_numpy in collator
    train_ds.set_format(
        "numpy", columns=["input_features"],
        output_all_columns=True,
    )
    if val_ds is not None:
        val_ds.set_format(
            "numpy", columns=["input_features"],
            output_all_columns=True,
        )

    result = {"train": train_ds, "_format": "hf_dataset"}

    if val_ds is not None:
        result["validation"] = val_ds
    elif eval_file and os.path.isfile(eval_file):
        eval_ds = load_dataset("json", data_files=eval_file, split="train")
        result["validation"] = eval_ds

    return result


# ══════════════════════════════════════════════════════════════════════
# DataCollators
# ══════════════════════════════════════════════════════════════════════

def _build_labels(input_ids: torch.Tensor, prefix_lens: List[int],
                  pad_token_id: int) -> torch.Tensor:
    """Create labels by masking prefix and padding regions with -100."""
    labels = input_ids.clone()
    for i, pl in enumerate(prefix_lens):
        labels[i, :pl] = -100
    labels[input_ids == pad_token_id] = -100
    return labels

@dataclass
class DataCollatorForVocalParse:
    """Collator for raw audio path mode (loads audio on-the-fly from NFS)."""
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["text"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        pad_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
        full_inputs["labels"] = _build_labels(full_inputs["input_ids"], prefix_lens, pad_id)
        return full_inputs


@dataclass
class DataCollatorForPrecomputedMel:
    """Collator for precomputed mel spectrogram mode (zero audio file I/O).

    Arrow data stores only raw metadata (syllables_json, bpm) and mel
    features.  This collator builds the prompt text, tokenizes, expands
    audio placeholders, and constructs labels — all online.

    Benefits:
    - Changing prompt format (e.g., bpm_position) only requires restarting
      training, NOT re-preprocessing.
    - json.loads() + tokenization overhead is ~0.1 ms/sample, negligible.
    """
    tokenizer: Any = None
    processor: Any = None
    prefix_text: str = ""
    eos: str = ""
    bpm_position: str = "last"
    asr_cot: bool = False        # Chain-of-Thought: prepend pure lyrics before AST
    pad_token_id: int = 151643
    audio_token_id: int = 151676  # <|audio_pad|>

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        import json as json_lib
        import numpy as np
        from torch.nn.utils.rnn import pad_sequence

        batch_mels = []
        batch_ids = []
        batch_prefix_lens = []

        for f in features:
            mel_frames = int(f["mel_frames"])
            mel_bins = int(f["mel_bins"])

            # 1. Reconstruct mel tensor
            mel_np = np.array(f["input_features"], dtype=np.float16, copy=True)
            mel = torch.from_numpy(mel_np.reshape(mel_frames, mel_bins))

            # 2. Build prompt text from metadata (online)
            syllables = json_lib.loads(f["syllables_json"])
            bpm = int(f["bpm"])

            ast_text = build_interleaved_text(
                syllables=syllables, bpm=bpm,
                bpm_position=self.bpm_position,
            )

            # ASR-CoT: prepend pure lyrics as Chain-of-Thought prefix
            if self.asr_cot:
                cot_lyrics = extract_lyrics_text(syllables)
                target_text = f"language Chinese<asr_text>{cot_lyrics}<|file_sep|>{ast_text}"
            else:
                target_text = f"language Chinese<asr_text>{ast_text}"

            sample_prefix = self.prefix_text

            full_text = sample_prefix + target_text + self.eos

            # 3. Tokenize
            full_ids = self.tokenizer(
                full_text, return_tensors="np", add_special_tokens=False,
            )["input_ids"][0].astype(np.int64)
            prefix_ids = self.tokenizer(
                sample_prefix, return_tensors="np", add_special_tokens=False,
            )["input_ids"][0]
            prefix_len = len(prefix_ids)

            # 4. Expand audio placeholder: 1 → N
            encoder_len = _get_encoder_output_length(mel_frames)
            audio_positions = np.where(full_ids == self.audio_token_id)[0]
            if len(audio_positions) == 1 and encoder_len > 1:
                pos = audio_positions[0]
                full_ids = np.concatenate([
                    full_ids[:pos],
                    np.full(encoder_len, self.audio_token_id, dtype=np.int64),
                    full_ids[pos + 1:],
                ])
                prefix_len += (encoder_len - 1)

            ids = torch.from_numpy(full_ids)
            batch_mels.append(mel)
            batch_ids.append(ids)
            batch_prefix_lens.append(prefix_len)

        # Pad mel via pad_sequence: (n_frames, n_mels) per sample
        padded_mels = pad_sequence(
            batch_mels, batch_first=True, padding_value=0.0
        )
        # Vectorized feature_attention_mask
        mel_lengths = torch.tensor([m.shape[0] for m in batch_mels])
        feature_attention_mask = (
            torch.arange(padded_mels.shape[1]).unsqueeze(0)
            < mel_lengths.unsqueeze(1)
        ).long()
        # Transpose to (batch, n_mels, n_frames) for WhisperEncoder
        padded_mels = padded_mels.transpose(1, 2)

        # Pad input IDs via pad_sequence
        padded_ids = pad_sequence(
            batch_ids, batch_first=True, padding_value=self.pad_token_id
        )
        id_lengths = torch.tensor([ids.shape[0] for ids in batch_ids])
        attention_mask = (
            torch.arange(padded_ids.shape[1]).unsqueeze(0)
            < id_lengths.unsqueeze(1)
        ).long()

        # Build labels (mask prefix region and padding with -100)
        labels = _build_labels(padded_ids, batch_prefix_lens, self.pad_token_id)

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "input_features": padded_mels,
            "feature_attention_mask": feature_attention_mask,
            "labels": labels,
        }
