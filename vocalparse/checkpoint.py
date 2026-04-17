# coding=utf-8
# VocalParse Checkpoint Utilities

import os
import re
import shutil
from pathlib import Path
from typing import Optional

from transformers import TrainerCallback, TrainingArguments


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def _resolve_hf_model_path(model_path: str) -> str:
    """Resolve a HuggingFace model ID to its local cache snapshot directory."""
    if os.path.isdir(model_path):
        return model_path

    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_path, local_files_only=True)
    except Exception:
        pass

    cache_root = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub",
    )
    model_dir_name = "models--" + model_path.replace("/", "--")
    snapshots_dir = os.path.join(cache_root, model_dir_name, "snapshots")
    if os.path.isdir(snapshots_dir):
        snaps = sorted(
            Path(snapshots_dir).iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if snaps:
            return str(snaps[0])

    return model_path


def copy_required_hf_files(src_dir: str, dst_dir: str):
    resolved_src = _resolve_hf_model_path(src_dir)
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(resolved_src, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = _resolve_hf_model_path(base_model_path)

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files(self.base_model_path, ckpt_dir)
        return control
