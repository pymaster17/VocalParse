# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# VocalParse Training Script
#
# Fine-tunes Qwen3-ASR to output AST-format sequences from singing audio:
#   <BPM_89> 感 <P_68><NOTE_DOT_16> 受 <P_68><NOTE_DOT_8> ...
#
# Data is loaded from standard SVS annotation formats (folder_based,
# json_file) and AST prompt text is built using the "aggregated" strategy.
#
# Usage:
#   python -m vocalparse.train --config configs/train.yaml
#
#   # Or with explicit arguments:
#   python train.py \
#     --model_path Qwen/Qwen3-ASR-1.7B \
#     --output_dir ./vocalparse-runs/experiment-1

import argparse
import os
import time
from pathlib import Path

import torch
import yaml
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)

# ── AST modules ─────────────────────────────────────────────────────
from vocalparse.model import patch_outer_forward, register_vocalparse_tokens
from vocalparse.checkpoint import find_latest_checkpoint, MakeEveryCheckpointInferableCallback
from vocalparse.prompts import build_interleaved_text, build_prefix_text, extract_lyrics_text
from vocalparse.data import (
    load_all_datasets,
    build_dataset,
    load_from_preprocessed,
    DataCollatorForVocalParse,
    DataCollatorForPrecomputedMel,
)
from vocalparse.validation import GenerateSamplesCallback


# ══════════════════════════════════════════════════════════════════════
# Custom Trainer
# ══════════════════════════════════════════════════════════════════════

class CastFloatInputsTrainer(Trainer):
    # Eval metrics to keep in TensorBoard (others are filtered out)
    _EVAL_KEEP = {"eval_loss", "eval_runtime"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_log_time = None
        self._last_log_step = None

    def log(self, logs, start_time=None, **kwargs):
        """Filter noisy eval metrics, keeping only eval_loss + val/* scalars.
        Also inject step/total_step and step/s into training logs."""
        # Normalize loss/grad_norm for multi-GPU (DDP all_reduce sums them)
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_world_size() > 1:
            ws = dist.get_world_size()
            if "loss" in logs:
                logs["loss"] = logs["loss"] / ws
            if "grad_norm" in logs:
                logs["grad_norm"] = logs["grad_norm"] / ws
        if any(k.startswith("eval_") for k in logs):
            logs = {k: v for k, v in logs.items()
                    if k in self._EVAL_KEEP or not k.startswith("eval_")}
        # Inject step/total_step and step/s into training logs
        if self.state is not None and "loss" in logs:
            step = self.state.global_step
            total = self.state.max_steps
            logs["step"] = f"{step}/{total}"
            now = time.perf_counter()
            if self._last_log_time is not None and step > self._last_log_step:
                elapsed = now - self._last_log_time
                steps_done = step - self._last_log_step
                logs["step/s"] = round(steps_done / elapsed, 2)
            self._last_log_time = now
            self._last_log_step = step
        return super().log(logs, start_time=start_time, **kwargs)

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def set_dynamic_batching(self, lengths, max_batch_tokens, max_batch_size,
                             num_workers, pin_memory):
        """Enable dynamic batching for training."""
        self._dyn_lengths = lengths
        self._dyn_max_batch_tokens = max_batch_tokens
        self._dyn_max_batch_size = max_batch_size
        self._dyn_num_workers = num_workers
        self._dyn_pin_memory = pin_memory

    def get_train_dataloader(self):
        """Override to use DynamicBatchSampler when configured."""
        if not hasattr(self, "_dyn_max_batch_tokens") or self._dyn_max_batch_tokens <= 0:
            return super().get_train_dataloader()

        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        batch_sampler = DynamicBatchSampler(
            lengths=self._dyn_lengths,
            max_batch_tokens=self._dyn_max_batch_tokens,
            max_batch_size=self._dyn_max_batch_size,
            shuffle=True,
            drop_last=True,
            rank=rank,
            world_size=world_size,
        )
        # Store for epoch management
        self._dynamic_sampler = batch_sampler

        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self._dyn_num_workers,
            pin_memory=self._dyn_pin_memory,
            persistent_workers=(self._dyn_num_workers > 0),
            prefetch_factor=4 if self._dyn_num_workers > 0 else None,
        )
        return loader

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        super()._save(output_dir, state_dict=state_dict)


# ══════════════════════════════════════════════════════════════════════
# Dynamic Batch Sampler
# ══════════════════════════════════════════════════════════════════════

import random as _random

class DynamicBatchSampler(torch.utils.data.Sampler):
    """Groups samples by length so total tokens per batch ≤ max_batch_tokens.

    Short samples → large batch → better GPU utilization.
    Long samples → small batch → avoid OOM.

    For DDP, batches are partitioned across ranks.
    """

    def __init__(self, lengths, max_batch_tokens, max_batch_size=64,
                 shuffle=True, drop_last=False, rank=0, world_size=1,
                 seed=42):
        self.lengths = lengths
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self._create_batches()

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._create_batches()

    def _create_batches(self):
        indices = list(range(len(self.lengths)))
        rng = _random.Random(self.seed + self.epoch)

        if self.shuffle:
            # Sort by length, then shuffle within chunks of similar length
            sorted_indices = sorted(indices, key=lambda i: self.lengths[i])
            chunk_size = max(100, len(indices) // 10)
            chunks = [sorted_indices[i:i + chunk_size]
                      for i in range(0, len(sorted_indices), chunk_size)]
            for chunk in chunks:
                rng.shuffle(chunk)
            rng.shuffle(chunks)
            indices = [idx for chunk in chunks for idx in chunk]

        # Build batches respecting token budget
        all_batches = []
        current_batch = []
        current_max_len = 0

        for idx in indices:
            sample_len = self.lengths[idx]
            new_max_len = max(current_max_len, sample_len)
            new_tokens = (len(current_batch) + 1) * new_max_len

            if ((new_tokens > self.max_batch_tokens or
                 len(current_batch) >= self.max_batch_size) and current_batch):
                all_batches.append(current_batch)
                current_batch = [idx]
                current_max_len = sample_len
            else:
                current_batch.append(idx)
                current_max_len = new_max_len

        if current_batch:
            if not self.drop_last or len(current_batch) >= 2:
                all_batches.append(current_batch)

        if self.shuffle:
            rng.shuffle(all_batches)

        # Pad so every rank gets the same number of batches (avoid DDP deadlock)
        if self.world_size > 1:
            remainder = len(all_batches) % self.world_size
            if remainder != 0:
                for i in range(self.world_size - remainder):
                    all_batches.append(all_batches[i % len(all_batches)])

        # Partition for DDP
        self._batches = all_batches[self.rank::self.world_size]

    def __iter__(self):
        for batch in self._batches:
            yield batch

    def __len__(self):
        return len(self._batches)


# ══════════════════════════════════════════════════════════════════════
# CLI Argument Parsing
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("VocalParse Training")

    # Paths
    p.add_argument("--model_path", type=str, default="")
    p.add_argument("--config", type=str, default="",
                   help="YAML config file with datasets and training params")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--preprocessed_dir", type=str, default="",
                   help="Path to preprocessed Arrow data (fast loading)")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Data
    p.add_argument("--max_samples", type=int, default=-1,
                   help="Max samples to load (-1 for all)")

    # Training hyper-params
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)

    # DataLoader
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=4)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)
    p.add_argument("--eval_steps", type=int, default=0,
                   help="Evaluation interval in steps (0 = use save_steps)")

    # Dynamic batching
    p.add_argument("--max_batch_mel_tokens", type=int, default=0,
                   help="Max total mel_frames per batch (0=disabled, use fixed batch_size)")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
# Main helpers
# ══════════════════════════════════════════════════════════════════════

def _load_config(args_cli):
    """Load YAML config and merge with CLI arguments (CLI takes precedence)."""
    config = {}
    if args_cli.config:
        with open(args_cli.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    return {
        "model_path": args_cli.model_path or config.get("model_path", "Qwen/Qwen3-ASR-1.7B"),
        "output_dir": args_cli.output_dir or config.get("output_dir", "./vocalparse-runs/default"),
        "max_samples": args_cli.max_samples if args_cli.max_samples != -1 else config.get("max_samples", -1),
        "preprocessed_dir": args_cli.preprocessed_dir or config.get("preprocessed_dir", ""),
        "bpm_position": config.get("bpm_position", "last"),
        "asr_cot": bool(config.get("asr_cot", False)),
        "config": config,
    }


def _load_model(model_path):
    """Load Qwen3-ASR model, apply patches, and register AST tokens."""
    from qwen_asr import Qwen3ASRModel

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    register_vocalparse_tokens(processor, model)

    return model, processor, use_bf16


def _load_data(cfg, config, model, processor, args_cli):
    """Load datasets and create the appropriate DataCollator.

    Returns (train_ds, val_ds, data_format, collator, prefix_text, eos).
    """
    preprocessed_dir = cfg["preprocessed_dir"]
    eval_file = args_cli.eval_file or config.get("eval_file", "")
    val_datasets_cfg = config.get("val_datasets", []) or []

    if preprocessed_dir and Path(preprocessed_dir).is_dir():
        print(f"\n=== Loading from preprocessed data: {preprocessed_dir} ===")
        datasets = load_from_preprocessed(
            preprocessed_dir=preprocessed_dir,
            processor=processor,
            val_datasets=val_datasets_cfg,
            eval_file=eval_file,
        )
    else:
        datasets_config = config.get("datasets", [])
        if not datasets_config:
            raise ValueError(
                "No datasets configured. Provide 'datasets' in YAML or --preprocessed_dir."
            )
        print("\n=== Loading AST datasets (raw scan) ===")
        all_samples = load_all_datasets(datasets_config, max_samples=cfg["max_samples"])
        if not all_samples:
            raise ValueError("No samples loaded. Check your datasets config.")
        datasets = build_dataset(
            all_samples, processor=processor, eval_file=eval_file,
            bpm_position=cfg["bpm_position"],
        )

    train_ds = datasets["train"]
    val_ds = datasets.get("validation", None)
    data_format = datasets.pop("_format", "raw")

    print(f"Train samples: {len(train_ds)}")
    if val_ds:
        print(f"Validation samples: {len(val_ds)}")

    if len(train_ds) > 0:
        sample = train_ds[0]
        print(f"\n=== Sample 0 ===")
        if data_format == "hf_dataset":
            print(f"  Format: precomputed (memory-mapped Arrow), zero audio I/O")
            print(f"  Mel: {sample['mel_frames']} frames x {sample['mel_bins']} bins")
            print(f"  BPM: {sample['bpm']}")
            print(f"  Syllables (first 120 chars): {sample['syllables_json'][:120]}...")
        else:
            print(f"  Audio: {sample['audio']}")
            print(f"  Target (first 200 chars): {sample['text'][:200]}...")

    # DataCollator
    prefix_text = ""
    eos = ""
    if data_format == "hf_dataset":
        prefix_text = build_prefix_text(processor)
        eos = processor.tokenizer.eos_token or ""
        pad_id = processor.tokenizer.pad_token_id or 151643
        audio_tid = model.thinker.config.audio_token_id
        collator = DataCollatorForPrecomputedMel(
            tokenizer=processor.tokenizer,
            processor=processor,
            prefix_text=prefix_text,
            eos=eos,
            bpm_position=cfg["bpm_position"],
            asr_cot=cfg["asr_cot"],
            pad_token_id=pad_id,
            audio_token_id=audio_tid,
        )
        print(f"Using DataCollatorForPrecomputedMel (bpm_position={cfg['bpm_position']}, asr_cot={cfg['asr_cot']})")
    else:
        collator = DataCollatorForVocalParse(
            processor=processor,
            sampling_rate=args_cli.sr,
        )
        print("Using DataCollatorForVocalParse (audio from NFS)")

    return train_ds, val_ds, data_format, collator, prefix_text, eos


def _build_trainer(cfg, config, model, processor, args_cli,
                   train_ds, val_ds, data_format, collator,
                   prefix_text, eos, use_bf16):
    """Build TrainingArguments, callbacks, and Trainer."""
    output_dir = cfg["output_dir"]
    batch_size = int(config.get("batch_size", args_cli.batch_size))
    grad_acc = int(config.get("grad_acc", args_cli.grad_acc))
    lr = float(config.get("lr", args_cli.lr))
    epochs = float(config.get("epochs", args_cli.epochs))
    save_steps = int(config.get("save_steps", args_cli.save_steps))
    save_total_limit = int(config.get("save_total_limit", args_cli.save_total_limit))
    eval_steps_cfg = int(config.get("eval_steps", args_cli.eval_steps) or save_steps)
    lr_scheduler_kwargs = config.get("lr_scheduler_kwargs", {}) or {}

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=config.get("lr_scheduler", args_cli.lr_scheduler_type),
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        warmup_ratio=float(config.get("warmup_ratio", args_cli.warmup_ratio)),
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1 and args_cli.num_workers > 0),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_safetensors=True,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=eval_steps_cfg if val_ds else None,
        do_eval=val_ds is not None,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="tensorboard",
        group_by_length=False,
        disable_tqdm=True,
    )

    val_generate_samples = int(config.get("val_generate_samples", 5))
    val_display_samples = int(config.get("val_display_samples", 5))
    max_batch_mel = int(config.get("max_batch_mel_tokens",
                                    args_cli.max_batch_mel_tokens))

    callbacks = [
        MakeEveryCheckpointInferableCallback(base_model_path=cfg["model_path"]),
    ]
    if val_ds is not None and val_generate_samples != 0:
        callbacks.append(GenerateSamplesCallback(
            val_ds=val_ds,
            tokenizer=processor.tokenizer,
            processor=processor,
            num_samples=val_generate_samples,
            num_display=val_display_samples,
            batch_size=batch_size,
            max_batch_mel_tokens=max_batch_mel,
            data_format=data_format,
            prefix_text=prefix_text if data_format == "hf_dataset" else "",
            eos=eos if data_format == "hf_dataset" else "",
            bpm_position=cfg["bpm_position"],
            asr_cot=cfg["asr_cot"],
        ))

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=callbacks,
    )

    for cb in callbacks:
        if isinstance(cb, GenerateSamplesCallback):
            cb.set_trainer(trainer)

    return trainer, max_batch_mel, batch_size


def _launch_training(trainer, output_dir, max_batch_mel, batch_size,
                     train_ds, data_format, args_cli):
    """Configure dynamic batching and launch training with auto-resume."""
    if max_batch_mel > 0 and data_format == "hf_dataset":
        mel_lengths = train_ds.with_format(None)["mel_frames"]
        if not isinstance(mel_lengths, list):
            mel_lengths = list(mel_lengths)

        trainer.set_dynamic_batching(
            lengths=mel_lengths,
            max_batch_tokens=max_batch_mel,
            max_batch_size=batch_size,
            num_workers=args_cli.num_workers,
            pin_memory=(args_cli.pin_memory == 1),
        )
        import torch.distributed as dist
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_rank0:
            _preview = DynamicBatchSampler(
                lengths=mel_lengths,
                max_batch_tokens=max_batch_mel,
                max_batch_size=batch_size,
                shuffle=False, drop_last=True,
            )
            n_batches = len(_preview)
            avg_bs = len(mel_lengths) / max(n_batches, 1)
            min_len, max_len = min(mel_lengths), max(mel_lengths)
            avg_len = sum(mel_lengths) / len(mel_lengths)
            print(f"\n=== Dynamic Batching ===")
            print(f"  max_batch_mel_tokens: {max_batch_mel}")
            print(f"  max_batch_size cap:   {batch_size}")
            print(f"  mel_frames range:     [{min_len}, {max_len}], avg={avg_len:.0f}")
            print(f"  Training batches:     {n_batches} (avg ~{avg_bs:.1f} samples/batch)")
            print()
            del _preview

    resume_from = find_latest_checkpoint(output_dir)
    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[auto-resume] Resuming from {resume_from}")
            print(f"  (to start fresh, change output_dir in your config)")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        if trainer.args.process_index == 0:
            print(f"[new training] No existing checkpoints in {output_dir}")
        trainer.train()


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    args_cli = parse_args()
    cfg = _load_config(args_cli)
    model, processor, use_bf16 = _load_model(cfg["model_path"])
    train_ds, val_ds, data_format, collator, prefix_text, eos = \
        _load_data(cfg, cfg["config"], model, processor, args_cli)

    # Print prompt structure (sample 0, rank 0 only)
    _rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if data_format == "hf_dataset" and len(train_ds) > 0 and _rank == 0:
        import json as _json
        _s0 = train_ds[0]
        _syllables = _json.loads(_s0["syllables_json"])
        _bpm = int(_s0["bpm"])
        _ast_text = build_interleaved_text(
            syllables=_syllables, bpm=_bpm,
            bpm_position=cfg["bpm_position"],
        )
        if cfg["asr_cot"]:
            _cot_lyrics = extract_lyrics_text(_syllables)
            _target = f"language Chinese<asr_text>{_cot_lyrics}<|file_sep|>{_ast_text}"
        else:
            _target = f"language Chinese<asr_text>{_ast_text}"
        _full = prefix_text + _target + eos
        print(f"\n{'='*60}")
        print(f"[Prompt] Structure (sample 0):")
        print(f"  bpm_position={cfg['bpm_position']}  asr_cot={cfg['asr_cot']}")
        print(f"  PREFIX ({len(prefix_text)} chars):")
        print(f"    {prefix_text}")
        print(f"  TARGET ({len(_target)} chars):")
        print(f"    {_target[:500]}")
        print(f"  FULL ({len(_full)} chars):")
        print(f"    {_full[:800]}")
        print(f"{'='*60}\n")

    trainer, max_batch_mel, batch_size = _build_trainer(
        cfg, cfg["config"], model, processor, args_cli,
        train_ds, val_ds, data_format, collator, prefix_text, eos, use_bf16,
    )
    _launch_training(
        trainer, cfg["output_dir"], max_batch_mel, batch_size,
        train_ds, data_format, args_cli,
    )


if __name__ == "__main__":
    main()
