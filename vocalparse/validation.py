# coding=utf-8
# VocalParse Validation and Visualization
#
# Provides the validation callback for AST fine-tuning:
# - Score condition visualization (GT vs Pred comparison figures)
# - GenerateSamplesCallback: multi-GPU batched validation with
#   TensorBoard logging of CER, Pitch MAE, Note MAE, Duration MAE, BPM MAE

import os
from pathlib import Path

import torch
from transformers import TrainerCallback, TrainingArguments

from vocalparse.evaluation import (
    NOTE_DUR_MAP as _DUR_UNITS,
    aggregate_metrics,
    compute_metrics,
    parse_transcription_text,
)
from vocalparse.model import _get_encoder_output_length
from vocalparse.prompts import build_interleaved_text, extract_lyrics_text
from vocalparse.distributed import pre_encode_audio_features, left_pad_input_ids


# ══════════════════════════════════════════════════════════════════════
# Score condition visualization
# ══════════════════════════════════════════════════════════════════════

def _draw_score_rows(ax_lyric, ax_midi, words, pitches, notes,
                     color_lyric, color_midi, chinese_font):
    """Draw lyric + MIDI rows on the given axes pair."""
    import numpy as np
    import matplotlib.pyplot as plt

    edge_color = '#8B6914'
    text_color = '#5A4A1A'
    note_durations = [_DUR_UNITS.get(n, 1.0) for n in notes]
    lyrics = [words[i] if i < len(words) else "." for i in range(len(pitches))]
    positions = np.cumsum([0.0] + note_durations[:-1])
    total_width = sum(note_durations)

    for pos, dur, lyric in zip(positions, note_durations, lyrics):
        rect = plt.Rectangle((pos, 0), dur, 1,
                              facecolor=color_lyric, edgecolor=edge_color, linewidth=1.5)
        ax_lyric.add_patch(rect)
        ax_lyric.text(pos + dur / 2, 0.5, lyric,
                      ha='center', va='center', fontsize=14, fontweight='bold',
                      color=text_color, fontproperties=chinese_font)

    for pos, dur, pitch in zip(positions, note_durations, pitches):
        rect = plt.Rectangle((pos, 0), dur, 1,
                              facecolor=color_midi, edgecolor=edge_color, linewidth=1.5)
        ax_midi.add_patch(rect)
        ax_midi.text(pos + dur / 2, 0.5, str(pitch),
                     ha='center', va='center', fontsize=12, fontweight='bold', color=text_color)

    for ax in (ax_lyric, ax_midi):
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    return total_width


def create_comparison_figure(gt_parsed, pred_parsed, sample_idx=0, step=None):
    """Create a combined GT vs Pred score condition figure.

    Layout (4 rows, top-to-bottom):
        GT Lyric  | GT MIDI  | Pred Lyric | Pred MIDI

    Both sections share the same x-axis for easy alignment comparison.

    Returns:
        matplotlib Figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np

    # Chinese font
    user_font_path = os.path.expanduser("~/.fonts/NotoSansSC-Regular.ttf")
    if os.path.exists(user_font_path):
        font_manager.fontManager.addfont(user_font_path)
        chinese_font = font_manager.FontProperties(fname=user_font_path)
    else:
        chinese_font = font_manager.FontProperties(family='sans-serif')

    text_color = '#5A4A1A'

    # Compute total widths for both to set shared xlim
    gt_widths = sum(_DUR_UNITS.get(n, 1.0) for n in gt_parsed["notes"])
    pred_widths = sum(_DUR_UNITS.get(n, 1.0) for n in pred_parsed["notes"])
    total_width = max(gt_widths, pred_widths)

    fig, axes = plt.subplots(
        4, 1, figsize=(max(12, total_width * 1.5), 6),
        gridspec_kw={'height_ratios': [1, 1, 1, 1], 'hspace': 0.08},
    )
    ax_gt_lyric, ax_gt_midi, ax_pred_lyric, ax_pred_midi = axes

    # Draw GT (warm orange)
    _draw_score_rows(ax_gt_lyric, ax_gt_midi,
                     gt_parsed["words"], gt_parsed["pitches"], gt_parsed["notes"],
                     color_lyric='#E8A038', color_midi='#F5D03A',
                     chinese_font=chinese_font)
    ax_gt_lyric.set_ylabel('GT\nLyric', fontsize=10, fontweight='bold',
                           color=text_color, rotation=0, labelpad=40, va='center')
    ax_gt_midi.set_ylabel('GT\nMIDI', fontsize=10, fontweight='bold',
                          color=text_color, rotation=0, labelpad=40, va='center')

    # Draw Pred (blue-green to distinguish)
    _draw_score_rows(ax_pred_lyric, ax_pred_midi,
                     pred_parsed["words"], pred_parsed["pitches"], pred_parsed["notes"],
                     color_lyric='#5DADE2', color_midi='#85C1E9',
                     chinese_font=chinese_font)
    ax_pred_lyric.set_ylabel('Pred\nLyric', fontsize=10, fontweight='bold',
                             color=text_color, rotation=0, labelpad=40, va='center')
    ax_pred_midi.set_ylabel('Pred\nMIDI', fontsize=10, fontweight='bold',
                            color=text_color, rotation=0, labelpad=40, va='center')

    # Align x-axes
    for ax in axes:
        ax.set_xlim(0, total_width)

    step_str = f" @ Step {step}" if step is not None else ""
    bpm = gt_parsed["bpm"]
    fig.suptitle(f'Sample {sample_idx} (BPM={bpm}){step_str}',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.06, 0, 1, 0.95])
    return fig


# ══════════════════════════════════════════════════════════════════════
# Validation sample generation callback
# ══════════════════════════════════════════════════════════════════════

class GenerateSamplesCallback(TrainerCallback):
    """Generate predictions on val samples and log to TensorBoard.

    At each evaluation step, runs model.generate() on the first N validation
    samples. In multi-GPU (DDP) mode, batches are sharded across all ranks
    for parallel inference, results gathered to rank 0 via /dev/shm.
    Rank 0 logs combined GT vs Pred score condition figures and AST metrics.
    """

    def __init__(self, val_ds, tokenizer, processor=None, num_samples: int = 5,
                 num_display: int = 5, batch_size: int = 32,
                 max_batch_mel_tokens: int = 0,
                 data_format: str = "hf_dataset",
                 prefix_text: str = "", eos: str = "",
                 bpm_position: str = "last",
                 include_dur: bool = False,
                 asr_cot: bool = False):
        self.val_ds = val_ds
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_samples = num_samples
        self.num_display = min(num_display, num_samples) if num_samples > 0 else num_display
        self.batch_size = batch_size
        self.max_batch_mel_tokens = max_batch_mel_tokens
        self.data_format = data_format
        self.prefix_text = prefix_text
        self.eos = eos
        self.bpm_position = bpm_position
        self.include_dur = include_dur
        self.asr_cot = asr_cot
        self._trainer = None

    def set_trainer(self, trainer):
        self._trainer = trainer

    def _get_tb_writer(self):
        if self._trainer is None:
            return None
        for cb in self._trainer.callback_handler.callbacks:
            if hasattr(cb, "tb_writer") and cb.tb_writer is not None:
                return cb.tb_writer
        return None

    def on_evaluate(self, args: TrainingArguments, state, control, model=None, **kwargs):
        if self.val_ds is None or self.num_samples == 0:
            return
        if model is None:
            return

        import torch.distributed as dist
        import pickle
        from pathlib import Path

        rank = args.process_index
        world_size = args.world_size if hasattr(args, "world_size") else 1
        is_distributed = dist.is_initialized() and world_size > 1

        step = state.global_step
        n = len(self.val_ds) if self.num_samples < 0 else min(self.num_samples, len(self.val_ds))
        device = args.device

        # Unwrap DDP model for independent per-rank generation
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.eval()

        if rank == 0:
            print(f"\n[GenerateSamples] Generating {n} val samples at step {step} "
                  f"({world_size} GPU(s))...")

        # === Phase 1: Sort & pack batches (all ranks compute the same batches) ===
        raw_samples = [(i, self.val_ds[i]) for i in range(n)]
        if self.data_format == "hf_dataset":
            raw_samples.sort(key=lambda x: int(x[1]["mel_frames"]))

        batches = []
        cur_batch, cur_mel_total = [], 0
        for idx, sample in raw_samples:
            mel_frames = int(sample["mel_frames"]) if self.data_format == "hf_dataset" else 1
            if cur_batch and (
                (self.max_batch_mel_tokens > 0 and cur_mel_total + mel_frames > self.max_batch_mel_tokens)
                or len(cur_batch) >= self.batch_size
            ):
                batches.append(cur_batch)
                cur_batch, cur_mel_total = [], 0
            cur_batch.append((idx, sample))
            cur_mel_total += mel_frames
        if cur_batch:
            batches.append(cur_batch)

        # === Phase 2: Shard batches across ranks (round-robin) ===
        my_batches = batches[rank::world_size]

        if rank == 0:
            batch_sizes = [len(b) for b in batches]
            print(f"  Packed {n} samples into {len(batches)} batches "
                  f"(sizes: {batch_sizes}), "
                  f"~{len(my_batches)} batches/GPU")

        # === Phase 3: Each rank generates its sharded batches ===
        import time as _t
        t0 = _t.perf_counter()

        my_results = []  # (original_idx, pred_text, gt_text)
        try:
            for batch in my_batches:
                chunk_samples = [s for _, s in batch]
                chunk_indices = [idx for idx, _ in batch]
                chunk_results = self._generate_batch(unwrapped, chunk_samples, device)
                for idx, (pred, gt) in zip(chunk_indices, chunk_results):
                    my_results.append((idx, pred, gt))
        except Exception as e:
            import traceback
            print(f"  [Rank {rank}] Batch generation failed: {e}")
            traceback.print_exc()

        elapsed = _t.perf_counter() - t0
        print(f"  [Rank {rank}] Generated {len(my_results)} samples in {elapsed:.1f}s")

        # === Phase 4: Gather results to rank 0 via /dev/shm ===
        if is_distributed:
            shm_path = Path(f"/dev/shm/_ast_val_rank{rank}_step{step}.pkl")
            try:
                with open(shm_path, "wb") as f:
                    pickle.dump(my_results, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"  [Rank {rank}] Failed to write results: {e}")

            dist.barrier()  # Ensure all ranks have written their files

            if rank == 0:
                all_results = []
                for r in range(world_size):
                    r_path = Path(f"/dev/shm/_ast_val_rank{r}_step{step}.pkl")
                    try:
                        if r_path.exists():
                            with open(r_path, "rb") as f:
                                all_results.extend(pickle.load(f))
                            r_path.unlink()
                    except Exception as e:
                        print(f"  [Gather] Failed to load rank {r} results: {e}")
                all_results.sort(key=lambda x: x[0])
                results = [(pred, gt) for _, pred, gt in all_results]
                print(f"  Gathered {len(results)} results from {world_size} GPUs")

            dist.barrier()  # Ensure rank 0 has read before non-rank-0 continues

            if rank != 0:
                return  # Non-rank-0: done
        else:
            my_results.sort(key=lambda x: x[0])
            results = [(pred, gt) for _, pred, gt in my_results]

        # === Phase 5: Rank 0 — TensorBoard logging & metrics ===
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        writer = self._get_tb_writer()
        if writer is None:
            print("[GenerateSamples] Warning: TensorBoard writer not found, skipping.")
            return

        all_metrics = []

        for i, (pred_text, gt_text) in enumerate(results):
            try:
                gt_parsed = parse_transcription_text(gt_text)
                pred_parsed = parse_transcription_text(pred_text)

                # Log figures/text only for the first num_display samples
                if i < self.num_display:
                    md = f"**Pred:** {pred_text[:300]}\n\n**GT:** {gt_text[:300]}"
                    writer.add_text(f"val_sample_{i}/text_summary", md, global_step=step)

                    if gt_parsed and pred_parsed:
                        fig = create_comparison_figure(
                            gt_parsed, pred_parsed, sample_idx=i, step=step,
                        )
                        writer.add_figure(f"val_sample_{i}", fig, global_step=step)
                        plt.close(fig)
                    elif not pred_parsed:
                        print(f"  [Sample {i}] Warning: prediction is not a valid AST sequence, skipping figure.")

                # Compute per-sample AST metrics
                sample_metrics = compute_metrics(
                    gt_text, pred_text, include_dur=self.include_dur,
                )
                if sample_metrics is not None:
                    all_metrics.append(sample_metrics)

                if i < self.num_display:
                    print(f"  [Sample {i}] pred={pred_text[:80]}...")
                    print(f"  [Sample {i}]   gt={gt_text[:80]}...")
            except Exception as e:
                import traceback
                print(f"  [Sample {i}] Error: {e}")
                traceback.print_exc()

        # Aggregate and log metrics
        if all_metrics:
            import math
            agg = aggregate_metrics(all_metrics, include_dur=self.include_dur)

            metric_keys = [
                ("cer", "eval/cer"),
                ("cer_singing", "eval/cer_singing"),
                ("pitch_mae", "eval/pitch_mae"),
                ("note_mae", "eval/note_mae"),
                ("abs_note_dur_mae", "eval/abs_note_dur_mae"),
                ("bpm_mae", "eval/bpm_mae"),
            ]
            if self.include_dur:
                metric_keys.append(("dur_mae", "eval/dur_mae"))

            parts = []
            for key, tb_key in metric_keys:
                val = agg.get(key, float("nan"))
                if not math.isnan(val):
                    writer.add_scalar(tb_key, val, global_step=step)
                    parts.append(f"{key}={val:.3f}")

            n_total = int(agg.get("n_samples", 0))
            n_ok = int(agg.get("n_parseable", 0))
            print(f"  Metrics ({n_ok}/{n_total} parseable): {', '.join(parts)}")

        writer.flush()
        print(f"[GenerateSamples] Done, logged to TensorBoard.")

    @torch.no_grad()
    def _generate_batch(self, model, samples, device):
        """Batched generation: process all samples in one model.generate() call.

        Steps:
        1. Pre-compute mel tensors and prefix input_ids for each sample.
        2. Pad mel (along time axis) and input_ids to the max length in the batch.
        3. Call model.generate() once with the full batch.
        4. Unpad generated sequences using per-sample prefix lengths.

        Returns:
            List of (pred_text, gt_text) tuples.
        """
        import json as json_lib
        import numpy as np
        from torch.nn.utils.rnn import pad_sequence
        import time as _t

        t0 = _t.perf_counter()

        # === Phase 1: Prepare per-sample tensors ===
        batch_mels = []       # (n_mels, n_frames) per sample
        batch_mel_lens = []
        batch_ids = []        # (seq_len,) per sample
        batch_prefix_lens = []
        gt_texts = []

        # Tokenize prefix once (shared across all samples)
        prefix_text = self.prefix_text
        base_prefix_ids = self.tokenizer(
            prefix_text, return_tensors="np", add_special_tokens=False,
        )["input_ids"][0].astype(np.int64)
        audio_token_id = 151676

        for sample in samples:
            if self.data_format != "hf_dataset":
                gt_texts.append(sample.get("text", ""))
                continue

            mel_frames = int(sample["mel_frames"])
            mel_bins = int(sample["mel_bins"])

            # Reconstruct mel: (n_mels, n_frames)
            mel_np = np.array(sample["input_features"], dtype=np.float16, copy=True)
            mel = torch.from_numpy(mel_np.reshape(mel_frames, mel_bins)).T  # (n_mels, n_frames)
            batch_mels.append(mel)
            batch_mel_lens.append(mel_frames)

            # Build GT text
            syllables = json_lib.loads(sample["syllables_json"])
            bpm = int(sample["bpm"])
            ast_text = build_interleaved_text(
                syllables=syllables, bpm=bpm,
                bpm_position=self.bpm_position,
                include_dur=self.include_dur,
            )
            # Include CoT lyrics in GT when asr_cot is enabled, so that
            # compute_metrics sees the full sequence for comparison.
            if self.asr_cot:
                cot_lyrics = extract_lyrics_text(syllables)
                gt_texts.append(f"language Chinese<asr_text>{cot_lyrics}<|file_sep|>{ast_text}")
            else:
                gt_texts.append(f"language Chinese<asr_text>{ast_text}")

            # Build prefix IDs — standard prefix (audio-only for validation)
            sample_prefix = prefix_text

            prefix_ids_np = self.tokenizer(
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

        if self.data_format != "hf_dataset":
            return [("[generation not supported for raw audio mode]", gt)
                    for gt in gt_texts]

        # === Phase 2: Pad input_ids and per-sample audio encoding ===
        # Left-pad input_ids
        pad_token_id = self.tokenizer.pad_token_id or 151643
        padded_ids, attention_mask, pad_offsets = left_pad_input_ids(batch_ids, pad_token_id)

        # Move to device
        padded_ids = padded_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Compute actual prefix lengths (after left-padding)
        actual_prefix_lens = [pad_offsets[i] + batch_prefix_lens[i]
                              for i in range(len(samples))]

        # Per-sample audio encoding (precision-safe)
        inputs_embeds = pre_encode_audio_features(
            model, padded_ids, batch_mels, batch_mel_lens, device,
        )

        # === Phase 3: Generate ===
        outputs = model.generate(
            input_ids=padded_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=512,
        )
        gen_seqs = outputs.sequences if hasattr(outputs, "sequences") else outputs

        elapsed = _t.perf_counter() - t0
        print(f"  [Batch generate] {len(samples)} samples in {elapsed:.1f}s")

        # === Phase 4: Decode each sample ===
        results = []
        for i in range(len(samples)):
            gen_ids = gen_seqs[i]
            # Actual prefix starts after left-padding
            actual_start = pad_offsets[i] + batch_prefix_lens[i]
            pred_tokens = gen_ids[actual_start:]
            pred_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=False)
            results.append((pred_text, gt_texts[i]))

        return results
