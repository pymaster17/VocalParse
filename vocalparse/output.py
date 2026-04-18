# coding=utf-8
# VocalParse Inference Output Strategies
#
# Handles formatting and saving inference results for different modes:
# - test_weak: CER-only evaluation
# - test_full: Full AST metrics evaluation
# - annotation: Opencpop.json-compatible annotation output

import json
import math
import os
from pathlib import Path


def _output_test_weak(results, cfg):
    """Test-weak mode: CER-only evaluation.

    Works with both raw_audio input (gt_lyrics from JSON) and preprocessed
    input (gt_lyrics extracted from gt_ast).
    """
    from vocalparse.evaluation import (
        parse_transcription_text, aggregate_to_words,
        SILENCE_WORDS, _needleman_wunsch,
    )

    output_file = cfg.get("output", "")
    display_count = int(cfg.get("display", 20))

    cer_values = []

    for r in results:
        # Get GT lyrics
        gt_lyrics = r.get("gt_lyrics")
        if gt_lyrics is None and r.get("gt_ast"):
            gt_parsed = parse_transcription_text(r["gt_ast"])
            if gt_parsed:
                gt_words = aggregate_to_words(gt_parsed)
                gt_lyrics = "".join(
                    w.char for w in gt_words if w.char not in SILENCE_WORDS
                )

        # Parse pred to extract predicted lyrics
        pred_parsed = parse_transcription_text(r["pred"])
        if pred_parsed:
            pred_words = aggregate_to_words(pred_parsed)
            pred_chars = [w.char for w in pred_words if w.char not in SILENCE_WORDS]
        else:
            pred_chars = []

        r["_gt_lyrics"] = gt_lyrics
        r["_pred_lyrics"] = "".join(pred_chars)
        r["_parseable"] = pred_parsed is not None

        if gt_lyrics:
            gt_chars = list(gt_lyrics)
            alignment = _needleman_wunsch(
                gt_chars, pred_chars, eq_fn=lambda a, b: a == b,
            )
            f_sub = sum(1 for g, p in alignment if g and p and g != p)
            f_del = sum(1 for g, p in alignment if g and p is None)
            f_ins = sum(1 for g, p in alignment if g is None and p)
            cer = (f_sub + f_del + f_ins) / max(len(gt_chars), 1)
            cer_values.append(cer)
        else:
            cer_values.append(None)

    # ── Display samples ─────────────────────────────────────────
    n_display = min(display_count, len(results))
    if n_display > 0:
        print(f"\n{'='*70}")
        print(f"Sample predictions (first {n_display})")
        print(f"{'='*70}")
    for i in range(n_display):
        r = results[i]
        label = os.path.basename(r["audio_path"]) if r.get("audio_path") else f"Sample {r['idx']}"
        print(f"\n[{label}]")
        print(f"  Pred lyric: {r['_pred_lyrics'][:200]}")
        if r["_gt_lyrics"]:
            print(f"  GT lyric:   {r['_gt_lyrics'][:200]}")
        if cer_values[i] is not None:
            print(f"  CER: {cer_values[i]:.4f}")

    # ── Aggregate CER ───────────────────────────────────────────
    valid_cer = [c for c in cer_values if c is not None]
    if valid_cer:
        mean_cer = sum(valid_cer) / len(valid_cer)
        print(f"\n{'='*70}")
        print(f"CER Results ({len(valid_cer)} samples with GT text):")
        print(f"{'='*70}")
        print(f"  {'cer':18s}: {mean_cer:.4f}")

    # ── Save output (aggregate metrics only) ────────────────────
    if output_file:
        n_parseable = sum(1 for r in results if r["_parseable"])
        output_data = {
            "cer": mean_cer if valid_cer else None,
            "n_total_samples": len(results),
            "n_parseable": n_parseable,
            "n_with_gt": len(valid_cer),
        }
        # Remove None values
        output_data = {k: v for k, v in output_data.items() if v is not None}

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to {output_file}")


def _output_test_full(results, cfg):
    """Test-full mode: full AST metrics evaluation.

    Requires preprocessed input (gt_ast with full pitch/note/BPM info).
    Uses ``compute_metrics()`` + ``aggregate_metrics()``.
    """
    from vocalparse.evaluation import compute_metrics, aggregate_metrics

    length_matched_lyric_eval = bool(cfg.get("length_matched_lyric_eval", False))
    output_file = cfg.get("output", "")
    display_count = int(cfg.get("display", 20))

    # ── Display samples ─────────────────────────────────────────
    n_display = min(display_count, len(results))
    if n_display > 0:
        print(f"\n{'='*70}")
        print(f"Sample predictions (first {n_display})")
        print(f"{'='*70}")
    for i in range(n_display):
        r = results[i]
        print(f"\n[Sample {r['idx']}]")
        print(f"  PRED: {r['pred'][:200]}...")
        print(f"    GT: {r['gt_ast'][:200]}...")

    # ── Compute metrics ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Computing metrics...")
    print(f"{'='*70}")

    all_metrics = []
    unparseable_samples = []
    for i, r in enumerate(results):
        try:
            m = compute_metrics(
                r["gt_ast"], r["pred"],
                inference_only_metrics=True,
                length_matched_lyric_eval=length_matched_lyric_eval,
            )
            if m is not None:
                all_metrics.append(m)
                if m["n_pred_words"] == 0 and len(unparseable_samples) < 5:
                    unparseable_samples.append((i, r["pred"], r["gt_ast"]))
        except Exception:
            pass

    if all_metrics:
        agg = aggregate_metrics(
            all_metrics, inference_only_metrics=True,
        )
        n_total = int(agg.get("n_samples", 0))
        n_ok = int(agg.get("n_parseable", 0))

        print(f"\nResults ({n_ok}/{n_total} parseable):")
        for key in ["cer", "pitch_mae", "note_mae",
                     "dur_mae", "bpm_mae",
                     "pitch_error_rate", "note_num_mean_error"]:
            val = agg.get(key, float("nan"))
            if not math.isnan(val):
                print(f"  {key:18s}: {val:.4f}")
        if unparseable_samples:
            print(f"\nUnparseable predictions ({n_total - n_ok} total, showing up to 5):")
            for idx, pred, gt in unparseable_samples:
                print(f"  [Sample {idx}]")
                print(f"    PRED: {pred[:300]}")
                print(f"      GT: {gt[:300]}")
    else:
        agg = {}
        print("No parseable predictions — metrics unavailable.")

    # ── Save output (aggregate metrics only) ────────────────────
    if output_file and agg:
        output_data = {
            k: v for k, v in agg.items()
            if not (isinstance(v, float) and math.isnan(v))
        }
        output_data["n_total_samples"] = len(results)

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to {output_file}")


def _output_annotation(results, cfg):
    """Annotation mode: output Opencpop.json-compatible per-sample annotations.

    Output format per sample::

        {
            "item_name": "2001000001",
            "word": ["感", "受", "SP", ...],
            "pitch": [68, 68, 0, ...],
            "note": ["<NOTE_DOT_16>", "<NOTE_DOT_8>", ...],
            "pitch2word": [0, 1, 2, ...],
            "wav_fn": "/path/to/2001000001.wav",
            "bpm": 89
        }

    SP/AP words get ``pitch=0``.  No metrics are computed.
    """
    from vocalparse.evaluation import (
        parse_transcription_text, aggregate_to_words, _merge_same_pitch_pairs,
        SILENCE_WORDS,
    )

    output_file = cfg.get("output", "")
    display_count = int(cfg.get("display", 20))

    annotation_results = []

    for r in results:
        pred = r["pred"]
        audio_path = r.get("audio_path")
        idx = r.get("idx", 0)

        parsed = parse_transcription_text(pred)
        if parsed is None:
            item_name = Path(audio_path).stem if audio_path else str(idx)
            annotation_results.append({
                "item_name": item_name,
                "word": [], "pitch": [], "note": [], "pitch2word": [],
                "wav_fn": audio_path or "",
                "bpm": 120,
            })
            continue

        words = aggregate_to_words(parsed)

        word_list, pitch_list, note_list, pitch2word = [], [], [], []
        for word_idx, w in enumerate(words):
            word_list.append(w.char)
            merged_pairs = _merge_same_pitch_pairs(w.pairs)

            if w.char in SILENCE_WORDS:
                pitch_list.append(0)
                note_list.append(merged_pairs[0][1] if merged_pairs else "<NOTE_4>")
                pitch2word.append(word_idx)
            else:
                for pitch, note_tok, _ in merged_pairs:
                    pitch_list.append(pitch)
                    note_list.append(note_tok)
                    pitch2word.append(word_idx)

        item_name = Path(audio_path).stem if audio_path else str(idx)
        annotation_results.append({
            "item_name": item_name,
            "word": word_list,
            "pitch": pitch_list,
            "note": note_list,
            "pitch2word": pitch2word,
            "wav_fn": audio_path or "",
            "bpm": parsed["bpm"],
        })

    # ── Display samples ─────────────────────────────────────────
    n_display = min(display_count, len(annotation_results))
    if n_display > 0:
        print(f"\n{'='*70}")
        print(f"Annotation predictions (first {n_display})")
        print(f"{'='*70}")
    for i in range(n_display):
        entry = annotation_results[i]
        print(f"\n[{entry['item_name']}]")
        print(f"  BPM: {entry['bpm']}")
        words_preview = " ".join(entry["word"][:15])
        print(f"  Words ({len(entry['word'])}): {words_preview}"
              f"{'...' if len(entry['word']) > 15 else ''}")
        pitch_preview = entry["pitch"][:15]
        print(f"  Pitch ({len(entry['pitch'])}): {pitch_preview}"
              f"{'...' if len(entry['pitch']) > 15 else ''}")

    n_parseable = sum(1 for e in annotation_results if len(e["word"]) > 0)
    print(f"\nAnnotation: {n_parseable}/{len(annotation_results)} parseable samples")

    # ── Save output ─────────────────────────────────────────────
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(annotation_results, f, ensure_ascii=False, indent=2)
        print(f"\nAnnotations saved to {output_file}")
