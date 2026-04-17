# coding=utf-8
# AST (Automatic Singing Transcription) Evaluation Metrics
#
# Quantitative evaluation of AST (Automatic Singing Transcription) accuracy by comparing
# ground-truth and predicted token sequences.
#
# Metrics:
#   - CER              : Character Error Rate on lyrics only (AP/SP excluded)
#   - CER (singing)    : Character Error Rate including AP/SP silence tokens
#   - Pitch MAE        : Mean Absolute Error in semitones
#   - Note MAE         : Mean Absolute Error in log2(note_duration) space
#   - Duration MAE     : Mean Absolute Error of physical duration (seconds)
#   - BPM MAE          : Tempo prediction error
#   - Pitch Error Rate : Fraction of aligned pairs with wrong pitch (inference-only)
#   - Note Num Mean Err: Mean |n_gt_pairs - n_pred_pairs| per word (inference-only)
#
# Algorithm:
#   1. Parse AST token sequences via parse_transcription_text()
#   2. Aggregate melisma entries into word-level structures (ASTWord)
#   3. Align GT/Pred word sequences using Needleman-Wunsch → CER
#   4. For aligned word pairs, align their pitch/note pairs → Pitch/Note/Dur MAE

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Silence/breath tokens in singing datasets (e.g. Opencpop)
# These are stripped before computing CER (lyrics-only).
SILENCE_WORDS = {"AP", "SP"}


# ══════════════════════════════════════════════════════════════════════
# Note duration map (quarter note = 1.0 beat)
# ══════════════════════════════════════════════════════════════════════

NOTE_DUR_MAP = {
    "<NOTE_32>": 0.125, "<NOTE_DOT_32>": 0.1875,
    "<NOTE_16>": 0.25,  "<NOTE_DOT_16>": 0.375,
    "<NOTE_8>": 0.5,    "<NOTE_DOT_8>": 0.75,
    "<NOTE_4>": 1.0,    "<NOTE_DOT_4>": 1.5,
    "<NOTE_2>": 2.0,    "<NOTE_DOT_2>": 3.0,
    "<NOTE_1>": 4.0,    "<NOTE_DOT_1>": 6.0,
}


# ══════════════════════════════════════════════════════════════════════
# AST text parsing
# ══════════════════════════════════════════════════════════════════════

def parse_transcription_text(text: str) -> Optional[Dict[str, Any]]:
    """Parse AST token sequence into structured components.

    Supports both BPM-first and BPM-last formats (auto-detected):
    - BPM-first: ``<BPM_89> 感 <P_68><NOTE_DOT_16> 受 <P_60><NOTE_8>``
    - BPM-last:  ``感 <P_68><NOTE_DOT_16> 受 <P_60><NOTE_8> <BPM_89>``
    - With duration: ``好 <dur_0.25> <P_68> <NOTE_4> 受 <dur_0.50> <P_60> <NOTE_8>``
    - Melisma (一字多音): ``感 <P_68><NOTE_8> <SLUR> <P_70><NOTE_4>``
      word "感" → first note, "⌒" → continuation notes
    - Error tolerance for malformed sequences from early training:
      Missing <NOTE> after <P>: default <NOTE_4>;
      Orphan <NOTE> without <P>: skipped;
      Garbled tokens: ignored.

    Returns:
        dict(bpm, words, pitches, notes, durations) or None if nothing parseable.
        ``durations`` is a list of float or None per entry (None when no dur token).
    """
    # Handle ASR-CoT format: "lyrics<|file_sep|>AST_sequence"
    # Strip the CoT prefix and parse only the AST sequence.
    _COT_SEP = '<|file_sep|>'
    if _COT_SEP in text:
        text = text.split(_COT_SEP)[-1]

    tokens = re.findall(r'<[^>]+>|[^\s<>]+', text)
    if not tokens:
        return None

    bpm = 120
    # Build list of [word, pitch, note_or_None, dur_or_None] entries
    entries = []
    cur_word = None
    last_is_slur = False
    pending_dur = None  # <dur_X.XX> seen before <P_xx>

    # Known meta tokens to skip
    _SKIP = {"<asr_text>", "language", "Chinese", "English", "Japanese", "Korean"}

    for tok in tokens:
        # EOS token
        if tok.endswith("/s>"):
            continue

        if tok.startswith("<BPM_"):
            m = re.match(r'<BPM_(\d+)>', tok)
            if m:
                bpm = int(m.group(1))

        elif tok.startswith("<dur_"):
            # Physical duration token: <dur_X.XX>
            m = re.match(r'<dur_(\d+\.\d+)>', tok)
            if m:
                pending_dur = float(m.group(1))

        elif tok.startswith("<P_"):
            m = re.match(r'<P_(\d+)>', tok)
            if m:
                pitch = int(m.group(1))
                # Previous entry missing note → fill default
                if entries and entries[-1][2] is None:
                    entries[-1][2] = "<NOTE_4>"
                word = cur_word if cur_word else ("⌒" if last_is_slur else "·")
                entries.append([word, pitch, None, pending_dur])
                cur_word = None
                last_is_slur = False
                pending_dur = None

        elif tok.startswith("<NOTE"):
            if tok in NOTE_DUR_MAP:
                if entries and entries[-1][2] is None:
                    entries[-1][2] = tok
                # else: orphan note → skip

        elif tok == "<SLUR>":
            last_is_slur = True

        elif tok in _SKIP:
            continue

        elif not tok.startswith("<"):
            # Lyric character (plain text)
            cur_word = tok

        # else: unknown special token → ignore

    # Fill default note for trailing entry
    if entries and entries[-1][2] is None:
        entries[-1][2] = "<NOTE_4>"

    if not entries:
        return None

    return {
        "bpm": bpm,
        "words": [e[0] for e in entries],
        "pitches": [e[1] for e in entries],
        "notes": [e[2] for e in entries],
        "durations": [e[3] for e in entries],
    }


# ══════════════════════════════════════════════════════════════════════
# Word-level aggregation (melisma grouping)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ASTWord:
    """A word (character) with its associated pitch/note/duration pairs."""
    char: str
    pairs: List[Tuple[int, str, Optional[float]]] = field(default_factory=list)
    # Each pair: (midi_pitch, note_token, duration_seconds_or_None)


def aggregate_to_words(parsed: Dict[str, Any]) -> List[ASTWord]:
    """Aggregate parse_transcription_text output into word-level structure.

    Two types of continuation entries are merged into the preceding word:
    - '⌒' (SLUR-based melisma): explicit continuation from ``<SLUR>`` token
    - '·' (orphan pitch): pitch appeared without a preceding character,
      common in model-generated pseudo-melisma

    This ensures a single word may carry multiple (pitch, note, dur) tuples.
    """
    words: List[ASTWord] = []
    for w, p, n, d in zip(
        parsed["words"], parsed["pitches"],
        parsed["notes"], parsed["durations"],
    ):
        if w in ("⌒", "·") and words:
            # Melisma / pseudo-melisma → append to previous word
            words[-1].pairs.append((p, n, d))
        else:
            words.append(ASTWord(char=w, pairs=[(p, n, d)]))
    return words


def _remove_silence_words(words: List[ASTWord]) -> List[ASTWord]:
    """Filter out AP/SP tokens from an ASTWord sequence."""
    return [w for w in words if w.char not in SILENCE_WORDS]


# ══════════════════════════════════════════════════════════════════════
# Tie resolution (merge consecutive same-pitch pairs)
# ══════════════════════════════════════════════════════════════════════

# Sorted by beat-duration for closest-match lookup
_BEATS_TO_NOTE = sorted(NOTE_DUR_MAP.items(), key=lambda x: x[1])


def _beats_to_note_token(beats: float) -> str:
    """Map a beat-duration to the closest NOTE token.

    For example 1.0 → <NOTE_4>, 0.5 → <NOTE_8>, 0.625 → <NOTE_8> (closest).
    """
    best_token = "<NOTE_4>"
    best_dist = float("inf")
    for tok, dur in _BEATS_TO_NOTE:
        dist = abs(dur - beats)
        if dist < best_dist:
            best_dist = dist
            best_token = tok
    return best_token


def _merge_same_pitch_pairs(
    pairs: List[Tuple[int, str, Optional[float]]],
) -> List[Tuple[int, str, Optional[float]]]:
    """Merge consecutive pairs that share the same pitch (tie resolution).

    Models sometimes predict "pseudo-melisma" — multiple consecutive notes
    at the **same** pitch, which is musically equivalent to a single longer
    note (a tie). For example::

        (68, <NOTE_8>), (68, <NOTE_8>)  →  (68, <NOTE_4>)   # 0.5+0.5 = 1.0

    This merge is applied to both GT and Pred before pair-level alignment,
    ensuring that the evaluation compares musical *intent* rather than
    notational *choices*.

    Beat-durations are summed and mapped back to the closest NOTE token.
    Physical durations (if present) are also summed.
    """
    if len(pairs) <= 1:
        return pairs

    merged: List[Tuple[int, str, Optional[float]]] = []
    i = 0
    while i < len(pairs):
        pitch, note, dur = pairs[i]
        acc_beats = NOTE_DUR_MAP.get(note, 1.0)
        acc_dur = dur  # physical duration (seconds) or None

        # Absorb consecutive pairs with the same pitch
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pitch:
            _, next_note, next_dur = pairs[j]
            acc_beats += NOTE_DUR_MAP.get(next_note, 1.0)
            if acc_dur is not None and next_dur is not None:
                acc_dur += next_dur
            elif next_dur is not None:
                acc_dur = next_dur
            j += 1

        merged_note = _beats_to_note_token(acc_beats) if j > i + 1 else note
        merged.append((pitch, merged_note, acc_dur))
        i = j

    return merged


# ══════════════════════════════════════════════════════════════════════
# Needleman-Wunsch global sequence alignment
# ══════════════════════════════════════════════════════════════════════

def _needleman_wunsch(seq_a, seq_b, eq_fn=None, gap_penalty=1):
    """Global sequence alignment using Needleman-Wunsch algorithm.

    Finds the minimum-cost alignment between two sequences, allowing
    matches (cost 0), substitutions (cost 1), insertions, and deletions.
    This is mathematically equivalent to Viterbi decoding on a pair-HMM
    over the 2D alignment grid.

    Args:
        seq_a: First sequence (treated as "reference" / GT).
        seq_b: Second sequence (treated as "hypothesis" / Pred).
        eq_fn: Equality function (a, b) → bool. Default: ``a == b``.
        gap_penalty: Cost of insertion/deletion.

    Returns:
        List of (a_item_or_None, b_item_or_None) tuples representing
        the optimal alignment. None indicates a gap.
    """
    if eq_fn is None:
        eq_fn = lambda a, b: a == b

    m, n = len(seq_a), len(seq_b)

    # Cost matrix (lower = better)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i * gap_penalty
    for j in range(n + 1):
        dp[0][j] = j * gap_penalty

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = 0 if eq_fn(seq_a[i - 1], seq_b[j - 1]) else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,   # match / substitution
                dp[i - 1][j] + gap_penalty,     # deletion (GT has, Pred missing)
                dp[i][j - 1] + gap_penalty,     # insertion (Pred extra)
            )

    # Traceback
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_cost = 0 if eq_fn(seq_a[i - 1], seq_b[j - 1]) else 1
            if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                alignment.append((seq_a[i - 1], seq_b[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + gap_penalty:
            alignment.append((seq_a[i - 1], None))
            i -= 1
        else:
            alignment.append((None, seq_b[j - 1]))
            j -= 1

    alignment.reverse()
    return alignment


# ══════════════════════════════════════════════════════════════════════
# Per-sample AST metrics
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(
    gt_text: str,
    pred_text: str,
    include_dur: bool = False,
    inference_only_metrics: bool = False,
    length_matched_lyric_eval: bool = False,
) -> Optional[Dict[str, Any]]:
    """Compute AST evaluation metrics between GT and Pred sequences.

    Two-layer alignment:
      Layer 1 — Word-level Needleman-Wunsch on aggregated characters → CER
      Layer 2 — Pair-level NW within each aligned word pair → Pitch/Note/Dur MAE

    Two CER variants are always computed:
      - ``cer``: Lyrics-only CER (AP/SP excluded), comparable to standard ASR.
      - ``cer_singing``: Full CER including AP/SP silence tokens.

    Args:
        gt_text: Ground-truth AST text (as produced by build_interleaved_text).
        pred_text: Predicted AST text (from model.generate).
        include_dur: Whether to compute duration MAE (requires <dur_X.XX> tokens).
        inference_only_metrics: When True, also compute pitch_error_rate and
            note_num_mean_error (skipped during validation to avoid overhead).
        length_matched_lyric_eval: For datasets whose GT "word" units are not
            in the same symbol space as model output (e.g. phoneme groups vs
            Chinese characters), require that GT/pred lyric lengths match after
            removing AP/SP, then align lyric words strictly by position for
            pair-level pitch/note/duration metrics. If lengths mismatch, skip
            the sample by returning None. CER remains the original literal
            token-level metric and is not corrected by this fallback.

    Returns:
        Dict with metrics, or None if either sequence cannot be parsed.
        Keys: cer, cer_singing, pitch_mae, note_mae, bpm_mae,
              dur_mae (if include_dur),
              n_gt_words, n_pred_words, n_aligned_pairs.
    """
    gt_parsed = parse_transcription_text(gt_text)
    pred_parsed = parse_transcription_text(pred_text)

    if gt_parsed is None:
        return None

    # Handle unparseable predictions (early training garbage)
    if pred_parsed is None:
        n_gt = len(set(range(len(gt_parsed["words"]))))  # count GT words
        gt_words = aggregate_to_words(gt_parsed)
        result = {
            "cer": 1.0,
            "cer_singing": 1.0,
            "pitch_mae": float("nan"),
            "note_mae": float("nan"),
            "bpm_mae": float("nan"),
            "n_gt_words": len(gt_words),
            "n_pred_words": 0,
            "n_aligned_pairs": 0,
        }
        if include_dur:
            result["dur_mae"] = float("nan")
        return result

    gt_words = aggregate_to_words(gt_parsed)
    pred_words = aggregate_to_words(pred_parsed)
    gt_lyric = _remove_silence_words(gt_words)
    pred_lyric = _remove_silence_words(pred_words)

    # ── Layer 1: Word-level alignment → CER ──────────────────────────

    word_alignment = _needleman_wunsch(
        gt_words, pred_words,
        eq_fn=lambda a, b: a.char == b.char,
    )

    substitutions = sum(1 for g, p in word_alignment if g and p and g.char != p.char)
    deletions = sum(1 for g, p in word_alignment if g and p is None)
    insertions = sum(1 for g, p in word_alignment if g is None and p)
    n_gt = len(gt_words)
    cer_singing = (substitutions + deletions + insertions) / max(n_gt, 1)

    # ── Layer 2: Pair-level alignment → Pitch / Note / Dur MAE ───────

    pitch_errors: List[float] = []
    note_errors: List[float] = []
    dur_errors: List[float] = []
    abs_note_dur_errors: List[float] = []  # absolute note duration (seconds)
    pitch_binary_errors: List[float] = []  # 0/1 per aligned pair (for PER)
    note_num_diffs: List[float] = []       # |n_gt - n_pred| per word pair

    gt_beat_dur = 60.0 / max(gt_parsed["bpm"], 1)    # seconds per beat
    pred_beat_dur = 60.0 / max(pred_parsed["bpm"], 1)

    pair_eval_alignment = word_alignment
    if length_matched_lyric_eval:
        if len(gt_lyric) != len(pred_lyric):
            return None
        pair_eval_alignment = list(zip(gt_lyric, pred_lyric))

    for g_word, p_word in pair_eval_alignment:
        if g_word is None or p_word is None:
            continue  # gap — no pair-level comparison possible

        # Merge consecutive same-pitch pairs (tie resolution) before
        # alignment so that e.g. (68, NOTE_8)(68, NOTE_8) ≡ (68, NOTE_4).
        g_pairs = _merge_same_pitch_pairs(g_word.pairs)
        p_pairs = _merge_same_pitch_pairs(p_word.pairs)

        # Note number mean error: |n_gt_pairs - n_pred_pairs| per word
        if inference_only_metrics:
            note_num_diffs.append(abs(len(g_pairs) - len(p_pairs)))

        # Align (pitch, note, dur) pairs within this word pair
        pair_alignment = _needleman_wunsch(
            g_pairs, p_pairs,
            eq_fn=lambda a, b: a[0] == b[0] and a[1] == b[1],
        )

        for g_pair, p_pair in pair_alignment:
            if g_pair is None or p_pair is None:
                continue  # pair-level gap

            # Pitch MAE (semitones)
            pitch_errors.append(abs(g_pair[0] - p_pair[0]))

            # Pitch Error Rate: binary 0/1 (inference-only)
            if inference_only_metrics:
                pitch_binary_errors.append(0.0 if g_pair[0] == p_pair[0] else 1.0)

            # Note MAE (log2 of note duration value)
            g_note_dur = NOTE_DUR_MAP.get(g_pair[1], 1.0)
            p_note_dur = NOTE_DUR_MAP.get(p_pair[1], 1.0)
            note_errors.append(abs(math.log2(g_note_dur) - math.log2(p_note_dur)))

            # Absolute note duration MAE (log2-seconds)
            # abs_dur = note_beats × 60 / BPM; log2 for perceptual uniformity
            g_abs = g_note_dur * gt_beat_dur
            p_abs = p_note_dur * pred_beat_dur
            abs_note_dur_errors.append(abs(math.log2(g_abs) - math.log2(p_abs)))

            # Duration MAE (seconds)
            if include_dur and g_pair[2] is not None and p_pair[2] is not None:
                dur_errors.append(abs(g_pair[2] - p_pair[2]))

    # ── CER (lyrics-only, AP/SP excluded) ─────────────────────────────

    if gt_lyric:
        lyric_alignment = _needleman_wunsch(
            gt_lyric, pred_lyric,
            eq_fn=lambda a, b: a.char == b.char,
        )
        f_sub = sum(1 for g, p in lyric_alignment if g and p and g.char != p.char)
        f_del = sum(1 for g, p in lyric_alignment if g and p is None)
        f_ins = sum(1 for g, p in lyric_alignment if g is None and p)
        cer_value = (f_sub + f_del + f_ins) / max(len(gt_lyric), 1)
    else:
        cer_value = 0.0 if not pred_lyric else 1.0

    # ── Assemble results ─────────────────────────────────────────────

    def _safe_mean(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    result = {
        "cer": cer_value,
        "cer_singing": cer_singing,
        "pitch_mae": _safe_mean(pitch_errors),
        "note_mae": _safe_mean(note_errors),
        "abs_note_dur_mae": _safe_mean(abs_note_dur_errors),
        "bpm_mae": float(abs(gt_parsed["bpm"] - pred_parsed["bpm"])),
        "n_gt_words": n_gt,
        "n_pred_words": len(pred_words),
        "n_aligned_pairs": len(pitch_errors),
    }
    if include_dur:
        result["dur_mae"] = _safe_mean(dur_errors)
    if inference_only_metrics:
        result["pitch_error_rate"] = _safe_mean(pitch_binary_errors)
        result["note_num_mean_error"] = _safe_mean(note_num_diffs)

    return result


# ══════════════════════════════════════════════════════════════════════
# Aggregate per-sample metrics to dataset level
# ══════════════════════════════════════════════════════════════════════

def aggregate_metrics(
    metrics_list: List[Dict[str, Any]],
    include_dur: bool = False,
    inference_only_metrics: bool = False,
) -> Dict[str, float]:
    """Aggregate per-sample AST metrics into dataset-level statistics.

    NaN values (from unparseable predictions) are excluded from averages.

    Both ``cer`` (lyrics-only) and ``cer_singing`` (with AP/SP) are always
    aggregated.

    Args:
        metrics_list: List of dicts from compute_metrics().
        include_dur: Whether dur_mae should be included.

    Returns:
        Dict with aggregated metrics (mean values) and sample count.
    """
    if not metrics_list:
        return {}

    keys = ["cer", "cer_singing", "pitch_mae", "note_mae", "abs_note_dur_mae", "bpm_mae"]
    if include_dur:
        keys.append("dur_mae")
    if inference_only_metrics:
        keys.extend(["pitch_error_rate", "note_num_mean_error"])

    agg: Dict[str, float] = {}
    for key in keys:
        values = [m[key] for m in metrics_list if not math.isnan(m.get(key, float("nan")))]
        agg[key] = sum(values) / len(values) if values else float("nan")
    agg["n_samples"] = float(len(metrics_list))
    agg["n_parseable"] = float(sum(1 for m in metrics_list if m["n_pred_words"] > 0))
    agg["n_total_samples"] = len(metrics_list)

    return agg
