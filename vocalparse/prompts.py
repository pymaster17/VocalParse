# coding=utf-8
# VocalParse Prompt Construction and Annotation Conversion

from typing import Any, Dict, List, Optional

from vocalparse.tokens import _quantize_dur


# Non-lyrics characters to exclude in lyrics-conditioned mode
_NON_LYRICS_CHARS = {"AP", "SP"}


def convert_annotation_to_syllables(
    words: List[str],
    pitches: List[int],
    notes: List[str],
    pitch2word: List[int],
    pitch_durs: Optional[List[float]] = None,
) -> List[Dict]:
    """Convert annotation format to syllables format."""
    syllables = []
    has_durs = pitch_durs is not None and len(pitch_durs) > 0

    word_to_pitches: Dict[int, List[int]] = {}
    for pitch_idx, word_idx in enumerate(pitch2word):
        if word_idx not in word_to_pitches:
            word_to_pitches[word_idx] = []
        word_to_pitches[word_idx].append(pitch_idx)

    for word_idx, word in enumerate(words):
        pitch_indices = word_to_pitches.get(word_idx, [])
        if not pitch_indices:
            continue

        if len(pitch_indices) == 1:
            pitch_idx = pitch_indices[0]
            pitch = pitches[pitch_idx] if pitch_idx < len(pitches) else 0
            note = notes[pitch_idx] if pitch_idx < len(notes) else "<NOTE_4>"
            syl: Dict[str, Any] = {"char": word, "pitch": pitch, "note": note}
            if has_durs:
                syl["pitch_dur"] = pitch_durs[pitch_idx] if pitch_idx < len(pitch_durs) else 0.0
            syllables.append(syl)
        else:
            pitch_list = [pitches[i] if i < len(pitches) else 0 for i in pitch_indices]
            note_list = [notes[i] if i < len(notes) else "<NOTE_4>" for i in pitch_indices]
            syl = {"char": word, "pitch": pitch_list, "note": note_list}
            if has_durs:
                syl["pitch_dur"] = [pitch_durs[i] if i < len(pitch_durs) else 0.0 for i in pitch_indices]
            syllables.append(syl)

    return syllables


def expand_syllables(syllables: List[Dict]) -> List[Dict]:
    """Expand syllables with array pitch/note (melisma) into individual rows."""
    expanded = []
    for word_idx, syl in enumerate(syllables):
        pitch = syl['pitch']
        note = syl['note']
        char = syl['char']
        pitch_dur = syl.get('pitch_dur')

        if isinstance(pitch, list):
            notes_list = note if isinstance(note, list) else [note] * len(pitch)
            durs = pitch_dur if isinstance(pitch_dur, list) else [pitch_dur] * len(pitch) if pitch_dur is not None else [None] * len(pitch)
            for p, n, d in zip(pitch, notes_list, durs):
                row: Dict[str, Any] = {'char': char, 'pitch': p, 'note': n, 'word_idx': word_idx}
                if d is not None:
                    row['pitch_dur'] = d
                expanded.append(row)
        else:
            row = {'char': char, 'pitch': pitch, 'note': note, 'word_idx': word_idx}
            if pitch_dur is not None:
                row['pitch_dur'] = pitch_dur
            expanded.append(row)

    return expanded


def extract_lyrics_text(syllables: List[Dict]) -> str:
    """Extract raw lyrics characters from syllables (no AP/SP)."""
    chars = []
    for syl in syllables:
        char = syl.get("char", "")
        if char and char not in ("AP", "SP", ""):
            chars.append(char)
    return "".join(chars)


def filter_lyrics_syllables(syllables: List[Dict]) -> List[Dict]:
    """Remove non-lyrics syllables (AP, SP) from syllable list."""
    return [syl for syl in syllables if syl.get("char", "") not in _NON_LYRICS_CHARS]


def build_interleaved_text(
    syllables: List[Dict],
    bpm: int,
    bpm_position: str = "last",
    include_dur: bool = False,
) -> str:
    """Build interleaved lyric-note text from syllables."""
    tokens: List[str] = []
    bpm_tok = f"<BPM_{bpm}>" if 0 <= bpm <= 255 else "<BPM_120>"

    if bpm_position == "first":
        tokens.append(bpm_tok)

    expanded = expand_syllables(syllables)

    prev_word_idx = None
    for syl in expanded:
        char = syl["char"]
        cur_word_idx = syl.get("word_idx", id(syl))
        is_melisma = prev_word_idx is not None and cur_word_idx == prev_word_idx

        if not is_melisma:
            tokens.append(char)

        if include_dur:
            pitch_dur = syl.get("pitch_dur")
            if pitch_dur is not None and isinstance(pitch_dur, (int, float)):
                tokens.append(_quantize_dur(float(pitch_dur)))

        pitch = syl.get("pitch", 0)
        note = syl.get("note", "<NOTE_4>")
        if isinstance(pitch, int) and 0 <= pitch <= 127:
            tokens.append(f"<P_{pitch}>")
        if isinstance(note, str) and note.startswith("<NOTE"):
            tokens.append(note)

        prev_word_idx = cur_word_idx

    if bpm_position == "last":
        tokens.append(bpm_tok)

    return " ".join(tokens)


def build_prefix_text(processor, lyrics_text: str = None) -> str:
    """Build chat template prefix text."""
    dummy_msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]
    base_prefix = processor.apply_chat_template(
        [dummy_msgs], add_generation_prompt=True, tokenize=False
    )[0]

    if lyrics_text:
        user_end_marker = "<|audio_end|><|im_end|>"
        if user_end_marker in base_prefix:
            base_prefix = base_prefix.replace(
                user_end_marker,
                f"<|audio_end|>{lyrics_text}<|im_end|>",
                1,
            )

    return base_prefix
