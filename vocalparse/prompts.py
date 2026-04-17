# coding=utf-8
# VocalParse Prompt Construction and Annotation Conversion

from typing import Dict, List


# Non-lyrics characters to exclude in lyrics-conditioned mode
_NON_LYRICS_CHARS = {"AP", "SP"}


def convert_annotation_to_syllables(
    words: List[str],
    pitches: List[int],
    notes: List[str],
    pitch2word: List[int],
) -> List[Dict]:
    """Convert annotation format to syllables format."""
    syllables = []

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
            syllables.append({"char": word, "pitch": pitch, "note": note})
        else:
            pitch_list = [pitches[i] if i < len(pitches) else 0 for i in pitch_indices]
            note_list = [notes[i] if i < len(notes) else "<NOTE_4>" for i in pitch_indices]
            syllables.append({"char": word, "pitch": pitch_list, "note": note_list})

    return syllables


def expand_syllables(syllables: List[Dict]) -> List[Dict]:
    """Expand syllables with array pitch/note (melisma) into individual rows."""
    expanded = []
    for word_idx, syl in enumerate(syllables):
        pitch = syl['pitch']
        note = syl['note']
        char = syl['char']

        if isinstance(pitch, list):
            notes_list = note if isinstance(note, list) else [note] * len(pitch)
            for p, n in zip(pitch, notes_list):
                expanded.append({'char': char, 'pitch': p, 'note': n, 'word_idx': word_idx})
        else:
            expanded.append({'char': char, 'pitch': pitch, 'note': note, 'word_idx': word_idx})

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
