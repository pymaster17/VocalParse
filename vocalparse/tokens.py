# coding=utf-8
# VocalParse Token Definitions


def get_token_maps():
    """Returns mapping for SVS/AST tokens.

    Returns:
        (pitch_tokens, note_tokens, bpm_tokens, dur_units)
    """
    pitch_tokens = [f"<P_{i}>" for i in range(128)]
    base_note_tokens = ["<NOTE_1>", "<NOTE_2>", "<NOTE_4>", "<NOTE_8>", "<NOTE_16>", "<NOTE_32>"]
    dotted_note_tokens = ["<NOTE_DOT_1>", "<NOTE_DOT_2>", "<NOTE_DOT_4>", "<NOTE_DOT_8>", "<NOTE_DOT_16>", "<NOTE_DOT_32>"]
    note_tokens = base_note_tokens + dotted_note_tokens

    # Value map for initialization (Unit: Quarter note = 1.0)
    dur_units = {
        "<NOTE_32>": 0.125,
        "<NOTE_DOT_32>": 0.1875,
        "<NOTE_16>": 0.25,
        "<NOTE_DOT_16>": 0.375,
        "<NOTE_8>": 0.5,
        "<NOTE_DOT_8>": 0.75,
        "<NOTE_4>": 1.0,
        "<NOTE_DOT_4>": 1.5,
        "<NOTE_2>": 2.0,
        "<NOTE_DOT_2>": 3.0,
        "<NOTE_1>": 4.0,
        "<NOTE_DOT_1>": 6.0,
    }

    bpm_tokens = [f"<BPM_{i}>" for i in range(256)]

    return pitch_tokens, note_tokens, bpm_tokens, dur_units
