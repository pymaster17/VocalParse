# VocalParse AST Token Reference

VocalParse extends the Qwen3-ASR tokenizer with approximately 1,400 AST-specific tokens.

## Pitch Tokens

128 tokens representing MIDI note numbers:

```
<P_0>, <P_1>, ..., <P_127>
```

| Token | Meaning |
|---|---|
| <P_60> | Middle C (C4) |
| <P_69> | Concert A (A4) |

## Note Duration Tokens

12 tokens representing symbolic note durations (quarter note = 1.0 beat):

| Token | Note Value | Nominal Duration @ 120 BPM |
|---|---|---|
| <NOTE_32> | 0.125 | 0.0625 s |
| <NOTE_DOT_32> | 0.1875 | 0.09375 s |
| <NOTE_16> | 0.25 | 0.125 s |
| <NOTE_DOT_16> | 0.375 | 0.1875 s |
| <NOTE_8> | 0.5 | 0.25 s |
| <NOTE_DOT_8> | 0.75 | 0.375 s |
| <NOTE_4> | 1.0 | 0.5 s |
| <NOTE_DOT_4> | 1.5 | 0.75 s |
| <NOTE_2> | 2.0 | 1.0 s |
| <NOTE_DOT_2> | 3.0 | 1.5 s |
| <NOTE_1> | 4.0 | 2.0 s |
| <NOTE_DOT_1> | 6.0 | 3.0 s |

## BPM Tokens

256 tokens representing beats per minute:

```
<BPM_0>, <BPM_1>, ..., <BPM_255>
```

A single global token per song. The nominal duration of a note in seconds is:

```
Note^d = 60 / BPM * Note^v
```

## Physical Duration Tokens

1,000 tokens quantizing physical duration in 0.01-second steps:

```
<dur_0.01>, <dur_0.02>, ..., <dur_10.00>
```

Used when `include_dur: true` in the training/inference configuration.

## Special Tokens

| Token | Meaning |
|---|---|
| <SLUR> | Melisma indicator (one word, multiple notes) |
| <SVS_MASK> | Masking token for singing voice synthesis |

## Example Sequences

### Standard (BPM-last, no duration)
```
感 <P_68> <NOTE_4> 受 <P_60> <NOTE_8> ... <BPM_89>
```

### With physical duration (Chain-of-Thought)
```
感 <dur_0.25> <P_68> <NOTE_4> 受 <dur_0.50> <P_60> <NOTE_8> ... <BPM_89>
```

### Chain-of-Thought (CoT) with lyrics prefix
```
感受<|file_sep|>感 <P_68> <NOTE_4> 受 <P_60> <NOTE_8> ... <BPM_89>
```
