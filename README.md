# VocalParse

**VocalParse** is a minimal open-source extraction of the VocalParse subsystem from [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR). It keeps the core training, validation, inference, preprocessing, and evaluation pipeline for unified singing voice transcription, while removing unrelated experimental branches and auxiliary baselines.

The model transcribes singing audio into a structured autoregressive sequence containing lyrics, pitch, note values, and global tempo.

## Features

- Unified singing transcription in one decoder stream
- CoT-style prompting with `asr_cot`
- Two inference conditions:
  - `audio-only`: predict lyrics + score
  - `audio-lyric`: predict score from audio with ground-truth lyrics provided
- Three inference outputs:
  - `test_weak`: lyric CER only
  - `test_full`: lyric + pitch + note + BPM metrics
  - `annotation`: export Opencpop-style annotation JSON
- Built-in validation callback with TensorBoard score comparison figures
- Dynamic batching by mel-frame budget for training and inference

## Architecture

```text
Singing Audio (16kHz) -> Whisper Encoder -> Qwen LLM Decoder -> AST Token Sequence
                                                               |
                                                               v
                     感 <P_68> <NOTE_4> 受 <P_60> <NOTE_8> ... <BPM_89>
```

VocalParse extends the Qwen3-ASR vocabulary with about 400 AST tokens for pitch (`<P_0>` to `<P_127>`), note value (`<NOTE_*>`), and tempo (`<BPM_*>`). See [docs/note_tokens.md](docs/note_tokens.md).

## Pretrained Model

A fine-tuned checkpoint (Qwen3-ASR-1.7B, CoT training) is available on HuggingFace:

| Model | HuggingFace |
|---|---|
| VocalParse-1.7B (CoT) | [pymaster/VocalParse](https://huggingface.co/pymaster/VocalParse) |

Download with `huggingface_hub`:

```python
from huggingface_hub import snapshot_download
snapshot_download("pymaster/VocalParse", local_dir="./vocalparse-weights")
```

Or with the CLI:

```bash
huggingface-cli download pymaster/VocalParse --local-dir ./vocalparse-weights
```

Point the inference config to the downloaded directory:

```yaml
checkpoint: ./vocalparse-weights
```

## Installation

[uv](https://docs.astral.sh/uv/) is recommended for fast, reproducible environment setup.

```bash
# Create and activate a virtual environment
uv venv --python 3.10
source .venv/bin/activate

# Install PyTorch (adjust the index URL for your CUDA version)
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install VocalParse and all dependencies
uv pip install -e .
```

Optional Flash Attention:

```bash
uv pip install -e ".[flash]"
```

Notes:
- `qwen-asr` and all other dependencies are installed automatically.
- If `flash-attn` is unavailable in your environment, use `sdpa` in the inference config instead.
- The pretrained checkpoint above is based on `Qwen/Qwen3-ASR-1.7B`.

## Quick Start

### 1. Preprocess Data

Prepare a config like [configs/preprocess.yaml](configs/preprocess.yaml):

```yaml
model_path: Qwen/Qwen3-ASR-0.6B
output_dir: "/path/to/preprocessed"

datasets:
  - name: opencpop
    type: json_file
    json_path: /path/to/Opencpop.json
    audio_root: /path/to/Opencpop
```

Run:

```bash
python scripts/preprocess.py --config configs/preprocess.yaml --num_workers 16
```

### 2. Train

Prepare a config like [configs/train.yaml](configs/train.yaml):

```yaml
model_path: Qwen/Qwen3-ASR-1.7B
output_dir: ./vocalparse-runs/experiment-1
preprocessed_dir: "/path/to/preprocessed"
val_datasets:
  - opencpop
bpm_position: "last"
asr_cot: true
batch_size: 64
lr: 2e-5
epochs: 10
```

Single GPU:

```bash
python -m vocalparse.train --config configs/train.yaml
```

Multi-GPU:

```bash
torchrun --nproc_per_node=2 -m vocalparse.train --config configs/train.yaml
```

Training automatically resumes from the latest `checkpoint-*` under `output_dir`.

### 3. Inference

Prepare a config like [configs/inference.yaml](configs/inference.yaml):

```yaml
checkpoint: /path/to/checkpoint-or-output-dir
preprocessed_dir: "/path/to/preprocessed"
val_datasets:
  - opencpop
mode: "test_full"
inference_mode: "audio-only"
bpm_position: "last"
```

Run:

```bash
python -m vocalparse.inference --config configs/inference.yaml
```

## Supported Data Inputs

### Training Inputs

Training supports two sources:

1. `preprocessed_dir`
   Recommended. Loads Arrow shards produced by `scripts/preprocess.py` and avoids audio I/O during training.
2. `datasets`
   Raw scan mode. The training script can directly scan source annotations when `preprocessed_dir` is not provided.

Raw dataset formats:

- `json_file`
  One JSON file with fields such as `word`, `pitch`, `note`, `pitch2word`, `pitch_dur`, `wav_fn`, and `bpm`
- `folder_based`
  `dataset_root/song_id/segment.audio + segment.json + metadata.json`

Example raw-scan training config:

```yaml
model_path: Qwen/Qwen3-ASR-1.7B
output_dir: ./vocalparse-runs/raw-scan
datasets:
  - name: opencpop
    type: json_file
    json_path: /path/to/Opencpop.json
    audio_root: /path/to/Opencpop
val_datasets:
  - opencpop
```

### Inference Inputs

Inference supports two sources:

1. `preprocessed_dir`
   Uses full ground-truth AST targets. Required for `mode: test_full`.
2. `audio_json`
   Loads raw audio file lists. Supports:

```json
["/abs/path/a.wav", "/abs/path/b.flac"]
```

or:

```json
[
  {"audio_path": "a.wav", "text": "歌词一"},
  {"audio_path": "b.wav", "lyrics": "歌词二"}
]
```

When paths are relative, `audio_root` is prepended if provided.

## Inference Modes And Outputs

### Condition Modes

- `audio-only`
  Model predicts lyrics and score from audio alone.
- `audio-lyric`
  Model receives ground-truth lyrics in the prompt and predicts only the score. This is mainly for CoT-trained checkpoints.

### Output Modes

- `test_weak`
  CER-only evaluation. Works with raw audio JSON or preprocessed input.
- `test_full`
  Full AST evaluation with CER, pitch MAE, note MAE, duration MAE, and BPM MAE. Requires `preprocessed_dir` (not supported with `audio_json`).
- `annotation`
  Exports Opencpop-style per-sample annotation JSON. Works with either input type.

## Token Format

The interleaved AST format directly encodes lyric-note correspondence:

```text
感 <P_68> <NOTE_4> 受 <P_60> <NOTE_8> ... <BPM_89>
  ^         ^         ^         ^
 lyric    pitch     lyric    pitch
          note                note
```

With `asr_cot: true`:

```text
感受<|file_sep|>感 <P_68> <NOTE_4> 受 <P_60> <NOTE_8> ... <BPM_89>
```

## Configuration Reference

Key parameters and their defaults:

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `Qwen/Qwen3-ASR-1.7B` | Base model (HuggingFace ID or local path) |
| `output_dir` | `./vocalparse-runs/default` | Checkpoint and TensorBoard log directory. Existing checkpoints trigger auto-resume. |
| `bpm_position` | `"last"` | Where to place the `<BPM_*>` token: `"first"` or `"last"`. `"last"` is recommended (see below). |
| `asr_cot` | `false` | Chain-of-Thought mode: output pure lyrics before the interleaved score |
| `batch_size` | `8` | Per-device batch size (upper bound when dynamic batching is on) |
| `max_batch_mel_tokens` | `0` | Dynamic batching mel-frame budget per batch (0 = disabled) |
| `val_generate_samples` | `5` | Number of validation samples to run generation on |
| `val_display_samples` | `5` | Number of GT vs Pred score figures to log to TensorBoard |
| `lr` | `2e-5` | Peak learning rate |
| `warmup_ratio` | `0.02` | Fraction of total steps for LR warmup |
| `save_steps` | `200` | Checkpoint save interval |
| `eval_steps` | `save_steps` | Validation interval (defaults to `save_steps` if unset) |

### BPM Position

Experiments show that `bpm_position: "last"` consistently outperforms `"first"`. Predicting BPM after the note sequence gives the model a complete view of the musical phrase before committing to a tempo estimate. `"last"` is therefore the recommended default.

### Configuration Invariants

Changing these prompt-format settings requires retraining the model, but not re-running preprocessing:

- `bpm_position`
- `asr_cot`

## Training Monitoring

VocalParse logs to TensorBoard in `output_dir`. The validation callback runs `model.generate()` every `eval_steps` and logs:

| TensorBoard Key | Metric | Notes |
|---|---|---|
| `eval/cer` | Lyric CER | `(S+D+I) / N_gt`, silence tokens excluded |
| `eval/pitch_mae` | Pitch MAE | Absolute error in MIDI semitones |
| `eval/note_mae` | Note MAE | Absolute error in log₂ note-value space |
| `eval/dur_mae` | Duration MAE | Absolute error in log₂ seconds, derived from `note × 60 / BPM` |
| `eval/bpm_mae` | BPM MAE | Absolute tempo error |

Score comparison figures (GT vs Pred lyric rows and MIDI rows) are also logged per `val_display_samples`.

## Evaluation Metrics

| Metric | Description |
|---|---|
| `CER` | Character error rate on lyrics, excluding silence tokens |
| `Pitch MAE` | Absolute pitch error in MIDI semitones |
| `Note MAE` | Absolute error in log₂ note-value space |
| `Duration MAE` | Absolute error in log₂ seconds (`note × 60 / BPM`) |
| `BPM MAE` | Absolute tempo error |
| `Pitch Error Rate` | Fraction of aligned pairs with mismatched pitch (inference only) |
| `Note Num Mean Error` | Mean `|n_gt − n_pred|` note count difference per word (inference only) |

Metrics are computed with two-stage Needleman-Wunsch alignment:

1. Word-level alignment for lyrics
2. Pair-level alignment inside matched words for pitch, note, and duration

## Project Structure

```text
vocalparse/
  train.py          # Training entry point
  inference.py      # Inference entry point
  data.py           # Dataset loading and collators
  evaluation.py     # AST metrics
  validation.py     # Validation callback and visualization
  output.py         # Inference output formatting
  tokens.py         # AST token definitions
  prompts.py        # Prompt construction
  model.py          # Model patching and audio helpers
  checkpoint.py     # Checkpoint helpers
  distributed.py    # Batched inference and DDP helpers
scripts/
  preprocess.py     # Mel -> Arrow preprocessing
configs/
  preprocess.yaml   # Example preprocessing config
  train.yaml        # Example training config
  inference.yaml    # Example inference config
docs/
  note_tokens.md    # Token reference
```

## Citation

If you use VocalParse in your research, please cite:

```bibtex
@article{vocalparse2026,
  title={VocalParse: Towards Unified and Scalable Singing Voice Transcription with Large Audio Language Models},
  year={2026}
}
```

## Acknowledgments

VocalParse is built on [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) by the Alibaba Qwen team.

## License

Apache 2.0
