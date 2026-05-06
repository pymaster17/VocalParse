# VocalParse

**VocalParse** is a minimal open-source extraction of the VocalParse subsystem from [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR). It keeps the core training, validation, inference, preprocessing, and evaluation pipeline for unified singing voice transcription, while removing unrelated experimental branches and auxiliary baselines.

The model transcribes singing audio into a structured autoregressive sequence containing lyrics, pitch, note values, and global tempo.

## Features

- Unified singing transcription in one decoder stream
- CoT-style prompting with `asr_cot`
- Two inference entry points:
  - `transcribe_one(audio, checkpoint)`: minimal single-sample demo
  - `VocalParseTranscriber`: production batch API with built-in multi-GPU torchrun support
- Built-in validation callback with TensorBoard score comparison figures
- Dynamic batching by mel-frame budget for training and inference

## What do you want to do?

VocalParse exposes three independent workflows; pick whichever matches your goal.

### 🎵 Quick try-out: single-sample inference

For first-time users who just want to see what the model produces on a wav file.

```python
from vocalparse import transcribe_one

text = transcribe_one(
    audio="path/to/song.wav",
    checkpoint="./vocalparse-weights",
)
print(text)
# 感 <P_68> <NOTE_4> 受 <P_60> <NOTE_8> ... <BPM_89>
```

Or via the command line:

```bash
python -m vocalparse.demo --audio path/to/song.wav --checkpoint ./vocalparse-weights
```

See [vocalparse/demo.py](vocalparse/demo.py). The module is intentionally compact — loading, preprocessing, generation, and decoding all live in one readable file.

### 🏭 Production batch: as a library inside another pipeline

For embedding VocalParse in SVS / TTS pipelines or running large-scale audio annotation. Internally uses multi-GPU torchrun, mel-budget batch packing, CPU prep ‖ GPU generate prefetch, and cross-rank work-steal scheduling — all hidden from the caller.

```python
from vocalparse import VocalParseTranscriber

trx = VocalParseTranscriber(checkpoint="./vocalparse-weights")
results = trx.transcribe([wav_a, wav_b, ...])  # list[np.float32 array] -> list[str]
```

Multi-GPU: launch your script with `torchrun --nproc_per_node=4 your_script.py` — `VocalParseTranscriber` reads `RANK` / `WORLD_SIZE` from env and gathers results to rank 0 automatically.

See [vocalparse/api.py](vocalparse/api.py) and the benchmark in [scripts/benchmark_api.py](scripts/benchmark_api.py).

### 🎓 Training / fine-tuning

For training on your own singing data. Pipeline: preprocess → train. Jump to [Quick Start § Preprocess Data](#1-preprocess-data) below.

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

Pass the downloaded path as `checkpoint` to either `transcribe_one(...)` or `VocalParseTranscriber(checkpoint=...)`.

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

Flash Attention is optional. The default install uses PyTorch SDPA, which works everywhere. If your environment supports it, you can install Flash Attention for a speedup:

```bash
uv pip install -e ".[flash]"
```

Notes:
- `qwen-asr` and all other dependencies are installed automatically.
- With the default install, pass `attn_implementation="sdpa"` to `transcribe_one` / `VocalParseTranscriber` (the training script does not require a flash-attn backend).
- The pretrained checkpoint above is based on `Qwen/Qwen3-ASR-1.7B`.

## Quick Start

### 1. Preprocess Data

Ready-to-use annotation JSONs for three public singing corpora are bundled under [data/](data/):

| File | Samples | Dataset |
|---|---|---|
| [data/Opencpop.json](data/Opencpop.json) | 3,756 | [Opencpop](https://wenet.org.cn/opencpop/) |
| [data/gtsinger.json](data/gtsinger.json) | 7,139 | [GTSinger](https://github.com/GTSinger/GTSinger) |
| [data/m4singer.json](data/m4singer.json) | 20,896 | [M4Singer](https://github.com/M4Singer/M4Singer) |

Download the raw audio from each dataset's official source and point `audio_root` at your local copy. A ready config is provided in [configs/preprocess.yaml](configs/preprocess.yaml):

```yaml
model_path: Qwen/Qwen3-ASR-1.7B
output_dir: "/path/to/preprocessed"

datasets:
  - name: opencpop
    type: json_file
    json_path: data/Opencpop.json
    audio_root: /path/to/Opencpop

  - name: gtsinger
    type: json_file
    json_path: data/gtsinger.json
    audio_root: /path/to/GTSinger

  - name: m4singer
    type: json_file
    json_path: data/m4singer.json
    audio_root: /path/to/m4singer
```

Run:

```bash
python scripts/preprocess.py --config configs/preprocess.yaml --num_workers 16
```

To use your own dataset, produce a JSON list with fields `word`, `pitch`, `note`, `pitch2word`, `pitch_dur`, `word_dur`, `wav_fn`, `bpm` (see the bundled files for reference), or use the `folder_based` layout described in [Supported Data Inputs](#supported-data-inputs).

### 2. Train

Prepare a config like [configs/train.yaml](configs/train.yaml):

```yaml
model_path: Qwen/Qwen3-ASR-1.7B
output_dir: ./vocalparse-runs/experiment-1
preprocessed_dir: "/path/to/preprocessed"

# Datasets listed under val_datasets are moved entirely out of the
# training set and used as validation. With the default three-corpus
# setup, GTSinger + m4singer train the model and Opencpop validates it.
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

See the two inference paths in the [What do you want to do?](#what-do-you-want-to-do) section above:

- **Single-sample quick inference**: [vocalparse/demo.py](vocalparse/demo.py) or `from vocalparse import transcribe_one`
- **Production batch inference**: [vocalparse/api.py](vocalparse/api.py) or `from vocalparse import VocalParseTranscriber`

For offline evaluation / annotation pipelines, build your own script on top of `VocalParseTranscriber` and the parsing utilities in [vocalparse/evaluation.py](vocalparse/evaluation.py) (`parse_transcription_text`, etc.). Use [scripts/benchmark_api.py](scripts/benchmark_api.py) as a starting template.

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

The inference APIs (`transcribe_one` / `VocalParseTranscriber`) take runtime data directly — no config file required:

- `transcribe_one`: a wav file path or a 1D `np.float32` array
- `VocalParseTranscriber.transcribe`: `list[np.float32 array]` (mono, 16 kHz)

Audio loading is the caller's responsibility (we recommend `librosa.load(path, sr=16000, mono=True)`).

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
  # Three user-facing entry points
  demo.py           # Single-sample quick inference (transcribe_one)
  api.py            # Production batch inference (VocalParseTranscriber)
  train.py          # Training entry point

  # Shared core
  model.py          # Model loading, patching, audio helpers
  prompts.py        # Prompt construction
  tokens.py         # AST token definitions
  evaluation.py     # AST metrics and parsing utilities
  data.py           # Dataset loading and collators
  validation.py    # Train-time validation callback and visualization
  checkpoint.py     # Checkpoint helpers
  distributed.py    # DDP / batch packing / per-sample encoding (used by api.py)
scripts/
  preprocess.py       # Mel -> Arrow preprocessing
  benchmark_api.py    # End-to-end benchmark for VocalParseTranscriber
configs/
  preprocess.yaml   # Example preprocessing config
  train.yaml        # Example training config
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
