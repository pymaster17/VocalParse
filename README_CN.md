# VocalParse

**VocalParse** 是从 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 中抽离出的最小开源子仓库，保留了 VocalParse 相关的核心训练、验证、推理、预处理和评估链路，同时去掉了无关实验分支和外围基线代码。

它将歌唱音频转录为统一的自回归结构化序列，包含歌词、音高、音符时值、可选物理时长和全局 BPM。

## 特性

- **统一转录**：单一解码器流同时建模歌词和旋律
- **CoT 提示**：通过 `asr_cot` 支持“先歌词、后乐谱”的训练目标
- **两种推理条件**：
  - `audio-only`：仅从音频预测歌词和乐谱
  - `audio-lyric`：给定歌词，仅预测乐谱
- **三种输出模式**：
  - `test_weak`：仅歌词 CER
  - `test_full`：完整 AST 指标
  - `annotation`：导出 Opencpop 风格标注 JSON
- **验证可视化**：TensorBoard 内置 GT vs Pred 乐谱对比图
- **动态批处理**：按 mel 帧预算进行训练和推理 batch 打包

## 架构

```text
歌唱音频 (16kHz) -> Whisper 编码器 -> Qwen LLM 解码器 -> AST Token 序列
                                                              |
                                                              v
                    感 <P_68> <NOTE_4> 受 <P_60> <NOTE_8> ... <BPM_89>
```

VocalParse 扩展了 Qwen3-ASR 的词表，新增约 1,400 个 AST token，包括音高（`<P_0>` 到 `<P_127>`）、音符时值（`<NOTE_*>`）、速度（`<BPM_*>`）和物理时长（`<dur_*>`）。完整列表参见 [docs/note_tokens.md](docs/note_tokens.md)。

## 预训练模型

基于 Qwen3-ASR-1.7B、CoT 训练的微调 checkpoint 已发布到 HuggingFace：

| 模型 | HuggingFace |
|---|---|
| VocalParse-1.7B (CoT) | [pymaster/VocalParse](https://huggingface.co/pymaster/VocalParse) |

通过 `huggingface_hub` 下载：

```python
from huggingface_hub import snapshot_download
snapshot_download("pymaster/VocalParse", local_dir="./vocalparse-weights")
```

或使用 CLI：

```bash
huggingface-cli download pymaster/VocalParse --local-dir ./vocalparse-weights
```

下载后在推理配置中指向该目录：

```yaml
checkpoint: ./vocalparse-weights
```

## 安装

先根据你的 CPU/CUDA 环境安装 PyTorch，再安装 VocalParse：

```bash
pip install torch torchaudio
pip install -e .
```

可选安装 Flash Attention：

```bash
pip install -e ".[flash]"
```

说明：
- `qwen-asr` 会作为依赖自动安装。
- 如果环境里无法安装 `flash-attn`，推理配置里改用 `sdpa` 即可。
- 预训练模型基于 `Qwen/Qwen3-ASR-1.7B` 微调，无需单独下载基座权重。

## 快速开始

### 1. 数据预处理

参考 [configs/preprocess.yaml](configs/preprocess.yaml)：

```yaml
model_path: Qwen/Qwen3-ASR-0.6B
output_dir: "/path/to/preprocessed"

datasets:
  - name: opencpop
    type: json_file
    json_path: /path/to/Opencpop.json
    audio_root: /path/to/Opencpop
```

运行：

```bash
python scripts/preprocess.py --config configs/preprocess.yaml --num_workers 16
```

### 2. 训练

参考 [configs/train.yaml](configs/train.yaml)：

```yaml
model_path: Qwen/Qwen3-ASR-1.7B
output_dir: ./vocalparse-runs/experiment-1
preprocessed_dir: "/path/to/preprocessed"
val_datasets:
  - opencpop
bpm_position: "last"
include_dur: false
asr_cot: true
batch_size: 64
lr: 2e-5
epochs: 10
```

单 GPU：

```bash
python -m vocalparse.train --config configs/train.yaml
```

多 GPU：

```bash
torchrun --nproc_per_node=2 -m vocalparse.train --config configs/train.yaml
```

如果 `output_dir` 下已有 `checkpoint-*`，训练会自动从最新 checkpoint 续训。

### 3. 推理

参考 [configs/inference.yaml](configs/inference.yaml)：

```yaml
checkpoint: /path/to/checkpoint-or-output-dir
preprocessed_dir: "/path/to/preprocessed"
val_datasets:
  - opencpop
mode: "test_full"
inference_mode: "audio-only"
bpm_position: "last"
include_dur: false
```

运行：

```bash
python -m vocalparse.inference --config configs/inference.yaml
```

## 支持的数据输入

### 训练输入

训练支持两种来源：

1. `preprocessed_dir`
   推荐。直接读取 `scripts/preprocess.py` 生成的 Arrow 数据，训练时无音频 I/O。
2. `datasets`
   原始扫描模式。未提供 `preprocessed_dir` 时，训练脚本会直接扫描标注数据。

原始数据支持两类格式：

- `json_file`
  单个 JSON 文件，包含 `word`、`pitch`、`note`、`pitch2word`、`pitch_dur`、`wav_fn`、`bpm` 等字段
- `folder_based`
  目录结构为 `dataset_root/song_id/segment.audio + segment.json + metadata.json`

原始扫描训练配置示例：

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

### 推理输入

推理支持两种来源：

1. `preprocessed_dir`
   使用完整 GT AST，`mode: test_full` 必须使用这种模式。
2. `audio_json`
   读取原始音频列表，支持：

```json
["/abs/path/a.wav", "/abs/path/b.flac"]
```

或者：

```json
[
  {"audio_path": "a.wav", "text": "歌词一"},
  {"audio_path": "b.wav", "lyrics": "歌词二"}
]
```

如果路径是相对路径，并且提供了 `audio_root`，则会自动拼接。

## 推理条件与输出模式

### 条件模式

- `audio-only`
  模型从音频直接预测歌词和乐谱。
- `audio-lyric`
  模型在 prompt 中接收 GT 歌词，只预测乐谱。主要用于 CoT 训练的模型。

### 输出模式

- `test_weak`
  只计算 CER。适用于原始音频 JSON 和预处理数据。
- `test_full`
  计算完整 AST 指标，包括 CER、Pitch MAE、Note MAE、可选 Duration MAE 和 BPM MAE。必须使用 `preprocessed_dir`。
- `annotation`
  导出 Opencpop 风格的逐样本标注 JSON。两种输入方式都支持。

## Token 格式

交错 AST 格式直接编码字音对应关系：

```text
感 <P_68> <NOTE_4> 受 <P_60> <NOTE_8> ... <BPM_89>
  ^         ^         ^         ^
 歌词     音高      歌词     音高
          音符                音符
```

当 `include_dur: true` 时：

```text
感 <dur_0.25> <P_68> <NOTE_4> 受 <dur_0.50> <P_60> <NOTE_8> ... <BPM_89>
```

当 `asr_cot: true` 时：

```text
感受<|file_sep|>感 <P_68> <NOTE_4> 受 <P_60> <NOTE_8> ... <BPM_89>
```

## 配置参数参考

常用参数及默认值：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_path` | `Qwen/Qwen3-ASR-1.7B` | 基座模型（HuggingFace ID 或本地路径） |
| `output_dir` | `./vocalparse-runs/default` | Checkpoint 和 TensorBoard 日志目录。目录下已有 checkpoint 时自动续训。 |
| `bpm_position` | `"last"` | `<BPM_*>` token 的位置：`"first"` 或 `"last"`。推荐使用 `"last"`（见下）。 |
| `include_dur` | `false` | 是否插入 `<dur_X.XX>` 物理时长 token |
| `asr_cot` | `false` | Chain-of-Thought 模式：先输出纯歌词，再输出交错乐谱 |
| `batch_size` | `8` | 每设备 batch size（动态批处理开启时为上限） |
| `max_batch_mel_tokens` | `0` | 动态批处理 mel 帧总数预算（0 = 禁用） |
| `val_generate_samples` | `5` | 验证时生成推理的样本数 |
| `val_display_samples` | `5` | 记录到 TensorBoard 的 GT vs Pred 对比图数量 |
| `lr` | `2e-5` | 峰值学习率 |
| `warmup_ratio` | `0.02` | LR warmup 占总步数的比例 |
| `save_steps` | `200` | Checkpoint 保存间隔 |
| `eval_steps` | `save_steps` | 验证间隔（未设置时与 `save_steps` 相同） |

### BPM 位置

实验表明，`bpm_position: "last"` 的效果稳定优于 `"first"`。将 BPM 预测放在音符序列之后，模型在给出 tempo 估计前已有完整的乐句信息，因此 `"last"` 作为推荐默认值。

### 配置约束

以下 prompt 格式相关配置一旦变化，需要重新训练模型，但不需要重新预处理数据：

- `bpm_position`
- `include_dur`
- `asr_cot`

## 训练监控

VocalParse 在 `output_dir` 下写入 TensorBoard 日志。验证回调每隔 `eval_steps` 步运行 `model.generate()`，记录以下指标：

| TensorBoard Key | 指标 | 说明 |
|---|---|---|
| `val/cer` | 歌词 CER | `(S+D+I) / N_gt`，排除静音 token |
| `val/cer_singing` | 歌词 CER（含静音） | 包含 AP/SP token |
| `val/pitch_mae` | Pitch MAE | MIDI 半音级绝对误差 |
| `val/note_mae` | Note MAE | log₂ 音符时值空间绝对误差 |
| `val/dur_mae` | Duration MAE | 物理时长绝对误差（秒），仅 `include_dur: true` 时 |
| `val/bpm_mae` | BPM MAE | 速度绝对误差 |

每次验证还会记录 `val_display_samples` 张 GT vs Pred 乐谱对比图。

## 评估指标

| 指标 | 说明 |
|---|---|
| `CER` | 排除静音 token 的歌词字符错误率 |
| `CER (singing)` | 包含静音 token 的歌词字符错误率 |
| `Pitch MAE` | MIDI 半音级绝对误差 |
| `Note MAE` | log2 音符时值空间绝对误差 |
| `Duration MAE` | 物理时长秒级绝对误差 |
| `BPM MAE` | 速度绝对误差 |

指标通过两层 Needleman-Wunsch 对齐计算：

1. 字级对齐计算歌词误差
2. 匹配字内部的音高/音符/时长对齐计算结构误差

## 项目结构

```text
vocalparse/
  train.py          # 训练入口
  inference.py      # 推理入口
  data.py           # 数据加载与 collator
  evaluation.py     # AST 指标
  validation.py     # 验证回调与可视化
  output.py         # 推理结果格式化
  tokens.py         # AST token 定义
  prompts.py        # Prompt 构建
  model.py          # 模型 patch 与音频工具
  checkpoint.py     # checkpoint 工具
  distributed.py    # DDP 与批处理推理辅助函数
scripts/
  preprocess.py     # Mel -> Arrow 预处理
configs/
  preprocess.yaml   # 预处理配置示例
  train.yaml        # 训练配置示例
  inference.yaml    # 推理配置示例
docs/
  note_tokens.md    # token 参考
```

## 引用

如果你在研究中使用了 VocalParse，请引用：

```bibtex
@article{vocalparse2026,
  title={VocalParse: Towards Unified and Scalable Singing Voice Transcription with Large Audio Language Models},
  year={2026}
}
```

## 致谢

VocalParse 基于阿里巴巴 Qwen 团队的 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 构建。

## 许可证

Apache 2.0
