# VocalParse

**VocalParse** 是从 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 中抽离出的最小开源子仓库，保留了 VocalParse 相关的核心训练、验证、推理、预处理和评估链路，同时去掉了无关实验分支和外围基线代码。

它将歌唱音频转录为统一的自回归结构化序列，包含歌词、音高、音符时值和全局 BPM。

## 特性

- **统一转录**：单一解码器流同时建模歌词和旋律
- **CoT 提示**：通过 `asr_cot` 支持“先歌词、后乐谱”的训练目标
- **两种推理入口**：
  - `transcribe_one(audio, checkpoint)`：单样本快速推理，最少代码即可上手
  - `VocalParseTranscriber`：生产级批量推理，原生支持多卡 torchrun
- **验证可视化**：TensorBoard 内置 GT vs Pred 乐谱对比图
- **动态批处理**：按 mel 帧预算进行训练和推理 batch 打包

## 你想做什么？

VocalParse 的代码库分为三条独立的工作路径，按需选用：

### 🎵 快速试用：单样本推理

适合：第一次接触 VocalParse、想快速验证模型在某个 wav 文件上的效果。

```python
from vocalparse import transcribe_one

text = transcribe_one(
    audio="path/to/song.wav",
    checkpoint="./vocalparse-weights",
)
print(text)
# 感 <P_68> <NOTE_4> 受 <P_60> <NOTE_8> ... <BPM_89>
```

也可以直接用命令行：

```bash
python -m vocalparse.demo --audio path/to/song.wav --checkpoint ./vocalparse-weights
```

详见 [vocalparse/demo.py](vocalparse/demo.py)。该模块刻意保持简洁——加载、预处理、生成、解码全在一个文件里，方便阅读和修改。

### 🏭 生产批量：作为外部库被调用

适合：把 VocalParse 嵌入 SVS / TTS 等下游 pipeline 做大批量音频识别 / 验证。
内部实现包含多卡 torchrun、按 mel 帧预算自动 batch 打包、CPU prep ‖ GPU generate 流水线、跨卡 work-steal 调度等优化，对调用者完全透明。

```python
from vocalparse import VocalParseTranscriber

trx = VocalParseTranscriber(checkpoint="./vocalparse-weights")
results = trx.transcribe([wav_a, wav_b, ...])  # list[np.float32 array] → list[str]
```

多卡：用 `torchrun --nproc_per_node=4 your_script.py` 启动调用方脚本即可，`VocalParseTranscriber` 会自动从环境读取 `RANK / WORLD_SIZE` 并把 batch 分片到各卡，最终结果聚合到 rank 0。

详见 [vocalparse/api.py](vocalparse/api.py) 与基准脚本 [scripts/benchmark_api.py](scripts/benchmark_api.py)。

### 🎓 训练 / 微调

适合：在自己的歌唱数据上训练或微调 VocalParse。流程：preprocess → train。
跳到下面的 [快速开始 § 数据预处理](#1-数据预处理) 章节即可。

## 架构

```text
歌唱音频 (16kHz) -> Whisper 编码器 -> Qwen LLM 解码器 -> AST Token 序列
                                                              |
                                                              v
                    感 <P_68> <NOTE_4> 受 <P_60> <NOTE_8> ... <BPM_89>
```

VocalParse 扩展了 Qwen3-ASR 的词表，新增约 400 个 AST token，包括音高（`<P_0>` 到 `<P_127>`）、音符时值（`<NOTE_*>`）和速度（`<BPM_*>`）。完整列表参见 [docs/note_tokens.md](docs/note_tokens.md)。

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

推荐使用 [uv](https://docs.astral.sh/uv/) 进行快速、可复现的环境配置。

```bash
# 创建并激活虚拟环境
uv venv --python 3.10
source .venv/bin/activate

# 安装 PyTorch（根据你的 CUDA 版本调整 index URL）
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装 VocalParse 及所有依赖
uv pip install -e .
```

Flash Attention 是可选依赖。默认安装使用 PyTorch SDPA，通用且无需额外配置。如果你的环境兼容 `flash-attn`，可以自行选装以获得加速：

```bash
uv pip install -e ".[flash]"
```

说明：
- `qwen-asr` 及其他所有依赖会自动安装。
- 使用默认安装时，在推理配置里设置 `attn_implementation: sdpa` 即可（训练脚本不强制要求 flash-attn 后端）。
- 预训练模型基于 `Qwen/Qwen3-ASR-1.7B` 微调，无需单独下载基座权重。

## 快速开始

### 1. 数据预处理

仓库在 [data/](data/) 下已内置三个公开歌唱数据集的标注 JSON，可直接使用：

| 文件 | 样本数 | 数据集 |
|---|---|---|
| [data/Opencpop.json](data/Opencpop.json) | 3,756 | [Opencpop](https://wenet.org.cn/opencpop/) |
| [data/gtsinger.json](data/gtsinger.json) | 7,139 | [GTSinger](https://github.com/GTSinger/GTSinger) |
| [data/m4singer.json](data/m4singer.json) | 20,896 | [M4Singer](https://github.com/M4Singer/M4Singer) |

音频请从各数据集官方渠道自行下载，`audio_root` 指向本地解压目录即可。[configs/preprocess.yaml](configs/preprocess.yaml) 已给出默认配置：

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

运行：

```bash
python scripts/preprocess.py --config configs/preprocess.yaml --num_workers 16
```

使用自定义数据集时，请产出包含 `word`、`pitch`、`note`、`pitch2word`、`pitch_dur`、`word_dur`、`wav_fn`、`bpm` 字段的 JSON 列表（可参考内置三个文件的字段结构），或使用 [支持的数据输入](#支持的数据输入) 中描述的 `folder_based` 目录式结构。

### 2. 训练

参考 [configs/train.yaml](configs/train.yaml)：

```yaml
model_path: Qwen/Qwen3-ASR-1.7B
output_dir: ./vocalparse-runs/experiment-1
preprocessed_dir: "/path/to/preprocessed"

# val_datasets 中列出的数据集会整体从训练集中剔除并作为验证集。
# 默认三数据集配置下：GTSinger + m4singer 用于训练，Opencpop 用于验证。
val_datasets:
  - opencpop

bpm_position: "last"
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

参见上方 [你想做什么？](#你想做什么) 章节中的两条推理路径：

- **单样本快速推理**：[vocalparse/demo.py](vocalparse/demo.py) 或 `from vocalparse import transcribe_one`
- **生产批量推理**：[vocalparse/api.py](vocalparse/api.py) 或 `from vocalparse import VocalParseTranscriber`

需要离线评测 / 标注？基于 `VocalParseTranscriber` 自己写脚本，搭配 [vocalparse/evaluation.py](vocalparse/evaluation.py) 中的 `parse_transcription_text` 等工具即可。可参考 [scripts/benchmark_api.py](scripts/benchmark_api.py) 作为模板。

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

推理 API（`transcribe_one` / `VocalParseTranscriber`）直接接受运行时数据，无需配置文件：

- `transcribe_one`：wav 文件路径或 1D `np.float32` 数组
- `VocalParseTranscriber.transcribe`：`list[np.float32 array]`（mono，16 kHz）

调用方负责音频加载（推荐 `librosa.load(path, sr=16000, mono=True)`）。

## Token 格式

交错 AST 格式直接编码字音对应关系：

```text
感 <P_68> <NOTE_4> 受 <P_60> <NOTE_8> ... <BPM_89>
  ^         ^         ^         ^
 歌词     音高      歌词     音高
          音符                音符
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
- `asr_cot`

## 训练监控

VocalParse 在 `output_dir` 下写入 TensorBoard 日志。验证回调每隔 `eval_steps` 步运行 `model.generate()`，记录以下指标：

| TensorBoard Key | 指标 | 说明 |
|---|---|---|
| `eval/cer` | 歌词 CER | `(S+D+I) / N_gt`，排除静音 token |
| `eval/pitch_mae` | Pitch MAE | MIDI 半音级绝对误差 |
| `eval/note_mae` | Note MAE | log₂ 音符时值空间绝对误差 |
| `eval/dur_mae` | Duration MAE | log₂ 秒空间绝对误差，由 `note × 60 / BPM` 派生 |
| `eval/bpm_mae` | BPM MAE | 速度绝对误差 |

每次验证还会记录 `val_display_samples` 张 GT vs Pred 乐谱对比图。

## 评估指标

| 指标 | 说明 |
|---|---|
| `CER` | 排除静音 token 的歌词字符错误率 |
| `Pitch MAE` | MIDI 半音级绝对误差 |
| `Note MAE` | log₂ 音符时值空间绝对误差 |
| `Duration MAE` | log₂ 秒空间绝对误差（`note × 60 / BPM`） |
| `BPM MAE` | 速度绝对误差 |
| `Pitch Error Rate` | 对齐 pair 中音高不匹配的比例（仅推理时报告） |
| `Note Num Mean Error` | 每个词的 `|n_gt − n_pred|` 平均值（仅推理时报告） |

指标通过两层 Needleman-Wunsch 对齐计算：

1. 字级对齐计算歌词误差
2. 匹配字内部的音高/音符/时长对齐计算结构误差

## 项目结构

```text
vocalparse/
  # 三个面向用户的入口
  demo.py           # 单样本快速推理（transcribe_one）
  api.py            # 批量生产推理（VocalParseTranscriber）
  train.py          # 训练入口

  # 共享核心
  model.py          # 模型加载、patch、音频工具
  prompts.py        # Prompt 构建
  tokens.py         # AST token 定义
  evaluation.py     # AST 指标与解析工具
  data.py           # 数据加载与 collator
  validation.py     # 训练时验证回调与可视化
  checkpoint.py     # checkpoint 工具
  distributed.py    # DDP / batch 打包 / per-sample 编码（仅供 api.py）
scripts/
  preprocess.py       # Mel -> Arrow 预处理
  benchmark_api.py    # VocalParseTranscriber 端到端基准
configs/
  preprocess.yaml   # 预处理配置示例
  train.yaml        # 训练配置示例
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
