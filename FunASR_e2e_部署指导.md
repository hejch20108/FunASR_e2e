# FunASR_e2e 部署指导

## 1. 项目用途

`FunASR_e2e` 是一个放在 FunASR 官方仓库根目录下运行的端到端音频转写工具，流程为：

```text
音频文件 -> FunASR ASR/VAD/标点/说话人识别 -> JSON -> cleaned 文本 -> 大模型润色 -> polished 文本
```

最终每个音频默认生成三份文件：

```text
<音频名>.json
<音频名>_cleaned.txt
<音频名>_polished.txt
```

## 2. 新服务器部署

在新 Linux 服务器上：

```bash
git clone https://github.com/modelscope/FunASR.git && \
cd FunASR && \
git clone https://github.com/hejch20108/FunASR_e2e.git && \
cd FunASR_e2e && \
cp .env.example .env
```

然后编辑 `.env`，填入百炼 API Key。

## 3. Python 环境安装

建议使用 `uv`。

### 3.1 安装 uv

```bash
sudo apt update && sudo apt install -y python3-pip && \
pip install uv
```

如果是非 root 用户，执行如下命令，使 uv 生效

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 3.2 创建虚拟环境

在 `FunASR_e2e` 目录中执行：

```bash
uv venv --python 3.11 .venv && \
source .venv/bin/activate
```

### 3.3 安装 PyTorch

有 NVIDIA GPU 且 CUDA 12.1：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CPU 环境：

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3.4 安装 FunASR 和运行依赖

`FunASR_e2e` 位于外层 `FunASR` 目录内，所以执行：

```bash
uv pip install -e ../  && \
uv pip install -U modelscope huggingface_hub soundfile librosa jieba pyyaml
```

如果系统缺少音频库，Ubuntu/Debian 可执行：

```bash
sudo apt update && \
sudo apt install -y ffmpeg libsndfile1
```

## 4. 配置 .env

`.env` 示例：

```env
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
API_KEY=your_api_key_here
MODEL_NAME=qwen3-max
```

也可以使用：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

## 5. 上传音频

把音频文件放入：

```text
FunASR_e2e/input_audio/
```

支持格式由 `settings.yaml` 的 `audio.supported_extensions` 控制，默认包括：

```text
.wav .mp3 .m4a .flac .aac .ogg
```

## 6. 执行

在 `FunASR_e2e` 目录中执行：

```bash
source .venv/bin/activate
python scripts/run_funasr_full_pipeline.py
```

默认读取：

```text
settings.yaml
```

也可以指定其他配置：

```bash
python scripts/run_funasr_full_pipeline.py --settings settings.yaml
```

## 7. 查看结果

默认 `output.per_audio_subdir: true`，结果位于：

```text
output/<音频名>/<音频名>.json
output/<音频名>/<音频名>_cleaned.txt
output/<音频名>/<音频名>_polished.txt
```

其中：

- `.json`：FunASR 原始结构化结果。
- `_cleaned.txt`：本地规则合并和保守清理结果。
- `_polished.txt`：大模型润色后的阅读版。

## 8. settings.yaml 常用参数

### 路径

```yaml
paths:
  input_audio_dir: input_audio
  output_dir: output
  env_file: .env
  prompt_dir: prompt
```

所有相对路径均相对于 `FunASR_e2e` 项目根目录。

### 单音频或多音频

```yaml
audio:
  mode: batch
  input_audio_file: null
```

- `batch`：扫描 `input_audio/` 下全部支持格式音频。
- `single`：只处理 `input_audio_file` 指定音频。
- `input_audio_file` 非空时，优先只处理该文件。

### FunASR 设备

```yaml
funasr:
  device: cuda
```

无 GPU 或 CUDA 不可用时可改为：

```yaml
funasr:
  device: cpu
```

### 大模型润色

```yaml
llm:
  skip_polish: false
  chunk_size: 20
  max_workers: 8
  max_retries: 3
  enable_thinking: false
```

- `skip_polish: true`：只生成 `.json` 和 `_cleaned.txt`。
- `chunk_size`：每次给大模型的段落数。
- `max_workers`：大模型并行请求数。
- `max_retries`：单个润色分块失败或格式校验不通过时的最大重试次数。
- `enable_thinking`：是否开启思考模式；开启会显著变慢，默认关闭。

## 9. 修改清洗规则和提示词

无需改 Python 代码，直接改 `prompt/` 目录。

```text
prompt/repeated_words.txt
prompt/drop_words.txt
prompt/filler_words.txt
prompt/polish_prompt_template.txt
```

- `repeated_words.txt`：连续重复时压缩的词。
- `drop_words.txt`：清洗后整段可丢弃的短词。
- `filler_words.txt`：句首句尾可清理的语气词。
- `polish_prompt_template.txt`：大模型润色提示词。

提示词中必须保留占位符：

```text
{{ chunk_text }}
```

脚本会把每个分块文本替换到该位置。

## 10. 多音频支持

默认支持多音频。把多个音频放入：

```text
input_audio/
```

执行：

```bash
python scripts/run_funasr_full_pipeline.py
```

脚本会：

1. 只加载一次 FunASR 模型；
2. 按文件名排序逐个处理音频；
3. 每个音频输出到独立目录；
4. 单个音频内部的大模型分块按 `llm.max_workers` 并行。

不建议同时并行处理多个音频，避免 GPU 显存不足。

## 11. 常见问题

### 找不到 funasr 包

确认目录结构是：

```text
FunASR/
  funasr/
  FunASR_e2e/
```

并且执行过：

```bash
uv pip install -e ../
```

### 没有找到音频

把音频放到：

```text
input_audio/
```

或在 `settings.yaml` 中设置：

```yaml
audio:
  input_audio_file: input_audio/your_audio.wav
```

### CUDA 不可用

把 `settings.yaml` 改成：

```yaml
funasr:
  device: cpu
```

### API Key 缺失

检查 `.env` 是否存在，并配置：

```env
API_KEY=your_api_key_here
```

或：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### LLM 段落数不一致

润色阶段只让大模型处理正文，脚本会原样拼回时间戳和说话人，并校验输入和输出段落数。如果仍然报错，可尝试降低分块大小：

```yaml
llm:
  chunk_size: 10
```

也可以收紧 `prompt/polish_prompt_template.txt`，强调不得合并、删除或新增 segment。

### 请求太慢

确认没有开启：

```yaml
llm:
  enable_thinking: false
```

可适当调大：

```yaml
llm:
  max_workers: 8
```

如果出现限流、超时或连接重置，可降低到 `5`。
