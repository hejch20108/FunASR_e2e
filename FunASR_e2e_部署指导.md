# FunASR_e2e 部署指导

## 1. 项目用途

`FunASR_e2e` 是一个放在 FunASR 官方仓库根目录下运行的端到端音频转写工具，流程为：

```text
音频文件 -> FunASR ASR/VAD/标点/说话人识别 -> JSON/evidence -> 说话人语义复核 -> reviewed -> cleaned -> 最终阅读整理 -> final 文本
```

启用语义说话人复核时，每个音频默认生成七份文件：

```text
<音频名>.json
<音频名>_evidence.txt
<音频名>_speaker_review.json
<音频名>_reviewed.txt
<音频名>_cleaned.txt
<音频名>_final.txt
<音频名>_final_audit.json
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

`MODEL_NAME` 是普通润色和说话人复核唯一使用的模型来源。请在 `.env` 中自行选择模型；`settings.yaml` 的两个 `model` 必须保持 `null`，脚本不会自行替换模型。

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

建议先只运行复核并验收证据层，再单独生成最终阅读版，避免为阅读整理重复调用说话人复核：

```bash
# 复用已有 JSON，只生成 evidence、reviewed 和 cleaned
python scripts/run_funasr_full_pipeline.py --settings settings.yaml --reuse-json --skip-polish

# 确认证据层后，仅从已验证的 reviewed spans 生成最终阅读版
python scripts/run_funasr_full_pipeline.py --settings settings.yaml --polish-only
```

`--polish-only` 会校验 `_speaker_review.json` 为 schema v3、源 JSON hash 未变化且所有完整性检查通过；它不会把可被人工编辑的 `_cleaned.txt` 作为事实来源。不满足时拒绝生成最终阅读版。

## 7. 查看结果

默认 `output.per_audio_subdir: true`，结果位于：

```text
output/<音频名>/<音频名>.json
output/<音频名>/<音频名>_evidence.txt
output/<音频名>/<音频名>_speaker_review.json
output/<音频名>/<音频名>_reviewed.txt
output/<音频名>/<音频名>_cleaned.txt
output/<音频名>/<音频名>_final.txt
output/<音频名>/<音频名>_final_audit.json
```

其中：

- `.json`：不可覆盖的 FunASR 原始结构化结果。
- `_evidence.txt`：逐条展开的机器转写稿，保留原 FunASR speaker，不做删词、合并、复核改标或润色。
- `_speaker_review.json`：schema v3 的说话人复核审计记录，包含 FunASR baseline、接受的改归、待回听项、显式 `unknown` 来源和逐句原文重建校验。
- `_reviewed.txt`：复核后的原文证据层；正文仅由原 JSON 精确切片生成，不接受 LLM 改写。
- `_cleaned.txt`：基于 `_reviewed.txt` 的本地规则中间产物，供核对或兼容使用，不是最终交付阅读版。
- `_final.txt`：唯一面向普通阅读的最终稿。它会在同一匿名 speaker 的连续表达内合并 ASR 碎片、重断句、补标点并删除纯口头禅；不会跨 speaker 合并或新增事实。
- `_final_audit.json`：最终稿的来源映射、清理记录、模型分块、fallback、保护事实/关系硬校验和低风险整理告警审计。

FunASR 的匿名 speaker 聚类是默认归属；LLM 只在存在明确局部反证时调整。`【待回听】` 表示归属或内容值得回听，但通常保留当前匿名 speaker；`unknown` 仅用于高影响且无法合理归属的显式复核结论；`overlap` 仅表示明确多人同时发言。关键内容仍应以原始音频、`.json` 和 `_evidence.txt` 核对。

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

### 说话人数

```yaml
funasr:
  preset_spk_num: null
```

- `null`：默认，由模型自动进行声纹聚类。
- 正整数：仅在明确知道人数时填写，例如双人录音填写 `2`。
- 该设置只控制匿名说话人聚类数量，不会映射为真实姓名。

### 最终阅读整理

```yaml
llm:
  skip_polish: false
  chunk_size: 8
  max_workers: 8
  max_retries: 3
  enable_thinking: true
```

- `skip_polish: true`：生成 `.json`、`_evidence.txt`、`_speaker_review.json`、`_reviewed.txt` 和 `_cleaned.txt`，但跳过 `_final.txt` 与 `_final_audit.json`。
- `chunk_size`：每次给最终阅读整理模型的 ReadingUnit 数量。
- `max_workers`：保留的兼容配置；最终阅读整理当前按分块顺序请求，以确保审计顺序稳定。
- `max_retries`：单个最终阅读分块的 JSON、覆盖、speaker 或保护事实/关系硬校验不通过时的最大重试次数；低风险整理告警会直接接受而不重试，耗尽后仅该分块回退为确定性整理结果。
- `enable_thinking`：是否开启思考模式；开启会显著变慢。部分模型会强制要求 `true`，应以 API 返回的参数约束为准。

### 语义说话人复核

```yaml
speaker_review:
  enabled: true
  enable_thinking: true
  model: null
  context_size: 4
  max_risk_core_sentences: 12
  max_boundary_candidates: 16
  max_workers: 8
  max_retries: 3
  request_timeout_s: 90
  auto_apply_confidence: 0.90
  allowed_speakers: null
  unknown_label: unknown
  overlap_label: overlap
  failure_policy: keep_original
```

- `enabled`：启用后先执行一次全稿稀疏复核，再只复核高风险连续片段；`skip_polish` 不会跳过该步骤。
- `enable_thinking`：语义复核请求是否开启思考模式；部分模型会强制要求 `true`，应以 API 返回的参数约束为准。
- `model: null`：必须保持 null，实际模型严格读取 `.env` 的 `MODEL_NAME`。
- `allowed_speakers: null`：从当前音频 FunASR 的匿名标签派生；只有明确配置时才固定集合，不映射真实姓名。
- `context_size`：高风险片段两侧的只读上下文句数。
- `max_risk_core_sentences`：单个高风险片段可决策的最大核心句数；超过且无法安全拆开时按失败策略保守处理。
- `max_boundary_candidates`：仅对疑似句内换人的核心句提供的最大可靠边界数；普通句不发送字符边界。
- `max_workers`：仅控制高风险片段请求的并发数；全稿复核始终只有一个请求。
- `auto_apply_confidence`：低于该值的 REASSIGN、SPLIT 或 OVERLAP 建议不会覆盖当前 baseline；高影响内容可附加 `【待回听】`。
- `failure_policy: keep_original`：全稿或单个高风险片段失败时保留 FunASR 或已接受的 baseline。`fail_closed` 会停止生成 reviewed、cleaned 和 final；不会因失败自动生成 `unknown`。
- `unknown` 仅来自通过置信度门槛、且高影响内容确实无法合理归属的显式 `REVIEW_REQUIRED`；低置信建议、guard 和请求失败均不生成 `unknown`。
- 任何原文重建、顺序、覆盖、speaker 集合或源 JSON hash 校验失败都会停止生成 reviewed、cleaned 和 final，不受 `failure_policy` 降级影响。

## 9. 修改清洗规则和提示词

无需改 Python 代码，直接改 `prompt/` 目录。

```text
prompt/repeated_words.txt
prompt/drop_words.txt
prompt/filler_words.txt
prompt/hotwords.txt
prompt/speaker_review_prompt_template.txt
prompt/polish_prompt_template.txt
```

- `repeated_words.txt`：连续重复时压缩的词。
- `drop_words.txt`：清洗后整段可丢弃的短词。
- `filler_words.txt`：句首句尾可清理的语气词。
- `hotwords.txt`：一行一个热词；空文件时不启用。空行和以 `#` 开头的注释会被忽略。
- `speaker_review_prompt_template.txt`：结构化说话人复核提示词，必须保留 `{{ review_input }}`，不得要求模型输出或改写正文。
- `polish_prompt_template.txt`：最终阅读整理提示词，必须保留 `{{ final_input }}`，并要求模型只输出约定的 JSON。

热词示例：

```text
# 人名、公司名、项目名或固定术语
经济补偿金
N加一
```

热词只帮助识别，不替代对金额、日期等关键内容的原始音频核验。

提示词中必须保留对应占位符：

```text
speaker_review_prompt_template.txt: {{ review_input }}
polish_prompt_template.txt: {{ final_input }}
```

脚本会把结构化输入替换到对应位置。

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

### 最终阅读整理校验失败

最终阅读层允许同一 speaker 的连续 ReadingUnit 合并，并会硬校验每个 unit 的覆盖、顺序、speaker、金额/日期/N+1、立场、承诺、条件及高置信主体关系。连接被标点或空白打断的原词、整理重复口语等低风险变化会记录告警并直接接受；同义替换、数值换算、主体或事实补写仍会重试，耗尽后仅该分块回退为确定性整理结果。若频繁发生，可降低分块大小：

```yaml
llm:
  chunk_size: 10
```

不要把提示词改回“不得合并或删除 segment”；应保留 `{{ final_input }}`、JSON 输出格式、unit 覆盖规则与事实锁约束。

### 请求太慢

若当前模型允许关闭思考模式，可设置：

```yaml
llm:
  enable_thinking: false
```

若 API 明确要求 `enable_thinking: true`，则必须保留开启状态，只能通过降低并发或分块数量控制请求压力。可适当调大：

```yaml
llm:
  max_workers: 8
```

如果出现限流、超时或连接重置，可降低到 `5`。
