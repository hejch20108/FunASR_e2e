# FunASR_e2e

本项目提供带审计链的本机音频转写流程，并保留两种相互隔离的使用方式：

- **CLI**：批量处理 `input_audio/` 中的音频，结果写入 `output/`。
- **Web UI**：在浏览器中上传录音、排队执行、查看审计产物和下载结果；所有 Web 数据只保存于 `app_data/`。

处理链路如下：

```text
音频 → FunASR ASR / VAD / 标点 / speaker 聚类 → evidence
     → speaker review → reviewed → cleaned → final + final audit
```

Web 服务只监听 `127.0.0.1`，不提供局域网访问。它不会扫描、登记、读取或删除 CLI 的 `input_audio/`、`output/` 内容；如需在 Web 中处理既有音频，请重新上传。

有关产品边界、数据模型和安全设计，见 [Web UI MVP 产品与技术方案](docs/Web_UI_MVP_产品与技术方案.md)。

## 系统要求

- Python **3.11**（项目最低支持 Python 3.10，推荐 3.11）。
- Node.js `^20.19.0 || >=22.12.0`，包含 npm。
- FFmpeg，命令行执行 `ffmpeg -version` 应成功。
- FunASR 运行环境：CPU 使用 CPU 版 PyTorch；GPU 使用与驱动、CUDA 兼容的 PyTorch。
- 若要生成 `final`，还需要可用的 LLM API 配置。

首次处理任务会下载 ASR、VAD、标点和 speaker 模型，可能耗时较长并占用额外磁盘空间。

## Windows：从全新电脑部署

以下命令均在项目根目录执行。先安装 Python、Node.js 和 FFmpeg，并重新打开终端使其进入 PATH。

### 1. 克隆项目并创建虚拟环境

```bat
git clone <仓库地址> FunASR_e2e
cd FunASR_e2e
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

不要提交 `.venv/`、`frontend/node_modules/`、`frontend/dist/` 或 `app_data/`；它们均已由 `.gitignore` 排除。

### 2. 安装 PyTorch、FunASR 与音频依赖

先按本机硬件选择 **一套** PyTorch 安装方式。CPU 示例：

```bat
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

如果使用 NVIDIA GPU，请按照本机显卡驱动和 CUDA 兼容性，从 PyTorch 官方安装指引选择对应命令；不要把 CPU 版与 CUDA 版混装。确认可用后再继续：

```bat
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pip install -U funasr modelscope huggingface_hub soundfile librosa jieba
```

若安装 FunASR 时提示额外系统或 Python 依赖缺失，应按该依赖的错误提示补齐。FFmpeg 是系统级前置条件，`setup_web.bat` 不会安装它。

### 3. 配置 LLM

复制示例配置，编辑 `.env` 后填写自己的密钥：

```bat
copy .env.example .env
```

`.env` 至少包含：

```env
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
API_KEY=your_api_key_here
MODEL_NAME=qwen3-max
```

`.env` 不能提交。`MODEL_NAME` 是普通阅读整理和 speaker review 的唯一模型来源；`settings.yaml` 中 `llm.model` 与 `speaker_review.model` 必须保持 `null`。

### 4. 安装 Web 依赖并构建前端

```bat
setup_web.bat
```

该脚本会：

1. 确认项目 `.venv`、Node.js 和 npm 存在；
2. 以 `.venv` 安装本项目的 Web Python 依赖；
3. 严格按 `frontend/package-lock.json` 执行 `npm ci`；
4. 构建 `frontend/dist/`。

它**不会**创建 `.venv`、选择或安装 PyTorch/CUDA、安装 FunASR、安装 FFmpeg，也不会创建或填写 `.env`。

### 5. 启动生产 Web UI

```bat
start_web.bat
```

脚本会使用 `.venv\Scripts\python.exe` 启动服务并打开浏览器。访问地址为：

```text
http://127.0.0.1:8000
```

首次启动会自动创建 `app_data/app.sqlite3` 并应用数据库迁移。停止该终端窗口或按 `Ctrl+C` 会同时有序停止 Web 服务和 worker。

### 6. 开发模式

```bat
start_web_dev.bat
```

开发模式启动 API 与 Vite 热更新前端，浏览器地址通常为 `http://127.0.0.1:5173`；它使用隔离的 `app_data_test/`，不会使用正式 `app_data/`。

## Linux：手工部署

安装 Python、Node.js、npm 和 FFmpeg 后，在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 按 CPU 或与本机 CUDA 匹配的方式安装 PyTorch
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -U funasr modelscope huggingface_hub soundfile librosa jieba
cp .env.example .env
# 编辑 .env，填写自己的 API_KEY 与 MODEL_NAME
python -m pip install -e .
npm --prefix frontend ci
npm --prefix frontend run build
python scripts/launch_web.py
```

服务仍只会监听 `127.0.0.1:8000`。需要从另一台电脑访问时，请使用 SSH 端口转发，不要直接改为公网或局域网监听：

```bash
ssh -L 8000:127.0.0.1:8000 <用户>@<服务器>
```

随后在本机浏览器访问 `http://127.0.0.1:8000`。

## CLI 快速开始

将音频放入 `input_audio/`，激活虚拟环境后执行：

```bat
.venv\Scripts\activate
python scripts\run_funasr_full_pipeline.py
```

Linux：

```bash
source .venv/bin/activate
python scripts/run_funasr_full_pipeline.py
```

默认输出路径为 `output/<音频名>/`。常用选项：

```bash
# 复用已有 JSON，仅生成 evidence、speaker review、reviewed 与 cleaned
python scripts/run_funasr_full_pipeline.py --reuse-json --skip-polish

# 从已验证的 reviewed 产物生成 final
python scripts/run_funasr_full_pipeline.py --polish-only
```

`--polish-only` 不会将可人工查看的 `_cleaned.txt` 作为事实来源；它会验证 speaker review、源 JSON hash 和完整性条件，不满足时拒绝生成 final。

## 产物与审计

启用完整处理链后，每个音频通常产生：

| 文件 | 作用 |
| --- | --- |
| `.json` / `raw_json` | FunASR 原始结构化识别结果。 |
| `_evidence.txt` / `evidence` | 保留时间与匿名 speaker 的证据文本。 |
| `_speaker_review.json` / `speaker_review` | speaker 复核决策和审计记录。 |
| `_reviewed.txt` / `reviewed` | 基于复核结果重建的逐句文本。 |
| `_cleaned.txt` / `cleaned` | 基于已复核文本生成的中间稿。 |
| `_final.txt` / `final` | 经过完整性校验的 canonical 最终稿。 |
| `_final_audit.json` / `final_audit` | final 的完整性、告警和 fallback 审计记录。 |

Web 中保存的 speaker 显示名只改变页面和“下载显示名版本”的呈现；不会修改 raw JSON、reviewed、canonical final 或任何审计产物，也不会进入 LLM 提示词。

`【待回听】` 表示归属或内容值得回听。重要内容仍应结合原始音频、原始 JSON 和 evidence 核验。

## `settings.yaml` 常用配置

### 设备

仓库默认值是 CPU：

```yaml
funasr:
  device: cpu
```

只有在 CUDA 版 PyTorch 已正确安装且 `torch.cuda.is_available()` 为 `True` 时，才将它改为：

```yaml
funasr:
  device: cuda
```

### 预设说话人数

```yaml
funasr:
  preset_spk_num: null
```

`null` 表示由模型自动进行匿名 speaker 聚类；只有明确知道人数时才填写正整数。该值不会映射真实姓名。

### 阅读整理与 speaker review

`llm.max_workers` 仅影响可并发的 speaker review 高风险片段；final 当前按分块顺序请求，以保证审计顺序稳定。模型名称始终读取 `.env` 的 `MODEL_NAME`，不要在 YAML 中填写模型。

## 数据、备份与隐私

- CLI 数据位于 `input_audio/` 与 `output/`。
- Web 上传文件、SQLite、处理产物和运行时文件位于 `app_data/`。
- SQLite 数据库为 `app_data/app.sqlite3`，启动时会自动创建和迁移。
- 备份前先停止 Web 服务，然后整体复制 `app_data/`；恢复时将完整目录放回项目根目录。
- Web 只信任本机操作系统账户。不要将端口暴露给局域网或互联网，也不要共享 `.env`、`app_data/`、音频或转写产物。

## 测试与构建

激活 `.venv` 后运行 Python 回归：

```bat
python -m unittest discover -s tests -p "test_*.py"
```

前端可重复安装与构建：

```bat
npm --prefix frontend ci
npm --prefix frontend run build
```

浏览器 E2E：

```bat
npm --prefix frontend run e2e
```

E2E 使用本机 Chrome；如果未安装在默认位置，请设置 `CHROME_PATH`。测试会使用隔离的 `app_data_e2e/` 和端口 `8002`，不会调用外部模型服务。

## 常见问题

### `start_web.bat` 提示找不到虚拟环境

在项目根目录执行：

```bat
python -m venv .venv
.venv\Scripts\activate
```

然后按“安装 PyTorch、FunASR 与音频依赖”和“安装 Web 依赖并构建前端”的顺序完成安装。

### `start_web.bat` 提示缺少生产前端

执行：

```bat
setup_web.bat
```

### 任务提示 `FunASR 运行环境不可用`

确认 Web 是通过 `start_web.bat` 启动的，而不是系统 Python；然后激活 `.venv` 并检查：

```bat
python -c "import funasr; print('FunASR 可用')"
```

### CUDA 不可用

保持 `settings.yaml` 的 `funasr.device: cpu`，或重新按显卡驱动和 CUDA 版本安装匹配的 PyTorch；不要只修改 YAML 就假定 CUDA 已配置完成。

### 找不到音频

CLI 读取 `input_audio/`；Web 不会读取该目录，应通过网页上传文件。

### LLM 请求失败或最终稿有告警

确认 `.env` 存在且 `API_KEY`、`BASE_URL`、`MODEL_NAME` 正确。对于重要转写内容，结合原音频和 evidence 审核 `【待回听】`、fallback 与审计摘要。