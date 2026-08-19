# FunASR_e2e Web UI MVP 产品与技术方案

> 状态：MVP 功能实现、按任务宽限期的强制停止协议、可直达页面路由、队列调整、显示名展示导出、审计摘要、浏览器自动化验收及受控删除均已完成。真实 FunASR → LLM 全链路浏览器验收依赖本机已配置的模型、音频与 `.env` 凭据，未在自动化测试中调用外部模型服务。
>
> 本文记录已确认的 MVP 方案与已确认的流程调整。

## 1. MVP 范围

### 包含

- 本机 `127.0.0.1` 浏览器访问；中文界面；无登录、无 LAN 访问。
- 单文件上传自动进入队列；批量上传后手动选择执行。
- 统一录音列表：每页 20 条、文件名模糊搜索、分页、状态、步骤、进度、失败原因、final 是否存在。
- FunASR 串行处理；JSON 与 evidence 提交后，自动按 FIFO 进入 speaker review、reviewed、cleaned、final 与 final audit。
- speaker 显示名可留空，并可在任意阶段修改；修改仅影响 UI 和下载呈现，不重跑 FunASR 或 LLM。
- 独立详情页：默认显示 final、支持本地音频播放和按时间范围跳转、查看/下载所有产物、下载 ZIP。
- SQLite 任务持久化、刷新恢复、重启恢复、重复提交保护、阶段原子提交。
- 常规取消与强制停止独立 worker。
- 失败后的单一“继续处理 / 推荐修复”主操作，以及折叠的专家操作。
- 受控应用数据目录；Web 上传、处理、下载和删除仅管理 `app_data/` 内的数据。
- 脱敏开发诊断区、fallback / 软告警 / `【待回听】` 摘要。

### MVP 明确不做

- LAN、登录、多用户、权限体系。
- final 人工编辑、逐句人工 speaker 纠错、手工合并/拆分 speaker 聚类。
- LLM 自主决定恢复路径。
- 文件大小、时长、批次总量的业务上限。
- WebSocket、实时帧级 FunASR 百分比、结果内全文搜索。
- 对既有 CLI 产物进行就地重命名、就地覆盖或修改已有 audit。

## 2. 推荐技术架构

```text
React + TypeScript + Vite
          │ 轮询 REST API
          ▼
FastAPI 服务进程
  ├─ SQLite：录音、运行版本、队列、状态、事件、映射、产物
  ├─ 应用受控目录：音频、临时阶段产物、已提交产物、临时 ZIP
  └─ 单独 Python worker 进程
       ├─ 唯一 FunASR 模型实例 / 严格串行 GPU 推理
       ├─ 调用既有 FunASR、speaker review、final 核心函数
       ├─ 读取 .env 的 MODEL_NAME / API Key / BASE_URL
       └─ 以阶段临时目录写入，校验后原子提交
```

| 决策 | 方案 | 原因 |
|---|---|---|
| 后端 | FastAPI | Python 复用现有核心逻辑最直接，未来可通过显式配置扩展到 LAN。 |
| 前端 | React + TypeScript + Vite | 上传进度、批量列表、speaker 显示名映射、详情审计区等交互较多。 |
| 状态存储 | SQLite | 适合本机单用户，支持任务恢复、事务与查询，不引入外部服务。 |
| 长任务 | 单独 worker 进程 | 避免 Web 请求超时；可隔离卡死的 FunASR/HTTP 调用；强制停止不带倒服务进程。 |
| GPU 并发 | 全局 1 个 FunASR 任务 | 现有 CLI 本身也是单模型、逐条处理。 |
| LLM 并发 | 单 worker 编排任务；保留当前单任务内风险片段受限并发 | 既保留现有 speaker review 能力，也避免多录音任务叠加放大并发。 |
| 进度传输 | 轮询 | 本机 MVP 简单可靠；上传用浏览器真实字节进度，处理用阶段/分块进度。 |

现有审计内核不重写：

- `_reviewed.txt` 的精确原文重建，继续由 `validate_spans` 保证：`scripts/review_funasr_speakers.py:1509-1542`。
- speaker review 的源 JSON hash、allowed speakers 与完整性校验继续保留：`scripts/review_funasr_speakers.py:1575-1795`。
- final 的 speaker、顺序、保护事实、关系校验与 fallback 继续保留：`scripts/postprocess_funasr_transcript.py:1068-1387`。
- `.env` 的 `MODEL_NAME` 仍是唯一模型来源；`settings.yaml` 中的 `llm.model` 与 `speaker_review.model` 继续保持 `null`。

## 3. 用户流程与状态机

```text
上传中
→ 已上传 / 等待执行
→ 排队等待 FunASR
→ FunASR 处理中
→ 排队等待后续处理
→ LLM speaker review 中
→ final 阅读整理中
→ 已完成
```

终态或异常分支：

```text
任意可取消状态
→ 取消请求中
→ 已取消
→ 强制停止后待恢复

任意执行状态
→ 失败
→ 推荐修复
→ 回到最近一个完整阶段继续

final 有 fallback 或软告警
→ 部分完成 / 有 fallback
```

关键规则：

1. FunASR 成功后，JSON 和 evidence 必须先完成校验并提交；当前任务随后原子完成，并自动创建按 FIFO 排队的后续任务。
2. speaker 显示名不改写 JSON，不改变 LLM 的匿名 speaker 输入，也不阻塞任何自动阶段。
3. speaker review 和 final 只能读取同一运行版本内、完整性通过的上游产物。
4. 强制停止时，当前临时目录废弃；只有已原子提交的阶段能被恢复流程复用。
5. 同一录音同时最多一个活跃任务；处理中禁止删除、重命名或重复启动。
6. 批量任务按全局 FIFO 串行执行；speaker 显示名可在任意阶段保存或修改。
7. FunASR 队列使用 FIFO；未开始任务可取消或调整位置。

## 4. 页面与路由

| 路由 | 页面 | MVP 内容 |
|---|---|---|
| `/recordings` | 录音列表主页 | 上传区、批量选择、搜索、分页、状态、队列操作。 |
| `/recordings/:recordingId` | 录音详情页 | 默认 final、音频播放器、时间跳转、产物下载、审计摘要、开发诊断。 |
| `/recordings/:recordingId/speakers` | speaker 显示名映射页或详情页内区域 | 匿名 speaker 卡片、代表片段、出现次数、时间范围、显示名输入与保存。 |

详情页的信息层级：

```text
最终可阅读版
├─ final 正文
├─ fallback / 软告警 / 待回听摘要
├─ 下载 final
└─ 音频回听与时间跳转

证据与审计
├─ 原始 JSON
├─ evidence
├─ speaker review audit
├─ reviewed
├─ cleaned
├─ final audit
└─ 下载全部 ZIP

开发诊断（折叠）
└─ 脱敏任务事件与异常摘要
```

独立详情页优于浮窗：长文本和多个产物更适合完整页面；刷新、前进后退和下载定位也更可靠。

## 5. 数据模型与目录

### SQLite 核心表

| 表 | 主要内容 |
|---|---|
| `recordings` | 不可变 `recording_id`、原始文件名、显示名称、扩展名、SHA-256、来源、上传时间、当前运行版本。 |
| `runs` | `run_id`、所属录音、运行版本、`preset_spk_num` 与配置快照、状态、开始/结束时间。 |
| `jobs` | 队列顺序、任务阶段、状态、进度摘要、推荐恢复动作、错误分类、取消状态。 |
| `stage_attempts` | 每次阶段尝试、开始/结束、输入完整性、产物提交结果、失败摘要。 |
| `artifacts` | 产物类型、内部相对路径、SHA-256、所属运行版本、提交状态。 |
| `speaker_mapping_versions` / `speaker_mapping_entries` | 匿名标签、显示名称和版本化映射。 |
| `task_events` | 脱敏状态事件、阶段进度、恢复推荐原因。 |
| `deletion_operations` | 受控删除操作的审计记录。 |

### 受控应用目录

```text
app_data/
├─ app.sqlite3
├─ recordings/
│  └─ <recording_id>/
│     ├─ source/
│     │  └─ audio.<extension>
│     └─ runs/
│        └─ <run_id>/
│           ├─ staging/
│           └─ artifacts/
│              ├─ raw.json
│              ├─ evidence.txt
│              ├─ speaker_review.json
│              ├─ reviewed.txt
│              ├─ cleaned.txt
│              ├─ final.txt
│              └─ final_audit.json
└─ runtime/
```

内部路径以 `recording_id` 与 `run_id` 为准，不使用用户文件名拼路径，从而避免路径穿越、重名覆盖和重命名冲突。

### 显示名与 final

- 系统保存不可变的匿名审计产物；speaker review 的 allowed speakers 继续使用 `SPEAKER_0` 等匿名标签。
- final 的系统原件及其 `final_audit.json` 保持一致，不能在用户改名后直接修改。
- 页面和下载时，根据当前 `speaker_display_mapping` 动态渲染显示标签。
- 使用显示名下载的 final 属于展示导出；它不替代原始 canonical final，也不修改 final audit。
- 显示名修改只新增映射版本，重新渲染 UI / 下载，不重跑 FunASR 或 LLM。

## 6. API 草案

```text
POST   /api/recordings/uploads
GET    /api/recordings?page=&page_size=20&query=
GET    /api/recordings/{recording_id}
PATCH  /api/recordings/{recording_id}
DELETE /api/recordings/{recording_id}

POST   /api/recordings/{recording_id}/funasr-jobs
POST   /api/recordings/batch/funasr-jobs
POST   /api/recordings/{recording_id}/speaker-mapping
POST   /api/recordings/{recording_id}/continue-jobs
POST   /api/recordings/batch/continue-jobs

GET    /api/jobs/{job_id}
GET    /api/jobs/queue
POST   /api/jobs/{job_id}/cancel
POST   /api/jobs/{job_id}/force-stop
POST   /api/jobs/{job_id}/recommended-recovery
POST   /api/jobs/{job_id}/advanced-recovery
POST   /api/jobs/reorder

GET    /api/recordings/{recording_id}/artifacts
GET    /api/recordings/{recording_id}/artifacts/{artifact_type}
GET    /api/recordings/{recording_id}/download/final?display_names=
GET    /api/recordings/{recording_id}/download/all
GET    /api/recordings/{recording_id}/audit-summary
GET    /api/recordings/{recording_id}/diagnostics
```

接口边界：

- API 永不返回 `.env`、API Key、base URL、完整请求内容或原始异常堆栈。
- 上传接口只接受 `settings.yaml` 当前列出的格式：`.wav`、`.mp3`、`.m4a`、`.flac`、`.aac`、`.ogg`。
- 无业务大小上限，但采用流式写入、可用磁盘空间预检、临时文件清理。
- SHA-256 仅用于仍存在的有效录音去重；删除录音及全部关联文件后，数据库记录和 hash 一并删除。
- 下载文件名使用显示名称，但内部文件访问只能由 artifact ID / recording ID 定位。

## 7. 后台任务、恢复、重试与取消

### 阶段提交

每个阶段均遵循：

```text
读取已提交上游产物
→ 写入 <run>/staging/<stage>/
→ 执行现有完整性校验
→ 计算 artifact hash
→ SQLite 事务更新状态
→ 原子移动到 artifacts/
```

这避免当前 CLI 直接 `write_text()` 到正式路径时，在强制停止下留下半成品的问题。

### 进度规则

- 上传：真实字节进度。
- 批量：真实完成数量，例如 `FunASR 2/8`。
- FunASR 单条：仅显示活动状态与已耗时，不伪造百分比。
- speaker risk segments：显示已完成段数 / 总段数。
- final：显示已完成分块 / 总分块。
- 前端轮询任务状态与事件，不依赖解析 stdout。

### 推荐恢复

| 条件 | 主按钮动作 |
|---|---|
| LLM 传输失败、限流、超时 | 重试当前阶段 |
| FunASR 与 evidence 已完成，后续任务尚未开始 | 继续处理 |
| speaker review / final 未完成 | 继续处理 |
| 强制停止且上游阶段完整 | 从最近完整阶段继续 |
| JSON hash 或审计完整性失败 | 从 FunASR 创建新运行版本 |
| 已完成 | 不显示继续处理 |

专家操作置于“更多方式”，显示影响范围和是否生成新运行版本。

### 取消

- 点击取消：写入 `cancel_requested`，worker 在阶段边界和 LLM 请求前后检查。
- 请求未自然返回：详情页在宽限期后显示“可强制停止”。
- 强制停止：终止独立 worker，不终止 FastAPI；当前阶段不提交，状态标为 `force_stopped`。
- 下次启动：worker 扫描遗留运行状态；仅重新领取未提交任务，已提交阶段根据 hash 和完整性决定是否可继续。

## 8. 安全与隐私边界

1. 仅监听 `127.0.0.1`；不提供 LAN 监听参数的 UI 开关。
2. 前端从不读取 `.env`；FastAPI 也不向浏览器返回其中的任何值。
3. API Key、base URL 仅由 worker 在执行 LLM 阶段读取；模型名严格来自 `.env` 的 `MODEL_NAME`。
4. 日志与任务事件做脱敏：不记录 Authorization header、请求正文、API Key、完整 URL 查询参数或环境变量。
5. 用户文件名不参与路径拼接；显示名和下载名都做文件名净化。
6. 删除需要二次确认；处理中禁止删除。确认后删除录音、所有运行版本、源音频、产物、ZIP、SQLite 记录与 SHA-256。
7. 原始 JSON、evidence、reviewed、canonical final 与 audit 不能通过前端编辑。
8. 本机无登录意味着“本机操作系统账户”是信任边界；未来允许 LAN 前必须补充认证、授权、CSRF/CORS 策略和传输安全。

## 9. 测试策略

| 层级 | 验证内容 |
|---|---|
| 现有核心回归 | 保留并执行 speaker review / final transcript 全部测试。 |
| 服务单元测试 | 状态机迁移、推荐恢复、显示名映射、SHA 去重、删除与 hash 清理、路径净化、敏感字段脱敏。 |
| worker 集成测试 | 原子提交、阶段失败、重启恢复、取消、强制停止、同一录音重复提交保护、单 GPU 锁。 |
| 审计集成测试 | UI worker 调用后仍通过 JSON hash、reviewed 重建、speaker schema、final audit 与 protected facts 校验。 |
| API 测试 | 上传格式、同名不同内容、相同 hash、分页、搜索、下载、受控删除、错误响应不泄密。 |
| 前端组件测试 | 列表、进度、speaker 命名、删除确认、fallback 摘要、专家操作折叠。 |
| 浏览器端到端测试 | 单/批上传、队列、speaker 暂停、继续 LLM、播放跳转、刷新恢复、取消、强制停止、下载与删除。 |

实施后必须启动本地服务，在浏览器实际验证 UI，而不是仅以测试通过作为完成依据。

## 10. 与现有 CLI 的隔离

1. 不修改现有 CLI 的默认路径、参数、执行方式或已有工作流；CLI 继续使用 `input_audio/` 与 `output/`。
2. Web 不扫描、登记、浏览或删除 `input_audio/`、`output/` 中的文件；所有 Web 数据均由上传进入 `app_data/`。
3. 新上传录音使用受控目录和不可变 ID，不使用 `<音频 stem>` 作为存储身份。
4. 若 UI 出现问题，CLI 可继续独立使用；应用数据目录可单独停用或删除，不影响 CLI 目录。

## 11. 分阶段实施、验收、风险与回滚

| 阶段 | 交付 | 验收 | 主要风险 | 回滚 |
|---|---|---|---|---|
| 1. 核心服务化 | 将 CLI 阶段拆为可调用的 FunASR、证据、speaker review、final 服务接口 | 现有回归测试全部通过；新接口生成的产物可通过现有审计 | 无意复制或绕过核心校验 | 保留 CLI 原入口；删除新服务层调用即可 |
| 2. 数据与 worker | SQLite、受控目录、单 worker、原子提交、恢复与取消 | 重启、重复提交、强制停止后无半成品被误用 | 进程锁与状态恢复边界 | 停止 worker，CLI 不受影响 |
| 3. FastAPI | 上传、列表、状态、任务、下载 API | 敏感信息不泄露；API 状态机测试通过 | 文件路径、ZIP、上传中断 | 禁用服务入口，不动 CLI 数据 |
| 4. React UI | 列表、上传、speaker 确认、详情、回听、下载、诊断 | 浏览器完成黄金路径与边界路径 | 前后端状态不一致 | 前端独立回退；后端和数据保持 |
| 5. 联调与隔离验证 | 完整 E2E、使用说明 | 单/批上传、暂停确认、恢复、删除、下载均实际验证 | 受控目录与 CLI 路径混用 | Web 仅使用 app_data/，不动 CLI 数据 |
| 6. 后续版本 | 人工 final 编辑、逐句人工校正、LAN/多用户 | 独立审计与权限方案获确认 | 破坏审计语义 | 不影响 MVP canonical artifacts |
