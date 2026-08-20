# FunASR_e2e 本机 Web UI MVP 产品与技术方案

> **实现快照（2026-08-20）**：Web UI MVP 的 React、FastAPI、SQLite、受控目录和独立 worker 主流程已经实现。当前仓库包含 96 个 Python `unittest` 测试，以及 3 个使用 mock API 的 Playwright UI smoke tests。真实 API 的完整浏览器黄金路径、真实 FunASR → LLM 自动化验收、服务冷启动后的遗留运行任务接管，以及跨多个服务实例的全局 worker/GPU 互斥尚未闭环。
>
> 本文以当前代码为事实来源，分别记录**已实现能力**、**当前 MVP 限制**和**后续方向**；后续方向不应被理解为已交付能力。

## 1. MVP 范围

### 当前已实现

- 仅允许本机访问：官方启动入口监听 `127.0.0.1`，没有 LAN 开关、登录、多用户或权限体系。
- 单文件上传后自动创建运行版本并进入队列；多文件上传后由用户选择录音，再批量创建任务。
- 统一录音列表：每页 20 条、按原始文件名和显示名称模糊搜索、分页、任务状态、阶段、final 是否存在、失败摘要和队列操作。
- 全局持久化 FIFO job 队列：`funasr` 与 `continuation` 两类 job 共享 `queue_seq`，单个 worker 串行领取任务。
- `funasr` job 依次生成并提交 `raw_json`、`evidence`；完成后自动在队尾创建 `continuation` job。
- `continuation` job 自动生成并提交 `speaker_review`、`reviewed`、`cleaned`、`final`、`final_audit`；当前没有 speaker 人工暂停或确认节点。
- speaker 显示名按运行版本保存，可留空；页面和显示名 final 下载按映射动态渲染，不改写匿名审计产物或 canonical final。
- 录音详情页支持 final 阅读、源音频播放、speaker 代表片段跳转、已提交产物下载、当前运行版本 ZIP、审计摘要和脱敏任务诊断。
- SQLite 记录录音、运行版本、job、阶段尝试、artifacts、speaker 映射和任务事件；阶段产物采用 staging、manifest 与 hash 校验提交。
- 支持普通取消、取消宽限期后的强制停止、同一运行版本的推荐恢复和受控删除。
- Web 上传、处理、下载和删除只管理 `app_data/`，不扫描、登记或修改 CLI 的 `input_audio/`、`output/`。

### 当前 MVP 限制

- `task_events` 已记录 speaker review 和 final 的细粒度进度，但 worker 尚未更新 `jobs.progress_completed` / `jobs.progress_total`；列表主视图通常只显示“处理中”，不会稳定显示段数、分块数或 FunASR 已耗时。
- 没有 batch 实体、批次 ID 或批次总体进度；单次批量启动和单次队列重排最多 200 项。
- speaker 显示名可在 run 已存在后保存；录音自身的 `display_name` 在存在 `queued`、`running` 或 `cancel_requested` job 时不可修改。
- 推荐恢复只按当前 run 中已提交 artifact 类型判断下一阶段，并重新排队原 job；没有按限流、超时或完整性错误分类的恢复策略，也没有高级恢复、批量恢复或自动创建新 run 的恢复路径。
- 启动恢复覆盖阶段 attempt 的 manifest 恢复；整个服务异常退出后的遗留 `running` / `cancel_requested` job 尚未由冷启动流程统一扫描和接管。
- 单 worker 串行执行是当前单服务实例的运行模型；跨多个 FastAPI 服务实例的全局 GPU/worker lease 尚未实现。
- 上传校验文件名和扩展名白名单，不探测真实媒体格式；上传以流式写入和逐块可用空间检查避免直接整文件读入内存。
- speaker 代表片段按钮只跳到 `start_ms` 并播放，不会在 `end_ms` 自动停止；没有 final 时间戳联动、波形或 AB 循环。
- final 在页面中阅读；其他 artifacts 主要提供下载链接，未实现通用的页面内查看器。

### 后续版本方向

- 人工 final 编辑、逐句 speaker 纠错、手工 speaker 聚类合并/拆分。
- 可解释的高级恢复、按错误类型恢复、创建新运行版本、批量恢复和批次进度。
- 主视图数值进度、FunASR 已耗时、真实浏览器黄金路径自动化和真实模型受控验收。
- LAN、多用户、认证、授权、CSRF/CORS 策略与传输安全。

## 2. 当前技术架构

```text
React + TypeScript + Vite（单页 MVP，History API 轻量路由）
          │ REST 轮询 / 上传字节进度
          ▼
FastAPI 服务进程
  ├─ SQLite：录音、运行版本、job 队列、阶段尝试、事件、映射、artifacts
  ├─ 受控应用目录：源音频、staging、已提交 artifacts、临时上传与 ZIP
  └─ WorkerSupervisor 管理的独立 Python worker
       ├─ 单服务实例内串行领取全局 FIFO job
       ├─ 调用既有 FunASR、speaker review、final 核心函数
       ├─ 仅在 LLM 阶段读取 .env 的 MODEL_NAME / 凭据
       └─ 以 manifest 驱动的 prepared → committed 流程发布 artifact
```

| 决策 | 当前实现 | 说明 |
|---|---|---|
| 后端 | FastAPI | 复用现有 Python 核心逻辑；服务端负责 API、持久化与 worker 监督。 |
| 前端 | React + TypeScript + Vite | 当前 UI 集中在 `frontend/src/main.tsx`，使用 History API 手工处理 SPA 路由。 |
| 状态存储 | SQLite | 本机单用户使用；启用 foreign keys、WAL 和 busy timeout。 |
| 长任务 | 独立 Python worker | 避免请求阻塞；强制停止只终止 worker，不终止 FastAPI。 |
| 队列并发 | 单服务实例内一个 worker | worker 每次领取一个全局 FIFO job；尚无跨服务实例全局锁。 |
| 进度传输 | 轮询 + 事件 | 上传使用浏览器真实字节进度；pipeline 细粒度进度写入 `task_events`，尚未完整接入主视图 job 进度字段。 |
| 模型来源 | `.env` | LLM 执行阶段读取 `.env` 的 `MODEL_NAME`；`settings.yaml` 的 `llm.model` 与 `speaker_review.model` 保持 `null`。 |

既有审计内核继续复用，不在 Web 层重写：

- `_reviewed.txt` 的精确原文重建继续由 `validate_spans` 保证：`scripts/review_funasr_speakers.py:1509-1542`。
- speaker review 的源 JSON hash、allowed speakers 与完整性校验继续保留：`scripts/review_funasr_speakers.py:1575-1795`。
- final 的 speaker、顺序、保护事实、关系校验与 fallback 继续保留：`scripts/postprocess_funasr_transcript.py:1068-1387`。

## 3. 用户流程、任务模型与状态机

### 正常流程

```text
上传录音
→ 创建 run 与 funasr job（单文件自动；批量由用户选择后创建）
→ 全局 FIFO 等待
→ funasr job：raw_json → evidence
→ funasr job succeeded，并自动在队尾创建 continuation job
→ continuation job：speaker_review + reviewed → cleaned → final + final_audit
→ continuation job succeeded，run completed
```

`funasr` 与 `continuation` 是当前仅有的两种 job。它们共享同一个全局 `queue_seq`，因此批量 A、B 的常见顺序是：

```text
A FunASR → B FunASR → A continuation → B continuation
```

而不是一条录音从 FunASR 一次连续执行到 final 后才处理下一条。

### 状态与异常分支

```text
queued → running → succeeded
                ├→ cancel_requested → cancelled
                ├→ failed
                ├→ interrupted
                └→ force_stopped

failed / interrupted / force_stopped
→ 按推荐阶段重新排队（同一 run、同一 job）
```

- queued job 取消后直接进入 `cancelled`；running job 先进入 `cancel_requested`，worker 在阶段边界与核心流程检查点响应取消。
- 取消请求达到 10 秒宽限期后，才允许服务端执行强制停止；强停后 job 为 `force_stopped`、run 为 `interrupted`，worker 会重启。
- `waiting_speaker` 是数据库保留状态，当前流程不进入该状态；FunASR 与 evidence 提交后会自动创建并处理 continuation，不需要用户确认继续 LLM。
- final audit 中的 fallback、软告警和 `【待回听】` 是已完成运行的审计属性，不构成“部分完成”终态。当前仍为 `run=completed`、`job=succeeded`，详情页通过审计摘要显示计数。
- 同一录音最多一个活跃 job；活跃状态为 `queued`、`running`、`cancel_requested`。
- speaker 显示名映射不阻塞 job，也不改写 canonical artifacts；录音 `display_name` 则在活跃 job 期间禁止修改。

## 4. 页面与路由

| 路由 | 当前页面 | MVP 内容 |
|---|---|---|
| `/recordings` | 录音列表主页 | 上传、上传字节进度、搜索、分页、批量选择启动、未开始队列和队列调整。 |
| `/recordings/:recordingId` | 录音详情页 | 音频播放器、显示名称、speaker 映射、final 阅读、审计摘要、artifact/ZIP 下载、诊断、取消/强停/推荐恢复/删除。 |
| `/recordings/:recordingId/speakers` | speaker 专用 URL | 复用同一详情页组件并聚焦 speaker 映射区域，不是独立的数据模型或独立组件树。 |

生产 FastAPI 对非 `/api/` 路径返回 SPA `index.html`。前端使用 `window.history.pushState` 与 `popstate` 解析上述路由；旧 `/imports` 等不匹配路径会回到录音列表。

详情页的信息层级：

```text
最终阅读版
├─ 按当前 speaker 显示名映射渲染的 final
├─ 下载显示名版本 final
└─ 源音频播放与 speaker 代表片段跳转到开始时间

证据与审计
├─ raw_json、evidence、speaker_review、reviewed、cleaned、final、final_audit 的下载链接
├─ 当前运行版本 committed artifacts 的 ZIP
└─ fallback / 软告警 / 待回听计数摘要

开发诊断（按 job 加载）
└─ 脱敏阶段事件、进度事件与异常摘要
```

## 5. 数据模型、目录与导出语义

### SQLite 核心表

| 表 | 当前主要内容 |
|---|---|
| `recordings` | 不可变 `recording_id`、原始文件名、显示名称、扩展名、SHA-256、受控源文件路径、创建时间、当前运行版本。 |
| `runs` | 所属录音、版本、`preset_spk_num`、settings 快照、状态、阶段与当前 speaker mapping 版本。 |
| `jobs` | `funasr` 或 `continuation`、全局 `queue_seq`、状态、阶段、取消请求、worker generation、错误码与摘要。`progress_completed` / `progress_total` 字段已存在，但当前未由 worker 写入。 |
| `stage_attempts` | 每次阶段尝试、staging 相对路径、manifest、输入 hash 与发布状态。 |
| `artifacts` | artifact 类型、variant、内部相对路径、SHA-256、字节数和 `prepared` / `committed` 状态。 |
| `speaker_mapping_versions` / `speaker_mapping_entries` | 匿名 speaker 标签、显示名称和按 run 递增的映射版本。 |
| `task_events` | 脱敏的阶段事件、细粒度完成数、总数和受限 details 字段。 |
| `deletion_operations` | 删除前创建的 pending/failed 过程记录。成功删除 recording 后该表会因外键级联清理，不是长期保留的删除审计日志。 |

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
│           │  └─ <stage>-<temporary-id>/
│           │     └─ manifest.json
│           └─ artifacts/
│              ├─ raw_json/raw.json
│              ├─ evidence/evidence.txt
│              ├─ speaker_review/speaker_review.json
│              ├─ reviewed/reviewed.txt
│              ├─ cleaned/cleaned.txt
│              ├─ final/final.txt
│              └─ final_audit/final_audit.json
└─ runtime/
   ├─ uploads/
   └─ exports/
```

内部路径始终由 `recording_id` 与 `run_id` 构造，用户文件名不参与路径拼接。artifact 的实际路径是 `artifacts/<type>/<filename>`，而不是平铺在 `artifacts/` 根目录。

### 显示名、canonical artifacts 与下载

- 匿名审计产物保持不可变；speaker review 的 allowed speakers 继续使用 `SPEAKER_0` 等匿名标签。
- speaker 显示名保存为当前 run 的新映射版本，允许空字符串；保存不重跑 FunASR 或 LLM。
- 页面 final 与 `display_names=true` 下载仅替换每行行首的 speaker 标签，不进行全文字符串替换；canonical final 与 `final_audit.json` 不会被修改。
- 当前下载名为 `final.txt`、`final-display.txt`、`artifacts.zip` 或 artifact 原文件名，不使用录音 `display_name` 生成下载文件名。
- ZIP 只包含当前 run 的 committed artifacts，不包含源音频、旧 run、speaker mapping、诊断事件或额外生成的显示名 final。ZIP 位于 `runtime/exports/`，正常响应结束后自动删除。
- Web 当前只展示并下载 `current_run_id` 对应的 artifacts；旧运行版本保留在受控目录和数据库中，但尚无 run history 或指定 run 浏览 API。

## 6. 当前 API

### 服务与录音

```text
GET    /api/health
POST   /api/recordings/uploads
GET    /api/recordings?page=&page_size=20&query=
GET    /api/recordings/{recording_id}
PATCH  /api/recordings/{recording_id}
DELETE /api/recordings/{recording_id}?confirm=true
POST   /api/recordings/{recording_id}/funasr-jobs
POST   /api/recordings/batch/funasr-jobs
```

### 队列与 job

```text
GET    /api/jobs/queue
POST   /api/jobs/reorder
GET    /api/jobs/{job_id}
GET    /api/jobs/{job_id}/diagnostics
POST   /api/jobs/{job_id}/cancel
POST   /api/jobs/{job_id}/force-stop
GET    /api/jobs/{job_id}/recommended-recovery
POST   /api/jobs/{job_id}/recommended-recovery
```

### artifacts、音频与 speaker

```text
GET    /api/recordings/{recording_id}/artifacts
GET    /api/recordings/{recording_id}/artifacts/{artifact_type}
GET    /api/recordings/{recording_id}/download/final?display_names=
GET    /api/recordings/{recording_id}/download/all
GET    /api/recordings/{recording_id}/audit-summary
GET    /api/recordings/{recording_id}/audio
GET    /api/recordings/{recording_id}/speaker-summary
GET    /api/recordings/{recording_id}/speaker-mapping
POST   /api/recordings/{recording_id}/speaker-mapping
```

### 接口边界

- 上传只接受 `settings.yaml` 当前列出的 `.wav`、`.mp3`、`.m4a`、`.flac`、`.aac`、`.ogg` 扩展名；文件以流式方式写入、计算 SHA-256，并逐块检查可用空间。
- SHA-256 用于有效受控录音去重；删除 recording 后其数据库记录及 hash 一并删除。
- 单次批量启动和队列重排最多 200 个 ID。重排请求必须包含当前全部且仅全部 queued jobs，不接受局部重排。
- `GET /audio` 支持完整音频流和单个 HTTP `Range` 请求，不支持多 range。
- 单产物、final、审计摘要和 ZIP 在读取 committed artifact 前核对路径仍在受控目录、文件存在性、字节数和 SHA-256；不匹配时拒绝返回产物。
- API 不返回 `.env`、API Key、base URL、完整请求正文、环境变量或原始异常堆栈。
- 当前没有 `continue-jobs`、batch continue、`advanced-recovery` 或 recording 维度 diagnostics API；诊断按 job 提供。

## 7. 后台任务、发布、恢复与取消

### 阶段发布协议

SQLite 与文件系统无法形成单一原子事务，当前实现使用 manifest 驱动的两阶段发布：

```text
在 staging 生成阶段产物
→ 计算 SHA-256 / size 并写入、fsync manifest.json
→ SQLite 写入 artifact=prepared、attempt=prepared
→ 移动到 artifacts/<type>/<filename>
→ 对目标文件重新校验 SHA-256 / size
→ SQLite 将 artifact 和 attempt 标为 committed
```

只有 `committed` artifacts 会被 API、ZIP、审计摘要和后续查询暴露。

### 启动恢复边界

- 启动时会扫描未完成的 `stage_attempts`。
- `running` attempt 或缺少 manifest 的 attempt 会被标记为 `abandoned`。
- `prepared` attempt 只有在 manifest、staging/目标文件、SHA-256 和 size 都匹配时才会继续完成发布；无法验证的 attempt 被放弃。
- 上述机制恢复的是阶段文件发布，不等同于服务冷启动时自动接管数据库中遗留的 `running` / `cancel_requested` jobs；后者仍是待补能力。

### 进度与诊断

- 上传使用浏览器 `XMLHttpRequest` 的真实字节进度。
- speaker review 风险片段与 final 分块的完成数会写入 `task_events`，可在 job 诊断区查看。
- `jobs.progress_completed` / `jobs.progress_total` 当前未由 worker 更新，列表/详情主视图不能据此稳定展示数值进度。
- 没有批次总体进度和 FunASR 已耗时显示。

### 推荐恢复

| 当前 run 中的 committed artifacts | 推荐 action / phase |
|---|---|
| 没有 `raw_json` | `run_funasr` / `funasr` |
| 有 `raw_json`、没有 `evidence` | `generate_evidence` / `evidence` |
| 缺少 `speaker_review` 或 `reviewed` | `continue_processing` / `speaker_review` |
| 缺少 `cleaned` | `continue_processing` / `cleaned` |
| 缺少 `final` 或 `final_audit` | `continue_processing` / `final` |
| 全部存在 | `none` |

推荐恢复仅对 `failed`、`interrupted`、`force_stopped` 的 job 生效，重新排队同一 run、同一 job。它不根据网络错误或限流分类，也不统一复验所有 committed artifacts 后创建新 run；高级恢复和批量恢复属于后续方向。

### 取消与强制停止

- queued job 的取消会直接标为 `cancelled`。
- running job 的取消写入 `cancel_requested`；worker 在阶段边界和 FunASR/LLM 调用周围的安全检查点响应。
- 前端在 `cancel_requested` 后即可展示“强制停止 worker”按钮，但服务端会在前 10 秒返回宽限期错误；达到 10 秒后，才会停止当前 supervisor 管理且匹配 job ID 的 worker。
- 强制停止会终止 worker、将 job 标为 `force_stopped`、将 run 标为 `interrupted`、按上述规则恢复或放弃阶段 attempt，并启动新 worker。

## 8. 安全与隐私边界

1. 官方启动入口固定绑定 `127.0.0.1`；应用 middleware 只接受 `127.0.0.1` 与 `localhost` Host。对 POST、PUT、PATCH、DELETE 请求，若带 Origin，则 Origin 主机也必须是本机主机名。
2. 本机无登录的信任边界仍是操作系统账户；允许 LAN 前必须补充认证、授权、CSRF/CORS 策略和传输安全。
3. 前端不读取 `.env`；worker 仅在 LLM 阶段读取模型名与凭据，API 不回传这些值。
4. 任务事件采用 stage/event 和 details 白名单，错误 API 统一返回脱敏文本；当前 worker 使用固定错误摘要，不向浏览器暴露原始堆栈。
5. 用户文件名会拒绝路径分隔符和控制字符，且不参与内部路径拼接；artifact 相对路径 resolve 后必须仍在 `app_data/` 内。
6. 上传仅以扩展名判定允许格式，仍可能在后续 FunASR 阶段发现伪装或损坏媒体；当前没有业务文件大小或时长上限。
7. 下载 committed artifact 前重新核对 SHA-256 与 size，避免篡改文件被作为已审计产物发送。
8. 删除必须传递 `confirm=true`，且存在活跃 job 时拒绝删除。删除会清理 recording 目录与关联数据库记录；成功删除不会留下长期 `deletion_operations` 审计行。
9. 原始 JSON、evidence、reviewed、canonical final 与审计文件没有前端编辑接口。speaker 显示名只形成版本化映射和展示导出。

## 9. 测试现状与后续验收

### 当前自动化覆盖

| 层级 | 当前覆盖 |
|---|---|
| 核心审计回归 | speaker review 与 final transcript 共 60 个 Python `unittest`，覆盖核心校验和 fallback 语义。 |
| pipeline 服务 | 2 个测试，覆盖 FunASR/evidence 服务化路径和匿名 speaker summary。 |
| 持久化与 worker | 17 个测试，覆盖迁移、FIFO、阶段发布/恢复、取消、部分 worker 编排和 Windows Job Object。 |
| Web/API | 17 个测试，覆盖 SPA fallback、上传、去重、队列、取消、强停宽限期、映射、下载、ZIP、删除、审计摘要与脱敏错误。 |
| Playwright UI | 3 个 UI smoke：搜索与单文件上传反馈、speaker 显示名即时渲染/保存、详情/speaker 直达路由与旧路径回退。业务 API 由 `page.route()` mock，不是完整真实后端业务流。 |
| 前端组件测试 | 当前没有 Vitest/Jest/Testing Library 等组件测试套件。 |
| 真实 FunASR → LLM | 当前没有默认自动化；需要本机模型、音频和 `.env` 凭据的受控手工验证。 |

### 后续完整验收方向

- 使用本地 fake pipeline/fake LLM 建立非 mock API 浏览器测试，覆盖真实上传、SQLite、worker、队列、取消、强停、推荐恢复、音频跳转、artifact/ZIP 下载、诊断和删除。
- 覆盖服务冷启动后的遗留 job 处理、多个服务实例的互斥、磁盘不足、上传中断、artifact 篡改与 prepared attempt 异常恢复。
- 为前端状态转换、列表、诊断、删除确认和 speaker 映射增加组件级测试。
- 在配置完成的受控机器上记录真实 FunASR → LLM 手工验收的音频、配置快照、产物 hash 和 final audit 摘要；不将外部模型调用纳入默认测试套件。

## 10. 与现有 CLI 的隔离

1. 不修改既有 CLI 的默认路径、参数、执行方式或工作流；CLI 继续使用 `input_audio/` 与 `output/`。
2. Web 不扫描、登记、浏览或删除 `input_audio/`、`output/`；Web 数据仅由上传进入 `app_data/`。
3. 新上传录音使用受控目录和不可变 ID，不使用音频 stem 作为存储身份。
4. Web 的旧录音导入入口已移除。若 UI 出现问题，CLI 可继续独立使用；停用或删除 `app_data/` 不影响 CLI 目录。
5. 当前 API/UI 仅展示 `current_run_id`；旧 run 作为受控历史数据保留，不支持从 Web 选择、浏览或下载旧运行版本。

## 11. 阶段现状、风险与回滚

| 阶段 | 当前状态 | 已完成主体 | 当前限制 / 后续 |
|---|---|---|---|
| 1. 核心服务化 | 基本完成 | FunASR、evidence、speaker review、cleaned、final 复用既有核心逻辑，保留 CLI 入口与审计内核。 | 继续保持 CLI 与 Web 的路径隔离；真实 Web worker 全链路审计集成验证仍可加强。 |
| 2. 数据与 worker | 主体完成 | SQLite、受控目录、两类 job、FIFO、manifest 发布、普通取消和强制停止。 | 服务冷启动 job 接管、跨服务实例全局 worker/GPU 锁、主视图数值进度尚未闭环。 |
| 3. FastAPI | 主要完成 | 上传、列表、任务、speaker、artifact、音频、删除、health 和推荐恢复 API。 | 无 continue、advanced recovery、批量恢复或 run history API。 |
| 4. React UI | MVP 完成 | 单页路由、列表、详情、speaker 映射、音频、下载、审计、诊断和任务操作。 | 主数值进度、批次进度、FunASR 已耗时、产物内嵌查看与更完整前端分层待补。 |
| 5. 联调与隔离验证 | 部分完成 | Python API/worker 测试和 mock API Playwright UI smoke。 | 真实 API 浏览器黄金路径、取消/强停/恢复/下载/删除 E2E、真实模型链路未自动化。 |
| 6. 后续版本 | 未开始 | 当前 canonical artifacts 和 CLI 隔离为后续演进保留边界。 | 人工编辑、逐句校正、LAN/多用户、认证与高级恢复需独立设计和审计。 |

回滚边界保持不变：Web 只使用 `app_data/`，不触碰 CLI 数据。停止 Web 服务或停用应用数据目录不会改变既有 CLI 行为。