# 说话人归属复核交接单

## 1. 当前结论

本轮改造在**流程效率、原文完整性和审计可追溯性**上已完成验证，但在真实协商录音的**关键说话人归属质量**上未通过验收。

- 不要提交或推送当前工作区。
- 不要把最新 `_reviewed.txt`、`_cleaned.txt` 当成可用于正式协商记录、证据整理或最终交付的说话人稿。
- 新会话的优先目标是提高 speaker 归属可靠性，尤其防止金额、补偿、立场、接受/拒绝等高影响内容被静默归给错误说话人。

当前提交状态：**没有创建 commit，也没有 push**。

## 面向通用场景的主逻辑

后续改造不应针对本样本或预设两人身份，而应适用于访谈、会议、客服、协商、课堂等任意多说话人录音。核心不是机械沿用 FunASR 的 `spk` 标签，而是把它视为弱声学先验，在不改动原文的前提下建立并验证整段对话的语义结构。

```text
不可变证据层（音频 / JSON / evidence）
→ 全稿语义梳理：匿名角色、持续立场、核心诉求、问答关系、称呼和对话主线
→ 以全局角色模型为依据，复核每句或精确切片的匿名 speaker 归属
→ 对金额、否定、承诺、争议、打断、短答、重叠等高影响或高歧义区域做局部深度复核
→ 保守地输出 reviewed、待回听队列和独立的上下文推断注释
→ 经人工验收后才生成面向阅读的 polished
```

### 1. 全稿梳理优先于逐句改标

模型首先应阅读完整时间线，建立**匿名**的 speaker registry。每个匿名角色的 registry 至少应包含：

- 相对稳定的角色摘要，例如提问方、解释方、谈判方、客服、客户、主持人；不得编造真实姓名或身份。
- 该角色的主要诉求、立场、反复主张和常用称呼。
- 用于支撑该判断的原始 `source_id`，以便审计和回听。
- 对角色不稳定、多人轮替或证据不足的明确不确定性说明。

然后才以这一全局结构处理局部句子：谁在回答问题、谁在延续先前诉求、哪句是打断、哪句可能属于交叠、哪些 FunASR 声学标签与长期立场相矛盾。这样才能减少连续串词，而不是只用相邻句的声学 speaker 标签投票。

### 2. “补充”只能是可追溯的推断，不能补写转写正文

用户所说的“补充模糊内容”应拆成两个不同层次：

- **允许补充：** 对话主线、匿名角色摘要、诉求/立场标签、问答关联、潜在交叠、归属置信度、待回听原因、可能的上下文关系。这些内容必须与原文分离保存为 audit 或独立 annotation，且要能定位回原始 `source_id`。
- **禁止补充：** 猜测或补写音频中未识别到的字词、数字、人名、日期、金额、否定词、条件、承诺或完整句子；也不得用上下文“纠正”原始 ASR 文本。

当原句被截断、抢话、听不清或仅靠上下文无法安全判断时，正确输出是 `unknown`、`overlap` 或 `【待回听】`，并附上结构化原因；不是把模型猜测写入正文。

建议后续增加独立的 `context_annotations` 审计字段或单独 JSON 输出，记录这类推断，但不要混入 `_evidence.txt`、`_reviewed.txt` 或原始 JSON。

## 2. 不可违反的约束

1. 模型必须严格读取项目 `.env` 的 `MODEL_NAME`；不得在 `settings.yaml` 或 Python 中固定、覆盖或回退模型。
2. 不读取、不回显 `.env` 内 API key、base URL 等敏感值。
3. LLM 只可调整匿名 speaker 归属或将原句精确切分；不得新增、删除、纠正、改写、总结原始 ASR 文本。
4. 原始 WAV、FunASR JSON、`_evidence.txt` 是证据层，必须保留。
5. `unknown`、`overlap` 与 `【待回听】` 可以保守使用；宁可待回听，不可自信错归。
6. 参考稿只能在所有生产模型请求完成后用于事后验收，绝不能放入任何生产 prompt。
7. 匿名 speaker 默认自动聚类；不要映射为真实姓名或把样本中的身份映射硬编码进程序。
8. 不要为了提高质量恢复旧的 anchor → registry → 双 Pass → arbitration → micro-split 全量多轮流程。旧流程 token 消耗过大且实际质量仍不足。

## 3. 当前真实样本与身份核验

测试音频：

```text
input_audio/08月05日_来瑞_港之龙.wav
```

事后参考稿：

```text
参考文档/2026年08月05日_与来瑞沟通协商_港之龙_整理稿.md
```

根据开场内容进行的**仅限本样本事后验收**身份映射：

```text
SPEAKER_0 ≈ 何景城
SPEAKER_1 ≈ 来瑞
```

这只是验收依据，不能进入生产逻辑或 prompt。

## 4. 已实现的架构

当前主流程：

```text
FunASR JSON/evidence
→ 一次 full_review（匿名角色 registry + 稀疏 override + 风险提示）
→ 本地结构风险融合并组成连续 risk segment
→ 只审核高风险 segment
→ 精确原文切片、speaker 白名单、逐句与全局重建校验
→ reviewed
→ cleaned
→ 人工验收后才 polish-only
```

已移除旧主路径中的：anchor、独立 registry、Pass A/B、arbitration、micro-split、全句全字符边界枚举。

### 关键文件

```text
scripts/review_funasr_speakers.py      说话人复核主逻辑
scripts/postprocess_funasr_transcript.py  原始句、清理、渲染与 API 调用基础能力
scripts/run_funasr_full_pipeline.py    命令行、模型凭据、阶段拆分
settings.yaml                           当前运行配置
prompt/speaker_review_prompt_template.txt  结构化复核提示词
tests/test_speaker_review.py            单元测试
FunASR_e2e_部署指导.md                  部署与使用说明
```

## 5. 已完成的安全与流程能力

### 原文与审计

- JSON 原始顺序和文本保持不变。
- `KEEP`、`REASSIGN`、`SPLIT`、`REVIEW_REQUIRED`、`OVERLAP` 都受严格 schema 验证。
- `SPLIT` 必须使用程序发送的可靠边界，连续、非空并完整覆盖原句。
- reviewed span 必须精确重建每个源句和整份源稿。
- audit 使用 schema v2，记录源 JSON hash、模型、请求数、prompt 字节数、决策、片段与完整性校验。

### 模型与阶段拆分

- `llm.model`、`speaker_review.model` 必须保持 `null`；否则程序拒绝运行。
- 模型只从项目 `.env` 的 `MODEL_NAME` 读取。
- 支持：

```bash
# 复用 JSON，只跑 speaker review，跳过普通润色
.venv/Scripts/python.exe scripts/run_funasr_full_pipeline.py --settings settings.yaml --reuse-json --skip-polish

# 只在现有 reviewed/cleaned 经 schema v2、源 hash、完整性验证后润色
.venv/Scripts/python.exe scripts/run_funasr_full_pipeline.py --settings settings.yaml --polish-only
```

- 生成新 `reviewed/cleaned` 时会清除旧 `_polished.txt`，避免润色失败后旧稿被误认作当前结果。

### 思考模式兼容

`speaker_review.enable_thinking` 已成为独立配置，并会传到 full review 和 risk segment 请求。

- `qwen3.7-max-preview` 实测 API 强制要求 `enable_thinking=true`，且 619 句全稿常超过 90 秒，前台等待不合适。
- 当前 `.env` 被用户改为 `qwen3.7-flash`；当前 `settings.yaml` 将 `llm.enable_thinking` 与 `speaker_review.enable_thinking` 都设为 `false`，`speaker_review.request_timeout_s` 为 `90`。
- 不要假定任意模型都可关闭或必须开启思考。若 API 返回参数限制，按返回的明确约束调整配置。

## 6. 本轮真实运行记录

### 失败尝试

1. `qwen3.7-max-preview` + `enable_thinking=false`：API 400，明确要求 `enable_thinking=true`。
2. `qwen3.7-max-preview` + `enable_thinking=true`：619 句全稿在 90 秒超时；调为 300 秒后用户中断等待。不要在前台无提示地运行可能长达多次重试的请求。
3. `qwen3.7-flash` 初次运行：模型生成的 `risk_item.source_ids` 超过 `max_risk_core_sentences=12`，严格 validator 使全稿结果被整体降级为 619 个 `unknown`。

### 已做的兼容修正

- 将 `max_risk_core_sentences` 传入 full_review 输入并在 prompt 明确要求超长风险项拆分。
- 对于长度为 0 或超过上限的**可选** `risk_item`，保留合法的 registry 和 overrides，仅丢弃该不合格风险项，并在 audit 的 `full_review.discarded_risk_items` 记录原因；其余结构风险仍由本地规则生成。
- 注意：这不是放松原文、speaker 白名单或切片完整性校验；只避免可选风险提示格式错误导致整份全稿无条件降级。

### 最后一次成功完成的运行

使用当前 `.env` 的：

```text
qwen3.7-flash
```

运行命令：

```bash
.venv/Scripts/python.exe scripts/run_funasr_full_pipeline.py --settings settings.yaml --reuse-json --skip-polish
```

输出审计：

```text
output/08月05日_来瑞_港之龙/08月05日_来瑞_港之龙_speaker_review.json
```

结果：

```text
full review：1 次，成功
risk segment：8 次实际请求
总请求：9 次
prompt 总字节数：137,992
边界候选数：52
KEEP：583
REASSIGN：0
SPLIT：1
unknown：35
overlap：0
```

完整性全部为 `true`：

```text
source_hash_verified
per_source_reconstruction_passed
global_reconstruction_passed
order_passed
coverage_passed
allowed_speaker_passed
```

与旧流程约 109 次请求、约 2.8 MB prompt 相比，新架构的请求数约减少 92%，prompt 体积约减少 95%。这部分改造是成功的。

## 7. 质量验收：未通过

不能把 `KEEP=583` 或 `unknown=35` 当作说话人准确率。

- `KEEP` 只表示沿用 FunASR 的原始匿名标签，不代表标签正确。
- `unknown` 约占 5.7%，但其余 94.2% 并不等于 94.2% speaker 准确率。
- 没有对 619 个 source sentence 做逐句 gold 标注和对齐，因此不能诚实地宣称整体 speaker 准确率是 80%、90% 或其他具体数值。

对参考稿重点区间的事后检查显示，仍存在高影响的静默错归：

| 时间段 | 参考稿中的正确归属 | 最新 reviewed 的问题 |
|---|---|---|
| 约 03:15–03:25 | 来瑞质疑“公积金怎么算、没有那么多” | 仍显示为 `SPEAKER_0`，应为 `SPEAKER_1` |
| 约 08:15–08:26 | 何景城说“公司应补缴将近六万、我个人也要补” | 仍显示为 `SPEAKER_1`，应为 `SPEAKER_0` |
| 约 15:13–15:42 | 何景城说明 N、补偿基数和 14600 元 | 连续显示为 `SPEAKER_1`，应主要为 `SPEAKER_0` |
| 约 19:34–20:11 | 来瑞与何景城围绕公积金、合法依据、公积金中心交替发言 | 存在双方反向归属和不足的 `unknown` 标记 |

可直接定位的输出证据：

```text
output/08月05日_来瑞_港之龙/08月05日_来瑞_港之龙_reviewed.txt

03:18 左右：约第 227 行
08:20 左右：约第 637 行
15:13–15:42：约第 1180 行
19:34–20:11：约第 1546 行、1579 行
```

结论：

- 可作为带时间戳的检索草稿和人工回听定位材料。
- 不可作为具有说话人归属的正式协商记录、证据整理稿或最终稿。
- 对本样本而言，即使不要求 100% 准确，也不能接受金额、补偿基数、诉求、接受/拒绝被静默归给另一方。

## 8. 现有质量缺口与下一步建议

当前失败不是文本重建失败，而是 risk coverage 和 speaker decision 质量不足：

1. full_review 输出稀疏，最终 `REASSIGN=0`，没有纠正若干明显连续错归。
2. 高风险片段没有覆盖所有关键连续错归区域。
3. 一个 24 句的风险组件因超过 `max_risk_core_sentences=12` 按失败策略被整体标为待回听；这是保守但覆盖粒度过粗。
4. 一个单句风险片段连续三次输出无效 SPLIT，最终 fallback 为 `unknown`。这比错误应用 SPLIT 安全，但说明 Flash 对精确边界 schema 的服从性有限。
5. 多个关键错误发生在同一原始 FunASR sentence 或快速交替的相邻句中；仅依赖 source speaker、短答、时间异常等本地启发式无法充分捕获。

建议新会话先进入计划阶段，然后重点评估以下方向，不要直接叠加更多全稿重复调用：

1. **高影响语义触发器**：对金额、补偿、社保、公积金、N/N+1、日期、工资基数、接受/拒绝、否定和“公司/个人”立场等内容，不能单独触发大量 LLM 请求，但当其与 speaker 快速切换、打断、相邻立场冲突同时出现时应强制进入 risk segment。
2. **连续错归组件拆分**：对超过 12 句的硬风险组件，研究按自然停顿、speaker 变化或语义边界安全切分，而不是整段 fallback。拆分后每个 source 仍只能属于一个 core，不能复制或重叠。
3. **full_review 输出约束**：评估如何让 Flash 可靠指出少量关键 override / risk item；可加强“不要把一整段或全稿列为 risk item、无风险必须输出空数组、只列最高风险连续句”的 prompt，而不是放宽关键 decision 校验。
4. **失败模式处理**：对模型无效 SPLIT，继续保守 fallback；不要根据模型未发送的边界猜测切点。可以调整该类句子的 allowed_operations，使没有足够可靠边界时只允许 `REVIEW_REQUIRED` 或 `OVERLAP`。
5. **验收方法**：生产请求结束后再读取参考稿，优先人工核验下列区间：

```text
02:30–03:32
07:26–10:15
14:59–16:14
18:59–20:43
```

验收重点：金额、日期、N/N+1、否定、接受/拒绝、补偿基数、社保/公积金论述的 speaker 是否正确；不确定的应为 `unknown` 或 `overlap`，不能错误归属。

## 9. 测试状态

已运行并通过：

```bash
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m py_compile scripts/postprocess_funasr_transcript.py scripts/review_funasr_speakers.py scripts/run_funasr_full_pipeline.py tests/test_speaker_review.py
git diff --check
```

最新结果：

```text
Ran 21 tests
OK
```

当前机器未安装 FFmpeg，启动时会提示 torchaudio fallback；本次输入为 WAV，未阻塞转写或复核。未来处理 MP3/M4A/AAC 时需要安装 FFmpeg。

## 10. 工作区状态

当前工作区有未提交改动与未跟踪文件；不要假定只包含本次 speaker review 改动。`git status --short` 显示的主要内容：

```text
修改：
- FunASR_e2e_部署指导.md
- prompt/drop_words.txt
- prompt/polish_prompt_template.txt
- scripts/postprocess_funasr_transcript.py
- scripts/run_funasr_full_pipeline.py
- settings.yaml

未跟踪：
- 2026-08-06_协商录音转写流程优化分析.md
- prompt/hotwords.txt
- prompt/speaker_review_prompt_template.txt
- scripts/review_funasr_speakers.py
- tests/
- 参考文档/
- speaker_review_handoff.md
```

在提交前必须再次审查完整 diff、确认哪些文件应纳入，并在 speaker 质量验收通过后再向用户请求或执行 commit/push。
