import React, { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Job = {
  id: string;
  status: string;
  phase: string;
  progress_completed: number | null;
  progress_total: number | null;
  error_message: string | null;
};

type Recording = {
  id: string;
  display_name: string;
  original_filename: string;
  run_status: string | null;
  phase: string | null;
  job_status: string | null;
  progress_completed: number | null;
  progress_total: number | null;
  final_exists: boolean;
  error_message: string | null;
};

type RecordingPage = { items: Recording[]; page: number; page_size: number; total: number };
type Artifact = { id: string; type: string; variant: string; size_bytes: number; sha256: string };

const artifactDescriptions: Record<string, string> = {
  raw_json: "FunASR 原始识别结果（JSON）",
  evidence: "按时间和匿名 speaker 整理的证据文本",
  speaker_review: "speaker 复核决策与审计记录（JSON）",
  reviewed: "按复核结果重建的逐句文本",
  cleaned: "基于已复核文本清洗的中间稿",
  final: "经完整性校验的 canonical 最终稿",
  final_audit: "最终稿的完整性、告警与 fallback 审计记录（JSON）",
};

type Detail = {
  recording: {
    id: string;
    display_name: string;
    original_filename: string;
    run: { id: string; status: string; phase: string } | null;
    job: Job | null;
  };
  artifacts: Artifact[];
};
type SpeakerMapping = {
  entries: Record<string, string>;
  speaker_prefix: string | null;
};

type SpeakerSummary = {
  anonymous_label: string;
  occurrence_count: number;
  start_ms: number;
  end_ms: number;
  excerpts: { start_ms: number; end_ms: number; text: string }[];
};
type EventItem = { id: string; stage: string; event: string; completed: number | null; total: number | null; message: string | null; created_at: string };
type AuditSummary = {
  speaker_review: { review_required_count?: number; review_queue_count?: number; full_review_fallback?: boolean; available?: boolean } | null;
  final: { fallback_chunk_count?: number; warning_chunk_count?: number; warning_count?: number; available?: boolean } | null;
};

type UploadProgress = { name: string; loaded: number; total: number; error?: string };
type QueueJob = Job & { display_name: string; original_filename: string };

const pageSize = 20;
const activeStatuses = new Set(["queued", "running", "cancel_requested"]);

function apiMessage(value: unknown): string {
  if (typeof value === "object" && value !== null && "error" in value) {
    const error = (value as { error?: { message?: unknown } }).error;
    if (typeof error?.message === "string") return error.message;
  }
  return "请求未完成。";
}

async function jsonRequest<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    let payload: unknown = null;
    try { payload = await response.json(); } catch { payload = null; }
    throw new Error(apiMessage(payload));
  }
  return response.json() as Promise<T>;
}

function uploadWithProgress(file: File, autoStart: boolean, onProgress: (loaded: number, total: number) => void): Promise<{ created: boolean; recording: { id: string } }> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.set("file", file);
    form.set("auto_start", String(autoStart));
    const request = new XMLHttpRequest();
    request.open("POST", "/api/recordings/uploads");
    request.upload.onprogress = (event) => onProgress(event.loaded, event.total || file.size);
    request.onerror = () => reject(new Error("上传连接失败。"));
    request.onload = () => {
      let payload: unknown = null;
      try { payload = JSON.parse(request.responseText); } catch { payload = null; }
      if (request.status >= 200 && request.status < 300 && payload) {
        resolve(payload as { created: boolean; recording: { id: string } });
      } else {
        reject(new Error(apiMessage(payload)));
      }
    };
    request.send(form);
  });
}

function formatTime(value: number): string {
  const seconds = Math.floor(value / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function progressLabel(item: Pick<Recording, "progress_completed" | "progress_total"> | Job): string {
  if (item.progress_completed === null || item.progress_total === null) return "处理中";
  return `${item.progress_completed}/${item.progress_total}`;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function DownloadIcon({ archive = false }: { archive?: boolean }) {
  if (archive) {
    return <svg className="download-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M4 8.5 12 4l8 4.5v8L12 21l-8-4.5v-8Z"/><path d="m4 8.5 8 4.5 8-4.5M12 13v5"/><path d="m9.5 15.5 2.5 2.5 2.5-2.5"/></svg>;
  }
  return <svg className="download-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="M6 3h8l4 4v14H6z"/><path d="M14 3v5h5M12 10v7m-3-3 3 3 3-3M8.5 20h7"/></svg>;
}

function renderFinalDisplay(canonicalText: string, entries: Record<string, string>, speakerPrefix: string | null): string {
  let rendered = canonicalText;
  for (const [anonymousLabel, displayName] of Object.entries(entries)) {
    if (!displayName || !anonymousLabel.startsWith("SPEAKER_")) continue;
    const suffix = anonymousLabel.slice("SPEAKER_".length);
    const labels = [anonymousLabel, ...(speakerPrefix ? [`${speakerPrefix}${suffix}`] : [])];
    for (const label of labels) {
      const expression = new RegExp(`^((?:\\[[^\\n]+\\]\\s+)?)${escapeRegExp(label)}(?=(?:【待回听】)?：)`, "gm");
      rendered = rendered.replace(expression, `$1${displayName}`);
    }
  }
  return rendered;
}

function App() {
  const initialParameters = new URLSearchParams(window.location.search);
  const [location, setLocation] = React.useState(() => `${window.location.pathname}${window.location.search}`);
  const [query, setQuery] = React.useState(initialParameters.get("query") ?? "");
  const [currentPage, setCurrentPage] = React.useState(Math.max(1, Number(initialParameters.get("page")) || 1));
  const [page, setPage] = React.useState<RecordingPage>({ items: [], page: 1, page_size: pageSize, total: 0 });
  const [selected, setSelected] = React.useState<Set<string>>(new Set());
  const [queueJobs, setQueueJobs] = React.useState<QueueJob[]>([]);
  const [detail, setDetail] = React.useState<Detail | null>(null);
  const [finalText, setFinalText] = React.useState("");
  const [speakerEntries, setSpeakerEntries] = React.useState<Record<string, string>>({});
  const [speakerPrefix, setSpeakerPrefix] = React.useState<string | null>(null);
  const [speakerSummaries, setSpeakerSummaries] = React.useState<SpeakerSummary[]>([]);
  const [displayName, setDisplayName] = React.useState("");
  const [diagnostics, setDiagnostics] = React.useState<EventItem[] | null>(null);
  const [auditSummary, setAuditSummary] = React.useState<AuditSummary | null>(null);
  const [message, setMessage] = React.useState("");
  const [uploading, setUploading] = React.useState<UploadProgress[]>([]);
  const audio = React.useRef<HTMLAudioElement>(null);
  const [pathname] = location.split("?");
  const routeMatch = pathname.match(/^\/recordings\/([^/]+)(?:\/(speakers))?\/?$/);
  const routeRecordingId = routeMatch?.[1] ?? null;
  const speakerRoute = routeMatch?.[2] === "speakers";

  function navigate(path: string): void {
    window.history.pushState(null, "", path);
    setLocation(path);
  }

  React.useEffect(() => {
    const onPopState = () => setLocation(`${window.location.pathname}${window.location.search}`);
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const loadRecordings = React.useCallback(async () => {
    try {
      const value = await jsonRequest<RecordingPage>(`/api/recordings?page=${currentPage}&page_size=${pageSize}&query=${encodeURIComponent(query)}`);
      setPage(value);
      if (value.page !== currentPage && value.total > 0) setCurrentPage(value.page);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "无法获取录音列表。");
    }
  }, [currentPage, query]);

  const loadQueue = React.useCallback(async () => {
    try {
      const value = await jsonRequest<{ jobs: QueueJob[] }>("/api/jobs/queue");
      setQueueJobs(value.jobs);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "无法获取任务队列。");
    }
  }, []);

  React.useEffect(() => { void loadRecordings(); void loadQueue(); }, [loadQueue, loadRecordings]);
  React.useEffect(() => {
    if (routeRecordingId) return;
    const next = new URLSearchParams();
    if (query) next.set("query", query);
    if (currentPage > 1) next.set("page", String(currentPage));
    const nextLocation = `/recordings${next.size ? `?${next}` : ""}`;
    if (location !== nextLocation) {
      window.history.replaceState(null, "", nextLocation);
      setLocation(nextLocation);
    }
  }, [currentPage, location, query, routeRecordingId]);

  const hasActiveListJob = page.items.some((item) => item.job_status !== null && activeStatuses.has(item.job_status));
  React.useEffect(() => {
    if (!hasActiveListJob) return;
    const interval = window.setInterval(() => {
      if (!document.hidden) void loadRecordings();
    }, 2500);
    return () => window.clearInterval(interval);
  }, [hasActiveListJob, loadRecordings]);

  const loadDetail = React.useCallback(async (recordingId: string, includeAncillary: boolean) => {
    try {
      const value = await jsonRequest<Detail>(`/api/recordings/${recordingId}`);
      setDetail(value);
      setDisplayName(value.recording.display_name);
      if (value.artifacts.some((item) => item.type === "final")) {
        const response = await fetch(`/api/recordings/${recordingId}/download/final`);
        if (response.ok) setFinalText(await response.text());
      } else {
        setFinalText("");
      }
      if (includeAncillary) {
        const [mapping, summary, audit] = await Promise.all([
          jsonRequest<SpeakerMapping>(`/api/recordings/${recordingId}/speaker-mapping`).catch(() => ({ entries: {}, speaker_prefix: null })),
          jsonRequest<{ items: SpeakerSummary[] }>(`/api/recordings/${recordingId}/speaker-summary`).catch(() => ({ items: [] })),
          jsonRequest<AuditSummary>(`/api/recordings/${recordingId}/audit-summary`).catch(() => ({ speaker_review: null, final: null })),
        ]);
        setAuditSummary(audit);
        setSpeakerPrefix(mapping.speaker_prefix);
        const mergedEntries = { ...mapping.entries };
        for (const item of summary.items) if (!(item.anonymous_label in mergedEntries)) mergedEntries[item.anonymous_label] = "";
        setSpeakerEntries(mergedEntries);
        setSpeakerSummaries(summary.items);
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "无法获取录音详情。");
    }
  }, []);

  React.useEffect(() => {
    if (!routeRecordingId) {
      setDetail(null);
      return;
    }
    setDiagnostics(null);
    setAuditSummary(null);
    setSpeakerEntries({});
    setSpeakerPrefix(null);
    setSpeakerSummaries([]);
    void loadDetail(routeRecordingId, true);
  }, [loadDetail, routeRecordingId]);

  React.useEffect(() => {
    if (!detail?.recording.job || !activeStatuses.has(detail.recording.job.status)) return;
    const recordingId = detail.recording.id;
    const interval = window.setInterval(() => {
      if (!document.hidden) void loadDetail(recordingId, false);
    }, 2500);
    return () => window.clearInterval(interval);
  }, [detail?.recording.id, detail?.recording.job?.status, loadDetail]);

  function openRecording(recordingId: string): void {
    navigate(`/recordings/${recordingId}`);
  }

  async function uploadFiles(files: FileList | null): Promise<void> {
    if (!files?.length) return;
    const values = Array.from(files);
    setUploading(values.map((file) => ({ name: file.name, loaded: 0, total: file.size })));
    const createdIds: string[] = [];
    const autoStart = values.length === 1;
    for (let index = 0; index < values.length; index += 1) {
      const file = values[index];
      try {
        const result = await uploadWithProgress(file, autoStart, (loaded, total) => {
          setUploading((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, loaded, total } : item));
        });
        if (result.created) createdIds.push(result.recording.id);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "上传失败。";
        setUploading((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, error: errorMessage } : item));
      }
    }
    setSelected(new Set(createdIds));
    setMessage(autoStart ? "已上传并自动进入队列。" : "文件已登记。请选择录音后批量启动。 ");
    await loadRecordings();
  }

  async function startSelected(): Promise<void> {
    if (!selected.size) return;
    try {
      await jsonRequest("/api/recordings/batch/funasr-jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recording_ids: [...selected] }),
      });
      setSelected(new Set());
      setMessage("已按当前队列顺序启动所选录音。");
      await loadRecordings();
      await loadQueue();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "批量启动失败。");
    }
  }

  async function moveQueuedJob(jobId: string, direction: -1 | 1): Promise<void> {
    const index = queueJobs.findIndex((item) => item.id === jobId);
    const target = index + direction;
    if (index < 0 || target < 0 || target >= queueJobs.length) return;
    const next = [...queueJobs];
    [next[index], next[target]] = [next[target], next[index]];
    setQueueJobs(next);
    try {
      await jsonRequest("/api/jobs/reorder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_ids: next.map((item) => item.id) }),
      });
      setMessage("已更新未开始任务的执行顺序。");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "队列调整失败。");
      await loadQueue();
    }
  }

  async function cancelJob(): Promise<void> {
    if (!detail?.recording.job) return;
    try {
      await jsonRequest(`/api/jobs/${detail.recording.job.id}/cancel`, { method: "POST" });
      setMessage("已请求取消任务。将在下一个安全检查点停止。");
      await loadDetail(detail.recording.id, false);
      await loadRecordings();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "取消任务失败。");
    }
  }

  async function forceStopJob(): Promise<void> {
    if (!detail?.recording.job || !window.confirm("将终止当前 worker 并自动重启。仅在普通取消长时间无响应时使用。")) return;
    try {
      await jsonRequest(`/api/jobs/${detail.recording.job.id}/force-stop`, { method: "POST" });
      setMessage("worker 已强制停止并重启；可按推荐方式恢复未完成阶段。 ");
      await loadDetail(detail.recording.id, false);
      await loadRecordings();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "强制停止失败。");
    }
  }

  async function recoverJob(): Promise<void> {
    if (!detail?.recording.job) return;
    try {
      await jsonRequest(`/api/jobs/${detail.recording.job.id}/recommended-recovery`, { method: "POST" });
      setMessage("已按推荐阶段重新排队。");
      await loadDetail(detail.recording.id, false);
      await loadRecordings();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "恢复任务失败。");
    }
  }

  async function saveSpeakers(): Promise<void> {
    if (!detail || !Object.keys(speakerEntries).length) return;
    try {
      await jsonRequest(`/api/recordings/${detail.recording.id}/speaker-mapping`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ entries: speakerEntries }),
      });
      setMessage("speaker 显示名映射已保存，不会改写审计产物。");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "speaker 映射保存失败。");
    }
  }

  async function saveDisplayName(): Promise<void> {
    if (!detail) return;
    try {
      const value = await jsonRequest<{ display_name: string }>(`/api/recordings/${detail.recording.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ display_name: displayName }),
      });
      setDetail((current) => current ? { ...current, recording: { ...current.recording, display_name: value.display_name } } : current);
      setMessage("显示名称已更新。");
      await loadRecordings();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "显示名称更新失败。");
    }
  }

  async function deleteRecording(): Promise<void> {
    if (!detail) return;
    if (!window.confirm(`确认删除“${detail.recording.display_name}”及其受控产物吗？此操作无法撤销。`)) return;
    try {
      await jsonRequest(`/api/recordings/${detail.recording.id}?confirm=true`, { method: "DELETE" });
      navigate("/recordings");
      setMessage("录音与其受控产物已删除。");
      await loadRecordings();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "删除失败。");
    }
  }

  async function toggleDiagnostics(): Promise<void> {
    if (!detail?.recording.job) return;
    if (diagnostics) {
      setDiagnostics(null);
      return;
    }
    try {
      const result = await jsonRequest<{ events: EventItem[] }>(`/api/jobs/${detail.recording.job.id}/diagnostics`);
      setDiagnostics(result.events);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "无法获取任务诊断。");
    }
  }

  function toggleSelected(recordingId: string): void {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(recordingId)) next.delete(recordingId); else next.add(recordingId);
      return next;
    });
  }

  function seek(startMs: number): void {
    if (!audio.current) return;
    audio.current.currentTime = startMs / 1000;
    void audio.current.play().catch(() => undefined);
  }

  const activeDetailJob = detail?.recording.job && activeStatuses.has(detail.recording.job.status);
  const failedDetailJob = detail?.recording.job && ["failed", "interrupted", "force_stopped"].includes(detail.recording.job.status);
  const displayedFinalText = renderFinalDisplay(finalText, speakerEntries, speakerPrefix);
  const pageCount = Math.max(1, Math.ceil(page.total / pageSize));

  if (detail && routeRecordingId) {
    return <main>
      <button className="secondary" onClick={() => navigate("/recordings")}>返回录音列表</button>
      <header className="detail-header">
        <div>
          <p className="eyebrow">{detail.recording.original_filename}</p>
          <h1>{detail.recording.display_name}</h1>
          <p>受控录音 · {detail.recording.run?.status ?? "尚未启动"} · {detail.recording.run?.phase ?? "-"}</p>
        </div>
        <div className="actions">
          {activeDetailJob && <button onClick={() => void cancelJob()}>取消任务</button>}
          {detail.recording.job?.status === "cancel_requested" && <button className="danger" onClick={() => void forceStopJob()}>强制停止 worker</button>}
          {failedDetailJob && <button onClick={() => void recoverJob()}>按推荐恢复</button>}
          <button className="secondary" onClick={() => void toggleDiagnostics()} disabled={!detail.recording.job}>诊断</button>
        </div>
      </header>
      {message && <p className="message">{message}</p>}
      <audio ref={audio} controls src={`/api/recordings/${detail.recording.id}/audio`}/>
      <section className="panel">
        <h2>显示名称</h2>
        <div className="inline-form">
          <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} disabled={Boolean(activeDetailJob)} />
          <button onClick={() => void saveDisplayName()} disabled={Boolean(activeDetailJob)}>保存名称</button>
        </div>
      </section>
      <section className="panel">
        <div className="section-title"><h2>speaker 显示名映射</h2><div className="actions">{!speakerRoute && <button className="secondary" onClick={() => navigate(`/recordings/${detail.recording.id}/speakers`)}>独立页面</button>}<button onClick={() => void saveSpeakers()} disabled={!Object.keys(speakerEntries).length}>保存映射</button></div></div>
        {speakerSummaries.length === 0 && <p>FunASR 完成后会显示匿名 speaker 与代表片段；显示名不会写入 canonical 审计产物。</p>}
        <div className="speaker-grid">{speakerSummaries.map((speaker) => <article className="speaker-card" key={speaker.anonymous_label}>
          <strong>{speaker.anonymous_label}</strong><small>{speaker.occurrence_count} 次 · {formatTime(speaker.start_ms)}–{formatTime(speaker.end_ms)}</small>
          <input aria-label={`${speaker.anonymous_label} 显示名`} value={speakerEntries[speaker.anonymous_label] ?? ""} placeholder="可留空" onChange={(event) => setSpeakerEntries((current) => ({ ...current, [speaker.anonymous_label]: event.target.value }))} />
          {speaker.excerpts.map((excerpt, index) => <button className="excerpt" key={`${speaker.anonymous_label}-${index}`} onClick={() => seek(excerpt.start_ms)}>{formatTime(excerpt.start_ms)} {excerpt.text}</button>)}
        </article>)}</div>
      </section>
      <section className="panel final-reading">
        <div className="section-title"><h2>最终阅读版</h2><div className="links">{finalText && <a className="download-link" href={`/api/recordings/${detail.recording.id}/download/final?display_names=true`}><DownloadIcon />下载显示名版本</a>}</div></div>
        <p>页面与显示名版本下载会按当前映射渲染，不会修改 canonical final 或 final audit。</p>
        <pre>{displayedFinalText || "最终稿尚未生成。完成前仅可下载已提交的阶段产物。"}</pre>
      </section>
      {auditSummary && (auditSummary.speaker_review || auditSummary.final) && <section className="panel audit-summary"><h2>回听与审计摘要</h2>
        {auditSummary.speaker_review && <p>说话人复核：{auditSummary.speaker_review.review_required_count ?? 0} 条【待回听】、{auditSummary.speaker_review.review_queue_count ?? 0} 条复核队列{auditSummary.speaker_review.full_review_fallback ? "；全稿复核已保守回退" : ""}。</p>}
        {auditSummary.final && <p>最终阅读整理：{auditSummary.final.fallback_chunk_count ?? 0} 个 fallback 分块、{auditSummary.final.warning_chunk_count ?? 0} 个软告警分块、{auditSummary.final.warning_count ?? 0} 条软告警。</p>}
      </section>}
      <section className="panel">
        <h2>已提交证据与审计</h2>
        <div className="links">{detail.artifacts.map((artifact) => <a key={artifact.id} href={`/api/recordings/${detail.recording.id}/artifacts/${artifact.type}`}>{artifact.type}（{artifactDescriptions[artifact.type] ?? "已提交产物"}）</a>)}</div>
        {detail.artifacts.length > 0 && <p><a className="download-link archive-download" href={`/api/recordings/${detail.recording.id}/download/all`}><DownloadIcon archive />下载当前运行全部 ZIP</a></p>}
      </section>
      {diagnostics && <section className="panel"><h2>脱敏任务诊断</h2><ul className="diagnostics">{diagnostics.map((event) => <li key={event.id}>{event.stage} · {event.event}{event.completed !== null && event.total !== null ? ` · ${event.completed}/${event.total}` : ""}{event.message ? ` · ${event.message}` : ""}</li>)}</ul></section>}
      <button className="danger" onClick={() => void deleteRecording()} disabled={Boolean(activeDetailJob)}>删除录音与受控产物</button>
    </main>;
  }


  return <main>
    <header><h1>FunASR_e2e</h1><p>仅本机运行的可审计转写</p></header>
    {message && <p className="message">{message}</p>}
    <section className="toolbar">
      <input value={query} onChange={(event) => { setQuery(event.target.value); setCurrentPage(1); }} placeholder="搜索文件名或显示名称" />
      <label className="upload">上传音频<input multiple type="file" accept=".wav,.mp3,.m4a,.flac,.aac,.ogg" onChange={(event) => { void uploadFiles(event.target.files); event.target.value = ""; }} /></label>
    </section>
    {uploading.length > 0 && <section className="panel upload-progress"><h2>上传进度</h2>{uploading.map((item, index) => <div key={`${item.name}-${index}`}><span>{item.name}</span><progress max={item.total || 1} value={item.loaded} /><small>{item.error ?? `${Math.round((item.loaded / Math.max(item.total, 1)) * 100)}%`}</small></div>)}</section>}
    {queueJobs.length > 0 && <section className="panel queue"><div className="section-title"><h2>等待队列</h2><button className="secondary" onClick={() => void loadQueue()}>刷新队列</button></div><p>仅未开始任务可调整位置；当前运行任务不受影响。</p>{queueJobs.map((job, index) => <div className="queue-item" key={job.id}><span>{index + 1}. {job.display_name}</span><small>{job.phase}</small><div className="actions"><button className="secondary" disabled={index === 0} onClick={() => void moveQueuedJob(job.id, -1)}>上移</button><button className="secondary" disabled={index === queueJobs.length - 1} onClick={() => void moveQueuedJob(job.id, 1)}>下移</button></div></div>)}</section>}
    <section className="panel list"><div className="section-title"><h2>录音</h2><button onClick={() => void startSelected()} disabled={!selected.size}>启动所选 {selected.size || ""}</button></div>{page.items.map((item) => <article className="recording" key={item.id}>
      <label className="checkbox"><input type="checkbox" checked={selected.has(item.id)} disabled={Boolean(item.job_status && activeStatuses.has(item.job_status))} onChange={() => toggleSelected(item.id)} /></label>
      <button className="recording-main" onClick={() => void openRecording(item.id)}><strong>{item.display_name}</strong><small>{item.original_filename} · 受控</small></button>
      <div>{item.job_status ?? "未排队"} · {item.phase ?? "-"}<small>{item.job_status && activeStatuses.has(item.job_status) ? progressLabel(item) : item.final_exists ? "最终稿已完成" : item.error_message ?? ""}</small></div>
    </article>)}{page.items.length === 0 && <p>暂无录音。</p>}
      <div className="pagination"><button className="secondary" disabled={currentPage <= 1} onClick={() => setCurrentPage((value) => value - 1)}>上一页</button><span>第 {currentPage}/{pageCount} 页，共 {page.total} 条</span><button className="secondary" disabled={currentPage >= pageCount} onClick={() => setCurrentPage((value) => value + 1)}>下一页</button></div>
    </section>
  </main>;
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
