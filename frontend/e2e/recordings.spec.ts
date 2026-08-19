import { expect, test } from "@playwright/test";

let speakerEntries: Record<string, string> = {};

const recordings = {
  items: [{
    id: "recording-1",
    display_name: "访谈录音",
    original_filename: "interview.wav",
    run_status: "completed",
    phase: "complete",
    job_status: "succeeded",
    progress_completed: null,
    progress_total: null,
    final_exists: true,
    error_message: null,
  }],
  page: 1,
  page_size: 20,
  total: 1,
};

test.beforeEach(async ({ page }) => {
  speakerEntries = { SPEAKER_0: "" };
  await page.route("**/api/recordings?**", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify(recordings) });
  });
  await page.route("**/api/recordings/uploads", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ created: true, recording: { id: "recording-uploaded" } }),
    });
  });
  await page.route("**/api/jobs/queue", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ jobs: [{
      id: "queued-1", run_id: "run-1", kind: "funasr", queue_seq: 1, status: "queued", phase: "funasr",
      progress_completed: null, progress_total: null, error_code: null, error_message: null,
      cancel_requested_at: null, created_at: "", updated_at: "", display_name: "等待录音", original_filename: "queued.wav",
    }] }) });
  });
  await page.route("**/api/jobs/reorder", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ jobs: [] }) });
  });
  await page.route("**/api/recordings/recording-1", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ recording: {
      id: "recording-1", display_name: "访谈录音", original_filename: "interview.wav",
      run: { id: "run-1", status: "completed", phase: "complete" }, job: null,
    }, artifacts: [{ id: "artifact-final", type: "final", variant: "canonical", size_bytes: 1, sha256: "hash" }] }) });
  });
  await page.route("**/api/recordings/recording-1/download/final", async (route) => {
    await route.fulfill({ contentType: "text/plain; charset=utf-8", body: "[00:00] 说话人0：第一句\n" });
  });
  await page.route("**/api/recordings/recording-1/speaker-mapping", async (route) => {
    if (route.request().method() === "POST") {
      speakerEntries = (route.request().postDataJSON() as { entries: Record<string, string> }).entries;
      await route.fulfill({ contentType: "application/json", body: JSON.stringify({ version: 1 }) });
      return;
    }
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ entries: speakerEntries, speaker_prefix: "说话人" }) });
  });
  await page.route("**/api/recordings/recording-1/speaker-summary", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ items: [{
      anonymous_label: "SPEAKER_0", occurrence_count: 1, start_ms: 0, end_ms: 1000, excerpts: [],
    }] }) });
  });
  await page.route("**/api/recordings/recording-1/audit-summary", async (route) => {
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ speaker_review: null, final: null }) });
  });
});

test("生产页面支持搜索与单文件上传反馈", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "FunASR_e2e" })).toBeVisible();
  await expect(page.getByText("访谈录音")).toBeVisible();

  const search = page.getByPlaceholder("搜索文件名或显示名称");
  await search.fill("访谈");
  await expect.poll(() => new URL(page.url()).searchParams.get("query")).toBe("访谈");

  await page.locator('input[type="file"]').setInputFiles({
    name: "single.wav",
    mimeType: "audio/wav",
    buffer: Buffer.from("audio"),
  });
  await expect(page.getByText("已上传并自动进入队列。")).toBeVisible();
});

test("speaker 显示名会即时更新最终阅读版并在保存后保留", async ({ page }) => {
  await page.goto("/recordings/recording-1");

  await expect(page.getByText("[00:00] 说话人0：第一句")).toBeVisible();
  await expect(page.locator(".final-reading pre")).toHaveCSS("overflow-y", "auto");
  await expect(page.getByRole("link", { name: "下载 canonical final" })).toHaveCount(0);
  await expect(page.getByRole("link", { name: "下载显示名版本" }).locator("svg.download-icon")).toHaveCount(1);
  await expect(page.getByRole("link", { name: "下载当前运行全部 ZIP" }).locator("svg.download-icon")).toHaveCount(1);
  await expect(page.getByRole("link", { name: "final（经完整性校验的 canonical 最终稿）" })).toBeVisible();
  const headings = await page.locator("h2").allTextContents();
  expect(headings.indexOf("显示名称")).toBeLessThan(headings.indexOf("speaker 显示名映射"));
  expect(headings.indexOf("speaker 显示名映射")).toBeLessThan(headings.indexOf("最终阅读版"));

  await page.getByLabel("SPEAKER_0 显示名").fill("甲");
  await expect(page.getByText("[00:00] 甲：第一句")).toBeVisible();
  await page.getByRole("button", { name: "保存映射" }).click();
  await page.reload();
  await expect(page.getByText("[00:00] 甲：第一句")).toBeVisible();
});

test("可直达详情和 speaker 页面，旧导入地址回到录音列表", async ({ page }) => {
  await page.goto("/recordings/recording-1");
  await expect(page.getByRole("heading", { name: "访谈录音" })).toBeVisible();
  await page.getByRole("button", { name: "独立页面" }).click();
  await expect.poll(() => new URL(page.url()).pathname).toBe("/recordings/recording-1/speakers");
  await expect(page.getByRole("heading", { name: "speaker 显示名映射" })).toBeVisible();

  await page.goto("/imports");
  await expect.poll(() => new URL(page.url()).pathname).toBe("/recordings");
  await expect(page.getByRole("heading", { name: "FunASR_e2e" })).toBeVisible();
  await expect(page.getByRole("button", { name: "旧录音导入" })).toHaveCount(0);
});
