import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";

const frontendDir = dirname(fileURLToPath(import.meta.url));
const projectDir = resolve(frontendDir, "..");
const chromePaths = [
  process.env.CHROME_PATH,
  process.env.PROGRAMFILES && join(process.env.PROGRAMFILES, "Google", "Chrome", "Application", "chrome.exe"),
  process.env["PROGRAMFILES(X86)"] && join(process.env["PROGRAMFILES(X86)"], "Google", "Chrome", "Application", "chrome.exe"),
].filter((value): value is string => Boolean(value));
const chromeExecutablePath = chromePaths.find((value) => existsSync(value));

if (!chromeExecutablePath) throw new Error("未找到 Chrome；请设置 CHROME_PATH 或安装受支持的浏览器。");

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  use: {
    baseURL: "http://127.0.0.1:8002",
    browserName: "chromium",
    headless: true,
    launchOptions: { executablePath: chromeExecutablePath },
  },
  webServer: {
    command: "python scripts/launch_web.py --app-data app_data_e2e --port 8002",
    cwd: projectDir,
    url: "http://127.0.0.1:8002/api/health",
    timeout: 30_000,
    reuseExistingServer: false,
  },
});
