from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from funasr_e2e.web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 FunASR_e2e 本机 Web 服务")
    parser.add_argument("--app-data", default="app_data")
    parser.add_argument("--settings", default="settings.yaml")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    app = create_app(
        app_data_dir=(project_dir / args.app_data).resolve(),
        project_dir=project_dir,
        settings_path=(project_dir / args.settings).resolve(),
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
