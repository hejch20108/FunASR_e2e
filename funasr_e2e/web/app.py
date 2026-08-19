from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from funasr_e2e.persistence.db import Database
from funasr_e2e.persistence.files import ManagedFileStore
from funasr_e2e.persistence.repository import Repository
from funasr_e2e.worker.supervisor import WorkerSupervisor
from scripts.run_funasr_full_pipeline import load_settings

from .errors import ApiError, install_exception_handlers
from .routes import artifacts, jobs, recordings, speakers
from .schemas import HealthResponse
from .security import LocalOnlyMiddleware


@dataclass(frozen=True)
class AppServices:
    repository: Repository
    store: ManagedFileStore
    supervisor: WorkerSupervisor
    settings: dict
    project_dir: Path


def create_app(
    *,
    app_data_dir: Path,
    project_dir: Path,
    settings_path: Path,
    supervisor_factory: Callable[..., WorkerSupervisor] = WorkerSupervisor,
    start_worker: bool = True,
) -> FastAPI:
    resolved_app_data_dir = app_data_dir.resolve()
    resolved_project_dir = project_dir.resolve()
    resolved_settings_path = settings_path.resolve()
    settings = load_settings(resolved_settings_path)
    repository = Repository(Database(resolved_app_data_dir / "app.sqlite3"))
    store = ManagedFileStore(resolved_app_data_dir, repository)
    supervisor = supervisor_factory(
        app_data_dir=resolved_app_data_dir,
        project_dir=resolved_project_dir,
        settings_path=resolved_settings_path,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        repository.initialize()
        store.initialize()
        store.recover_attempts()
        if start_worker:
            supervisor.start()
        try:
            yield
        finally:
            if start_worker:
                supervisor.stop()

    app = FastAPI(title="FunASR_e2e 本机服务", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.services = AppServices(repository=repository, store=store, supervisor=supervisor, settings=settings, project_dir=resolved_project_dir)
    app.add_middleware(LocalOnlyMiddleware)
    install_exception_handlers(app)
    app.include_router(recordings.router)
    app.include_router(jobs.router)
    app.include_router(artifacts.router)
    app.include_router(speakers.router)

    @app.get("/api/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        services: AppServices = request.app.state.services
        return HealthResponse(worker_running=services.supervisor.refresh().running)

    frontend_dist = resolved_project_dir / "frontend" / "dist"
    frontend_index = frontend_dist / "index.html"
    if frontend_index.is_file():
        assets_dir = frontend_dist / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str = "") -> FileResponse:
            if path.startswith("api/"):
                raise ApiError(status_code=404, code="API_NOT_FOUND", message="接口不存在")
            return FileResponse(frontend_index)

    return app


def services_for(request: Request) -> AppServices:
    return request.app.state.services
