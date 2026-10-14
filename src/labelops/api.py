"""Local REST interface for LabelOps. Run one worker against a trusted workspace."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, StrictInt

from labelops.service import Workbench


class AnnotationRequest(BaseModel):
    sample_id: StrictInt = Field(ge=0)
    reviewer: str = Field(min_length=1, max_length=80)
    label: StrictInt = Field(ge=0, le=9)
    note: str = Field(default="", max_length=2000)


def create_app(workspace: str | Path | None = None) -> FastAPI:
    """Create a service with an isolated, durable local annotation workspace."""
    target = Path(workspace or os.environ.get("LABELOPS_WORKSPACE", ".labelops"))
    lock = RLock()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.workbench = Workbench(target)
        yield

    application = FastAPI(
        title="LabelOps",
        version="0.1.0",
        description="Local annotation, active learning, and model quality workbench.",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @application.get("/summary")
    def summary():
        with lock:
            return application.state.workbench.summary()

    @application.get("/queue")
    def queue(limit: int = Query(default=12, ge=1, le=50)):
        with lock:
            return {"items": application.state.workbench.queue(limit=limit)}

    @application.post("/annotations", status_code=201)
    def annotate(request: AnnotationRequest):
        with lock:
            try:
                application.state.workbench.annotate(**request.model_dump())
            except ValueError as error:
                raise HTTPException(status_code=422, detail=str(error)) from error
        return {"status": "recorded", "sample_id": request.sample_id}

    @application.get("/annotations")
    def annotations():
        with lock:
            return {"items": application.state.workbench.store.annotations()}

    @application.get("/annotations/export")
    def export():
        with lock:
            content = application.state.workbench.store.export_csv()
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="annotations.csv"'},
        )

    @application.post("/retrain")
    def retrain():
        with lock:
            return application.state.workbench.retrain()

    @application.post("/evaluate")
    def evaluate():
        with lock:
            return application.state.workbench.evaluation()

    return application


app = create_app()
