"""FastAPI UI with live LangGraph event streaming."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import AsyncIterator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ASSIGNMENT_DIR.parent
if str(ASSIGNMENT_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT_DIR))

from src.supervisor_workers import build_graph

load_dotenv(ASSIGNMENT_DIR / ".env")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Supervisor Workers RAG UI", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RunRequest(BaseModel):
    question: str = Field(min_length=3, max_length=3000)
    use_llm: bool = False


def event(event_type: str, **payload: object) -> bytes:
    return (
        json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"
    ).encode("utf-8")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "assignment_dir": str(ASSIGNMENT_DIR)}


@app.post("/api/run")
async def run(request: RunRequest) -> StreamingResponse:
    async def stream() -> AsyncIterator[bytes]:
        started_at = perf_counter()
        graph = build_graph()
        effective_use_llm = request.use_llm and bool(
            os.getenv("OPENROUTER_API_KEY")
        )
        yield event(
            "run_started",
            question=request.question,
            mode="openrouter" if effective_use_llm else "offline",
            requested_llm=request.use_llm,
        )
        if request.use_llm and not effective_use_llm:
            yield event(
                "configuration_warning",
                message=(
                    "Chưa có OPENROUTER_API_KEY; hệ thống tự chuyển sang "
                    "offline synthesis."
                ),
            )
        yield event("node_status", node="supervisor", status="running")

        try:
            async for update in graph.astream(
                {
                    "question": request.question,
                    "use_llm": effective_use_llm,
                    "plan": [],
                    "worker_results": [],
                    "logs": [],
                    "final_answer": "",
                },
                stream_mode="updates",
            ):
                for node, values in update.items():
                    if node == "supervisor":
                        yield event(
                            "supervisor_plan",
                            plan=values.get("plan", []),
                            logs=values.get("logs", []),
                        )
                        yield event(
                            "node_status",
                            node="supervisor",
                            status="completed",
                        )
                        for worker in values.get("plan", []):
                            yield event(
                                "node_status",
                                node=worker,
                                status="running",
                            )
                    elif node in {
                        "legal_worker",
                        "news_worker",
                        "verification_worker",
                    }:
                        results = values.get("worker_results", [])
                        result = results[0] if results else {}
                        yield event(
                            "worker_result",
                            node=node,
                            result=result,
                            logs=values.get("logs", []),
                        )
                        yield event(
                            "node_status",
                            node=node,
                            status="completed",
                        )
                        yield event(
                            "node_status",
                            node="synthesizer",
                            status="running",
                        )
                    elif node == "synthesizer":
                        yield event(
                            "node_status",
                            node="synthesizer",
                            status="completed",
                        )
                        yield event(
                            "final_answer",
                            answer=values.get("final_answer", ""),
                            logs=values.get("logs", []),
                        )

            yield event(
                "done",
                latency=round(perf_counter() - started_at, 3),
            )
        except Exception as exc:
            yield event(
                "error",
                message=str(exc),
                error_type=type(exc).__name__,
                latency=round(perf_counter() - started_at, 3),
            )

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")


if __name__ == "__main__":
    main()
