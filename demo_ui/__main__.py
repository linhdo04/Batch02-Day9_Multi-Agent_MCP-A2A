"""FastAPI web UI for demonstrating Stage 5 A2A agent interactions."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from time import perf_counter
from typing import AsyncIterator
from uuid import uuid4

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from common.a2a_client import delegate

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:10000")
CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")

app = FastAPI(title="Legal Multi-Agent Demo", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class DemoRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)


def _event(event_type: str, **payload: object) -> bytes:
    return (
        json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"
    ).encode("utf-8")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "registry_url": REGISTRY_URL,
        "customer_agent_url": CUSTOMER_AGENT_URL,
    }


async def _registered_agents() -> list[dict]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{REGISTRY_URL}/agents")
        response.raise_for_status()
        return response.json().get("agents", [])


async def _inspect_agent(endpoint: str) -> dict:
    started_at = perf_counter()
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            response = await client.get(f"{endpoint}/.well-known/agent.json")
            card = response.json() if response.is_success else {}
            return {
                "online": response.is_success,
                "latency_ms": round((perf_counter() - started_at) * 1000),
                "http_status": response.status_code,
                "card_name": card.get("name", ""),
                "version": card.get("version", ""),
            }
    except httpx.HTTPError as exc:
        return {
            "online": False,
            "latency_ms": round((perf_counter() - started_at) * 1000),
            "http_status": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _routes(question: str) -> dict[str, bool]:
    question_lower = question.lower()
    return {
        "tax": any(
            keyword in question_lower
            for keyword in ["tax", "irs", "thuế", "fbar", "fatca"]
        ),
        "compliance": any(
            keyword in question_lower
            for keyword in [
                "compliance",
                "sec",
                "regulation",
                "sox",
                "aml",
                "fcpa",
                "gdpr",
                "data",
                "privacy",
                "dữ liệu",
            ]
        ),
    }


@app.post("/api/run")
async def run_demo(request: DemoRequest) -> StreamingResponse:
    async def stream() -> AsyncIterator[bytes]:
        started_at = perf_counter()
        trace_id = str(uuid4())
        context_id = str(uuid4())
        question_preview = request.question.replace("\n", " ")[:180]
        yield _event(
            "run_started",
            trace_id=trace_id,
            context_id=context_id,
            delegation_depth=0,
            question_length=len(request.question),
        )
        yield _event(
            "log",
            level="info",
            phase="request",
            message="Đã tạo A2A request context",
            details={
                "trace_id": trace_id,
                "context_id": context_id,
                "delegation_depth": 0,
                "question_preview": question_preview,
            },
        )
        yield _event("agent_status", agent="registry", status="running")
        yield _event(
            "log",
            level="info",
            phase="registry",
            message="Đang truy vấn danh sách agent đã đăng ký",
            details={"method": "GET", "url": f"{REGISTRY_URL}/agents"},
        )

        try:
            registry_started_at = perf_counter()
            try:
                agents = await _registered_agents()
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    "Registry chưa chạy. Hãy khởi động Stage 5 bằng ./start_all.sh."
                ) from exc
            registry_latency_ms = round(
                (perf_counter() - registry_started_at) * 1000
            )
            yield _event(
                "log",
                level="success",
                phase="registry",
                message=f"Registry trả về {len(agents)} agent",
                details={
                    "latency_ms": registry_latency_ms,
                    "registered_agents": [
                        agent.get("agent_name", "unknown") for agent in agents
                    ],
                },
            )
            endpoint_by_name = {
                agent.get("agent_name", ""): agent.get("endpoint", "")
                for agent in agents
            }
            endpoint_by_name.setdefault("customer-agent", CUSTOMER_AGENT_URL)

            checks = await asyncio.gather(
                *[
                    _inspect_agent(endpoint)
                    for endpoint in endpoint_by_name.values()
                    if endpoint
                ]
            )
            names = [name for name, endpoint in endpoint_by_name.items() if endpoint]
            inspections = dict(zip(names, checks, strict=False))
            online = {
                name: inspection.get("online", False)
                for name, inspection in inspections.items()
            }

            yield _event(
                "registry",
                agents=[
                    {
                        "name": name,
                        "endpoint": endpoint_by_name[name],
                        "online": online.get(name, False),
                        **inspections.get(name, {}),
                    }
                    for name in names
                ],
            )
            for name in names:
                inspection = inspections.get(name, {})
                yield _event(
                    "log",
                    level="success" if inspection.get("online") else "error",
                    phase="health_check",
                    message=(
                        f"{name} online"
                        if inspection.get("online")
                        else f"{name} không phản hồi"
                    ),
                    details={
                        "endpoint": endpoint_by_name[name],
                        "agent_card": inspection.get("card_name", ""),
                        "version": inspection.get("version", ""),
                        "http_status": inspection.get("http_status"),
                        "latency_ms": inspection.get("latency_ms"),
                        "error": inspection.get("error"),
                    },
                )
            yield _event("agent_status", agent="registry", status="completed")

            if not online.get("customer-agent", False):
                raise RuntimeError(
                    "Customer Agent chưa chạy. Hãy khởi động Stage 5 bằng ./start_all.sh."
                )

            routes = _routes(request.question)
            selected_agents = [
                agent for agent, enabled in routes.items() if enabled
            ]
            yield _event(
                "log",
                level="info",
                phase="routing",
                message="Đã phân tích từ khóa để dự đoán nhánh specialist",
                details={
                    "tax": routes["tax"],
                    "compliance": routes["compliance"],
                    "selected_agents": selected_agents or ["none"],
                    "note": (
                        "Routing thực tế cuối cùng do Law Agent quyết định bằng LLM."
                    ),
                },
            )
            yield _event("agent_status", agent="customer", status="running")
            yield _event("agent_status", agent="law", status="running")
            yield _event(
                "agent_status",
                agent="tax",
                status=(
                    "running"
                    if routes["tax"] and online.get("tax-agent", False)
                    else "available"
                ),
            )
            yield _event(
                "agent_status",
                agent="compliance",
                status=(
                    "running"
                    if routes["compliance"] and online.get("compliance-agent", False)
                    else "available"
                ),
            )
            yield _event("routing", **routes)
            yield _event(
                "log",
                level="info",
                phase="a2a",
                message="Đang gửi message tới Customer Agent",
                details={
                    "endpoint": CUSTOMER_AGENT_URL,
                    "agent_card_url": (
                        f"{CUSTOMER_AGENT_URL}/.well-known/agent.json"
                    ),
                    "trace_id": trace_id,
                    "context_id": context_id,
                    "delegation_depth": 0,
                    "payload_chars": len(request.question),
                },
            )

            delegation_started_at = perf_counter()
            answer = await delegate(
                endpoint=CUSTOMER_AGENT_URL,
                question=request.question,
                context_id=context_id,
                trace_id=trace_id,
                depth=0,
            )
            delegation_latency = round(
                perf_counter() - delegation_started_at, 2
            )
            yield _event(
                "log",
                level="success",
                phase="a2a",
                message="Đã nhận A2A response từ Customer Agent",
                details={
                    "delegation_latency_s": delegation_latency,
                    "response_chars": len(answer),
                    "response_empty": not bool(answer),
                },
            )

            for agent in ("customer", "law"):
                yield _event("agent_status", agent=agent, status="completed")
            for agent, enabled in routes.items():
                if enabled:
                    yield _event("agent_status", agent=agent, status="completed")

            yield _event("result", answer=answer)
            total_latency = round(perf_counter() - started_at, 2)
            yield _event(
                "log",
                level="success",
                phase="summary",
                message="Hoàn tất toàn bộ request",
                details={
                    "total_latency_s": total_latency,
                    "registry_latency_ms": registry_latency_ms,
                    "a2a_delegation_latency_s": delegation_latency,
                    "response_chars": len(answer),
                },
            )
            yield _event(
                "done",
                latency=total_latency,
            )
        except Exception as exc:
            yield _event(
                "error",
                message=str(exc),
                error_type=type(exc).__name__,
                elapsed=round(perf_counter() - started_at, 2),
                trace_id=trace_id,
                context_id=context_id,
            )

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
