"""Supervisor-Workers orchestration for the Day 8 RAG pipeline.

Topology:
    START -> supervisor
          -> [legal_worker, news_worker, verification_worker] (parallel)
          -> synthesizer -> END

The workers use the existing standardized Markdown corpus. LLM synthesis is
optional; the default offline synthesizer keeps the assignment runnable without
network access or an API key.
"""

from __future__ import annotations

import operator
import os
import re
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

PROJECT_DIR = Path(__file__).resolve().parents[1]
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"

WorkerName = Literal["legal_worker", "news_worker", "verification_worker"]


class Evidence(TypedDict):
    content: str
    score: float
    source: str
    document_type: str


class WorkerResult(TypedDict):
    worker: WorkerName
    role: str
    summary: str
    evidence: list[Evidence]
    warnings: list[str]


class SupervisorState(TypedDict):
    question: str
    use_llm: bool
    plan: list[WorkerName]
    worker_results: Annotated[list[WorkerResult], operator.add]
    logs: Annotated[list[str], operator.add]
    final_answer: str


STOPWORDS = {
    "ai",
    "bị",
    "các",
    "cho",
    "có",
    "của",
    "đã",
    "đến",
    "được",
    "gì",
    "khi",
    "là",
    "một",
    "nào",
    "những",
    "theo",
    "thì",
    "trong",
    "từ",
    "và",
    "về",
    "với",
}


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text for lightweight offline retrieval."""
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def _document_type(path: Path) -> str:
    if "legal" in path.parts:
        return "legal"
    if "news" in path.parts:
        return "news"
    return "unknown"


def _load_corpus(document_type: str | None = None) -> list[dict]:
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        current_type = _document_type(path)
        if document_type and current_type != document_type:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if content:
            documents.append(
                {
                    "content": content,
                    "source": path.name,
                    "document_type": current_type,
                }
            )
    return documents


def _chunk_text(text: str, size: int = 900, overlap_words: int = 20) -> list[str]:
    """Split documents with word-aligned overlap and no embedding dependency."""
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > size:
            if current:
                chunks.append(current)
                current = ""

            words = paragraph.split()
            window: list[str] = []
            for word in words:
                candidate = " ".join([*window, word])
                if window and len(candidate) > size:
                    chunks.append(" ".join(window))
                    window = window[-overlap_words:]
                window.append(word)
            if window:
                current = " ".join(window)
            continue

        candidate = f"{current}\n\n{paragraph}".strip()
        if current and len(candidate) > size:
            chunks.append(current)
            overlap = " ".join(current.split()[-overlap_words:])
            current = f"{overlap}\n\n{paragraph}".strip()
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def local_search(
    query: str,
    *,
    document_type: str | None = None,
    top_k: int = 4,
) -> list[Evidence]:
    """Retrieve evidence using normalized token overlap."""
    query_terms = set(tokenize(query))
    candidates: list[Evidence] = []

    for document in _load_corpus(document_type):
        for chunk in _chunk_text(document["content"]):
            chunk_terms = set(tokenize(chunk))
            if not query_terms or not chunk_terms:
                continue
            matched = query_terms & chunk_terms
            score = len(matched) / len(query_terms)
            if score <= 0:
                continue
            candidates.append(
                {
                    "content": chunk,
                    "score": round(score, 4),
                    "source": document["source"],
                    "document_type": document["document_type"],
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:top_k]


def supervisor(state: SupervisorState) -> dict:
    """Create a work plan and delegate to three specialist workers."""
    plan: list[WorkerName] = [
        "legal_worker",
        "news_worker",
        "verification_worker",
    ]
    return {
        "plan": plan,
        "logs": [
            "Supervisor đã phân rã câu hỏi thành 3 nhiệm vụ độc lập.",
            "Các workers sẽ chạy song song bằng LangGraph Send API.",
        ],
    }


def dispatch_workers(state: SupervisorState) -> list[Send]:
    """Fan out work to every worker selected by the supervisor."""
    return [
        Send(
            worker_name,
            {
                "question": state["question"],
                "use_llm": state.get("use_llm", False),
                "plan": state["plan"],
                "worker_results": [],
                "logs": [],
                "final_answer": "",
            },
        )
        for worker_name in state["plan"]
    ]


def legal_worker(state: SupervisorState) -> dict:
    """Worker 1: retrieve statutes and legal provisions."""
    evidence = local_search(state["question"], document_type="legal", top_k=4)
    summary = (
        f"Tìm thấy {len(evidence)} đoạn văn bản pháp luật liên quan."
        if evidence
        else "Không tìm thấy văn bản pháp luật đủ liên quan."
    )
    result: WorkerResult = {
        "worker": "legal_worker",
        "role": "Tra cứu luật và điều khoản",
        "summary": summary,
        "evidence": evidence,
        "warnings": [] if evidence else ["Thiếu evidence từ corpus pháp luật."],
    }
    return {
        "worker_results": [result],
        "logs": [f"Legal Worker hoàn tất: {summary}"],
    }


def news_worker(state: SupervisorState) -> dict:
    """Worker 2: retrieve news reports and event context."""
    evidence = local_search(state["question"], document_type="news", top_k=4)
    summary = (
        f"Tìm thấy {len(evidence)} đoạn tin tức liên quan."
        if evidence
        else "Không tìm thấy tin tức đủ liên quan."
    )
    result: WorkerResult = {
        "worker": "news_worker",
        "role": "Tra cứu tin tức và sự kiện",
        "summary": summary,
        "evidence": evidence,
        "warnings": [] if evidence else ["Không có news evidence phù hợp."],
    }
    return {
        "worker_results": [result],
        "logs": [f"News Worker hoàn tất: {summary}"],
    }


def verification_worker(state: SupervisorState) -> dict:
    """Worker 3: independently verify source coverage and evidence diversity."""
    evidence = local_search(state["question"], top_k=6)
    source_types = {item["document_type"] for item in evidence}
    warnings = []
    if not evidence:
        warnings.append("Không có bằng chứng để kiểm chứng câu trả lời.")
    if evidence and "legal" not in source_types:
        warnings.append("Chưa có nguồn pháp luật trong nhóm bằng chứng.")
    if evidence and len({item["source"] for item in evidence}) < 2:
        warnings.append("Bằng chứng chỉ đến từ một tài liệu.")

    result: WorkerResult = {
        "worker": "verification_worker",
        "role": "Kiểm chứng nguồn và độ bao phủ",
        "summary": (
            f"Kiểm tra độc lập {len(evidence)} evidence từ "
            f"{len({item['source'] for item in evidence})} tài liệu."
        ),
        "evidence": evidence,
        "warnings": warnings,
    }
    return {
        "worker_results": [result],
        "logs": ["Verification Worker đã đánh giá độ bao phủ của nguồn."],
    }


def _deduplicate_evidence(results: list[WorkerResult], limit: int = 8) -> list[Evidence]:
    best_by_key: dict[tuple[str, str], Evidence] = {}
    for result in results:
        for item in result["evidence"]:
            key = (item["source"], item["content"][:160])
            if key not in best_by_key or item["score"] > best_by_key[key]["score"]:
                best_by_key[key] = item
    return sorted(
        best_by_key.values(),
        key=lambda item: item["score"],
        reverse=True,
    )[:limit]


def _offline_synthesis(question: str, results: list[WorkerResult]) -> str:
    evidence = _deduplicate_evidence(results)
    warnings = [
        warning
        for result in results
        for warning in result["warnings"]
    ]

    if not evidence:
        return (
            "Tôi không thể xác minh câu trả lời từ corpus hiện có. "
            "Cần bổ sung tài liệu hoặc điều chỉnh câu hỏi."
        )

    lines = [
        "## Kết quả phân tích",
        "",
        f"**Câu hỏi:** {question}",
        "",
        "### Bằng chứng được các workers tìm thấy",
    ]
    for index, item in enumerate(evidence[:5], 1):
        excerpt = " ".join(item["content"].split())[:420]
        lines.append(
            f"{index}. {excerpt} "
            f"[Nguồn: {item['source']}; score={item['score']:.2f}]"
        )

    lines.extend(
        [
            "",
            "### Kết luận",
            (
                "Các nguồn trên là evidence được truy xuất tự động. "
                "Cần đối chiếu toàn văn văn bản pháp luật trước khi sử dụng "
                "cho tư vấn hoặc quyết định thực tế."
            ),
        ]
    )
    if warnings:
        lines.extend(["", "### Cảnh báo kiểm chứng"])
        lines.extend(f"- {warning}" for warning in dict.fromkeys(warnings))
    return "\n".join(lines)


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "1200")),
    )


async def synthesizer(state: SupervisorState) -> dict:
    """Combine worker outputs into a Vietnamese answer with citations."""
    results = state.get("worker_results", [])
    evidence = _deduplicate_evidence(results)
    use_llm = (
        state.get("use_llm", False)
        or os.getenv("SUPERVISOR_USE_LLM", "false").lower() == "true"
    ) and bool(os.getenv("OPENROUTER_API_KEY"))

    if not use_llm:
        answer = _offline_synthesis(state["question"], results)
        return {
            "final_answer": answer,
            "logs": ["Synthesizer hoàn tất ở chế độ offline."],
        }

    context = "\n\n".join(
        (
            f"[Nguồn: {item['source']} | loại: {item['document_type']} "
            f"| score: {item['score']:.4f}]\n{item['content']}"
        )
        for item in evidence
    )
    warnings = "\n".join(
        f"- {warning}"
        for result in results
        for warning in result["warnings"]
    ) or "- Không có cảnh báo."

    response = await _llm().ainvoke(
        [
            SystemMessage(
                content=(
                    "Bạn là Supervisor tổng hợp kết quả từ nhiều RAG workers. "
                    "Luôn trả lời bằng tiếng Việt. Chỉ sử dụng evidence được cung cấp. "
                    "Mọi nhận định thực tế phải có citation dạng [Nguồn: tên_file]. "
                    "Nếu evidence không đủ, phải nói rõ không thể xác minh."
                )
            ),
            HumanMessage(
                content=(
                    f"Câu hỏi:\n{state['question']}\n\n"
                    f"Evidence:\n{context}\n\n"
                    f"Cảnh báo từ Verification Worker:\n{warnings}"
                )
            ),
        ]
    )
    return {
        "final_answer": response.content,
        "logs": ["Synthesizer hoàn tất bằng OpenRouter LLM."],
    }


def build_graph():
    """Build and compile the Supervisor-Workers LangGraph."""
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("legal_worker", legal_worker)
    graph.add_node("news_worker", news_worker)
    graph.add_node("verification_worker", verification_worker)
    graph.add_node("synthesizer", synthesizer)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        dispatch_workers,
        ["legal_worker", "news_worker", "verification_worker"],
    )
    graph.add_edge("legal_worker", "synthesizer")
    graph.add_edge("news_worker", "synthesizer")
    graph.add_edge("verification_worker", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()


async def run_supervisor(question: str, use_llm: bool = False) -> dict:
    """Run the complete Supervisor-Workers workflow."""
    graph = build_graph()
    return await graph.ainvoke(
        {
            "question": question,
            "use_llm": use_llm,
            "plan": [],
            "worker_results": [],
            "logs": [],
            "final_answer": "",
        }
    )
