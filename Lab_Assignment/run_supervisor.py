"""CLI demo for the Day 9 Supervisor-Workers assignment."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from src.supervisor_workers import run_supervisor

DEFAULT_QUESTION = (
    "Pháp luật Việt Nam quy định hình phạt thế nào đối với hành vi "
    "tàng trữ trái phép chất ma túy và báo chí đã phản ánh vấn đề này ra sao?"
)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Supervisor-Workers RAG workflow."
    )
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    result = await run_supervisor(args.question)

    print("=" * 80)
    print("SUPERVISOR PLAN")
    print("=" * 80)
    for worker in result["plan"]:
        print(f"- {worker}")

    print("\n" + "=" * 80)
    print("EXECUTION LOG")
    print("=" * 80)
    for log in result["logs"]:
        print(f"- {log}")

    print("\n" + "=" * 80)
    print("WORKER RESULTS")
    print("=" * 80)
    for worker_result in result["worker_results"]:
        print(
            f"- {worker_result['worker']}: {worker_result['summary']} "
            f"(evidence={len(worker_result['evidence'])})"
        )

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(result["final_answer"])


if __name__ == "__main__":
    load_dotenv(Path(__file__).with_name(".env"))
    asyncio.run(main())
