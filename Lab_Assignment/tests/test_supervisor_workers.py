"""Offline tests for the Supervisor-Workers assignment."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.supervisor_workers import build_graph, local_search, run_supervisor


class TestSupervisorWorkers(unittest.TestCase):
    def test_graph_contains_supervisor_and_three_workers(self):
        nodes = set(build_graph().get_graph().nodes)
        self.assertIn("supervisor", nodes)
        self.assertIn("legal_worker", nodes)
        self.assertIn("news_worker", nodes)
        self.assertIn("verification_worker", nodes)
        self.assertIn("synthesizer", nodes)

    def test_local_search_uses_existing_corpus(self):
        results = local_search("ma túy", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIn("source", results[0])
        self.assertIn("score", results[0])

    def test_end_to_end_offline(self):
        result = asyncio.run(
            run_supervisor(
                "Hình phạt đối với hành vi tàng trữ trái phép chất ma túy?"
            )
        )
        self.assertEqual(len(result["plan"]), 3)
        self.assertEqual(len(result["worker_results"]), 3)
        self.assertTrue(result["final_answer"])
        worker_names = {
            worker_result["worker"]
            for worker_result in result["worker_results"]
        }
        self.assertEqual(
            worker_names,
            {"legal_worker", "news_worker", "verification_worker"},
        )


if __name__ == "__main__":
    unittest.main()
