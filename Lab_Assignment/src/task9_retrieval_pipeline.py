"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF)
    3. Rerank (optional)
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from task5_semantic_search import semantic_search
from task6_lexical_search import lexical_search
from task7_reranking import rerank_rrf
from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.02   # RRF scores thường nhỏ, điều chỉnh threshold
DEFAULT_TOP_K = 5
USE_RERANKING = True  # Đã dùng RRF để merge rồi


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = USE_RERANKING,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: RRF được sử dụng mặc định

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    print(f"\n🔍 Query: {query}")

    # Step 1: Chạy semantic + lexical search
    print("  → Running semantic search...")
    dense_results = semantic_search(query, top_k=top_k * 2)

    print("  → Running lexical search...")
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: Merge bằng RRF
    print("  → Merging with RRF...")
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k)

    # Tag source as hybrid
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Check threshold → fallback
    if not merged or merged[0]["score"] < score_threshold:
        best_score = merged[0]["score"] if merged else 0
        print(f"  ⚠ Best hybrid score ({best_score:.4f}) < threshold ({score_threshold})")
        print(f"  → Fallback to PageIndex...")

        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            print(f"  ✓ PageIndex returned {len(fallback)} results")
            return fallback
        else:
            print(f"  ⚠ PageIndex unavailable, returning hybrid results anyway")
            return merged[:top_k]

    print(f"  ✓ Hybrid search success (best score: {merged[0]['score']:.4f})")
    return merged[:top_k]


if __name__ == "__main__":
    print("=" * 80)
    print("Task 9: Retrieval Pipeline - Hybrid Search with Fallback")
    print("=" * 80)

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý",
        "Luật phòng chống ma tuý quy định gì về cai nghiện",
    ]

    for idx, q in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test Query {idx}/3")
        results = retrieve(q, top_k=3)

        print(f"\n📋 Top 3 Results:")
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] Score: {r['score']:.4f} | Source: {r['source']}")
            print(f"      Metadata: {r['metadata']['source']} ({r['metadata']['type']})")
            print(f"      Content: {r['content'][:150]}...")

    print(f"\n{'='*80}")
    print("✓ Task 9 Complete - Retrieval Pipeline Working")
    print("=" * 80)
