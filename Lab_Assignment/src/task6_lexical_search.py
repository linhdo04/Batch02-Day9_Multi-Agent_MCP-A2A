"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import chromadb
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

# Global variables for BM25 index
_bm25 = None
_corpus = None


def _load_corpus():
    """Load corpus từ ChromaDB."""
    db_path = Path(__file__).parent.parent / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection("drug_law_docs")

    # Get all documents
    results = collection.get()

    corpus = []
    for i in range(len(results['ids'])):
        corpus.append({
            "content": results['documents'][i],
            "metadata": results['metadatas'][i]
        })
    return corpus


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    # Tokenize - đơn giản với split() cho tiếng Việt
    # Note: có thể dùng underthesea để tokenize tốt hơn
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global _bm25, _corpus

    # Lazy load corpus and build index
    if _bm25 is None:
        print("Building BM25 index...")
        _corpus = _load_corpus()
        _bm25 = build_bm25_index(_corpus)
        print(f"✓ Indexed {len(_corpus)} documents")

    # Tokenize query
    tokenized_query = query.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    # Get top_k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": _corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": _corpus[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    # Test
    print("Testing lexical search (BM25)...")
    print("Query: 'Điều 248 tàng trữ trái phép chất ma tuý'\n")

    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] BM25 Score: {r['score']:.3f}")
        print(f"Source: {r['metadata']['source']} ({r['metadata']['type']})")
        print(f"Content: {r['content'][:200]}...")
        print("-" * 80)
