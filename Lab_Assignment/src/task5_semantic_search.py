"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer
    from pathlib import Path

    # Load embedding model (same as Task 4)
    model = SentenceTransformer("BAAI/bge-m3")

    # Embed query
    query_embedding = model.encode(query).tolist()

    # Connect to ChromaDB
    db_path = Path(__file__).parent.parent / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_collection("drug_law_docs")

    # Query with cosine similarity
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Format results
    output = []
    for i in range(len(results['ids'][0])):
        output.append({
            "content": results['documents'][0][i],
            "score": 1 - results['distances'][0][i],  # ChromaDB returns L2 distance, convert to similarity
            "metadata": results['metadatas'][0][i]
        })

    return output


if __name__ == "__main__":
    # Test
    print("Testing semantic search...")
    print("Query: 'hình phạt cho tội tàng trữ ma tuý'\n")

    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] Score: {r['score']:.3f}")
        print(f"Source: {r['metadata']['source']} ({r['metadata']['type']})")
        print(f"Content: {r['content'][:200]}...")
        print("-" * 80)
