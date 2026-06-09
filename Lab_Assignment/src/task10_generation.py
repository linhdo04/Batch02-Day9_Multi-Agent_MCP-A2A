"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)
    """
    if len(chunks) <= 2:
        return chunks

    reordered = []
    # Lấy các vị trí lẻ trước (quan trọng → đầu)
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])
    # Lấy các vị trí chẵn sau, đảo ngược (quan trọng → cuối)
    for i in range(len(chunks) - 1 - (len(chunks) % 2 == 0), 0, -2):
        reordered.append(chunks[i])

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context string cho prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources
    """
    print(f"\n🔍 Retrieving context for: {query[:100]}...")

    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không tìm thấy thông tin liên quan từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none"
        }

    # Step 2: Reorder
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""

    # Step 5: Call LLM (nếu có API key)
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("  ⚠ OPENAI_API_KEY not set. Returning mock response.")
        answer = f"""[MOCK RESPONSE - Set OPENAI_API_KEY in .env to get real answer]

Dựa trên {len(chunks)} nguồn được tìm thấy:
- {chunks[0]['metadata']['source']} ({chunks[0]['metadata']['type']})

Để có câu trả lời thực tế với citation, vui lòng:
1. Thêm OPENAI_API_KEY vào file .env
2. Chạy lại script này"""
    else:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)

            print("  → Generating answer with citations...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

            answer = response.choices[0].message.content
            print("  ✓ Generated answer with citations")
        except Exception as e:
            answer = f"Error calling OpenAI API: {e}"

    # Step 6: Return
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }


if __name__ == "__main__":
    print("=" * 80)
    print("Task 10: RAG Generation with Citation")
    print("=" * 80)

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
    ]

    for idx, q in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test Query {idx}")
        print("=" * 80)
        print(f"Q: {q}\n")

        result = generate_with_citation(q)

        print(f"\n{'='*80}")
        print("📝 ANSWER WITH CITATIONS:")
        print("=" * 80)
        print(result['answer'])

        print(f"\n{'='*80}")
        print(f"📚 SOURCES USED: {len(result['sources'])} chunks (via {result['retrieval_source']})")
        print("=" * 80)
        for i, src in enumerate(result['sources'][:3], 1):
            print(f"  {i}. {src['metadata']['source']} ({src['metadata']['type']}) - Score: {src['score']:.4f}")

    print(f"\n{'='*80}")
    print("✓ Task 10 Complete - RAG Pipeline End-to-End")
    print("=" * 80)
