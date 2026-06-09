"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PageIndex API key not set. Skipping upload.")
        print("  1. Sign up at: https://pageindex.ai/")
        print("  2. Get your API key")
        print("  3. Add to .env: PAGEINDEX_API_KEY=your_key_here")
        return

    try:
        from pageindex import PageIndex

        pi = PageIndex(api_key=PAGEINDEX_API_KEY)

        uploaded = 0
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            pi.upload(
                content=content,
                metadata={"filename": md_file.name, "type": md_file.parent.name}
            )
            print(f"  ✓ Uploaded: {md_file.name}")
            uploaded += 1

        print(f"\n✓ Uploaded {uploaded} documents to PageIndex")
    except ImportError:
        print("⚠ PageIndex SDK not installed. Run: pip install pageindex")
    except Exception as e:
        print(f"⚠ Error uploading to PageIndex: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PageIndex API key not set. Returning empty results.")
        return []

    try:
        from pageindex import PageIndex

        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        results = pi.query(query=query, top_k=top_k)

        return [
            {
                "content": r.text,
                "score": r.score,
                "metadata": r.metadata,
                "source": "pageindex"
            }
            for r in results
        ]
    except ImportError:
        print("⚠ PageIndex SDK not installed. Run: pip install pageindex")
        return []
    except Exception as e:
        print(f"⚠ Error querying PageIndex: {e}")
        return []


if __name__ == "__main__":
    print("=" * 80)
    print("Task 8: PageIndex Vectorless RAG")
    print("=" * 80)

    if not PAGEINDEX_API_KEY:
        print("\n⚠ PAGEINDEX_API_KEY not set in .env file")
        print("\nTo use PageIndex:")
        print("  1. Sign up at: https://pageindex.ai/")
        print("  2. Get your API key from dashboard")
        print("  3. Add to .env file: PAGEINDEX_API_KEY=your_key_here")
        print("  4. Run: python src/task8_pageindex_vectorless.py")
        print("\n" + "=" * 80)
        print("NOTE: PageIndex is optional. Task 9 will use hybrid search as primary method.")
        print("=" * 80)
    else:
        print("\nUploading documents to PageIndex...")
        upload_documents()

        print("\n" + "=" * 80)
        print("Testing PageIndex query...")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)

        if results:
            for i, r in enumerate(results, 1):
                print(f"\n[{i}] Score: {r['score']:.3f}")
                print(f"Content: {r['content'][:200]}...")
        else:
            print("No results returned.")

        print("\n" + "=" * 80)
        print("✓ Task 8 setup complete")
        print("=" * 80)
