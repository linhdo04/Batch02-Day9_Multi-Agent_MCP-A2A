# Day 9 Assignment - Supervisor Workers over Day 8 RAG

## Cải tiến đã thực hiện

- Giữ nguyên corpus và RAG pipeline Day 8.
- Thêm LangGraph Supervisor điều phối công việc.
- Thêm 3 workers chạy song song:
  - Legal Research Worker.
  - News Research Worker.
  - Verification Worker.
- Thêm Synthesizer tạo câu trả lời tiếng Việt có citation.
- Dùng `Send` API cho fan-out và reducer `operator.add` cho fan-in.
- Có chế độ offline để demo không cần API key.
- Có tùy chọn OpenRouter cho bước tổng hợp cuối.
- Có 3 automated tests cho graph, retrieval và end-to-end workflow.

### Kết quả test

```text
Ran 3 tests
OK
```

### Lệnh demo

```bash
.venv/bin/python Lab_Assignment/run_supervisor.py
```

---

# Day 8 - RAG Pipeline Implementation Summary

## ✅ All Tasks Completed (Task 1-10)

### Task 2: Crawl News Articles ✅
- **Completed**: Crawled 5 news articles about Vietnamese artists and drugs
- **Tool**: Crawl4AI
- **Output**: `data/landing/news/` (5 JSON files)
- **Commit**: `0a8983d`

### Task 3: Convert to Markdown ✅
- **Completed**: Converted all documents to Markdown using MarkItDown
- **Files**: 5 legal docs + 5 news articles → 10 markdown files
- **Output**: `data/standardized/`
- **Commit**: `66ec45b`

### Task 4: Chunking & Indexing ✅
- **Completed**: 3,406 chunks indexed to ChromaDB
- **Chunking**: RecursiveCharacterTextSplitter (size=500, overlap=50)
- **Embedding**: BAAI/bge-m3 (1024 dim, multilingual, excellent for Vietnamese)
- **Vector Store**: ChromaDB (local, persistent)
- **Commit**: `f966d3a`

### Task 5: Semantic Search ✅
- **Completed**: Dense retrieval using BGE-M3 embeddings
- **Method**: Cosine similarity search on ChromaDB
- **Test**: Query "hình phạt cho tội tàng trữ ma tuý" → Found relevant legal articles
- **Commit**: `d65dac2`

### Task 6: Lexical Search ✅
- **Completed**: BM25 keyword search
- **Implementation**: rank-bm25 library with 3,406 documents
- **Test**: Query "Điều 248 tàng trữ trái phép chất ma tuý" → Found exact keyword matches
- **Commit**: `1ea685e`

### Task 7: Reranking ✅
- **Completed**: RRF (Reciprocal Rank Fusion) reranking
- **Method**: Fuses semantic + lexical search results
- **Formula**: RRF(d) = Σ 1/(k+rank) with k=60
- **Advantage**: Simple, no training needed, balances both search methods
- **Commit**: `4cd8de5`

### Task 8: PageIndex Vectorless RAG ✅
- **Completed**: Implementation ready (optional fallback)
- **Status**: API key not set (graceful handling)
- **Usage**: Add `PAGEINDEX_API_KEY` to `.env` if needed
- **Commit**: `1927def`

### Task 9: Full Retrieval Pipeline ✅
- **Completed**: Hybrid search with fallback logic
- **Pipeline**: Semantic + Lexical → RRF Merge → Fallback (PageIndex if needed)
- **Threshold**: 0.02 (RRF scores typically small)
- **Commit**: `7157dbf`

### Task 10: Generation with Citation ✅
- **Completed**: RAG generation with citations
- **Features**:
  - Document reordering (lost-in-the-middle prevention)
  - Context formatting with source labels
  - System prompt enforcing citations
- **Config**:
  - TOP_K=5 (sufficient evidence)
  - Temperature=0.3 (factual, not creative)
  - TOP_P=0.9 (balanced diversity)
- **LLM**: OpenAI GPT-4o-mini (set `OPENAI_API_KEY` in `.env`)
- **Commit**: `387f8a0`

---

## 📊 System Architecture

```
Query
  │
  ├→ Semantic Search (BGE-M3 embeddings) ────┐
  │                                           ├→ RRF Fusion
  ├→ Lexical Search (BM25)  ─────────────────┘
  │
  ├→ Reranked Results
  │
  ├→ Check Threshold
  │   └→ If score < 0.02 → PageIndex Fallback (optional)
  │
  ├→ Reorder Documents (lost-in-the-middle prevention)
  │
  └→ LLM Generation with Citations
```

---

## 🎯 Key Technologies

| Component | Technology | Reason |
|-----------|-----------|---------|
| **Crawling** | Crawl4AI | Async, markdown output |
| **Conversion** | MarkItDown (Microsoft) | Unified markdown format |
| **Chunking** | RecursiveCharacterTextSplitter | Good for structured legal text |
| **Embedding** | BAAI/bge-m3 | State-of-the-art multilingual |
| **Vector DB** | ChromaDB | Simple, local, persistent |
| **Lexical** | BM25Okapi | Classic keyword search |
| **Fusion** | RRF | Simple, no training needed |
| **LLM** | GPT-4o-mini | Cost-effective, citation support |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
playwright install  # For Crawl4AI
```

### 2. Setup Environment
```bash
cp .env.example .env
# Add API keys (optional):
# OPENAI_API_KEY=sk-...
# PAGEINDEX_API_KEY=...
```

### 3. Run Individual Tasks
```bash
# Task 4-10 are ready to run
python src/task5_semantic_search.py
python src/task6_lexical_search.py
python src/task7_reranking.py
python src/task9_retrieval_pipeline.py
python src/task10_generation.py
```

### 4. Query the System
```python
from src.task10_generation import generate_with_citation

result = generate_with_citation(
    "Hình phạt cho tội tàng trữ trái phép chất ma tuý?"
)
print(result['answer'])  # Answer with citations
```

---

## 📈 Performance

- **Corpus**: 10 documents → 3,406 chunks
- **Indexing Time**: ~5 minutes (one-time)
- **Query Time**:
  - Semantic search: ~1-2 seconds
  - Lexical search: ~0.5 seconds
  - RRF fusion: <0.1 seconds
  - Generation (w/ LLM): ~3-5 seconds

---

## 🔧 Configuration

All configurable parameters are documented in each task file:

- `task4_chunking_indexing.py`: CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
- `task9_retrieval_pipeline.py`: SCORE_THRESHOLD, DEFAULT_TOP_K
- `task10_generation.py`: TOP_K, TEMPERATURE, TOP_P

---

## ✨ Features Implemented

✅ **Dual Search**: Semantic (dense) + Lexical (sparse)
✅ **Hybrid Fusion**: RRF algorithm
✅ **Smart Fallback**: PageIndex if hybrid fails
✅ **Lost-in-Middle Prevention**: Document reordering
✅ **Citation Enforcement**: System prompt + source labels
✅ **Multilingual**: Vietnamese text support (BGE-M3)
✅ **Production Ready**: Error handling, logging, graceful degradation

---

## 📝 Next Steps (Group Project)

Choose one:
1. **RAG Chatbot**: Streamlit/Chainlit interface with conversation memory
2. **RAG Evaluation**: DeepEval/RAGAS/TruLens with golden dataset

---

## 📚 References

- Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts*
- Cormack et al. (2009), *Reciprocal Rank Fusion*
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- Crawl4AI: https://github.com/unclecode/crawl4ai

---

**Status**: All 10 individual tasks completed ✅
**Ready for**: Group project phase 🚀
