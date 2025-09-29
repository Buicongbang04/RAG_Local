import os, json, uuid, hashlib, time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss
from pypdf import PdfReader
from dotenv import load_dotenv
import chainlit as cl
from google import genai

# ========= Load .env & config =========
load_dotenv()
API_KEY  = os.getenv("GEMINI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH", "").strip() or None

EMBED_MODEL    = os.getenv("EMBED_MODEL", "gemini-embedding-001")  # dùng khi không bật local
GEN_MODEL      = os.getenv("GEN_MODEL", "gemini-2.5-flash")
TOP_K          = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBED_BATCH    = int(os.getenv("EMBED_BATCH", "100"))               # batch cho Gemini (<=100)
USE_LOCAL      = os.getenv("USE_LOCAL_EMBEDDING", "false").lower() == "true"
LOCAL_MODEL    = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LOCAL_BATCH    = int(os.getenv("LOCAL_EMBED_BATCH", "64"))

INDEX_DIR = Path("index_store"); INDEX_DIR.mkdir(exist_ok=True)

if not API_KEY:
    raise RuntimeError("Thiếu GEMINI_API_KEY trong .env (dùng cho phần sinh trả lời)")

client = genai.Client(api_key=API_KEY)

# ========= PDF utils =========
def read_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(pdf_path)
    out = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if text:
            out.append((i, text))
    return out

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + chunk_size)
        chunk = text[i:end]
        if end < n:
            last_dot = chunk.rfind(".")
            if last_dot > int(chunk_size * 0.6):
                chunk = chunk[:last_dot+1]
                end = i + last_dot + 1
        if chunk.strip():
            chunks.append(chunk)
        i = max(end - overlap, end)
    return chunks

# ========= Embedding backends =========
class EmbeddingBackend:
    """Gói 2 cách embed: Local (sentence-transformers) hoặc Gemini API."""

    def __init__(self):
        self.local_model = None
        if USE_LOCAL:
            # Lazy import để không bắt buộc cài nếu không dùng
            from sentence_transformers import SentenceTransformer
            self.local_model = SentenceTransformer(LOCAL_MODEL)

    def embed_corpus(self, texts: List[str]) -> np.ndarray:
        """Trả về mảng (n, d) float32 đã L2-normalized."""
        if not texts:
            return np.zeros((0, 1), dtype="float32")

        if USE_LOCAL:
            # encode batch tại chỗ, normalize ở sbert cho nhanh
            # convert_to_numpy + normalize_embeddings=True => đã chuẩn hóa L2
            from sentence_transformers.util import batch_to_device
            vecs = self.local_model.encode(
                texts,
                batch_size=LOCAL_BATCH,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            return vecs

        # --- Gemini path (batch <= 100 + retry) ---
        all_vecs: List[np.ndarray] = []
        total = len(texts)

        def _embed_batch(batch: List[str]) -> List[np.ndarray]:
            result = client.models.embed_content(model=EMBED_MODEL, contents=batch)
            return [np.array(e.values, dtype="float32") for e in result.embeddings]

        start = 0
        while start < total:
            end = min(total, start + EMBED_BATCH)
            batch = texts[start:end]

            for attempt in range(4):
                try:
                    vecs = _embed_batch(batch)
                    all_vecs.extend(vecs)
                    break
                except Exception as e:
                    msg = str(e)
                    if (("429" in msg) or ("rate" in msg.lower()) or ("unavailable" in msg.lower()) or ("500" in msg)) and attempt < 3:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise
            print(f"[embed] {end}/{total}")
            start = end

        arr = np.stack(all_vecs).astype("float32")
        faiss.normalize_L2(arr)
        return arr

    def embed_query(self, text: str) -> np.ndarray:
        """Trả về vector (1, d) float32 đã L2-normalized cho truy vấn."""
        if USE_LOCAL:
            q = self.local_model.encode(
                [text],
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            return q

        q_vec = client.models.embed_content(model=EMBED_MODEL, contents=text).embeddings[0].values
        q = np.array(q_vec, dtype="float32")[None, :]
        faiss.normalize_L2(q)
        return q

embedder = EmbeddingBackend()

# ========= FAISS Index =========
def _pdf_fingerprint(pdf_path: Path) -> str:
    stat = pdf_path.stat()
    h = hashlib.sha256()
    h.update(str(pdf_path.resolve()).encode())
    h.update(str(stat.st_size).encode())
    h.update(str(int(stat.st_mtime)).encode())
    return h.hexdigest()[:16]

def build_index_from_pdf(pdf_path: Path):
    pages = read_pdf_text(str(pdf_path))
    records = []
    for page_no, page_text in pages:
        for ch in chunk_text(page_text):
            records.append({"page": page_no, "text": ch})
    if not records:
        raise RuntimeError("Không trích xuất được nội dung từ PDF.")

    texts = [r["text"] for r in records]
    mat = embedder.embed_corpus(texts)
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)
    return index, records

def ensure_index(pdf_path: Path):
    fp = _pdf_fingerprint(pdf_path)
    base = INDEX_DIR / f"{pdf_path.stem}_{fp}"
    meta_fp = base.with_suffix(".meta.json")
    index_fp = base.with_suffix(".faiss")

    if meta_fp.exists() and index_fp.exists():
        index = faiss.read_index(str(index_fp))
        with open(meta_fp, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta["records"], meta["pdf"]

    index, records = build_index_from_pdf(pdf_path)
    faiss.write_index(index, str(index_fp))
    with open(meta_fp, "w", encoding="utf-8") as f:
        json.dump({"pdf": str(pdf_path), "records": records}, f, ensure_ascii=False)
    return index, records, str(pdf_path)

def search(index, records, query: str, k=TOP_K):
    if not query.strip():
        return []
    q = embedder.embed_query(query)
    scores, idxs = index.search(q, k)
    hits = []
    for score, idx in zip(scores[0], idxs[0]):
        rec = records[int(idx)]
        hits.append((float(score), rec))
    return hits

def answer_with_context(query: str, hits) -> str:
    context = ""
    for i, (score, rec) in enumerate(hits, start=1):
        context += f"\n[Đoạn {i} | trang {rec['page']} | score {score:.3f}]\n{rec['text']}\n"
    system_hint = (
        "Bạn là trợ lý chỉ dùng NGỮ CẢNH để trả lời. "
        "Nếu thiếu dữ liệu, nói 'không có trong tài liệu'. Luôn chèn [trang X] khi trích dẫn."
    )
    prompt = (
        f"{system_hint}\n\n"
        f"NGỮ CẢNH:\n{context}\n"
        f"CÂU HỎI: {query}\n"
        f"YÊU CẦU: Trả lời ngắn gọn, rõ ràng, có bullet nếu phù hợp."
    )
    resp = client.models.generate_content(model=GEN_MODEL, contents=prompt)
    return resp.text

# ========= Chainlit lifecycle =========
@cl.on_chat_start
async def start():
    await cl.Message(content="RAG Chainlit").send()

    pdf_path = None
    if PDF_PATH and Path(PDF_PATH).exists():
        pdf_path = Path(PDF_PATH)
    else:
        msg = await cl.AskFileMessage(
            content="Hãy tải lên 1 file PDF (hoặc đặt PDF_PATH trong .env).",
            accept=["application/pdf"],
            max_size_mb=30,
            timeout=180,
        ).send()
        if msg and msg[0]:
            pdf_file = msg[0]
            pdf_path = Path(pdf_file.path)

    if not pdf_path:
        await cl.Message(content="Không có PDF. Hãy cấu hình PDF_PATH trong .env hoặc upload.").send()
        return

    backend_name = f"Local: {LOCAL_MODEL}" if USE_LOCAL else f"Gemini: {EMBED_MODEL}"
    await cl.Message(content=f"Đang lập chỉ mục: **{pdf_path.name}** …\nEmbedding backend: **{backend_name}**").send()

    try:
        index, records, src = ensure_index(pdf_path)
    except Exception as e:
        await cl.Message(content=f"Lỗi khi lập chỉ mục: {e}").send()
        return

    cl.user_session.set("index", index)
    cl.user_session.set("records", records)
    cl.user_session.set("pdf_src", src)

    await cl.Message(content="Xong! Bạn có thể đặt câu hỏi về tài liệu.").send()

@cl.on_message
async def on_message(message: cl.Message):
    index = cl.user_session.get("index")
    records = cl.user_session.get("records")

    if not index or not records:
        await cl.Message(content="Chưa sẵn sàng. Vui lòng tải PDF hoặc cấu hình .env rồi /restart.").send()
        return

    query = message.content.strip()
    if not query:
        await cl.Message(content="Bạn hãy nhập câu hỏi nhé.").send()
        return

    await cl.Message(content="Đang truy xuất ngữ cảnh…").send()

    try:
        hits = search(index, records, query, k=TOP_K)

        elements = []
        src_texts = []
        for i, (score, rec) in enumerate(hits, start=1):
            snippet = rec["text"][:800]
            src_texts.append(f"- Đoạn {i} (trang {rec['page']}, score {score:.3f})")
            elements.append(
                cl.Text(
                    name=f"Đoạn {i} • trang {rec['page']} • score {score:.3f}",
                    content=snippet,
                    display="inline",
                )
            )

        await cl.Message(content="Đang tổng hợp câu trả lời…").send()
        answer = answer_with_context(query, hits)

        await cl.Message(
            content=answer + ("\n\nNguồn: " + ", ".join(src_texts) if src_texts else ""),
            elements=elements if elements else None,
        ).send()

    except Exception as e:
        await cl.Message(content=f"Lỗi khi trả lời: {e}").send()
