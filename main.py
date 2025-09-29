import os, json, hashlib, time
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
from pypdf import PdfReader
from dotenv import load_dotenv
import chainlit as cl

# ====== Hugging Face / Transformers ======
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ========= Load .env & config =========
load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "").strip() or None

# Retrieval (embedding)
USE_LOCAL_EMBEDDING = os.getenv("USE_LOCAL_EMBEDDING", "true").strip().lower() == "true"
LOCAL_EMBED_MODEL   = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LOCAL_EMBED_BATCH   = int(os.getenv("LOCAL_EMBED_BATCH", "64"))

TOP_K          = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", "150"))

# Generation (LLM local)
USE_LOCAL_LLM       = os.getenv("USE_LOCAL_LLM", "true").strip().lower() == "true"
LOCAL_LLM_MODEL     = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_MAX_NEW_TOKENS  = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "512"))
LLM_TEMPERATURE     = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.2"))

INDEX_DIR = Path("index_store"); INDEX_DIR.mkdir(exist_ok=True)

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
    i, n = 0, len(text)
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

# ========= Embedding backend (local) =========
class LocalEmbedder:
    def __init__(self, model_name: str, batch: int):
        self.model = SentenceTransformer(model_name)
        self.batch = batch

    def embed_corpus(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        v = self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        return v

# ========= Local LLM (transformers pipeline) =========
class LocalLLM:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto"
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

    def _format_prompt(self, context: str, question: str) -> str:
        system = (
            "Bạn là trợ lý chỉ trả lời dựa trên NGỮ CẢNH cung cấp. "
            "Nếu thông tin không có trong ngữ cảnh, hãy nói 'không có trong tài liệu'. "
            "Khi trích dẫn, ghi [trang X]. Trả lời ngắn gọn, có bullet khi phù hợp."
        )
        # Nếu tokenizer có chat template thì dùng (nhiều model Instruct hỗ trợ)
        if hasattr(self.tokenizer, "apply_chat_template"):
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}"}
            ]
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # Fallback: prompt thường
        return (
            f"[HỆ THỐNG]\n{system}\n\n"
            f"[NGỮ CẢNH]\n{context}\n\n"
            f"[CÂU HỎI]\n{question}\n\n[TRẢ LỜI]"
        )

    def generate(self, context: str, question: str,
                 max_new_tokens: int = LLM_MAX_NEW_TOKENS,
                 temperature: float = LLM_TEMPERATURE) -> str:
        prompt = self._format_prompt(context, question)
        out = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]
        # Cắt phần prompt nếu model trả về cả prompt
        return out[len(prompt):].strip() if out.startswith(prompt) else out.strip()

# ======== init backends ========
embedder = LocalEmbedder(LOCAL_EMBED_MODEL, LOCAL_EMBED_BATCH)
llm = LocalLLM(LOCAL_LLM_MODEL)

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

def generate_answer(question: str, hits) -> str:
    context = ""
    for i, (score, rec) in enumerate(hits, start=1):
        context += f"\n[Đoạn {i} | trang {rec['page']} | score {score:.3f}]\n{rec['text']}\n"
    return llm.generate(context=context, question=question)

# ========= Chainlit lifecycle =========
@cl.on_chat_start
async def start():
    await cl.Message(content="RAG Chainlit (100% local)").send()

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

    await cl.Message(content=f"Đang lập chỉ mục: **{pdf_path.name}** …").send()

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

        await cl.Message(content="Đang tổng hợp câu trả lời (local)…").send()
        answer = generate_answer(query, hits)

        await cl.Message(
            content=answer + ("\n\nNguồn: " + ", ".join(src_texts) if src_texts else ""),
            elements=elements if elements else None,
        ).send()

    except Exception as e:
        await cl.Message(content=f"Lỗi khi trả lời: {e}").send()
