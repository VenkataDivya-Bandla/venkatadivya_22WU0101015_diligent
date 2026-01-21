import os
import uuid
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Optional PDF support
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# ----------------------------
# 1) Load env reliably
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "jarvis-knowledge")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not PINECONE_API_KEY:
    raise RuntimeError(
        "PINECONE_API_KEY is missing. Ensure backend/.env exists and contains:\n"
        "PINECONE_API_KEY=pcsk_...\n"
    )

# ----------------------------
# 2) Embeddings model
# ----------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)
DIM = embedder.get_sentence_embedding_dimension()

# ----------------------------
# 3) Pinecone client
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)


def ensure_index() -> None:
    """Create Pinecone index if it doesn't exist."""
    existing = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Simple chunking. For interview/MVP this is enough.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(0, end - overlap)

    return chunks


def embed(text: str) -> List[float]:
    vec = embedder.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


def read_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(file_path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError(
            "pypdf is not installed. Install it with:\n"
            "pip install pypdf\n"
        )
    reader = PdfReader(str(file_path))
    full_text = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            full_text.append(t)
    return "\n".join(full_text)


def load_documents(data_dir: Path) -> List[Tuple[str, str]]:
    """
    Returns list of (source_name, text)
    Supports: .pdf, .txt
    """
    docs = []
    if not data_dir.exists():
        print(f"⚠️ Data folder not found: {data_dir}")
        return docs

    for fp in sorted(data_dir.iterdir()):
        if fp.is_file() and fp.suffix.lower() in [".txt", ".pdf"]:
            try:
                if fp.suffix.lower() == ".txt":
                    text = read_txt(fp)
                else:
                    text = read_pdf(fp)

                if text.strip():
                    docs.append((fp.name, text))
                else:
                    print(f"⚠️ Skipped (empty text): {fp.name}")

            except Exception as e:
                print(f"❌ Failed to read {fp.name}: {e}")

    return docs


def upsert_document(source_name: str, text: str) -> None:
    ensure_index()
    index = pc.Index(INDEX_NAME)

    chunks = chunk_text(text)
    if not chunks:
        print(f"⚠️ No chunks created for {source_name} (empty text).")
        return

    vectors = []
    for i, ch in enumerate(chunks):
        vid = str(uuid.uuid4())
        vectors.append(
            (
                vid,
                embed(ch),
                {
                    "source": source_name,
                    "chunk_id": str(i),
                    "text": ch,
                },
            )
        )

    index.upsert(vectors=vectors)
    print(f"✅ Ingested {len(chunks)} chunks into '{INDEX_NAME}' from '{source_name}'")


def main():
    data_dir = BASE_DIR / "data"
    print(f"📂 Loading documents from: {data_dir}")

    docs = load_documents(data_dir)

    # If no files exist, ingest a fallback sample (so demo always works)
    if not docs:
        print("ℹ️ No PDFs/TXTs found in data/. Ingesting sample text for demo...")
        sample_doc = """
Company Leave Policy:
- Employees get 12 paid leaves per year.
- Leave requests must be applied 2 days in advance.
- Sick leaves can be applied on the same day with manager approval.

Overtime Policy:
- Standard work week is 40 hours.
- Overtime is paid only if pre-approved by manager.
- Overtime rate: 1.5x hourly pay for extra hours.

Security:
- Never share passwords or OTP.
- Report suspicious emails to IT immediately.
"""
        upsert_document("Handbook.pdf", sample_doc)
        return

    for name, text in docs:
        upsert_document(name, text)


if __name__ == "__main__":
    main()
