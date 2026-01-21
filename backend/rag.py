import os
import requests
from typing import List, Dict
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "jarvis-knowledge")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Embedding model (local embeddings; fast & interview-friendly)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def embed_text(text: str) -> List[float]:
    vec = embedder.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


def retrieve_context(query: str, top_k: int = 4) -> List[Dict]:
    qvec = embed_text(query)
    res = index.query(vector=qvec, top_k=top_k, include_metadata=True)
    matches = res.get("matches", []) if isinstance(res, dict) else res.matches

    chunks = []
    for m in matches:
        md = m["metadata"] if isinstance(m, dict) else m.metadata
        score = m["score"] if isinstance(m, dict) else m.score
        chunks.append({
            "score": float(score),
            "text": md.get("text", ""),
            "source": md.get("source", "unknown"),
            "chunk_id": md.get("chunk_id", "NA"),
        })
    return chunks


def build_prompt(user_query: str, contexts: List[Dict]) -> str:
    context_text = "\n\n".join(
        [f"[Source: {c['source']} | Chunk: {c['chunk_id']}]\n{c['text']}" for c in contexts]
    )

    prompt = f"""
You are an enterprise personal assistant.
You must answer ONLY using the provided CONTEXT.
If the answer is not in the context, say: "I don't have enough information in the knowledge base."

CONTEXT:
{context_text}

USER QUESTION:
{user_query}

ANSWER (be clear and concise):
"""
    return prompt.strip()


def call_ollama(prompt: str) -> str:
    # Ollama generate API
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def answer_query(user_query: str) -> Dict:
    contexts = retrieve_context(user_query, top_k=4)

    # Basic “no context” safety
    if not contexts or all(c["text"].strip() == "" for c in contexts):
        return {
            "answer": "I don't have enough information in the knowledge base.",
            "sources": []
        }

    prompt = build_prompt(user_query, contexts)
    answer = call_ollama(prompt)

    # Return top sources (for interview: show grounding)
    sources = [{
        "source": c["source"],
        "chunk_id": c["chunk_id"],
        "score": c["score"]
    } for c in contexts]

    return {"answer": answer, "sources": sources}
