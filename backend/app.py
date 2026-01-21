from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import answer_query

app = FastAPI(title="Jarvis RAG Assistant")

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    result = answer_query(req.message)
    return result
