import { useState } from "react";
import "./styles.css";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

export default function App() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Hi! I’m Jarvis. Ask me something from the knowledge base." }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    const msg = input.trim();
    if (!msg || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: msg }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
      });

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: data.answer || "No response",
          sources: data.sources || []
        }
      ]);
    } catch (err) {
  console.error(err);   // ✅ now err is used
  setMessages((prev) => [
    ...prev,
    { role: "bot", text: "Error: Could not reach backend. Is it running on port 8000?" }
  ]);
}
 finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter") sendMessage();
  }

  return (
    <div className="page">
      <header className="header">
        <h2>Jarvis (Self-hosted LLM + Pinecone)</h2>
        <p className="sub">RAG Chatbot UI in React</p>
      </header>

      <div className="chat">
        {messages.map((m, idx) => (
          <div key={idx} className={`bubbleRow ${m.role}`}>
            <div className="bubble">
              <div className="role">{m.role === "user" ? "You" : "Jarvis"}</div>
              <div className="text">{m.text}</div>

              {m.sources?.length ? (
                <div className="sources">
                  <span>Sources: </span>
                  {m.sources.map((s, i) => (
                    <span key={i} className="sourceTag">
                      {s.source}#{s.chunk_id}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        ))}
      </div>

      <div className="composer">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question..."
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </div>
    </div>
  );
}
