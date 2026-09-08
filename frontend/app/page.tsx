"use client";
import { useState } from "react";

interface Source {
  page: number | string;
  content: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  sub_queries?: string[];
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploading, setUploading] = useState(false);
  
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setUploadStatus("Processing document with advanced chunking...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setUploadStatus(data.message || "Upload successful!");
    } catch (err) {
      console.error(err);
      setUploadStatus("Error uploading file.");
    } finally {
      setUploading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input;
    setInput("");
    
    const updatedMessages = [...messages, { role: "user" as const, content: userMessage }];
    setMessages(updatedMessages);
    setLoading(true);

    // Add empty assistant slot for streaming tokens
    setMessages((prev) => [...prev, { role: "assistant", content: "", sources: [], sub_queries: [] }]);

    try {
      const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          question: userMessage,
          history: updatedMessages.map(m => ({ role: m.role, content: m.content }))
        }),
      });

      if (!res.body) return;

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunkText = decoder.decode(value, { stream: true });
        const lines = chunkText.split("\n").filter(Boolean);

        for (const line of lines) {
          const data = JSON.parse(line);

          if (data.type === "meta") {
            setMessages((prev) => {
              const updated = [...prev];
              const lastMsg = updated[updated.length - 1];
              lastMsg.sources = data.sources;
              lastMsg.sub_queries = data.sub_queries_used;
              return updated;
            });
          } else if (data.type === "token") {
            accumulatedContent += data.content;
            setMessages((prev) => {
              const updated = [...prev];
              const lastMsg = updated[updated.length - 1];
              lastMsg.content = accumulatedContent;
              return updated;
            });
          }
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex flex-col h-screen bg-gray-900 text-white p-4 max-w-3xl mx-auto">
      <header className="py-4 border-b border-gray-800 mb-4 flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold">CloudVault AI — Advanced Agentic RAG</h1>
          <p className="text-xs text-blue-400 mt-0.5">Powered by Query Decomposition, Re-ranking, Streaming & Memory</p>
        </div>
      </header>

      {/* Dynamic File Upload Section */}
      <form onSubmit={handleFileUpload} className="bg-gray-800 p-4 rounded-lg mb-4 flex items-center gap-4">
        <input
          type="file"
          accept=".pdf"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-500 cursor-pointer"
        />
        <button
          type="submit"
          disabled={!file || uploading}
          className="bg-green-600 px-4 py-2 rounded-lg text-sm font-medium hover:bg-green-500 disabled:opacity-50"
        >
          {uploading ? "Processing..." : "Upload & Index PDF"}
        </button>
      </form>
      {uploadStatus && <p className="text-xs text-green-400 mb-2 px-1">{uploadStatus}</p>}

      {/* Chat Messages Section */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-2 mb-4">
        {messages.length === 0 && (
          <p className="text-gray-500 text-center mt-10">Upload a PDF and ask follow-up questions with full conversational memory...</p>
        )}
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`p-4 rounded-lg ${
              msg.role === "user" ? "bg-blue-600 ml-auto max-w-[80%]" : "bg-gray-800 mr-auto max-w-[90%]"
            }`}
          >
            {msg.sub_queries && msg.sub_queries.length > 0 && (
              <div className="mb-2 pb-2 border-b border-gray-700 text-xs text-yellow-400">
                <span className="font-semibold">Agentic Decomposition: </span>
                {msg.sub_queries.join(" | ")}
              </div>
            )}
            <p className="whitespace-pre-wrap">{msg.content}</p>
            {msg.sources && msg.sources.length > 0 && (
              <div className="mt-3 pt-2 border-t border-gray-700 text-xs text-gray-400">
                <p className="font-semibold text-gray-300">Re-ranked Sources (Cross-Encoder Verified):</p>
                {msg.sources.map((src, sIdx) => (
                  <div key={sIdx} className="mt-1">
                    <span className="bg-gray-700 text-gray-200 px-1.5 py-0.5 rounded">
                      Page {src.page}
                    </span>
                    <span className="ml-2 italic">{src.content.substring(0, 80)}...</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && <div className="text-gray-400 italic">Streaming response with context memory...</div>}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question or follow-up..."
          className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 focus:outline-none focus:border-blue-500"
        />
        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 px-6 py-2 rounded-lg font-medium hover:bg-blue-500 disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </main>
  );
}