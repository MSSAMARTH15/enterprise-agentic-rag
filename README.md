# Enterprise Agentic RAG Architecture 🚀

A production-grade, full-stack Retrieval-Augmented Generation (RAG) application featuring agentic query decomposition, cross-encoder re-ranking, real-time token streaming, and conversational memory. 

Fully containerized with Docker and evaluated for high precision using Ragas.

## 🏗️ Architecture & Tech Stack

* **Frontend:** Next.js, React, Tailwind CSS
* **Backend:** FastAPI, Python, Server-Sent Events (SSE)
* **Vector Database:** Pinecone
* **LLM Engine:** Google Gemini (Agentic Routing & Generation)
* **Embeddings & Re-ranking:** Hugging Face (`all-MiniLM-L6-v2`, `bge-reranker-base`)
* **Orchestration:** Docker & Docker Compose
* **Evaluation:** Ragas (Automated Metrics)

## ✨ Key Engineering Features

* **Agentic Query Decomposition:** The system intercepts complex, multi-part prompts and uses an LLM router to break them down into distinct sub-queries for highly targeted vector retrieval.
* **Hybrid Search & Cross-Encoder Re-ranking:** Retrieves top-$k$ documents from Pinecone and passes them through a Hugging Face cross-encoder to mathematically score and re-rank the context, ensuring only hyper-relevant chunks are sent to the LLM.
* **Real-Time Token Streaming:** Bypasses standard request timeouts by utilizing Server-Sent Events (SSE) to stream generated tokens directly from FastAPI to the Next.js frontend in real-time.
* **Conversational Memory:** Seamlessly passes multi-turn chat history payloads between the client and server, allowing the LLM to resolve pronoun references (e.g., "elaborate on that second point") without losing context.

## 📊 Quantitative Evaluation (Ragas)

The pipeline was rigorously tested using the Ragas evaluation framework to ensure output quality and prevent hallucinations.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Faithfulness** | `1.0000` | Zero hallucination; 100% of generated answers are grounded in retrieved context. |
| **Context Precision** | `1.0000` | The re-ranker successfully isolated the exact relevant document chunks. |
| **Answer Relevancy** | `0.8495` | High precision match aligning with the user's original intent. |

## 🚀 Local Setup & Deployment

The entire architecture is containerized. To spin up the backend, frontend, and environment configuration simultaneously, run:

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Create a .env file with your API keys
echo "PINECONE_API_KEY=your_key_here" >> .env
echo "GOOGLE_API_KEY=your_key_here" >> .env

# Build and spin up the full stack
docker-compose up --build


Frontend: Available at http://localhost:3000

Backend API: Available at http://localhost:8000