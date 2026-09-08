import os
import shutil
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Environment Variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Heavy Models Once on Startup
print("Initializing Advanced RAG Components...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name="rag-assistant", embedding=embeddings)

# Initialize robust local Cross-Encoder for Re-ranking
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

# Initialize Gemini LLMs
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
router_llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.0)

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []

@app.post("/ask")
async def ask_question(request: QueryRequest):
    user_query = request.question
    chat_history = request.history or []
    
    # 1. Agentic Decomposition
    router_prompt = f"Decompose the following user query into 1 to 3 distinct sub-queries for document search if it's complex, otherwise return just the original query separated by semicolons: '{user_query}'"
    router_response = router_llm.invoke(router_prompt)
    sub_queries = [q.strip() for q in router_response.content.split(";")]

    # 2. Broad Retrieval & Re-ranking
    retrieved_docs_map = {}
    for q in sub_queries:
        docs = vectorstore.similarity_search(q, k=6)
        for doc in docs:
            retrieved_docs_map[doc.page_content] = doc

    candidate_docs = list(retrieved_docs_map.values())
    if not candidate_docs:
        async def empty_stream():
            yield json.dumps({"type": "meta", "sources": [], "sub_queries_used": sub_queries}) + "\n"
            yield json.dumps({"type": "token", "content": "I could not find any relevant information in the uploaded documents."}) + "\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    pairs = [[user_query, doc.page_content] for doc in candidate_docs]
    scores = model.score(pairs)
    scored_docs = list(zip(candidate_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    final_docs = [doc for doc, score in scored_docs[:3]]

    sources = [
        {
            "page": doc.metadata.get("page", "Unknown"),
            "content": doc.page_content[:200]
        }
        for doc in final_docs
    ]

    # 3. Format Conversation History
    history_text = ""
    for msg in chat_history[-4:]: # Keep last 4 turns for efficient context windows
        role_label = "User" if msg.role == "user" else "Assistant"
        history_text += f"{role_label}: {msg.content}\n"

    # 4. Stream Tokens via Async Generator with Memory & Context
    async def generate_tokens():
        meta_payload = json.dumps({
            "type": "meta",
            "sources": sources,
            "sub_queries_used": sub_queries
        }) + "\n"
        yield meta_payload

        context_text = "\n\n".join([doc.page_content for doc in final_docs])
        
        full_prompt = (
            f"You are an advanced enterprise RAG assistant.\n"
            f"Conversation History:\n{history_text}\n"
            f"Retrieved Context:\n{context_text}\n\n"
            f"Answer the user's latest question accurately using the context and history above."
        )
        
        async for chunk in llm.astream(full_prompt):
            token_payload = json.dumps({"type": "token", "content": chunk.content}) + "\n"
            yield token_payload

    return StreamingResponse(generate_tokens(), media_type="text/event-stream")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        PineconeVectorStore.from_documents(
            chunks, 
            embeddings, 
            index_name="rag-assistant"
        )
        
        return {
            "message": f"Successfully processed '{file.filename}' and uploaded {len(chunks)} chunks to Pinecone!"
        }
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)