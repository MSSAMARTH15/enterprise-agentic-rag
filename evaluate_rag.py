import os
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

print("Setting up RAGAS Evaluation Pipeline...")

# 1. Initialize Evaluator LLM & Embeddings
eval_llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name="rag-assistant", embedding=embeddings)

# 2. Define a small test dataset based on your uploaded PDF content
test_questions = [
    "What is the main objective of Atlas?",
    "What happens during Phase 4 Repository Analysis?"
]

test_ground_truths = [
    "Atlas is designed to analyze code repositories and answer user questions about source code.",
    "Phase 4 combines information across individual files to build a repository-level representation and map relationships between entities."
]

# Run retrieval to populate evaluation data
eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": test_ground_truths}

for q in test_questions:
    docs = vectorstore.similarity_search(q, k=3)
    contexts = [doc.page_content for doc in docs]
    
    # Generate answer using Gemini
    prompt = f"Answer the question based only on these contexts:\n\nContext:\n" + "\n".join(contexts) + f"\n\nQuestion: {q}"
    response = eval_llm.invoke(prompt)
    
    eval_data["question"].append(q)
    eval_data["answer"].append(response.content)
    eval_data["contexts"].append(contexts)

# 3. Convert to Hugging Face Dataset format required by Ragas
dataset = Dataset.from_dict(eval_data)

print("\nRunning Ragas Evaluation Metrics (Faithfulness, Answer Relevancy, Context Precision)...")
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=eval_llm,
    embeddings=embeddings
)

print("\n--- RAGAS EVALUATION RESULTS ---")
print(result)