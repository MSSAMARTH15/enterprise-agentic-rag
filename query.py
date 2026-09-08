import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Setup API Keys
os.environ['PINECONE_API_KEY'] = 'pcsk_2x9hsr_RhsZpuoziaYDFg4XU81kM6PCd46ErPUkDT4RnJtzRb1V3e7gQyYYgc5PaXYXL3F'
api_key = os.getenv("GOOGLE_API_KEY")

def ask_rag(question):
    print("\n--- Processing Question ---")
    
    # 2. Re-connect to the same Hugging Face embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Connect to the existing Pinecone Index
    vectorstore = PineconeVectorStore(
        index_name="rag-assistant", 
        embedding=embeddings
    )
    
    # Configure it as a retriever (fetches top 3 relevant chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 4. Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        temperature=0.2
    )
    
    # 5. Define System Prompt forcing answers grounded strictly in context
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 6. Combine Retrieval and Generation
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # 7. Execute Search and Generate Response
    response = rag_chain.invoke({"input": question})
    
    print("\n[AI Answer]:")
    print(response["answer"])
    
    print("\n[Sources Retrieved]:")
    for doc in response["context"]:
        page_num = doc.metadata.get('page', 'Unknown')
        print(f"- Page {page_num}: {doc.page_content[:100]}...")

if __name__ == "__main__":
    # Test question about the CloudVault AI report
    ask_rag("How does CloudVault AI handle missing or failed chunks?")