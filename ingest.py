import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Set your API Key
os.environ['PINECONE_API_KEY'] = 'pcsk_2x9hsr_RhsZpuoziaYDFg4XU81kM6PCd46ErPUkDT4RnJtzRb1V3e7gQyYYgc5PaXYXL3F'

def process_pdf(file_path):
    print(f"Loading document from {file_path}...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split the document into {len(chunks)} chunks.")
    return chunks

def upload_to_pinecone(chunks):
    # 2. Initialize the embedding model
    # We use "all-MiniLM-L6-v2" because it's fast, runs locally, 
    # and outputs exactly 384 dimensions (matching your Pinecone index)
    print("Downloading/Initializing embedding model (this happens once)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Upload chunks and embeddings to Pinecone
    print("Uploading to Pinecone database... this might take a minute.")
    index_name = "rag-assistant"
    
    # This automatically converts your text chunks into vectors and pushes them to the cloud
    PineconeVectorStore.from_documents(
        chunks, 
        embeddings, 
        index_name=index_name
    )
    print("Upload 100% complete! Your database is loaded.")

if __name__ == "__main__":
    # Pointing directly to your report.pdf
    pdf_chunks = process_pdf(r"C:\Users\Samarth\CloudVault_AI\report.pdf")
    
    # Trigger the upload
    upload_to_pinecone(pdf_chunks)