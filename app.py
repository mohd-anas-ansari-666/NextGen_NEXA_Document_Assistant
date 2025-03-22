import fitz  # PyMuPDF for text extraction
import os
import numpy as np
from pymongo import MongoClient
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["pdf_db"]
collection = db["pdf_chunks"]
chat_history = db["chat_history"]

# Create indexes for faster retrieval
collection.create_index("pdf_name")
collection.create_index("chunk_text")

# Initialize models
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    temperature=0.3,
    max_output_tokens=2048,
    top_p=0.95,
    top_k=40
)

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file"""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def create_text_chunks(text):
    """Split text into manageable chunks with overlap"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return text_splitter.split_text(text)

def embed_chunks(chunks):
    """Generate embeddings for text chunks"""
    return embedding_model.embed_documents(chunks)

def store_chunks_in_db(pdf_name, chunks, embeddings):
    """Store chunks and their embeddings in MongoDB"""
    for chunk, embedding in zip(chunks, embeddings):
        # Convert numpy array to list if needed
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
            
        # Create document
        document = {
            "pdf_name": pdf_name,
            "chunk_text": chunk,
            "embeddings": embedding,
            "timestamp": datetime.now()
        }
        
        # Insert into MongoDB
        collection.insert_one(document)
    
    return len(chunks)

def process_pdf(pdf_path):
    """Process a PDF file and store its chunks and embeddings"""
    # Extract PDF name from path
    pdf_name = os.path.basename(pdf_path)
    
    # Check if PDF has already been processed
    if collection.count_documents({"pdf_name": pdf_name}) > 0:
        print(f"PDF {pdf_name} has already been processed.")
        return f"PDF {pdf_name} already exists in database."
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Create chunks from text
    chunks = create_text_chunks(text)
    
    # Generate embeddings for chunks
    embeddings = embed_chunks(chunks)
    
    # Store chunks and embeddings in MongoDB
    num_chunks = store_chunks_in_db(pdf_name, chunks, embeddings)
    
    return f"Successfully processed {pdf_name}: {num_chunks} chunks stored in database."

def retrieve_relevant_chunks(query, top_n=5):
    """Find the most relevant text chunks for a query using vector similarity"""
    # Generate embedding for query
    query_embedding = np.array(embedding_model.embed_documents([query])[0])
    
    # Retrieve all chunks from database
    all_docs = list(collection.find(
        {}, 
        {"chunk_text": 1, "embeddings": 1, "pdf_name": 1, "_id": 0}
    ))
    
    # Calculate similarity scores
    results = []
    for doc in all_docs:
        chunk_text = doc["chunk_text"]
        chunk_embedding = np.array(doc["embeddings"])
        pdf_name = doc["pdf_name"]
        
        # Calculate cosine similarity
        similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
        
        # Add to results
        results.append((chunk_text, similarity, pdf_name))
    
    # Sort by similarity score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top matches
    return results[:top_n]

def chat_with_pdf(query, user_id="default_user", session_id=None):
    """Chat with PDF content using relevant context chunks"""
    # Create new session ID if none provided
    if not session_id:
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Find relevant chunks
    relevant_chunks = retrieve_relevant_chunks(query, top_n=5)
    
    # Extract text and sources
    context_texts = [chunk[0] for chunk in relevant_chunks]
    sources = list(set([chunk[2] for chunk in relevant_chunks]))
    
    # Combine context chunks
    context = "\n\n---\n\n".join(context_texts)
    
    # Create prompt for LLM
    prompt = f"""
    You are an expert document assistant that helps users understand information from PDF documents.
    Answer the following question based on the provided context sections.
    
    Your answer should:
    1. Be thorough and complete, covering all relevant information
    2. Present information in a clear, structured format
    3. Include all items from any lists mentioned in the context
    4. Only use information contained in the context
    
    If the context doesn't contain enough information to answer the question fully, 
    simply state: "I don't have enough information to answer that question completely."
    
    CONTEXT:
    {context}
    
    QUESTION:
    {query}
    """
    
    # Get response from LLM
    response = chat_model.invoke(prompt)
    answer = response.content
    
    # Add source information
    if sources:
        answer += f"\n\nSources: {', '.join(sources)}"
    
    # Store conversation in chat history
    chat_entry = {
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": datetime.now(),
        "query": query,
        "response": answer,
        "sources": sources
    }
    chat_history.insert_one(chat_entry)
    
    return answer, session_id

def get_chat_history(user_id, session_id=None):
    """Retrieve chat history for a user"""
    query = {"user_id": user_id}
    if session_id:
        query["session_id"] = session_id
    
    history = list(chat_history.find(
        query,
        {"query": 1, "response": 1, "timestamp": 1, "_id": 0}
    ).sort("timestamp", 1))
    
    return history

def interactive_chat():
    """Run an interactive chat session in the console"""
    print("=" * 50)
    print("PDF Chat Assistant")
    print("=" * 50)
    
    user_id = input("Enter user ID (or press Enter for default): ") or "default_user"
    session_id = None
    
    while True:
        query = input("\nYour question (type 'exit' to quit): ")
        
        if query.lower() in ["exit", "quit", "q"]:
            print("Thank you for using PDF Chat Assistant!")
            break
        
        print("\nSearching documents...")
        answer, session_id = chat_with_pdf(query, user_id, session_id)
        
        print("\nAssistant:")
        print(answer)

# Example usage
if __name__ == "__main__":
    # Process a PDF file
    pdf_path = "testSummary.pdf"
    result = process_pdf(pdf_path)
    print(result)
    
    # Start interactive chat
    interactive_chat()