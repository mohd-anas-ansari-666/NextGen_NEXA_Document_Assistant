import fitz  # PyMuPDF for text and image extraction
import pdfplumber  # For table extraction
import pytesseract  # OCR for images
from PIL import Image
import os
from datetime import datetime
import numpy as np
from pymongo import MongoClient
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables and configure Google Generative AI
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["pdf_embeddings"]
collection = db["documents"]
chat_history_collection = db["chat_history"]  # New collection for chat history

# Ensure indexing for faster retrieval
collection.create_index("pdf_name")
collection.create_index("chunk_text")

# Load embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")

# Initialize Gemini Pro for chat functionality
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")

def extract_text(pdf_path):
    """Extracts text from PDF and returns it as a string."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def extract_images(pdf_path, output_folder="images"):
    """Extracts images from PDF and saves them locally."""
    os.makedirs(output_folder, exist_ok=True)
    images = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_path = os.path.join(output_folder, f"page_{i}_img_{img_index}.png")
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                images.append(img_path)
    return images

def extract_tables(pdf_path):
    """Extracts tables and returns them as JSON."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                tables.append(table)
    return tables

def perform_ocr(image_path):
    """Performs OCR on an image and returns extracted text."""
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def chunk_text(text):
    """Splits text into smaller chunks using LangChain."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def generate_embeddings(chunks):
    """Generates embeddings for text chunks using Gemini."""
    return embedding_model.embed_documents(chunks)

def store_text_chunk_in_mongodb(pdf_name, chunk, embeddings):
    """Stores a single text chunk and its embeddings into MongoDB."""

    # # Debugging: Print the chunk length and first 100 characters before storing it
    # print(f"Storing chunk: {chunk[:100]}...")  # First 100 characters
    # print(f"Chunk length: {len(chunk)} characters")  # Length of the chunk

    # Make sure the embedding is a list for proper MongoDB storage
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()

    data = {
        "pdf_name": pdf_name,
        "chunk_text": chunk,
        "embeddings": embeddings
    }
    collection.insert_one(data)

def store_image_in_mongodb(pdf_name, image_path):
    """Stores an image reference in MongoDB."""
    data = {
        "pdf_name": pdf_name,
        "image_path": image_path
    }
    collection.insert_one(data)

def store_table_in_mongodb(pdf_name, table_data):
    """Stores a table reference in MongoDB."""
    data = {
        "pdf_name": pdf_name,
        "table_data": table_data
    }
    collection.insert_one(data)

def find_similar_chunks(query, top_n=10):
    """Finds the most similar text chunk for a given query using cosine similarity."""
    # Get query embedding as a 1D array
    query_embedding = np.array(embedding_model.embed_documents([query])[0])
    
    docs = list(collection.find({}, {"chunk_text": 1, "embeddings": 1, "_id": 0}))
    
    # List to hold all matches with their scores
    matches = []

    # best_match = None
    # best_score = -1
    
    for doc in docs:
        if "chunk_text" in doc and "embeddings" in doc:
            # Ensure we're dealing with a single chunk and its embedding
            chunk = doc["chunk_text"]
            emb = np.array(doc["embeddings"])
            pdf_name = doc.get("pdf_name", "Unknown")
            
            # # Debugging: Print chunk content, length, and score
            # print(f"Processing chunk (first 100 characters): {chunk[:100]}...")
            # print(f"Chunk length: {len(chunk)}")
            # print(f"Embedding shape: {emb.shape}")

            # Make sure embeddings are correctly shaped
            if isinstance(chunk, list) and isinstance(emb, list):
                # Handle case where we have lists of chunks and embeddings
                for c, e in zip(chunk, emb):
                    e = np.array(e)
                    if e.ndim == 1:  # If embedding is already 1D
                        score = cosine_similarity([query_embedding], [e])[0][0]
                    else:
                        # Try to flatten or reshape as needed
                        e = e.flatten() if e.ndim > 2 else e
                        score = cosine_similarity([query_embedding], [e])[0][0]
                    
                    matches.append((c, score, pdf_name))

                    # if score > best_score:
                    #     best_score = score
                    #     best_match = c
            else:
                # Handle single chunk case
                emb = np.array(emb)
                if emb.ndim == 1:  # If embedding is already 1D
                    score = cosine_similarity([query_embedding], [emb])[0][0]
                else:
                    # Try to flatten or reshape as needed
                    emb = emb.flatten() if emb.ndim > 2 else emb
                    score = cosine_similarity([query_embedding], [emb])[0][0]
                
                matches.append((chunk, score, pdf_name))

                # if score > best_score:
                #     best_score = score
                #     best_match = chunk
    
    # return best_match, best_score

    # Sort matches by score in descending order
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # Get the top_n matches
    top_matches = matches[:top_n]
    
    # # Debugging: Print top matches for verification
    # print("\nTop matches:")
    # for i, (match, score) in enumerate(top_matches):
    #     print(f"Top {i+1}: (Score: {score})")
    #     print(f"Chunk: {match[:200]}...")  # Print first 200 characters of the matched chunk

    # Return top 10 results
    return top_matches

def chat_with_pdf(query, user_id="default_user", session_id=None):
    """
    Provides a chat interface using the PDF content as context.
    
    Args:
        query (str): The user's question
        user_id (str): Identifier for the user
        session_id (str): Identifier for the current chat session
    
    Returns:
        str: The AI's response
    """
    # If no session_id is provided, create a new one
    if not session_id:
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Find relevant chunks
    top_matches = find_similar_chunks(query, top_n=3)
    
    # Extract the text content from matches
    context = "\n\n".join([match[0] for match in top_matches])
    
    # Prepare the prompt for Gemini
    prompt = f"""
    You are an AI assistant that helps users understand information from PDF documents. 
    Answer the following question based ONLY on the provided context. If you cannot answer the 
    question based solely on the context, say "I don't have enough information to answer that question."
    
    CONTEXT:
    {context}
    
    QUESTION:
    {query}
    
    SOURCES:
    {', '.join(set([match[2] for match in top_matches]))}
    """
    
    # Get response from Gemini
    response = gemini_model.invoke(prompt)
    
    # Get the text from the response
    answer = response.content
    
    # Store the conversation in MongoDB
    chat_entry = {
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": datetime.now(),
        "query": query,
        "response": answer,
        "context_chunks": [(match[0][:100] + "..." if len(match[0]) > 100 else match[0], match[1], match[2]) for match in top_matches]
    }
    chat_history_collection.insert_one(chat_entry)
    
    return answer, session_id

def get_chat_history(user_id, session_id=None):
    """
    Retrieves chat history for a user.
    
    Args:
        user_id (str): The user identifier
        session_id (str, optional): If provided, get history for this session only
    
    Returns:
        list: Chat history as a list of entries
    """
    query = {"user_id": user_id}
    if session_id:
        query["session_id"] = session_id
    
    # Sort by timestamp to get chronological order
    chat_history = list(chat_history_collection.find(
        query, 
        {"query": 1, "response": 1, "timestamp": 1, "_id": 0}
    ).sort("timestamp", 1))
    
    return chat_history

# Main processing function
def process_pdf(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    text = extract_text(pdf_path)
    text_chunks = chunk_text(text)
    embeddings = generate_embeddings(text_chunks)
    tables = extract_tables(pdf_path)
    images = extract_images(pdf_path)
    # image_texts = [perform_ocr(img) for img in images]
    # for chunk, embeddings in zip(text_chunks, embeddings):
    #     store_text_chunk_in_mongodb(pdf_name, chunk, embeddings) #text_chunks+image_texts

    for i, chunk in enumerate(text_chunks):
        store_text_chunk_in_mongodb(pdf_name, chunk, embeddings[i])

     # Store images in MongoDB
    for img_path in images:
        store_image_in_mongodb(pdf_name, img_path)
    
    # Store tables in MongoDB
    for table in tables:
        store_table_in_mongodb(pdf_name, table)

    print(f"PDF {pdf_name} processed and stored in MongoDB.")

# Interactive chat function
def interactive_chat():
    """Interactive console-based chat with the PDF content."""
    print("Welcome to PDF Chat Assistant! (Type 'exit' to quit)")
    
    user_id = input("Enter your user ID (or press Enter for default): ") or "default_user"
    session_id = None
    
    while True:
        query = input("\nYour question: ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Thank you for using PDF Chat Assistant!")
            break
        
        answer, session_id = chat_with_pdf(query, user_id, session_id)
        print("\nAssistant:", answer)

# Example Usage
if __name__ == "__main__":
    pdf_path = r"C:\Users\HP\Desktop\V3.0\NextGenNEXA_Doc_Assistant\testSummary.pdf"
    process_pdf(pdf_path)

    # Start the interactive chat
    interactive_chat()

    # # Example Query Search
    # query = "satellite engineeering"
    # # match, score = find_similar_chunks(query)
    # # print(f"Best match: {match} (Score: {score})")
    # top_matches = find_similar_chunks(query, top_n=5)

    # # Print top 10 best matches
    # for i, (match, score) in enumerate(top_matches):
    #     print(f"Match {i+1}: {match} (Score: {score})")










