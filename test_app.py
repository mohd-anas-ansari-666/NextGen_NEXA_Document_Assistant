import fitz  # PyMuPDF for text and image extraction
import pdfplumber  # For table extraction
import pytesseract  # OCR for images
from PIL import Image
import os
import numpy as np
from pymongo import MongoClient
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# Load environment variables and configure Google Generative AI
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["pdf_embeddings"]
collection = db["documents"]

# Ensure indexing for faster retrieval
collection.create_index("pdf_name")
collection.create_index("chunk_text")

# Load embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")

def extract_text_parallel(pdf_path):
    """Extracts text from PDF using multi-threading."""
    with fitz.open(pdf_path) as doc:
        with ThreadPoolExecutor() as executor:
            text = executor.map(lambda page: page.get_text("text"), doc)
    return "\n".join(text)

def extract_images_parallel(pdf_path, output_folder="images"):
    os.makedirs(output_folder, exist_ok=True)
    images = []
    with fitz.open(pdf_path) as doc:
        def process_page(page, page_num):
            extracted_images = []
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_path = os.path.join(output_folder, f"page_{page_num}_img_{img_index}.png")
                with open(img_path, "wb") as f:
                    f.write(base_image["image"])
                extracted_images.append(img_path)
            return extracted_images
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_page, doc, range(len(doc)))
            for img_list in results:
                images.extend(img_list)
    return images

def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables.extend(page.extract_tables())
    return tables

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return text_splitter.split_text(text)

def generate_embeddings_batch(chunks, batch_size=10):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    return embeddings

def store_text_chunks_bulk(pdf_name, text_chunks, embeddings):
    bulk_data = [
        {"pdf_name": pdf_name, "chunk_text": chunk, "embeddings": emb}
        for chunk, emb in zip(text_chunks, embeddings)
    ]
    collection.insert_many(bulk_data)

def store_images_bulk(pdf_name, images):
    bulk_data = [{"pdf_name": pdf_name, "image_path": img} for img in images]
    collection.insert_many(bulk_data)

def store_tables_bulk(pdf_name, tables):
    bulk_data = [{"pdf_name": pdf_name, "table_data": table} for table in tables]
    collection.insert_many(bulk_data)

def find_similar_chunks(query):
    query_embedding = embedding_model.embed_documents([query])[0]
    docs = list(collection.find({}, {"chunk_text": 1, "embeddings": 1}))
    
    best_match = None
    best_score = -1
    
    for doc in docs:
        for chunk, emb in zip(doc.get("chunk_text", []), doc.get("embeddings", [])):
            score = cosine_similarity([query_embedding], [emb])[0][0]
            if score > best_score:
                best_score = score
                best_match = chunk
    
    return best_match, best_score

def process_pdf(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    text = extract_text_parallel(pdf_path)
    text_chunks = chunk_text(text)
    embeddings = generate_embeddings_batch(text_chunks)
    tables = extract_tables(pdf_path)
    images = extract_images_parallel(pdf_path)
    store_text_chunks_bulk(pdf_name, text_chunks, embeddings)
    store_images_bulk(pdf_name, images)
    store_tables_bulk(pdf_name, tables)
    print(f"PDF {pdf_name} processed and stored in MongoDB.")

# Example Usage
if __name__ == "__main__":
    pdf_path = r"C:\Users\HP\Desktop\V3.0\NextGenNEXA_Doc_Assistant\Missile Guidance And Control Systems.pdf"
    process_pdf(pdf_path)
    query = "Missile guidance systems"
    match, score = find_similar_chunks(query)
    print(f"Best match: {match} (Score: {score})")
