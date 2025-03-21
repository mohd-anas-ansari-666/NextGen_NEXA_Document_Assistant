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

# Load environment variables and configure Google Generative AI
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["pdf_embeddings"]
collection = db["documents"]

# Load embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return text_splitter.split_text(text)

def generate_embeddings(chunks):
    """Generates embeddings for text chunks using Gemini."""
    return embedding_model.embed_documents(chunks)

def store_text_chunk_in_mongodb(pdf_name, chunk, embeddings):
    """Stores a single text chunk and its embeddings into MongoDB."""
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

# Main processing function
def process_pdf(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    text = extract_text(pdf_path)
    text_chunks = chunk_text(text)
    embeddings = generate_embeddings(text_chunks)
    tables = extract_tables(pdf_path)
    images = extract_images(pdf_path)
    # image_texts = [perform_ocr(img) for img in images]
    for chunk, embeddings in zip(text_chunks, embeddings):
        store_text_chunk_in_mongodb(pdf_name, chunk, embeddings) #text_chunks+image_texts
     # Store images in MongoDB
    for img_path in images:
        store_image_in_mongodb(pdf_name, img_path)
    
    # Store tables in MongoDB
    for table in tables:
        store_table_in_mongodb(pdf_name, table)
        
    print(f"PDF {pdf_name} processed and stored in MongoDB.")

# Example Usage
if __name__ == "__main__":
    pdf_path = r"C:\Users\HP\Desktop\V3.0\NextGenNEXA_Doc_Assistant\Missile Guidance And Control Systems.pdf"
    process_pdf(pdf_path)
