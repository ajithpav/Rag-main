import os
import fitz  
import faiss
import numpy as np
import groq
from flask import Flask, request, render_template, flash, session
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

index = None
chunks = []

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def chunk_text(text, chunk_size=300):
    """Splits text into smaller chunks."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def create_vector_store(chunks):
    """Creates FAISS vector store from text chunks."""
    embeddings = embedding_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, chunks

def retrieve_relevant_chunks(query, index, chunks, top_k=5):
    """Retrieves top-k relevant chunks from FAISS index."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def rerank_with_bm25(query, retrieved_chunks):
    """Reranks retrieved chunks using BM25."""
    tokenized_chunks = [chunk.split() for chunk in retrieved_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(query.split())
    ranked_chunks = [x for _, x in sorted(zip(scores, retrieved_chunks), reverse=True)]
    return ranked_chunks[:3] 

def generate_response(query, context):
    """Generates a response using Groq API."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        raise ValueError(" Missing Groq API key. Set 'GROQ_API_KEY' as an environment variable.")

    client = groq.Client(api_key=groq_api_key)

    prompt = f"""
    You are a highly accurate AI assistant. Answer the query strictly based on the provided context. 
    If the answer is not in the context, say 'I don't know'.

    Context: {context}

    Query: {query}

    Response:
    """

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

@app.route("/", methods=["GET", "POST"])
def home():
    """Handles file uploads and query processing."""
    global index, chunks
    response = ""

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file.filename.endswith(".pdf"):
                pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

                try:
                    print(f" Attempting to save file at: {pdf_path}") 
                    file.save(pdf_path)  
                    
                    if os.path.exists(pdf_path):  
                        print(f" File successfully saved at: {pdf_path}")
                    else:
                        print(f" File not found after saving. Check permissions.")

                    text = extract_text_from_pdf(pdf_path)
                    chunks = chunk_text(text)
                    index, chunks = create_vector_store(chunks)
                    session["index_created"] = True
                    flash(" PDF uploaded and processed successfully!", "success")

                except Exception as e:
                    flash(f" Error saving file: {str(e)}", "danger")
                    print(f" Error saving file: {str(e)}")

            else:
                flash(" Invalid file type. Please upload a PDF.", "warning")
        
        elif "query" in request.form:
            query = request.form["query"]

            if not session.get("index_created", False):
                flash(" Please upload a PDF first.", "danger")
            else:
                retrieved_chunks = retrieve_relevant_chunks(query, index, chunks)
                reranked_chunks = rerank_with_bm25(query, retrieved_chunks)
                response = generate_response(query, reranked_chunks)

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
