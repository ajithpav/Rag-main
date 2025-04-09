# 🧠 PDF Question-Answering System with FAISS, BM25, and Groq

This project is a Flask web application that allows users to upload PDF documents and ask questions based on the content. It uses a combination of:

- 📄 **PDF text extraction** with `PyMuPDF (fitz)`
- 🧩 **Text chunking** for context window management
- 🔍 **Semantic search** using `FAISS` and `SentenceTransformers`
- 📊 **BM25 reranking** for relevance
- 🧠 **Answer generation** with `Groq` using `llama3-8b-8192`

---

## ✨ Features

- Upload a PDF and convert it into meaningful text chunks
- Perform vector-based search with FAISS
- Re-rank search results using BM25 for better context
- Generate accurate, context-bound answers with the Groq API
- Warns the user when the answer is not found in the document

---

## 📁 Project Structure

