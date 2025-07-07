# 🤖 RAG Chatbot – Ask Questions About Any File!

This is a **Retrieval-Augmented Generation (RAG)** chatbot powered by **LangChain** and **Streamlit**. Just **upload any file you want** — PDF, DOCX, TXT (you name it) — and ask it questions. The chatbot intelligently understands your documents and provides accurate answers.

---

## 📦 Features

- ✅ Upload any file (PDF, DOCX, TXT)
- 🧠 Ask natural-language questions about the file
- 🗃️ Stores and indexes documents with FAISS vector database
- 🔍 Uses HuggingFace embeddings for semantic search
- 🌐 Easy-to-use web interface via Streamlit

---

## 🚀 Quick Start (Local)

### Clone the repo
```bash
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot

---

🧠 **Folder Structure**
Upload any file (PDF, DOCX, TXT)

The app processes the file and splits it into chunks

Converts chunks into embeddings using HuggingFace

Stores them in a FAISS vectorstore

When you ask a question:

Retrieves the most relevant chunks

Uses an LLM to answer based on your document context

---

🙌 **Credits**

1.LangChain
2.FAISS
3.Streamlit
4.HuggingFace

