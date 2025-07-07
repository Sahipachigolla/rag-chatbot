import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Setup Streamlit page
st.set_page_config(page_title="📄 Local RAG Chatbot", layout="centered")
st.title("🧠 Local RAG Chatbot (No OpenAI, Fully Offline)")

@st.cache_resource
def load_vectorstore():
    loader = DirectoryLoader("docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documents)
    if len(docs) == 0:
        st.error("❌ No valid text found in PDF. Upload text-based PDFs.")
        st.stop()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

# Load vectorstore once
vectorstore = load_vectorstore()

# Ask input
user_query = st.text_input("💬 Ask a question about your PDFs:")

if user_query:
    with st.spinner("Searching..."):
        results = vectorstore.similarity_search(user_query, k=2)
    st.subheader("📚 Top Matching Chunks:")
    for i, doc in enumerate(results, 1):
        st.markdown(f"**{i}.** {doc.page_content[:700]}...")

    st.info("🔍 Retrieved from local PDF using embeddings.")
