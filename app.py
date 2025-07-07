import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load PDF documents from the docs/ folder
loader = DirectoryLoader("docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Step 2: Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# ⚠️ Safety check
if len(docs) == 0:
    print("❌ No valid text chunks found. Make sure your PDFs contain text.")
    exit()

# Step 3: Use FREE local embeddings from HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# Step 4: Ask questions in terminal (retrieval only, no generation)
print("\n🤖 RAG Retriever Ready! (No OpenAI key needed)\nAsk about your documents or type 'exit'\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break

    results = vectorstore.similarity_search(query, k=2)
    print("\n📚 Top Matching Chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:500]}...\n")

    print("🔍 (These are the best chunks your future LLM could answer from.)\n")
