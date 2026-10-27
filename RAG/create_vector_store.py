import json

# Paths are relative to the repository root, so the script can be run from anywhere.
from pathlib import Path as _P
REPO_ROOT = _P(__file__).resolve().parents[1]
RAG_DIR = REPO_ROOT / 'RAG'
VECTOR_STORE = str(RAG_DIR / 'vector_store' / 'faiss_nomic_ollama')  # created by create_vector_store.py
from langchain_core.documents import Document
#from langchain.embeddings import SentenceTransformerEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

FILE_PATH = str(RAG_DIR / "merged_output.json")

with open(FILE_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []





for item in data:
    meta = item.get("metadata", {})

    section = meta.get("section") or ""
    subsection = meta.get("subsection") or ""
    subsubsection = meta.get("subsubsection") or ""
    content = item.get("content") or ""

    formatted_text = (
        f"Section: {section}\n"
        f"Subsection: {subsection}\n"
        f"Category: {subsubsection}\n\n"
        f"{content}"
    ).strip()

    doc = Document(
        page_content=formatted_text,
        metadata=meta
    )

    documents.append(doc)




print("Number of documents:", len(documents))

model_name = "nomic-embed-text"
embeddings = OllamaEmbeddings(
    model=model_name
)

#print (documents[:3])
#model_name = "BAAI/bge-m3"
#embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={"normalize_embeddings": True})



# Build vector store
vectorstore = FAISS.from_documents(documents, embeddings)

vectorstore.save_local(VECTOR_STORE)


