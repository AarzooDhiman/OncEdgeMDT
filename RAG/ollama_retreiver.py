###run on langchain container
import re
from pathlib import Path
import os

import json

# Paths are relative to the repository root, so the script can be run from anywhere.
from pathlib import Path as _P
REPO_ROOT = _P(__file__).resolve().parents[1]
RAG_DIR = REPO_ROOT / 'RAG'
VECTOR_STORE = str(RAG_DIR / 'vector_store' / 'faiss_nomic_ollama')  # created by create_vector_store.py
OUT_DIR = REPO_ROOT / 'outputs' / 'ollama_retrieved'
OUT_DIR.mkdir(parents=True, exist_ok=True)
##reading embeddings

from langchain.vectorstores import FAISS


from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.load_local(
    VECTOR_STORE,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ques ="What is recommended for high-risk women regarding chemoprevention?"
retrieved_docs = retriever.invoke(ques)

#print (retrieved_docs)


def save_retrieved(question, docs):
	data = {
        "question": question,
        "docs": [
            {
                "content": d.page_content,
                "metadata": d.metadata
            }
            for d in docs
        ]
        }

	with open(OUT_DIR / 'test2.json' , "w") as f:
        	f.write(json.dumps(data) + "\n")
        

save_retrieved(ques, retrieved_docs)





