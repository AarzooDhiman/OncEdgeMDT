###run on langchain container
import re
from pathlib import Path
import os

import json
##reading embeddings

from langchain.vectorstores import FAISS


from langchain_community.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.load_local(
    '/home/data/LLM/vector_store/faiss_nomic_ollama',
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

	with open(f'/home/data/LLM/LLM_outputs/medgemma/real_case/medgemma_8kxl/ollama_retrieved/test2.json' , "w") as f:
        	f.write(json.dumps(data) + "\n")
        

save_retrieved(ques, retrieved_docs)





