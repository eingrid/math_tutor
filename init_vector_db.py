import chromadb
import numpy as np
from langchain_chroma import Chroma
import os 
import json
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from uuid import uuid4
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v3",model_kwargs = {"trust_remote_code": True})

problems = []

train_path = "MATH_TRANSLATED/train"
for topic in os.listdir(train_path):
    for problem in os.listdir(train_path+f"/{topic}"):
        problem_path = train_path+f"/{topic}" + f"/{problem}"
        with open(problem_path) as f:
            problem_dict = json.load(f)
            problems.append(Document(page_content=problem_dict['task'],
                            metadata={"topic": topic,
                                      "solution":problem_dict['solution'],
                                      "answer": problem_dict['answer']
                                    }))
            

vector_store = Chroma(
    collection_name="test",
    persist_directory="./chroma_langchain_db",
    embedding_function=embeddings,
)
uuids = [str(uuid4()) for _ in range(len(problems))]


vector_store.add_documents(documents=problems, ids=uuids)


res = vector_store.similarity_search("sample math problem",k=5)

print(res)