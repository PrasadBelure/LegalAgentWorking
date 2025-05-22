# utils/embedding_utils.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings(chunks):
    return model.encode(chunks)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_top_chunks(query, chunks, embeddings, index, k=5):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector), k)
    return [chunks[i] for i in I[0]]
