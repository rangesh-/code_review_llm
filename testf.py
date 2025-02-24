import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

FAISS_INDEX_FILE = "faiss_index.bin"
EMBEDDING_DIM = 768  # Set according to your embedding model
K = 5  # Number of similar documents to retrieve

# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to load FAISS index
def load_faiss_index():
    if os.path.exists(FAISS_INDEX_FILE):
        print("Loading FAISS index from disk...")
        return faiss.read_index(FAISS_INDEX_FILE)
    else:
        print("No FAISS index found. Creating a new one...")
        return faiss.IndexFlatL2(EMBEDDING_DIM)

faiss_index = load_faiss_index()

# Function to add embeddings to FAISS
def add_to_faiss(texts):
    global faiss_index
    embeddings = np.array([embed_model.encode(text) for text in texts], dtype=np.float32)

    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    print(f"Added {len(texts)} documents to FAISS and saved.")

# Function to search FAISS
def search_faiss(query):
    if faiss_index.ntotal == 0:
        print("FAISS index is empty. Add documents first!")
        return []
    
    query_embedding = np.array([embed_model.encode(query)], dtype=np.float32)
    distances, indices = faiss_index.search(query_embedding, K)

    print(f"Query: {query}")
    print(f"Similar Document Indices: {indices}")
    print(f"Distances: {distances}")

    return indices[0]
