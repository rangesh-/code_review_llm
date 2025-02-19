import os
import psycopg2
import redis
import faiss
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from celery import Celery
from sentence_transformers import SentenceTransformer
from datasketch import MinHash, MinHashLSH
from contextlib import contextmanager
import streamlit as st

# Load environment variables
load_dotenv()

# PostgreSQL Configuration
POSTGRES_DB = "DVDRENTAL"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "password"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5433

# Redis Configuration
REDIS_HOST ="localhost"
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Hugging Face Embedding Model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Redis Cache Client with connection pooling
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)

# FAISS Index for Vector Storage (Requires Numeric IDs)
DIMENSION = 384  # Embedding size
faiss_index = faiss.IndexFlatL2(DIMENSION)  # Base index
faiss_index = faiss.IndexIDMap(faiss_index)  # Wrap with IndexIDMap to support custom IDs

# String-to-Numeric ID Mapping
doc_id_map = {}  # Maps string ID → Numeric FAISS ID
reverse_doc_id_map = {}  # Reverse mapping for retrieval

# MinHash LSH for Approximate Search
LSH_THRESHOLD = 0.8
LSH_NUM_PERM = 128
lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=LSH_NUM_PERM)

# Celery Configuration
celery_app = Celery('tasks', broker=f'redis://{REDIS_HOST}:{REDIS_PORT}/0')

# Context manager for PostgreSQL connections
@contextmanager
def get_db_connection():
    conn = psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
    )
    try:
        yield conn
    finally:
        conn.close()

# Fetch batch of records dynamically handling multiple columns
def fetch_documents(batch_size=1000, offset=0):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM film LIMIT %s OFFSET %s", (batch_size, offset))
        records = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(records, columns=columns)

# Generate Embeddings using Hugging Face Model
def get_embedding(text):
    return embedding_model.encode(text, convert_to_numpy=True)

# Prepare text for embedding by concatenating relevant columns
def prepare_text_for_embedding(row):
    return " ".join(str(row[col]) for col in row.index if isinstance(row[col], str))

# Store embedding in Redis cache with TTL (Time-to-Live)
def store_in_cache(query, embedding, ttl=3600):  # Default TTL: 1 hour
    redis_client.set(query, pickle.dumps(embedding), ex=ttl)

# Retrieve embedding from Redis cache
def get_from_cache(query):
    cached_result = redis_client.get(query)
    return pickle.loads(cached_result) if cached_result else None

# Store embedding in FAISS with a string ID mapping
def store_in_faiss(embedding, doc_id):
    numeric_id = len(doc_id_map) + 1  # Assign sequential numeric ID
    doc_id_map[doc_id] = numeric_id
    reverse_doc_id_map[numeric_id] = doc_id
    faiss_index.add_with_ids(np.array([embedding], dtype=np.float32), np.array([numeric_id], dtype=np.int64))

# Search FAISS for relevant embeddings
def search_faiss(query_embedding, top_k=5):
    distances, indices = faiss_index.search(np.array([query_embedding], dtype=np.float32), top_k)
    return [reverse_doc_id_map.get(idx, None) for idx in indices[0] if idx in reverse_doc_id_map]

# Add Document to MinHash LSH
def add_to_lsh(doc_id, text):
    minhash = MinHash(num_perm=LSH_NUM_PERM)
    for word in text.split():
        minhash.update(word.encode('utf8'))
    lsh.insert(doc_id, minhash)

# Search in LSH
def search_lsh(query):
    minhash = MinHash(num_perm=LSH_NUM_PERM)
    for word in query.split():
        minhash.update(word.encode('utf8'))
    return lsh.query(minhash)

# Asynchronous task for embedding computation
@celery_app.task
def compute_embedding_task(text):
    return get_embedding(text)

# Query Processing Pipeline
def process_query(query):
    # 1. Check Redis Cache
    cached_embedding = get_from_cache(query)
    if cached_embedding:
        return cached_embedding

    # 2. Perform Approximate Search via MinHash LSH
    similar_docs = search_lsh(query)
    if similar_docs:
        return [faiss_index.reconstruct(doc_id_map[doc_id]) for doc_id in similar_docs if doc_id in doc_id_map]

    # 3. On-Demand Embedding Computation
    embedding = compute_embedding_task.apply_async(args=[query]).get()

    # 4. Store Results
    store_in_cache(query, embedding)
    return embedding

# Streamlit App
def main():
    st.title("Hybrid Search System")
    st.sidebar.header("Options")

    # Button to load data from the database
    if st.sidebar.button("Load Data from Database"):
        st.write("Loading data from PostgreSQL...")
        offset = 0
        batch_size = 1000
        
        while True:
            df = fetch_documents(batch_size=batch_size, offset=offset)
            if df.empty:
                break
            
            st.write(f"Processing batch {offset} to {offset + batch_size}")
            
            for _, row in df.iterrows():
                doc_id = str(row['title'])  # Ensure ID is a string
                text = prepare_text_for_embedding(row)
                embedding = get_embedding(text)
                store_in_faiss(embedding, doc_id)
                add_to_lsh(doc_id, text)

            offset += batch_size

        st.success("Data ingestion complete!")

    # Input for user query
    query = st.text_input("Enter your query:")
    if query:
        st.write(f"Processing query: {query}")
        embedding = process_query(query)
        st.write("Query Embedding:", embedding)

        # Display similar documents
        similar_docs = search_faiss(embedding)
        if similar_docs:
            st.subheader("Similar Documents:")
            for doc_id in similar_docs:
                st.write(f"Document ID: {doc_id}")

# Run the Streamlit app
if __name__ == "__main__":
    main()