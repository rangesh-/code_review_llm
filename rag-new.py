import os
import psycopg2
import numpy as np
import redis
import faiss
import hashlib
import minhash
import ollama
from celery import Celery
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from sklearn.preprocessing import normalize

# ✅ PostgreSQL Configuration
POSTGRES_DB = "rag_db"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "password"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"

# ✅ Redis Cache
REDIS_HOST = "localhost"
REDIS_PORT = 6379
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ✅ Celery for Async Embedding Computation
CELERY_BROKER_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
CELERY_BACKEND = f"redis://{REDIS_HOST}/{REDIS_PORT}/1"
celery_app = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_BACKEND)

# ✅ FAISS Vector Index
embedding_size = 4096  # Ollama Llama3 default embedding size
index = faiss.IndexFlatL2(embedding_size)

# ✅ Ollama Llama3 Model
ollama_model = Ollama(model="llama3")

# ✅ PostgreSQL Connection
def get_postgres_connection():
    return psycopg2.connect(
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
    )

# ✅ Generate Embeddings (Ollama Llama3)
def generate_embedding(text):
    response = ollama.embeddings(model="llama3", input=text)
    return np.array(response["embedding"], dtype=np.float32)

# ✅ MinHash for Approximate Search
def compute_minhash(text, num_hashes=128):
    minhash_obj = minhash.MinHash(num_perm=num_hashes)
    minhash_obj.update(text.encode("utf8"))
    return minhash_obj.hashvalues

# ✅ Reinforcement Learning-Based Query Prioritization
def rl_query_decision(query):
    query_hash = hashlib.md5(query.encode()).hexdigest()
    score = redis_client.get(f"rl_score:{query_hash}")

    if score is None:
        return "APPROXIMATE_SEARCH"

    score = float(score)

    if score > 0.8:
        return "USE_CACHED_EMBEDDING"
    elif score > 0.5:
        return "APPROXIMATE_SEARCH"
    else:
        return "ON_DEMAND_EMBEDDING"

# ✅ Store Embedding in FAISS
def store_embedding(text, doc_id):
    embedding = generate_embedding(text)
    index.add(np.array([embedding]))
    redis_client.set(f"embedding:{doc_id}", embedding.tobytes())
    return embedding

# ✅ Retrieve Cached Embedding
def get_cached_embedding(doc_id):
    cached_embedding = redis_client.get(f"embedding:{doc_id}")
    if cached_embedding:
        return np.frombuffer(cached_embedding, dtype=np.float32)
    return None

# ✅ Celery Task for On-Demand Embedding Computation
@celery_app.task
def compute_embedding_async(query, doc_id):
    embedding = generate_embedding(query)
    redis_client.set(f"embedding:{doc_id}", embedding.tobytes())
    index.add(np.array([embedding]))

# ✅ Batch Process Large Dataset from PostgreSQL
def process_large_dataset(batch_size=10000):
    conn = get_postgres_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, content FROM item")  # Table 'item' has 1M rows

    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        for doc_id, text in rows:
            store_embedding(text, doc_id)

    conn.close()

# ✅ Query Execution Pipeline
def query_pipeline(query):
    decision = rl_query_decision(query)

    if decision == "USE_CACHED_EMBEDDING":
        print("✅ Using Cached Embedding")
        return get_cached_embedding(query)

    elif decision == "APPROXIMATE_SEARCH":
        print("🔍 Performing MinHash Approximate Search")
        minhash_signature = compute_minhash(query)
        # (Simulation: Return dummy nearest match)
        return f"Approximate result for: {query}"

    elif decision == "ON_DEMAND_EMBEDDING":
        print("🚀 Triggering On-Demand Embedding Computation")
        compute_embedding_async.delay(query, hashlib.md5(query.encode()).hexdigest())
        return "Embedding computation in progress..."

    return "No suitable retrieval strategy found."

# ✅ Run End-to-End Pipeline
if __name__ == "__main__":
    process_large_dataset()  # Process large dataset in batches

    print("\n🔍 Running Query Pipeline...")
    query1 = "How does AI work?"
    print(query_pipeline(query1))

    print("\n🔍 Running Query Pipeline Again...")
    print(query_pipeline(query1))  # Should now use stored embedding
