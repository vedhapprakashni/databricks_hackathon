# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3: Embedding and Vector Store
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads the enriched facility data
# MAGIC 2. Generates embeddings using sentence-transformers (runs locally on the cluster)
# MAGIC 3. Builds a FAISS vector index for fast semantic search
# MAGIC 4. Saves the index and metadata to DBFS for reuse
# MAGIC
# MAGIC **Run notebooks 00, 01, 02 first!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3A. Load Data and Prepare Texts

# COMMAND ----------

import json
import numpy as np
import pickle
from pyspark.sql import functions as F

# Load enriched data
df = spark.table("hackathon.facilities_enriched")
print(f"Loaded {df.count()} facilities")

# Collect to pandas for embedding (FAISS works with numpy/pandas)
pdf = df.select(
    "pk_unique_id", "name", "search_text", "address_city",
    "address_stateOrRegion", "facilityTypeId", "specialties",
    "procedure", "equipment", "capability", "description",
    "numberDoctors", "capacity", "operatorTypeId",
    "num_procedures", "num_equipment", "num_capabilities", "num_specialties",
    "flag_procedures_no_doctors", "flag_capacity_no_equipment",
    "flag_clinic_claims_surgery", "flag_too_many_specialties", "flag_sparse_record"
).toPandas()

print(f"Converted {len(pdf)} rows to pandas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3B. Generate Embeddings
# MAGIC
# MAGIC Using `all-MiniLM-L6-v2` -- a small, fast model that produces 384-dimension embeddings.
# MAGIC Runs entirely on the cluster CPU, no API calls needed.

# COMMAND ----------

from sentence_transformers import SentenceTransformer

# Load embedding model (downloads ~90MB on first run, cached after that)
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")

# COMMAND ----------

# Prepare text for each facility
texts = pdf["search_text"].fillna("").tolist()

# Filter out empty texts (keep track of indices)
valid_indices = [i for i, t in enumerate(texts) if len(t.strip()) > 10]
valid_texts = [texts[i] for i in valid_indices]
valid_pdf = pdf.iloc[valid_indices].reset_index(drop=True)

print(f"Facilities with valid search text: {len(valid_texts)} / {len(texts)}")

# COMMAND ----------

# Generate embeddings (batched for efficiency)
print("Generating embeddings... (this may take 1-2 minutes)")
embeddings = embed_model.encode(
    valid_texts,
    show_progress_bar=True,
    batch_size=64,
    normalize_embeddings=True  # Normalize for cosine similarity
)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3C. Build FAISS Index

# COMMAND ----------

import faiss

# Build the index using Inner Product (since embeddings are normalized, this = cosine similarity)
dimension = embeddings.shape[1]  # 384
index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
index.add(embeddings.astype(np.float32))

print(f"FAISS index built with {index.ntotal} vectors of dimension {dimension}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3D. Test Semantic Search

# COMMAND ----------

def search_facilities(query: str, k: int = 5):
    """Search for facilities matching a natural language query."""
    # Encode query
    query_embedding = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    
    # Search FAISS
    scores, indices = index.search(query_embedding, k)
    
    # Get results
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx >= 0 and idx < len(valid_pdf):
            row = valid_pdf.iloc[idx]
            results.append({
                "rank": rank + 1,
                "score": float(score),
                "name": row["name"],
                "city": row["address_city"],
                "region": row["address_stateOrRegion"],
                "type": row["facilityTypeId"],
                "specialties": row["specialties"],
                "doctors": row["numberDoctors"],
                "capacity": row["capacity"],
            })
    return results

# COMMAND ----------

# Test query 1: Equipment search
print("=== Query: 'hospitals with CT scanners' ===")
results = search_facilities("hospitals with CT scanners")
for r in results:
    print(f"  {r['rank']}. {r['name']} ({r['city']}, {r['region']}) - Score: {r['score']:.3f}")

# COMMAND ----------

# Test query 2: Regional search
print("\n=== Query: 'surgical facilities in Northern Ghana' ===")
results = search_facilities("surgical facilities in Northern Ghana")
for r in results:
    print(f"  {r['rank']}. {r['name']} ({r['city']}, {r['region']}) - Score: {r['score']:.3f}")

# COMMAND ----------

# Test query 3: Service search
print("\n=== Query: 'dental clinics in Accra' ===")
results = search_facilities("dental clinics in Accra")
for r in results:
    print(f"  {r['rank']}. {r['name']} ({r['city']}, {r['region']}) Type: {r['type']} - Score: {r['score']:.3f}")

# COMMAND ----------

# Test query 4: Capability search
print("\n=== Query: 'emergency medicine and trauma care' ===")
results = search_facilities("emergency medicine and trauma care")
for r in results:
    print(f"  {r['rank']}. {r['name']} ({r['city']}, {r['region']}) - Score: {r['score']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3E. Save Index and Metadata to DBFS

# COMMAND ----------

import os
import tempfile

# Create a temp directory to save files, then copy to DBFS
local_dir = tempfile.mkdtemp()

# Save FAISS index
faiss_path = os.path.join(local_dir, "facility_index.faiss")
faiss.write_index(index, faiss_path)
print(f"FAISS index saved locally: {faiss_path}")

# Save the metadata (facility data aligned with index)
meta_path = os.path.join(local_dir, "facility_metadata.pkl")
metadata = {
    "texts": valid_texts,
    "dataframe": valid_pdf,
    "embedding_model": "all-MiniLM-L6-v2",
    "dimension": dimension,
    "total_vectors": index.ntotal,
}
with open(meta_path, "wb") as f:
    pickle.dump(metadata, f)
print(f"Metadata saved locally: {meta_path}")

# Copy to DBFS
dbfs_dir = "/FileStore/hackathon/vector_store"
dbutils.fs.mkdirs(dbfs_dir)
dbutils.fs.cp(f"file:{faiss_path}", f"{dbfs_dir}/facility_index.faiss")
dbutils.fs.cp(f"file:{meta_path}", f"{dbfs_dir}/facility_metadata.pkl")

print(f"\nFiles saved to DBFS: {dbfs_dir}/")
print(f"  facility_index.faiss ({index.ntotal} vectors)")
print(f"  facility_metadata.pkl ({len(valid_pdf)} facilities)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3F. Verify Save

# COMMAND ----------

# Verify files exist on DBFS
files = dbutils.fs.ls(dbfs_dir)
for f in files:
    print(f"  {f.name}: {f.size} bytes")

print("\nStep 3 complete! Vector store is ready.")
print("Next: Run notebook 04_rag_chain.py")
