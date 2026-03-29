# Databricks notebook source
# MAGIC %md
# MAGIC # Step 3: Databricks Vector Search Index
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Creates a Databricks Vector Search index on the enriched facilities table
# MAGIC 2. Uses Delta Sync to keep the index in sync with the source table
# MAGIC 3. Uses the built-in Databricks embedding model (or sentence-transformers as fallback)
# MAGIC 4. Tests semantic search queries
# MAGIC
# MAGIC **Run notebooks 00, 01, 02 first!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3A. Configuration

# COMMAND ----------

import time
from databricks.vector_search.client import VectorSearchClient

# ============================================================
# CONFIG: Must match what was set in 00_setup
# ============================================================
CATALOG = "hackathon_vf"
SCHEMA = "healthcare"
TABLE_PREFIX = f"{CATALOG}.{SCHEMA}"
VS_ENDPOINT = "vf_facility_search"

try:
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql(f"USE SCHEMA {SCHEMA}")
except Exception:
    CATALOG = "hive_metastore"
    SCHEMA = "hackathon"
    TABLE_PREFIX = f"{CATALOG}.{SCHEMA}"

SOURCE_TABLE = f"{TABLE_PREFIX}.facilities_enriched"
VS_INDEX_NAME = f"{TABLE_PREFIX}.facilities_vs_index"

print(f"Source table: {SOURCE_TABLE}")
print(f"VS Index: {VS_INDEX_NAME}")
print(f"VS Endpoint: {VS_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3B. Wait for Vector Search Endpoint to be Ready

# COMMAND ----------

vsc = VectorSearchClient()

# Wait for endpoint to be ready (can take a few minutes on first creation)
for i in range(20):
    try:
        endpoint = vsc.get_endpoint(VS_ENDPOINT)
        state = endpoint.get("endpoint_status", {}).get("state", "UNKNOWN")
        print(f"Endpoint status: {state}")
        if state == "ONLINE":
            print("Endpoint is ready!")
            break
    except Exception as e:
        print(f"Waiting for endpoint... ({e})")
    time.sleep(30)
else:
    print("WARNING: Endpoint may not be ready. Continuing anyway...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3C. Create Vector Search Index
# MAGIC
# MAGIC We use a **Delta Sync Index** that automatically stays in sync with the source Delta table.
# MAGIC The `search_text` column is embedded using Databricks' built-in embedding model.

# COMMAND ----------

# Check if index already exists
try:
    existing_index = vsc.get_index(VS_ENDPOINT, VS_INDEX_NAME)
    print(f"Index '{VS_INDEX_NAME}' already exists!")
    print(f"Status: {existing_index.describe()}")
    INDEX_EXISTS = True
except Exception:
    INDEX_EXISTS = False
    print(f"Index '{VS_INDEX_NAME}' does not exist. Creating...")

# COMMAND ----------

if not INDEX_EXISTS:
    try:
        # Create Delta Sync Index with Databricks-managed embeddings
        index = vsc.create_delta_sync_index(
            endpoint_name=VS_ENDPOINT,
            index_name=VS_INDEX_NAME,
            source_table_name=SOURCE_TABLE,
            primary_key="pk_unique_id",
            pipeline_type="TRIGGERED",  # Sync on demand (use "CONTINUOUS" for auto-sync)
            embedding_source_column="search_text",
            embedding_model_endpoint_name="databricks-bge-large-en"  # Built-in embedding model
        )
        print(f"Vector Search index created: {VS_INDEX_NAME}")
        print("Index is syncing... this may take 5-10 minutes.")
    except Exception as e:
        error_msg = str(e)
        print(f"Databricks BGE model failed: {error_msg}")
        print("\nTrying with databricks-gte-large-en instead...")
        try:
            index = vsc.create_delta_sync_index(
                endpoint_name=VS_ENDPOINT,
                index_name=VS_INDEX_NAME,
                source_table_name=SOURCE_TABLE,
                primary_key="pk_unique_id",
                pipeline_type="TRIGGERED",
                embedding_source_column="search_text",
                embedding_model_endpoint_name="databricks-gte-large-en"
            )
            print(f"Vector Search index created with GTE model: {VS_INDEX_NAME}")
        except Exception as e2:
            print(f"\nDatabricks embedding models not available: {e2}")
            print("Falling back to self-managed embeddings (see section 3C-ALT below)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3C-ALT. Fallback: Self-Managed Embeddings with sentence-transformers
# MAGIC
# MAGIC If Databricks embedding models are not available, we compute embeddings ourselves
# MAGIC and create a Direct Access Index.

# COMMAND ----------

# Only run this if the Delta Sync index creation failed above
FALLBACK_MODE = False

try:
    vsc.get_index(VS_ENDPOINT, VS_INDEX_NAME)
    print("Delta Sync index exists. Skipping fallback.")
except Exception:
    print("Using fallback: self-managed embeddings with sentence-transformers")
    FALLBACK_MODE = True

# COMMAND ----------

if FALLBACK_MODE:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # Load embedding model
    print("Loading sentence-transformers model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDING_DIM = 384

    # Load data
    pdf = spark.table(SOURCE_TABLE).select("pk_unique_id", "search_text").toPandas()
    texts = pdf["search_text"].fillna("").tolist()
    ids = pdf["pk_unique_id"].tolist()

    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} facilities...")
    embeddings = embed_model.encode(texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # Create a Direct Access Index
    try:
        index = vsc.create_direct_access_index(
            endpoint_name=VS_ENDPOINT,
            index_name=VS_INDEX_NAME,
            primary_key="pk_unique_id",
            embedding_dimension=EMBEDDING_DIM,
            embedding_vector_column="embedding",
            schema={
                "pk_unique_id": "string",
                "search_text": "string",
                "embedding": "array<float>"
            }
        )
        print(f"Direct Access index created: {VS_INDEX_NAME}")

        # Upsert embeddings in batches
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            batch = [
                {
                    "pk_unique_id": ids[j],
                    "search_text": texts[j],
                    "embedding": embeddings[j].tolist()
                }
                for j in range(i, min(i + batch_size, len(ids)))
            ]
            index.upsert(batch)
            print(f"  Upserted batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size}")

        print(f"All {len(ids)} embeddings uploaded to Vector Search index")
    except Exception as e:
        print(f"Direct Access index creation failed: {e}")
        print("\nFalling back to local FAISS index...")
        
        # Ultimate fallback: FAISS
        import faiss
        import pickle
        import tempfile
        
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        faiss_index.add(embeddings.astype(np.float32))
        
        local_dir = tempfile.mkdtemp()
        faiss.write_index(faiss_index, f"{local_dir}/facility_index.faiss")
        with open(f"{local_dir}/facility_metadata.pkl", "wb") as f:
            pickle.dump({"texts": texts, "ids": ids, "dataframe": pdf}, f)
        
        dbutils.fs.mkdirs("/FileStore/hackathon/vector_store")
        dbutils.fs.cp(f"file:{local_dir}/facility_index.faiss", "/FileStore/hackathon/vector_store/facility_index.faiss")
        dbutils.fs.cp(f"file:{local_dir}/facility_metadata.pkl", "/FileStore/hackathon/vector_store/facility_metadata.pkl")
        print("FAISS index saved to /FileStore/hackathon/vector_store/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3D. Wait for Index Sync (Delta Sync only)

# COMMAND ----------

if not FALLBACK_MODE:
    print("Waiting for index to sync...")
    for i in range(30):
        try:
            idx = vsc.get_index(VS_ENDPOINT, VS_INDEX_NAME)
            status = idx.describe()
            state = status.get("status", {}).get("detailed_state", "UNKNOWN")
            num_rows = status.get("status", {}).get("num_rows_updated", 0)
            print(f"  [{i+1}] State: {state}, Rows indexed: {num_rows}")
            if state == "ONLINE_NO_PENDING_UPDATE":
                print("Index is fully synced!")
                break
            if "ONLINE" in str(state) and num_rows > 0:
                print("Index is online with data!")
                break
        except Exception as e:
            print(f"  [{i+1}] Checking... ({e})")
        time.sleep(20)
    else:
        print("Index may still be syncing. You can continue and check back later.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3E. Test Semantic Search

# COMMAND ----------

def search_facilities(query: str, k: int = 5, filters: dict = None):
    """Search facilities using Databricks Vector Search."""
    try:
        idx = vsc.get_index(VS_ENDPOINT, VS_INDEX_NAME)
        
        # Build search kwargs
        search_kwargs = {
            "query_text": query,
            "columns": ["pk_unique_id", "search_text"],
            "num_results": k,
        }
        if filters:
            search_kwargs["filters"] = filters
        
        results = idx.similarity_search(**search_kwargs)
        return results
    except Exception as e:
        print(f"Vector Search query failed: {e}")
        print("Index may still be syncing. Try again in a few minutes.")
        return None

# COMMAND ----------

# Test 1: Equipment search
print("=== Query: 'hospitals with CT scanners' ===")
results = search_facilities("hospitals with CT scanners")
if results:
    for row in results.get("result", {}).get("data_array", []):
        print(f"  ID: {row[0]}, Score: {row[-1]:.3f}")
        print(f"  Text: {str(row[1])[:150]}...")
        print()

# COMMAND ----------

# Test 2: Regional search
print("=== Query: 'surgical facilities in Northern Ghana' ===")
results = search_facilities("surgical facilities in Northern Ghana")
if results:
    for row in results.get("result", {}).get("data_array", []):
        print(f"  ID: {row[0]}, Score: {row[-1]:.3f}")
        print(f"  Text: {str(row[1])[:150]}...")
        print()

# COMMAND ----------

# Test 3: Service search
print("=== Query: 'dental clinics in Accra' ===")
results = search_facilities("dental clinics in Accra")
if results:
    for row in results.get("result", {}).get("data_array", []):
        print(f"  ID: {row[0]}, Score: {row[-1]:.3f}")
        print(f"  Text: {str(row[1])[:150]}...")
        print()

# COMMAND ----------

# Config for later use
print(f"Index: {VS_INDEX_NAME}")
print(f"Fallback mode: {FALLBACK_MODE}")
print("Next: Run notebook 04_rag_chain")
