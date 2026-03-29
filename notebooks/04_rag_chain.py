# Databricks notebook source
# MAGIC %md
# MAGIC # Step 4: RAG Chain (Query + LLM Answer with Citations)
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads the FAISS index and metadata from DBFS
# MAGIC 2. Builds a RAG pipeline: question -> vector search -> LLM answer
# MAGIC 3. Returns answers with facility-level citations
# MAGIC 4. Tests against the Must-Have questions from the agent questions doc
# MAGIC
# MAGIC **Run notebooks 00, 01, 02, 03 first!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4A. Load Vector Store and LLM

# COMMAND ----------

import os
import json
import pickle
import tempfile
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

# ============================================================
# SET YOUR GROQ API KEY
# ============================================================
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"  # <-- Replace!
# ============================================================

# COMMAND ----------

# Load FAISS index from DBFS
dbfs_dir = "/FileStore/hackathon/vector_store"
local_dir = tempfile.mkdtemp()

# Copy from DBFS to local
dbutils.fs.cp(f"{dbfs_dir}/facility_index.faiss", f"file:{local_dir}/facility_index.faiss")
dbutils.fs.cp(f"{dbfs_dir}/facility_metadata.pkl", f"file:{local_dir}/facility_metadata.pkl")

# Load FAISS index
index = faiss.read_index(os.path.join(local_dir, "facility_index.faiss"))
print(f"FAISS index loaded: {index.ntotal} vectors")

# Load metadata
with open(os.path.join(local_dir, "facility_metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

texts = metadata["texts"]
facility_df = metadata["dataframe"]
print(f"Metadata loaded: {len(facility_df)} facilities")

# COMMAND ----------

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded")

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0,
    max_tokens=2048
)
print("Groq LLM initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4B. Build the RAG Pipeline

# COMMAND ----------

def format_facility_context(row, rank):
    """Format a single facility's data for the LLM context."""
    parts = [f"[Facility {rank}]"]
    parts.append(f"Name: {row.get('name', 'Unknown')}")

    if row.get("address_city"):
        parts.append(f"City: {row['address_city']}")
    if row.get("address_stateOrRegion"):
        parts.append(f"Region: {row['address_stateOrRegion']}")
    if row.get("facilityTypeId"):
        parts.append(f"Type: {row['facilityTypeId']}")
    if row.get("operatorTypeId"):
        parts.append(f"Operator: {row['operatorTypeId']}")
    if row.get("numberDoctors") and str(row.get("numberDoctors")) != "nan":
        parts.append(f"Doctors: {int(row['numberDoctors'])}")
    if row.get("capacity") and str(row.get("capacity")) != "nan":
        parts.append(f"Bed Capacity: {int(row['capacity'])}")

    # Parse JSON array fields
    for label, field in [("Specialties", "specialties"), 
                          ("Procedures", "procedure"),
                          ("Equipment", "equipment"), 
                          ("Capabilities", "capability")]:
        val = row.get(field, "[]")
        if val and val not in ("null", "[]", None, "nan"):
            try:
                items = json.loads(val)
                if isinstance(items, list) and items:
                    parts.append(f"{label}: {', '.join(str(i) for i in items)}")
            except (json.JSONDecodeError, TypeError):
                if str(val).strip() not in ("", "nan"):
                    parts.append(f"{label}: {val}")

    if row.get("description") and str(row.get("description")) not in ("None", "nan", ""):
        desc = str(row["description"])[:300]  # Truncate long descriptions
        parts.append(f"Description: {desc}")

    return "\n".join(parts)

# COMMAND ----------

# RAG prompt template
RAG_PROMPT = ChatPromptTemplate.from_template("""You are a healthcare facility intelligence agent for Ghana.
You help the Virtue Foundation identify medical gaps and coordinate healthcare resources.

Answer the user's question using ONLY the facility data provided below.
Be specific: include facility names, locations, and data points in your answer.
If the data does not contain enough information to answer fully, say so honestly.

For each claim you make, cite the facility by name (e.g., "[Facility Name, Region]").

FACILITY DATA (from vector search):
{context}

ADDITIONAL SQL DATA (if available):
{sql_context}

USER QUESTION: {question}

DETAILED ANSWER:""")

# COMMAND ----------

def search_and_answer(question: str, k: int = 10, include_sql: bool = True):
    """
    Full RAG pipeline:
    1. Embed the question
    2. Search FAISS for relevant facilities
    3. Optionally run SQL for structured data
    4. Send context + question to LLM
    5. Return answer with sources
    """
    # Step 1: Embed and search
    query_embedding = embed_model.encode([question], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query_embedding, k)

    # Step 2: Build context from search results
    context_parts = []
    sources = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx >= 0 and idx < len(facility_df):
            row = facility_df.iloc[idx].to_dict()
            context_parts.append(format_facility_context(row, rank))
            sources.append({
                "name": row.get("name", "Unknown"),
                "pk_unique_id": row.get("pk_unique_id", ""),
                "region": row.get("address_stateOrRegion", ""),
                "score": float(score)
            })

    context = "\n\n".join(context_parts)

    # Step 3: SQL context for structured queries
    sql_context = "No SQL data available."
    if include_sql:
        sql_context = get_sql_context(question)

    # Step 4: Generate answer with LLM
    chain = RAG_PROMPT | llm
    response = chain.invoke({
        "context": context,
        "sql_context": sql_context,
        "question": question
    })

    return {
        "answer": response.content,
        "sources": sources,
        "num_sources": len(sources)
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4C. SQL Context Helper
# MAGIC
# MAGIC For structured queries (counts, aggregations), we also pull data from Spark SQL
# MAGIC to give the LLM accurate numbers.

# COMMAND ----------

def get_sql_context(question: str) -> str:
    """Run relevant SQL queries based on the question to provide structured context."""
    context_parts = []
    q_lower = question.lower()

    try:
        # If asking about counts or "how many"
        if any(word in q_lower for word in ["how many", "count", "number of", "total"]):
            # Total facilities
            total = spark.sql("SELECT COUNT(*) as total FROM hackathon.facilities_enriched").collect()[0]["total"]
            context_parts.append(f"Total facilities in database: {total}")

            # By type
            type_counts = spark.sql("""
                SELECT facilityTypeId, COUNT(*) as count 
                FROM hackathon.facilities_enriched 
                GROUP BY facilityTypeId 
                ORDER BY count DESC
            """).collect()
            type_str = ", ".join([f"{r['facilityTypeId']}: {r['count']}" for r in type_counts])
            context_parts.append(f"Facilities by type: {type_str}")

        # If asking about regions
        if any(word in q_lower for word in ["region", "area", "where", "location", "geographic"]):
            region_counts = spark.sql("""
                SELECT address_stateOrRegion, COUNT(*) as count, 
                       SUM(COALESCE(numberDoctors, 0)) as doctors
                FROM hackathon.facilities_enriched 
                GROUP BY address_stateOrRegion 
                ORDER BY count DESC
            """).collect()
            for r in region_counts:
                context_parts.append(f"Region {r['address_stateOrRegion']}: {r['count']} facilities, {r['doctors']} doctors")

        # If asking about specialties
        if any(word in q_lower for word in ["cardiology", "surgery", "pediatric", "emergency", 
                                             "dental", "specialty", "specialties", "specialization"]):
            # Find the specific specialty mentioned
            specialties_to_check = [
                "cardiology", "surgery", "pediatrics", "emergencyMedicine",
                "gynecologyAndObstetrics", "ophthalmology", "dentistry",
                "internalMedicine", "orthopedicSurgery", "radiology"
            ]
            for spec in specialties_to_check:
                if spec.lower() in q_lower or spec.lower().replace("and", "").replace("medicine", "") in q_lower:
                    count = spark.sql(f"""
                        SELECT COUNT(*) as count 
                        FROM hackathon.facilities_enriched 
                        WHERE LOWER(specialties) LIKE '%{spec.lower()}%'
                    """).collect()[0]["count"]
                    context_parts.append(f"Facilities with {spec}: {count}")

        # If asking about anomalies
        if any(word in q_lower for word in ["anomal", "suspicious", "mismatch", "unrealistic", "inconsisten"]):
            anomaly_counts = spark.sql("""
                SELECT 
                    SUM(CASE WHEN flag_procedures_no_doctors THEN 1 ELSE 0 END) as procedures_no_docs,
                    SUM(CASE WHEN flag_capacity_no_equipment THEN 1 ELSE 0 END) as capacity_no_equip,
                    SUM(CASE WHEN flag_clinic_claims_surgery THEN 1 ELSE 0 END) as clinic_surgery,
                    SUM(CASE WHEN flag_too_many_specialties THEN 1 ELSE 0 END) as too_many_specs,
                    SUM(CASE WHEN flag_sparse_record THEN 1 ELSE 0 END) as sparse
                FROM hackathon.facilities_enriched
            """).collect()[0]
            context_parts.append(f"Anomaly flags - Procedures but no doctors: {anomaly_counts['procedures_no_docs']}")
            context_parts.append(f"Anomaly flags - High capacity but no equipment: {anomaly_counts['capacity_no_equip']}")
            context_parts.append(f"Anomaly flags - Clinics claiming surgery: {anomaly_counts['clinic_surgery']}")
            context_parts.append(f"Anomaly flags - Too many specialties for size: {anomaly_counts['too_many_specs']}")
            context_parts.append(f"Anomaly flags - Sparse records: {anomaly_counts['sparse']}")

        # If asking about medical deserts
        if any(word in q_lower for word in ["desert", "underserved", "gap", "lacking", "shortage"]):
            deserts = spark.sql("""
                SELECT address_stateOrRegion, facility_count, total_doctors, total_beds,
                       has_surgery, has_emergency, desert_score
                FROM hackathon.regional_analysis
                ORDER BY desert_score DESC
                LIMIT 5
            """).collect()
            context_parts.append("Top underserved regions (highest desert score):")
            for r in deserts:
                context_parts.append(
                    f"  {r['address_stateOrRegion']}: score={r['desert_score']}, "
                    f"facilities={r['facility_count']}, doctors={r['total_doctors']}, "
                    f"surgery={'Yes' if r['has_surgery']>0 else 'No'}"
                )
    except Exception as e:
        context_parts.append(f"SQL context unavailable: {str(e)}")

    return "\n".join(context_parts) if context_parts else "No structured data pulled."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4D. Test: Must-Have Questions from Agent Questions Doc

# COMMAND ----------

# Test Question 1.1: Basic facility count
print("=" * 70)
print("Q1.1: How many hospitals in the dataset offer cardiology services?")
print("=" * 70)
result = search_and_answer("How many hospitals in the dataset offer cardiology services?")
print(f"\nAnswer:\n{result['answer']}")
print(f"\nSources used: {result['num_sources']}")
for s in result['sources'][:3]:
    print(f"  - {s['name']} ({s['region']}) [score: {s['score']:.3f}]")

# COMMAND ----------

# Test Question 1.3: Facility lookup
print("=" * 70)
print("Q1.3: What specific services does Korle Bu Teaching Hospital offer?")
print("=" * 70)
result = search_and_answer("What specific services does Korle Bu Teaching Hospital offer?")
print(f"\nAnswer:\n{result['answer']}")

# COMMAND ----------

# Test Question 1.4: Regional search
print("=" * 70)
print("Q1.4: Are there any clinics in Tamale that perform surgery?")
print("=" * 70)
result = search_and_answer("Are there any clinics in Tamale that perform surgery or surgical procedures?")
print(f"\nAnswer:\n{result['answer']}")

# COMMAND ----------

# Test Question 1.5: Regional comparison
print("=" * 70)
print("Q1.5: Which region has the most hospital-type facilities?")
print("=" * 70)
result = search_and_answer("Which region has the most hospital-type facilities?")
print(f"\nAnswer:\n{result['answer']}")

# COMMAND ----------

# Test Medical Desert question
print("=" * 70)
print("Q2.3: Which regions are medical deserts with no surgery capability?")
print("=" * 70)
result = search_and_answer("Which geographic regions have no surgical capability and could be considered medical deserts?")
print(f"\nAnswer:\n{result['answer']}")

# COMMAND ----------

# Test Anomaly question
print("=" * 70)
print("Q4.4: Which facilities claim unrealistic procedures for their size?")
print("=" * 70)
result = search_and_answer("Which facilities claim an unrealistically high number of procedures relative to their size, number of doctors, or facility type?")
print(f"\nAnswer:\n{result['answer']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4E. Convenience Function for Interactive Use

# COMMAND ----------

def ask(question: str):
    """Quick helper to ask a question and print the answer."""
    print(f"Q: {question}")
    print("-" * 70)
    result = search_and_answer(question)
    print(result["answer"])
    print(f"\n[Sources: {result['num_sources']} facilities searched]")
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Try It Yourself!
# MAGIC
# MAGIC Use the `ask()` function to ask any question:
# MAGIC ```python
# MAGIC ask("Which hospitals in Accra have the most doctors?")
# MAGIC ask("What equipment is available at Cape Coast Teaching Hospital?")
# MAGIC ask("Which regions lack pediatric care?")
# MAGIC ```

# COMMAND ----------

# Example: Try your own question
ask("Which hospitals have the highest bed capacity in Ghana?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4F. Save RAG Config for Later Notebooks

# COMMAND ----------

# Save a reference so later notebooks (supervisor, app) can import this setup
rag_config = {
    "faiss_path": f"{dbfs_dir}/facility_index.faiss",
    "metadata_path": f"{dbfs_dir}/facility_metadata.pkl",
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_model": "llama-3.1-70b-versatile",
    "llm_provider": "groq",
}

# Save as JSON to DBFS
import json
config_json = json.dumps(rag_config, indent=2)
dbutils.fs.put("/FileStore/hackathon/rag_config.json", config_json, overwrite=True)
print("RAG config saved to /FileStore/hackathon/rag_config.json")

# COMMAND ----------

print("=" * 60)
print("STEP 4 COMPLETE: RAG Chain is working!")
print("=" * 60)
print("""
What we built:
  1. Vector search over facility data (FAISS + sentence-transformers)
  2. SQL context helper for structured data (Spark SQL)
  3. RAG pipeline: question -> search -> LLM answer with citations
  4. Tested against Must-Have questions from the agent doc

Next steps:
  - Step 5: SQL Agent (more precise structured queries)
  - Step 6: Medical Reasoning Agent (anomaly analysis)
  - Step 7: Supervisor Agent (route questions to the right sub-agent)
""")
