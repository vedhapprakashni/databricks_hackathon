# Databricks notebook source
# MAGIC %md
# MAGIC # Step 4: RAG Chain (Databricks Vector Search + Groq LLM)
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Connects to the Databricks Vector Search index
# MAGIC 2. Builds a RAG pipeline: question -> Vector Search -> Groq LLM -> answer with citations
# MAGIC 3. Includes SQL context helper for structured data
# MAGIC 4. Logs queries with MLflow for traceability
# MAGIC 5. Tests against Must-Have questions from the agent questions doc
# MAGIC
# MAGIC **Run notebooks 00, 01, 02, 03 first!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4A. Load Vector Search Index and LLM

# COMMAND ----------

import os
import json
import mlflow
from databricks.vector_search.client import VectorSearchClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Config
# ============================================================
# CONFIG: Must match what was set in 00_setup
# ============================================================
CATALOG = "hackathon_vf"
SCHEMA = "healthcare"
TABLE_PREFIX = f"{CATALOG}.{SCHEMA}"
VS_ENDPOINT = "vf_facility_search"
VS_INDEX_NAME = f"{TABLE_PREFIX}.facilities_vs_index"

try:
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql(f"USE SCHEMA {SCHEMA}")
except Exception:
    CATALOG = "hive_metastore"
    SCHEMA = "hackathon"
    TABLE_PREFIX = f"{CATALOG}.{SCHEMA}"
    VS_INDEX_NAME = f"{TABLE_PREFIX}.facilities_vs_index"

# Load API key
try:
    GROQ_API_KEY = dbutils.secrets.get(scope="hackathon", key="GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

print(f"Config: {TABLE_PREFIX} | VS Index: {VS_INDEX_NAME}")

# COMMAND ----------

# Initialize Vector Search client
vsc = VectorSearchClient()

try:
    vs_index = vsc.get_index(VS_ENDPOINT, VS_INDEX_NAME)
    print(f"Vector Search index connected: {VS_INDEX_NAME}")
    USING_VS = True
except Exception as e:
    print(f"Vector Search not available: {e}")
    print("Falling back to FAISS...")
    USING_VS = False
    
    # Load FAISS fallback
    import faiss, pickle, tempfile, numpy as np
    from sentence_transformers import SentenceTransformer
    
    local_dir = tempfile.mkdtemp()
    dbutils.fs.cp("/FileStore/hackathon/vector_store/facility_index.faiss", f"file:{local_dir}/facility_index.faiss")
    dbutils.fs.cp("/FileStore/hackathon/vector_store/facility_metadata.pkl", f"file:{local_dir}/facility_metadata.pkl")
    
    faiss_index = faiss.read_index(f"{local_dir}/facility_index.faiss")
    with open(f"{local_dir}/facility_metadata.pkl", "rb") as f:
        faiss_meta = pickle.load(f)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"FAISS fallback loaded: {faiss_index.ntotal} vectors")

# COMMAND ----------

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=2048
)
print("Groq LLM initialized")

# Set up MLflow
experiment_name = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/vf_healthcare_agent"
mlflow.set_experiment(experiment_name)
print(f"MLflow experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4B. Vector Search Function

# COMMAND ----------

def vector_search(query: str, k: int = 10):
    """Search for relevant facilities using Vector Search or FAISS fallback."""
    if USING_VS:
        # Databricks Vector Search
        results = vs_index.similarity_search(
            query_text=query,
            columns=[
                "pk_unique_id", "name", "search_text", "address_city",
                "address_stateOrRegion", "facilityTypeId", "specialties",
                "procedure", "equipment", "capability", "description",
                "numberDoctors", "capacity", "operatorTypeId"
            ],
            num_results=k
        )
        
        # Parse results into list of dicts
        cols = results["manifest"]["columns"]
        col_names = [c["name"] for c in cols]
        facilities = []
        for row in results["result"]["data_array"]:
            facility = dict(zip(col_names, row))
            facilities.append(facility)
        return facilities
    else:
        # FAISS fallback
        query_emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = faiss_index.search(query_emb, k)
        
        facilities = []
        pdf = faiss_meta.get("dataframe")
        if pdf is not None:
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(pdf):
                    row = pdf.iloc[idx].to_dict()
                    row["score"] = float(score)
                    facilities.append(row)
        else:
            texts = faiss_meta["texts"]
            ids = faiss_meta["ids"]
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    facilities.append({
                        "pk_unique_id": ids[idx],
                        "search_text": texts[idx],
                        "score": float(score)
                    })
        return facilities

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4C. Format Context for LLM

# COMMAND ----------

def format_facility_context(facility: dict, rank: int) -> str:
    """Format a facility dict into a readable context block for the LLM."""
    parts = [f"[Facility {rank}]"]
    parts.append(f"Name: {facility.get('name', 'Unknown')}")
    
    for label, key in [("City", "address_city"), ("Region", "address_stateOrRegion"),
                        ("Type", "facilityTypeId"), ("Operator", "operatorTypeId")]:
        val = facility.get(key)
        if val and str(val) not in ("None", "nan", "null", ""):
            parts.append(f"{label}: {val}")
    
    for label, key in [("Doctors", "numberDoctors"), ("Bed Capacity", "capacity")]:
        val = facility.get(key)
        if val and str(val) not in ("None", "nan", "null", ""):
            try:
                parts.append(f"{label}: {int(float(val))}")
            except (ValueError, TypeError):
                pass
    
    for label, key in [("Specialties", "specialties"), ("Procedures", "procedure"),
                        ("Equipment", "equipment"), ("Capabilities", "capability")]:
        val = facility.get(key, "[]")
        if val and str(val) not in ("None", "nan", "null", "[]", ""):
            try:
                items = json.loads(val) if isinstance(val, str) else val
                if isinstance(items, list) and items:
                    parts.append(f"{label}: {', '.join(str(i) for i in items)}")
            except (json.JSONDecodeError, TypeError):
                parts.append(f"{label}: {val}")
    
    desc = facility.get("description")
    if desc and str(desc) not in ("None", "nan", "null", ""):
        parts.append(f"Description: {str(desc)[:300]}")
    
    return "\n".join(parts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4D. SQL Context Helper

# COMMAND ----------

ENRICHED_TABLE = f"{TABLE_PREFIX}.facilities_enriched"
DESERT_TABLE = f"{TABLE_PREFIX}.regional_analysis"

def get_sql_context(question: str) -> str:
    """Pull structured data from Spark SQL based on the question."""
    context_parts = []
    q_lower = question.lower()

    try:
        if any(word in q_lower for word in ["how many", "count", "number of", "total"]):
            total = spark.sql(f"SELECT COUNT(*) as total FROM {ENRICHED_TABLE}").collect()[0]["total"]
            context_parts.append(f"Total facilities in database: {total}")
            type_counts = spark.sql(f"""
                SELECT facilityTypeId, COUNT(*) as count FROM {ENRICHED_TABLE}
                GROUP BY facilityTypeId ORDER BY count DESC
            """).collect()
            context_parts.append(f"By type: {', '.join(f'{r.facilityTypeId}: {r.count}' for r in type_counts)}")

        if any(word in q_lower for word in ["region", "area", "where", "location", "geographic"]):
            regions = spark.sql(f"""
                SELECT address_stateOrRegion, COUNT(*) as count,
                       SUM(COALESCE(numberDoctors, 0)) as doctors
                FROM {ENRICHED_TABLE} GROUP BY address_stateOrRegion ORDER BY count DESC
            """).collect()
            for r in regions:
                context_parts.append(f"Region {r.address_stateOrRegion}: {r.count} facilities, {r.doctors} doctors")

        specialties_map = {
            "cardiology": "cardiology", "surgery": "surgery", "pediatric": "pediatrics",
            "emergency": "emergency", "dental": "dentistry", "obstetrics": "obstetrics",
            "ophthalmology": "ophthalmology", "radiology": "radiology"
        }
        for keyword, spec in specialties_map.items():
            if keyword in q_lower:
                count = spark.sql(f"""
                    SELECT COUNT(*) as c FROM {ENRICHED_TABLE}
                    WHERE LOWER(specialties) LIKE '%{spec}%'
                """).collect()[0]["c"]
                context_parts.append(f"Facilities with {spec}: {count}")

        if any(word in q_lower for word in ["anomal", "suspicious", "mismatch", "unrealistic"]):
            flags = spark.sql(f"""
                SELECT
                    SUM(CASE WHEN flag_procedures_no_doctors THEN 1 ELSE 0 END) as proc_no_doc,
                    SUM(CASE WHEN flag_capacity_no_equipment THEN 1 ELSE 0 END) as cap_no_equip,
                    SUM(CASE WHEN flag_clinic_claims_surgery THEN 1 ELSE 0 END) as clinic_surg,
                    SUM(CASE WHEN flag_too_many_specialties THEN 1 ELSE 0 END) as many_specs,
                    SUM(CASE WHEN flag_sparse_record THEN 1 ELSE 0 END) as sparse
                FROM {ENRICHED_TABLE}
            """).collect()[0]
            context_parts.append(f"Anomalies - Procedures but no doctors: {flags.proc_no_doc}")
            context_parts.append(f"Anomalies - High capacity but no equipment: {flags.cap_no_equip}")
            context_parts.append(f"Anomalies - Clinics claiming surgery: {flags.clinic_surg}")
            context_parts.append(f"Anomalies - Too many specialties: {flags.many_specs}")
            context_parts.append(f"Anomalies - Sparse records: {flags.sparse}")

        if any(word in q_lower for word in ["desert", "underserved", "gap", "shortage", "lacking"]):
            deserts = spark.sql(f"""
                SELECT address_stateOrRegion, facility_count, total_doctors, total_beds,
                       has_surgery, has_emergency, desert_score
                FROM {DESERT_TABLE} ORDER BY desert_score DESC LIMIT 5
            """).collect()
            context_parts.append("Top underserved regions (highest desert score):")
            for r in deserts:
                context_parts.append(
                    f"  {r.address_stateOrRegion}: score={r.desert_score}, "
                    f"facilities={r.facility_count}, doctors={r.total_doctors}, "
                    f"surgery={'Yes' if r.has_surgery > 0 else 'No'}"
                )
    except Exception as e:
        context_parts.append(f"SQL context error: {str(e)}")

    return "\n".join(context_parts) if context_parts else "No structured data pulled."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4E. Build the RAG Chain

# COMMAND ----------

RAG_PROMPT = ChatPromptTemplate.from_template("""You are a healthcare facility intelligence agent for Ghana.
You help the Virtue Foundation identify medical gaps and coordinate healthcare resources.

Answer the user's question using ONLY the facility data provided below.
Be specific: include facility names, locations, and data points in your answer.
If the data does not contain enough information to answer fully, say so honestly.

For each claim you make, cite the facility by name (e.g., "[Facility Name, Region]").

FACILITY DATA (from semantic search):
{context}

STRUCTURED DATA (from database):
{sql_context}

USER QUESTION: {question}

DETAILED ANSWER:""")

# COMMAND ----------

def search_and_answer(question: str, k: int = 10):
    """
    Full RAG pipeline with MLflow logging:
    1. Vector Search for relevant facilities
    2. SQL context for structured data
    3. Groq LLM generates answer with citations
    4. MLflow logs the query for traceability
    """
    with mlflow.start_run(nested=True):
        mlflow.log_param("question", question[:250])
        mlflow.log_param("search_k", k)
        mlflow.log_param("using_vector_search", USING_VS)

        # Step 1: Vector Search
        facilities = vector_search(question, k=k)
        mlflow.log_metric("num_results", len(facilities))

        # Step 2: Build context
        context_parts = []
        sources = []
        for rank, fac in enumerate(facilities, 1):
            context_parts.append(format_facility_context(fac, rank))
            sources.append({
                "name": fac.get("name", "Unknown"),
                "pk_unique_id": fac.get("pk_unique_id", ""),
                "region": fac.get("address_stateOrRegion", ""),
                "score": fac.get("score", 0)
            })
        context = "\n\n".join(context_parts)

        # Step 3: SQL context
        sql_context = get_sql_context(question)
        mlflow.log_param("sql_context_length", len(sql_context))

        # Step 4: LLM answer
        chain = RAG_PROMPT | llm
        response = chain.invoke({
            "context": context,
            "sql_context": sql_context,
            "question": question
        })

        answer = response.content
        mlflow.log_param("answer_length", len(answer))
        mlflow.log_text(answer, "answer.txt")

        return {
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
            "sql_context": sql_context
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4F. Test: Must-Have Questions

# COMMAND ----------

print("=" * 70)
print("Q1.1: How many hospitals offer cardiology services?")
print("=" * 70)
result = search_and_answer("How many hospitals in the dataset offer cardiology services?")
print(f"\n{result['answer']}")
print(f"\n[{result['num_sources']} sources searched]")

# COMMAND ----------

print("=" * 70)
print("Q1.3: What services does Korle Bu Teaching Hospital offer?")
print("=" * 70)
result = search_and_answer("What specific services does Korle Bu Teaching Hospital offer?")
print(f"\n{result['answer']}")

# COMMAND ----------

print("=" * 70)
print("Q1.4: Are there any clinics in Tamale that perform surgery?")
print("=" * 70)
result = search_and_answer("Are there any clinics in Tamale that perform surgery or surgical procedures?")
print(f"\n{result['answer']}")

# COMMAND ----------

print("=" * 70)
print("Q1.5: Which region has the most hospital-type facilities?")
print("=" * 70)
result = search_and_answer("Which region has the most hospital-type facilities?")
print(f"\n{result['answer']}")

# COMMAND ----------

print("=" * 70)
print("Q2.3: Which regions are medical deserts?")
print("=" * 70)
result = search_and_answer("Which geographic regions have no surgical capability and could be considered medical deserts?")
print(f"\n{result['answer']}")

# COMMAND ----------

print("=" * 70)
print("Q4.4: Which facilities claim unrealistic procedures for their size?")
print("=" * 70)
result = search_and_answer("Which facilities claim an unrealistically high number of procedures relative to their size, number of doctors, or facility type?")
print(f"\n{result['answer']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4G. Interactive Query Function

# COMMAND ----------

def ask(question: str):
    """Quick helper -- ask any question and get an answer."""
    print(f"Q: {question}")
    print("-" * 70)
    result = search_and_answer(question)
    print(result["answer"])
    print(f"\n[{result['num_sources']} sources | VS: {USING_VS}]")
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Try It!
# MAGIC
# MAGIC ```python
# MAGIC ask("Which hospitals in Accra have the most doctors?")
# MAGIC ask("What equipment is available at Cape Coast Teaching Hospital?")
# MAGIC ask("Which regions lack pediatric care?")
# MAGIC ask("Where should the Virtue Foundation send emergency medicine volunteers?")
# MAGIC ```

# COMMAND ----------

ask("Which hospitals have the highest bed capacity in Ghana?")

# COMMAND ----------

ask("Where should the Virtue Foundation prioritize sending doctors to address medical deserts?")

# COMMAND ----------

print("=" * 60)
print("STEP 4 COMPLETE: RAG Chain with Databricks Vector Search")
print("=" * 60)
print(f"""
What we built:
  - Databricks Vector Search semantic retrieval (or FAISS fallback)
  - SQL context helper for structured counts and stats
  - Groq LLM (Llama 3.1 70B) for answer generation with citations
  - MLflow logging for every query (traceability)
  - Tested against Must-Have agent questions

Using Vector Search: {USING_VS}
MLflow Experiment: {experiment_name}

Next steps:
  - Step 5: SQL Agent (precise structured queries)
  - Step 6: Medical Reasoning Agent (anomaly analysis)
  - Step 7: Supervisor Agent (route to right sub-agent)
""")
