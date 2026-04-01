# Databricks notebook source
# MAGIC %md
# MAGIC # Step 9: MLflow Tracing and Citations
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Sets up comprehensive MLflow tracing for the multi-agent system
# MAGIC 2. Logs every agent step with inputs, outputs, and metadata
# MAGIC 3. Provides citation provenance (which data supported which answer)
# MAGIC 4. Creates an evaluation dataset for agent quality assessment
# MAGIC
# MAGIC **Run notebooks 00-08 first!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9A. Install Packages

# COMMAND ----------

%pip install -U typing-extensions pydantic
%pip install langchain langchain-groq databricks-vectorsearch mlflow sentence-transformers faiss-cpu

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9B. Configuration

# COMMAND ----------

import os, json, time, mlflow
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from databricks.vector_search.client import VectorSearchClient

CATALOG = "hackathon_vf"
SCHEMA = "healthcare"
TABLE_PREFIX = f"{CATALOG}.{SCHEMA}"

try:
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql(f"USE SCHEMA {SCHEMA}")
except Exception:
    CATALOG = "hive_metastore"
    SCHEMA = "hackathon"
    TABLE_PREFIX = f"{CATALOG}.{SCHEMA}"
    spark.sql(f"USE {SCHEMA}")

ENRICHED_TABLE = f"{TABLE_PREFIX}.facilities_enriched"
DESERT_TABLE = f"{TABLE_PREFIX}.regional_analysis"
VS_ENDPOINT = "vf_facility_search"
VS_INDEX_NAME = f"{TABLE_PREFIX}.facilities_vs_index"
TRACE_TABLE = f"{TABLE_PREFIX}.agent_traces"

try:
    GROQ_API_KEY = dbutils.secrets.get(scope="hackathon", key="GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0, max_tokens=2048)

experiment_name = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/vf_healthcare_agent"
mlflow.set_experiment(experiment_name)
print(f"MLflow experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9C. Load Components

# COMMAND ----------

vsc = VectorSearchClient()
USING_VS = False
vs_index = None
try:
    vs_index = vsc.get_index(VS_ENDPOINT, VS_INDEX_NAME)
    USING_VS = True
    print("Vector Search connected")
except Exception as e:
    print(f"VS not available: {e}, trying FAISS...")
    try:
        import faiss, pickle, tempfile, numpy as np
        from sentence_transformers import SentenceTransformer
        local_dir = tempfile.mkdtemp()
        dbutils.fs.cp("/FileStore/hackathon/vector_store/facility_index.faiss", f"file:{local_dir}/facility_index.faiss")
        dbutils.fs.cp("/FileStore/hackathon/vector_store/facility_metadata.pkl", f"file:{local_dir}/facility_metadata.pkl")
        faiss_index = faiss.read_index(f"{local_dir}/facility_index.faiss")
        with open(f"{local_dir}/facility_metadata.pkl", "rb") as f:
            faiss_meta = pickle.load(f)
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"FAISS loaded: {faiss_index.ntotal} vectors")
    except Exception as e2:
        print(f"FAISS also unavailable: {e2}")

facilities_pdf = spark.table(ENRICHED_TABLE).toPandas()
print(f"Loaded {len(facilities_pdf)} facilities")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9D. Sub-Agent Functions (Compact)

# COMMAND ----------

CLASSIFY_PROMPT = ChatPromptTemplate.from_template(
    "Classify into ONE: STRUCTURED (counts/aggregations), SEMANTIC (service lookups), "
    "REASONING (anomalies/inference), DESERT (gaps/underserved). Question: {question}\nCategory:")

def classify_query(question):
    cat = llm.invoke(CLASSIFY_PROMPT.format(question=question)).content.strip().upper()
    valid = {"STRUCTURED", "SEMANTIC", "REASONING", "DESERT"}
    return cat if cat in valid else next((v for v in valid if v in cat), "SEMANTIC")

def sql_sub_agent(question):
    prompt = ChatPromptTemplate.from_template(
        f"Convert to Spark SQL (SELECT only, no backticks). Table: {ENRICHED_TABLE}. "
        f"Columns: pk_unique_id, name, facilityTypeId, address_stateOrRegion, numberDoctors, capacity, "
        f"specialties, procedure, equipment, num_procedures, num_equipment, num_specialties, "
        f"flag_procedures_no_doctors, flag_capacity_no_equipment, flag_clinic_claims_surgery. "
        f"Regional: {DESERT_TABLE} (desert_score, facility_count, total_doctors). "
        "Search text: LOWER(col) LIKE '%term%'. Question: {q}\nSQL:")
    sql = llm.invoke(prompt.format(q=question)).content.strip()
    if sql.startswith("```"): sql = "\n".join(l for l in sql.split("\n") if not l.strip().startswith("```"))
    sql = sql.strip().rstrip(";")
    for kw in ["DROP","DELETE","UPDATE","INSERT","ALTER","CREATE","TRUNCATE"]:
        if kw in sql.upper(): return f"BLOCKED: {kw}", sql, []
    try:
        rows = spark.sql(sql).limit(30).collect()
        cols = spark.sql(sql).columns
        text = "\n".join(" | ".join(str(r[c]) for c in cols) for r in rows)
        sids = [str(r["pk_unique_id"]) for r in rows if "pk_unique_id" in cols] if "pk_unique_id" in cols else []
        return text, sql, sids
    except Exception as e:
        return f"Error: {e}", sql, []

def rag_sub_agent(question):
    facilities, sids = [], []
    if USING_VS and vs_index:
        res = vs_index.similarity_search(query_text=question, columns=["pk_unique_id","name","search_text","address_stateOrRegion","facilityTypeId","specialties","numberDoctors"], num_results=10)
        cols = [c["name"] for c in res["manifest"]["columns"]]
        for row in res["result"]["data_array"]:
            f = dict(zip(cols, row)); facilities.append(f); sids.append(f.get("pk_unique_id",""))
    if not facilities: return "No semantic search results.", [], []
    ctx = "\n".join(f"[{i+1}] {f.get('name','?')} ({f.get('facilityTypeId','?')}) {f.get('address_stateOrRegion','?')} Specs:{f.get('specialties','[]')}" for i,f in enumerate(facilities))
    ans = llm.invoke(ChatPromptTemplate.from_template("Healthcare agent. Answer from data. Cite names.\nDATA:\n{c}\nQ: {q}\nA:").format(c=ctx,q=question)).content
    return ans, sids, facilities

def reasoning_sub_agent(question):
    rel = facilities_pdf.sort_values("source_count", ascending=False).head(15)
    txt = "\n".join(f"- {r.get('name','?')} [{r.get('facilityTypeId','?')}] {r.get('address_stateOrRegion','?')} Docs:{r.get('numberDoctors','N/A')} Procs:{r.get('num_procedures',0)} Equip:{r.get('num_equipment',0)}" for _,r in rel.iterrows())
    ans = llm.invoke(ChatPromptTemplate.from_template("Medical reasoning agent.\nDATA:\n{d}\nQ: {q}\nA:").format(d=txt,q=question)).content
    return ans, rel["pk_unique_id"].dropna().tolist()

def desert_sub_agent(question):
    try:
        rows = spark.table(DESERT_TABLE).orderBy("desert_score", ascending=False).collect()
        data = "\n".join(f"{r['address_stateOrRegion']}: Fac={r['facility_count']}, Docs={r['total_doctors']}, Surgery={'Y' if r['has_surgery']>0 else 'N'}, Score={r['desert_score']}" for r in rows)
    except: data = "N/A"
    return llm.invoke(ChatPromptTemplate.from_template("Desert analyst.\nDATA:\n{d}\nQ: {q}\nA:").format(d=data,q=question)).content, []

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9E. Fully Traced Agent Query

# COMMAND ----------

def traced_agent_query(question: str) -> dict:
    """Full multi-agent pipeline with comprehensive MLflow tracing."""
    with mlflow.start_run(run_name=f"query_{datetime.now().strftime('%H%M%S')}"):
        trace = {"question": question, "timestamp": datetime.now().isoformat(), "steps": []}
        mlflow.log_param("question", question[:250])

        # Step 1: Classify
        t0 = time.time()
        category = classify_query(question)
        trace["steps"].append({"step": "classify", "output": category, "ms": round((time.time()-t0)*1000)})
        mlflow.log_param("intent_category", category)
        print(f"[1] Intent: {category}")

        # Step 2: Route & Execute
        t1 = time.time()
        source_ids = []
        if category == "STRUCTURED":
            raw, sql, source_ids = sql_sub_agent(question)
            mlflow.log_text(sql, "generated_sql.sql"); agent = "sql"
        elif category == "SEMANTIC":
            raw, source_ids, _ = rag_sub_agent(question); agent = "rag"
        elif category == "REASONING":
            raw, source_ids = reasoning_sub_agent(question); agent = "reasoning"
        else:
            raw, source_ids = desert_sub_agent(question); agent = "desert"

        trace["steps"].append({"step": agent, "sources": len(source_ids), "ms": round((time.time()-t1)*1000)})
        mlflow.log_param("agent_used", agent)
        print(f"[2] Agent: {agent} ({len(source_ids)} sources)")

        # Step 3: Synthesize (SQL only)
        if category == "STRUCTURED" and not raw.startswith(("BLOCKED","Error")):
            final = llm.invoke(ChatPromptTemplate.from_template("Answer naturally.\nQ: {q}\nData:\n{d}\nA:").format(q=question,d=raw[:3000])).content
        else:
            final = raw

        # Step 4: Citations
        citations = []
        for sid in source_ids[:10]:
            if sid:
                match = facilities_pdf[facilities_pdf["pk_unique_id"]==str(sid)]
                if len(match)>0:
                    r = match.iloc[0]
                    citations.append({"id": str(sid), "name": str(r.get("name","")), "region": str(r.get("address_stateOrRegion",""))})

        total = time.time()-t0
        mlflow.log_metric("total_time_ms", round(total*1000))
        mlflow.log_metric("citation_count", len(citations))
        mlflow.log_text(final, "final_answer.txt")
        if citations: mlflow.log_text(json.dumps(citations, indent=2), "citations.json")
        mlflow.log_text(json.dumps(trace, indent=2, default=str), "trace.json")
        print(f"[3] Done: {total:.2f}s | {len(citations)} citations")

        return {"question": question, "category": category, "agent": agent,
                "answer": final, "citations": citations, "trace": trace, "total_time": round(total,2)}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9F. Test Traced Queries

# COMMAND ----------

print("=" * 70)
print("TRACED QUERY 1: Structured")
print("=" * 70)
r1 = traced_agent_query("How many hospitals have cardiology?")
print(f"\n{r1['answer']}\nCitations: {json.dumps(r1['citations'], indent=2)}")

# COMMAND ----------

print("=" * 70)
print("TRACED QUERY 2: Semantic")
print("=" * 70)
r2 = traced_agent_query("What services does Korle Bu Teaching Hospital offer?")
print(f"\n{r2['answer']}")

# COMMAND ----------

print("=" * 70)
print("TRACED QUERY 3: Reasoning")
print("=" * 70)
r3 = traced_agent_query("Which facilities have suspicious mismatches between procedures and equipment?")
print(f"\n{r3['answer']}")

# COMMAND ----------

print("=" * 70)
print("TRACED QUERY 4: Desert")
print("=" * 70)
r4 = traced_agent_query("Where should the Virtue Foundation prioritize sending doctors?")
print(f"\n{r4['answer']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9G. Save Trace Log to Delta

# COMMAND ----------

trace_records = []
for result in [r1, r2, r3, r4]:
    trace_records.append({
        "question": result["question"], "category": result["category"],
        "agent_used": result["agent"], "answer_length": len(result["answer"]),
        "citation_count": len(result["citations"]), "total_time_seconds": float(result["total_time"]),
        "timestamp": datetime.now().isoformat(), "answer_preview": result["answer"][:500]
    })

schema = StructType([
    StructField("question", StringType()), StructField("category", StringType()),
    StructField("agent_used", StringType()), StructField("answer_length", IntegerType()),
    StructField("citation_count", IntegerType()), StructField("total_time_seconds", FloatType()),
    StructField("timestamp", StringType()), StructField("answer_preview", StringType()),
])

spark.createDataFrame(trace_records, schema=schema).write.format("delta").mode("append").option("mergeSchema","true").saveAsTable(TRACE_TABLE)
print(f"Traces saved to {TRACE_TABLE}")
display(spark.table(TRACE_TABLE).orderBy("timestamp", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9H. Must-Have Evaluation Dataset

# COMMAND ----------

eval_questions = [
    {"id": "1.1", "question": "How many hospitals have cardiology?", "type": "STRUCTURED"},
    {"id": "1.3", "question": "What services does Korle Bu Teaching Hospital offer?", "type": "SEMANTIC"},
    {"id": "1.5", "question": "Which region has the most hospital-type facilities?", "type": "STRUCTURED"},
    {"id": "2.3", "question": "Where are geographic cold spots for critical procedures?", "type": "DESERT"},
    {"id": "4.4", "question": "Which facilities claim unrealistic procedures for their size?", "type": "REASONING"},
    {"id": "4.9", "question": "Where do we see things that shouldn't move together?", "type": "REASONING"},
    {"id": "6.1", "question": "Where is the surgical workforce practicing in Ghana?", "type": "REASONING"},
    {"id": "7.5", "question": "Which procedures depend on only 1-2 facilities per region?", "type": "STRUCTURED"},
    {"id": "7.6", "question": "Where is oversupply vs scarcity of procedures?", "type": "REASONING"},
    {"id": "8.3", "question": "Where are gaps where no NGOs work despite need?", "type": "DESERT"},
]
print(f"Evaluation: {len(eval_questions)} Must-Have questions")
for eq in eval_questions:
    print(f"  [{eq['id']}] {eq['type']:12s} | {eq['question']}")

# COMMAND ----------

print("=" * 60)
print("STEP 9 COMPLETE: MLflow Tracing & Citations")
print("=" * 60)
print(f"""
Built: Step-level MLflow tracing, citation provenance, trace Delta table,
       evaluation dataset with {len(eval_questions)} Must-Have questions.

MLflow artifacts per query: trace.json, citations.json, final_answer.txt
Table: {TRACE_TABLE}
Function: traced_agent_query(question)

Next: Run notebook 10_final_testing
""")
