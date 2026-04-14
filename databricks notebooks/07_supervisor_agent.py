# Databricks notebook source
# MAGIC %md
# MAGIC # Step 7: Supervisor Agent (Router + LangGraph Orchestration)
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Builds a supervisor agent that classifies user intent
# MAGIC 2. Routes queries to the appropriate sub-agent (SQL, RAG, Reasoning, Desert)
# MAGIC 3. Implements a LangGraph state machine for multi-agent orchestration
# MAGIC 4. Provides a unified `ask()` interface for all query types
# MAGIC 5. Logs routing decisions and end-to-end pipeline with MLflow
# MAGIC
# MAGIC **Run notebooks 00–06 first!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7A. Install Packages

# COMMAND ----------

# MAGIC %pip install -U typing-extensions pydantic
# MAGIC %pip install langchain langchain-community langchain-community langgraph databricks-vectorsearch mlflow sentence-transformers faiss-cpu

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7B. Configuration & Load All Components

# COMMAND ----------

import os
import json
import time
import mlflow
from typing import TypedDict, Literal, Annotated
from pyspark.sql import functions as F
from langchain_community.chat_models import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate
from databricks.vector_search.client import VectorSearchClient

# ============================================================
# CONFIG: Must match what was set in 00_setup
# ============================================================
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

# Load API key
try:
    DATABRICKS_TOKEN = dbutils.secrets.get(scope="hackathon", key="DATABRICKS_TOKEN")
except Exception:
    DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "YOUR_TOKEN_HERE")
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# Initialize LLM
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    
    temperature=0,
    max_tokens=2048
)

# MLflow
experiment_name = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/vf_healthcare_agent"
mlflow.set_experiment(experiment_name)

print(f"Config: {TABLE_PREFIX}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7C. Initialize Sub-Agent Components

# COMMAND ----------

# --- Vector Search ---
vsc = VectorSearchClient()
USING_VS = False
vs_index = None

try:
    vs_index = vsc.get_index(VS_ENDPOINT, VS_INDEX_NAME)
    USING_VS = True
    print(f"Vector Search connected: {VS_INDEX_NAME}")
except Exception as e:
    print(f"Vector Search not available: {e}")
    print("Loading FAISS fallback...")
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
        print(f"FAISS fallback loaded: {faiss_index.ntotal} vectors")
    except Exception as e2:
        print(f"FAISS also unavailable: {e2}")

# --- Facilities DataFrame ---
facilities_pdf = spark.table(ENRICHED_TABLE).toPandas()
print(f"Loaded {len(facilities_pdf)} facilities for reasoning agent")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7D. Intent Classification

# COMMAND ----------

CLASSIFY_PROMPT = ChatPromptTemplate.from_template("""Classify this healthcare facility question into EXACTLY ONE category.

Categories:

STRUCTURED: Questions about counts, aggregations, rankings, filtering by specific fields.
  Examples: "How many hospitals have cardiology?", "Which region has the most hospitals?",
  "What is the average bed capacity?", "How many facilities are public vs private?"

SEMANTIC: Questions asking about specific facility services, descriptions, or free-text content.
  Examples: "What services does Korle Bu offer?", "Find clinics that do surgery in Tamale",
  "Which facilities have CT scanners?", "Are there dental clinics in Accra?"

REASONING: Questions requiring medical inference, anomaly detection, cross-validation of claims.
  Examples: "Which facilities have suspicious claims?", "What mismatches exist between
  procedures and equipment?", "Which facilities claim unrealistic capabilities?",
  "Where do we see things that shouldn't move together?"

DESERT: Questions about medical deserts, healthcare gaps, underserved regions, resource distribution.
  Examples: "Where are the biggest healthcare gaps?", "Which areas lack emergency care?",
  "Where should the Virtue Foundation send doctors?", "Which regions are medical deserts?"

Question: {question}

Respond with ONLY ONE WORD: STRUCTURED, SEMANTIC, REASONING, or DESERT""")

def classify_query(question: str) -> str:
    """Classify the user's question into an intent category."""
    response = llm.invoke(CLASSIFY_PROMPT.format(question=question))
    category = response.content.strip().upper()

    # Validate
    valid_categories = {"STRUCTURED", "SEMANTIC", "REASONING", "DESERT"}
    if category not in valid_categories:
        # Try to extract a valid category from the response
        for cat in valid_categories:
            if cat in category:
                return cat
        return "SEMANTIC"  # Default fallback

    return category

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7E. Sub-Agent Implementations

# COMMAND ----------

# =============================================
# SUB-AGENT 1: SQL Agent (Structured Queries)
# =============================================

SCHEMA_REFERENCE = f"""
TABLE: {ENRICHED_TABLE}
KEY COLUMNS: pk_unique_id, name, facilityTypeId (hospital/clinic/dentist/pharmacy),
  operatorTypeId (public/private), address_city, address_stateOrRegion,
  numberDoctors (int, nullable), capacity (int, nullable),
  specialties (JSON array string), procedure (JSON array string),
  equipment (JSON array string), capability (JSON array string),
  num_procedures (int), num_equipment (int), num_capabilities (int), num_specialties (int),
  source_count (int), flag_procedures_no_doctors (bool), flag_capacity_no_equipment (bool),
  flag_clinic_claims_surgery (bool), flag_too_many_specialties (bool), flag_sparse_record (bool)
REGIONAL TABLE: {DESERT_TABLE}
  address_stateOrRegion, facility_count, total_doctors, total_beds, hospital_count,
  has_surgery, has_emergency, has_obstetrics, has_pediatrics, sparse_records, desert_score
NOTES: For specialty/procedure search use LOWER(specialties) LIKE '%term%'.
  Use COALESCE(numberDoctors, 0) when summing. SELECT only, no mutations.
"""

SQL_GEN_PROMPT = ChatPromptTemplate.from_template("""Convert this question to Spark SQL.
{schema}
Return ONLY the SQL query, no explanation, no backticks. SELECT only, limit 50.
Question: {question}
SQL:""")


def sql_sub_agent(question: str) -> str:
    """Execute a structured query via Text-to-SQL."""
    response = llm.invoke(SQL_GEN_PROMPT.format(schema=SCHEMA_REFERENCE, question=question))
    sql = response.content.strip()

    # Clean markdown formatting
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(l for l in lines if not l.strip().startswith("```"))
    sql = sql.strip().rstrip(";")

    # Safety check
    blocked = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"]
    for kw in blocked:
        if kw in sql.upper():
            return f"SQL blocked: contains {kw}"

    try:
        result_df = spark.sql(sql)
        rows = result_df.limit(30).collect()
        cols = result_df.columns

        result_text = f"SQL: {sql}\n\nResults ({len(rows)} rows):\n"
        result_text += " | ".join(cols) + "\n"
        for row in rows:
            result_text += " | ".join(str(row[c]) for c in cols) + "\n"
        return result_text
    except Exception as e:
        return f"SQL Error: {e}\nGenerated SQL: {sql}"

# COMMAND ----------

# =============================================
# SUB-AGENT 2: RAG Agent (Semantic Search)
# =============================================

def vector_search_fn(query: str, k: int = 10):
    """Search for relevant facilities."""
    if USING_VS and vs_index:
        results = vs_index.similarity_search(
            query_text=query,
            columns=["pk_unique_id", "name", "search_text", "address_city",
                      "address_stateOrRegion", "facilityTypeId", "specialties",
                      "procedure", "equipment", "capability", "description",
                      "numberDoctors", "capacity", "operatorTypeId"],
            num_results=k
        )
        cols = results["manifest"]["columns"]
        col_names = [c["name"] for c in cols]
        return [dict(zip(col_names, row)) for row in results["result"]["data_array"]]
    else:
        # FAISS fallback
        try:
            import numpy as np
            query_emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
            scores, indices = faiss_index.search(query_emb, k)
            pdf = faiss_meta.get("dataframe")
            facilities = []
            if pdf is not None:
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(pdf):
                        row = pdf.iloc[idx].to_dict()
                        row["score"] = float(score)
                        facilities.append(row)
            return facilities
        except Exception:
            return []


RAG_PROMPT = ChatPromptTemplate.from_template("""You are a healthcare facility intelligence agent for Ghana.
Answer using ONLY the facility data below. Be specific with names and locations. Cite facilities.
If data is insufficient, say so.

FACILITY DATA:
{context}

QUESTION: {question}

ANSWER:""")


def rag_sub_agent(question: str) -> str:
    """Execute a semantic search query with RAG."""
    facilities = vector_search_fn(question, k=10)

    if not facilities:
        return "No facilities found via semantic search. The vector search index may not be available."

    # Build context
    context_parts = []
    for i, fac in enumerate(facilities, 1):
        parts = [f"[{i}] {fac.get('name', 'Unknown')}"]
        for key, label in [("address_city", "City"), ("address_stateOrRegion", "Region"),
                            ("facilityTypeId", "Type"), ("numberDoctors", "Doctors"),
                            ("capacity", "Beds")]:
            val = fac.get(key)
            if val and str(val) not in ("None", "nan", "null", ""):
                parts.append(f"{label}: {val}")
        for key, label in [("specialties", "Specialties"), ("procedure", "Procedures"),
                            ("equipment", "Equipment"), ("capability", "Capabilities")]:
            val = fac.get(key, "[]")
            if val and str(val) not in ("None", "nan", "null", "[]", ""):
                try:
                    items = json.loads(val) if isinstance(val, str) else val
                    if isinstance(items, list) and items:
                        parts.append(f"{label}: {', '.join(str(x) for x in items)}")
                except (json.JSONDecodeError, TypeError):
                    parts.append(f"{label}: {val}")
        context_parts.append(" | ".join(parts))

    context = "\n".join(context_parts)
    response = llm.invoke(RAG_PROMPT.format(context=context, question=question))
    return response.content

# COMMAND ----------

# =============================================
# SUB-AGENT 3: Reasoning Agent
# =============================================

REASON_PROMPT = ChatPromptTemplate.from_template("""You are a medical reasoning agent with deep knowledge of healthcare systems.
Analyze the Ghana healthcare data to answer this question with medical reasoning.

FACILITY DATA (sample):
{facility_data}

REGIONAL SUMMARY:
{regional_summary}

QUESTION: {question}

Provide a thorough, evidence-based answer. Cite specific facilities and regions.

ANSWER:""")


def reasoning_sub_agent(question: str) -> str:
    """Perform medical reasoning analysis."""
    # Select relevant facilities
    q_lower = question.lower()
    relevant = facilities_pdf.copy()

    # Filter by region if mentioned
    for region in facilities_pdf["address_stateOrRegion"].dropna().unique():
        if region and region.lower() in q_lower:
            relevant = relevant[relevant["address_stateOrRegion"] == region]
            break

    if len(relevant) > 20:
        # Prioritize flagged or data-rich facilities
        flag_cols = [c for c in relevant.columns if c.startswith("flag_")]
        relevant["has_flag"] = relevant[flag_cols].any(axis=1).astype(int)
        relevant = relevant.sort_values(["has_flag", "source_count"], ascending=[False, False]).head(20)

    # Format
    fac_lines = []
    for _, row in relevant.iterrows():
        def safe(val, default="N/A"):
            if val is None or (isinstance(val, float) and str(val) == "nan"):
                return default
            return str(val)[:150]
        fac_lines.append(
            f"- {safe(row.get('name'))} [{safe(row.get('facilityTypeId'))}] "
            f"{safe(row.get('address_city'))}, {safe(row.get('address_stateOrRegion'))} | "
            f"Docs: {safe(row.get('numberDoctors'))} | Beds: {safe(row.get('capacity'))} | "
            f"Specs: {safe(row.get('specialties'), '[]')} | "
            f"Procs: {safe(row.get('num_procedures'), '0')} | Equip: {safe(row.get('num_equipment'), '0')}"
        )
    facility_data = "\n".join(fac_lines) if fac_lines else "No matching facilities."

    # Regional summary
    try:
        desert_rows = spark.table(DESERT_TABLE).collect()
        reg_lines = [
            f"{r['address_stateOrRegion']}: {r['facility_count']} fac, {r['total_doctors']} docs, "
            f"Surgery: {'Y' if r['has_surgery'] > 0 else 'N'}, "
            f"Emergency: {'Y' if r['has_emergency'] > 0 else 'N'}, "
            f"Score: {r['desert_score']}"
            for r in desert_rows
        ]
        regional_summary = "\n".join(reg_lines)
    except Exception:
        regional_summary = "Not available"

    response = llm.invoke(REASON_PROMPT.format(
        facility_data=facility_data,
        regional_summary=regional_summary,
        question=question
    ))
    return response.content

# COMMAND ----------

# =============================================
# SUB-AGENT 4: Medical Desert Agent
# =============================================

DESERT_PROMPT = ChatPromptTemplate.from_template("""You are a medical desert analyst advising the Virtue Foundation on Ghana.

REGIONAL DATA:
{regional_data}

QUESTION: {question}

Provide a data-driven answer focused on healthcare access gaps, underserved populations,
and actionable recommendations for the Virtue Foundation. Be specific about regions and numbers.

ANSWER:""")


def desert_sub_agent(question: str) -> str:
    """Analyze medical deserts and healthcare gaps."""
    try:
        desert_rows = spark.table(DESERT_TABLE).orderBy("desert_score", ascending=False).collect()
        lines = []
        for r in desert_rows:
            lines.append(
                f"{r['address_stateOrRegion']}: "
                f"Facilities={r['facility_count']}, Doctors={r['total_doctors']}, Beds={r['total_beds']}, "
                f"Hospitals={r['hospital_count']}, "
                f"Surgery={'Yes' if r['has_surgery'] > 0 else 'NO'}, "
                f"Emergency={'Yes' if r['has_emergency'] > 0 else 'NO'}, "
                f"Obstetrics={'Yes' if r['has_obstetrics'] > 0 else 'NO'}, "
                f"Pediatrics={'Yes' if r['has_pediatrics'] > 0 else 'NO'}, "
                f"Desert Score={r['desert_score']}"
            )
        regional_data = "\n".join(lines)
    except Exception as e:
        regional_data = f"Error loading regional data: {e}"

    response = llm.invoke(DESERT_PROMPT.format(
        regional_data=regional_data,
        question=question
    ))
    return response.content

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7F. Supervisor Router

# COMMAND ----------

def supervisor_agent(question: str) -> dict:
    """
    Main supervisor agent: classifies intent, routes to sub-agent,
    synthesizes response, and logs everything with MLflow.
    """
    with mlflow.start_run(nested=True):
        mlflow.log_param("agent", "supervisor")
        mlflow.log_param("question", question[:250])

        start_time = time.time()

        # Step 1: Classify intent
        category = classify_query(question)
        mlflow.log_param("intent_category", category)
        print(f"[Supervisor] Intent: {category}")

        # Step 2: Route to sub-agent
        if category == "STRUCTURED":
            print("[Supervisor] Routing to SQL Agent...")
            raw_answer = sql_sub_agent(question)
            agent_used = "sql"
        elif category == "SEMANTIC":
            print("[Supervisor] Routing to RAG Agent...")
            raw_answer = rag_sub_agent(question)
            agent_used = "rag"
        elif category == "REASONING":
            print("[Supervisor] Routing to Reasoning Agent...")
            raw_answer = reasoning_sub_agent(question)
            agent_used = "reasoning"
        elif category == "DESERT":
            print("[Supervisor] Routing to Medical Desert Agent...")
            raw_answer = desert_sub_agent(question)
            agent_used = "desert"
        else:
            print("[Supervisor] Unknown intent, fallback to RAG Agent...")
            raw_answer = rag_sub_agent(question)
            agent_used = "rag_fallback"

        mlflow.log_param("agent_used", agent_used)
        elapsed = time.time() - start_time
        mlflow.log_metric("response_time_seconds", elapsed)

        # Step 3: If SQL agent returned raw data, synthesize a natural language answer
        if agent_used == "sql" and "Results" in raw_answer:
            synth_prompt = ChatPromptTemplate.from_template("""Based on this SQL query result,
provide a clear natural language answer to the user's question.

Question: {question}
Data:
{data}

ANSWER:""")
            synth_response = llm.invoke(synth_prompt.format(question=question, data=raw_answer[:3000]))
            final_answer = synth_response.content
            mlflow.log_text(raw_answer, "raw_sql_result.txt")
        else:
            final_answer = raw_answer

        mlflow.log_text(final_answer, "final_answer.txt")
        mlflow.log_param("answer_length", len(final_answer))

        return {
            "question": question,
            "category": category,
            "agent_used": agent_used,
            "answer": final_answer,
            "raw_answer": raw_answer,
            "response_time": round(elapsed, 2)
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7G. LangGraph State Machine (Advanced Orchestration)

# COMMAND ----------

from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    question: str
    category: str
    raw_result: str
    final_answer: str

def classify_node(state: AgentState) -> AgentState:
    """Classify the user's question intent."""
    state["category"] = classify_query(state["question"])
    print(f"[LangGraph] Classified as: {state['category']}")
    return state

def sql_node(state: AgentState) -> AgentState:
    """Execute SQL sub-agent."""
    print("[LangGraph] Running SQL Agent...")
    state["raw_result"] = sql_sub_agent(state["question"])
    return state

def rag_node(state: AgentState) -> AgentState:
    """Execute RAG sub-agent."""
    print("[LangGraph] Running RAG Agent...")
    state["raw_result"] = rag_sub_agent(state["question"])
    return state

def reasoning_node(state: AgentState) -> AgentState:
    """Execute Reasoning sub-agent."""
    print("[LangGraph] Running Reasoning Agent...")
    state["raw_result"] = reasoning_sub_agent(state["question"])
    return state

def desert_node(state: AgentState) -> AgentState:
    """Execute Desert sub-agent."""
    print("[LangGraph] Running Desert Agent...")
    state["raw_result"] = desert_sub_agent(state["question"])
    return state

def response_node(state: AgentState) -> AgentState:
    """Synthesize final response."""
    if state["category"] == "STRUCTURED" and "Results" in state.get("raw_result", ""):
        synth_prompt = ChatPromptTemplate.from_template(
            "Answer this question naturally based on the data.\nQuestion: {q}\nData:\n{d}\nANSWER:"
        )
        resp = llm.invoke(synth_prompt.format(q=state["question"], d=state["raw_result"][:3000]))
        state["final_answer"] = resp.content
    else:
        state["final_answer"] = state.get("raw_result", "No answer generated.")
    return state

def route_by_intent(state: AgentState) -> str:
    """Route based on classified intent."""
    category = state.get("category", "SEMANTIC")
    routing = {
        "STRUCTURED": "sql_agent",
        "SEMANTIC": "rag_agent",
        "REASONING": "reasoning_agent",
        "DESERT": "desert_agent"
    }
    return routing.get(category, "rag_agent")

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_node)
workflow.add_node("sql_agent", sql_node)
workflow.add_node("rag_agent", rag_node)
workflow.add_node("reasoning_agent", reasoning_node)
workflow.add_node("desert_agent", desert_node)
workflow.add_node("respond", response_node)

workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", route_by_intent, {
    "sql_agent": "sql_agent",
    "rag_agent": "rag_agent",
    "reasoning_agent": "reasoning_agent",
    "desert_agent": "desert_agent",
})

for node in ["sql_agent", "rag_agent", "reasoning_agent", "desert_agent"]:
    workflow.add_edge(node, "respond")
workflow.add_edge("respond", END)

# Compile
agent_graph = workflow.compile()
print("LangGraph agent compiled successfully!")

# COMMAND ----------

def ask_langgraph(question: str) -> dict:
    """Query the LangGraph-based multi-agent system."""
    result = agent_graph.invoke({"question": question, "category": "", "raw_result": "", "final_answer": ""})
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7H. Unified Ask Function

# COMMAND ----------

def ask(question: str, use_langgraph: bool = False):
    """
    Unified interface to the healthcare intelligence agent.
    Routes to the right sub-agent automatically.
    """
    print(f"\nQ: {question}")
    print("=" * 70)

    if use_langgraph:
        result = ask_langgraph(question)
        category = result["category"]
        answer = result["final_answer"]
        print(f"[Agent: {category} via LangGraph]")
    else:
        result = supervisor_agent(question)
        category = result["category"]
        answer = result["answer"]
        print(f"[Agent: {result['agent_used']} | Time: {result['response_time']}s]")

    print(f"\n{answer}")
    print("-" * 70)
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7I. Test the Supervisor Agent

# COMMAND ----------

print("TEST 1: STRUCTURED query")
ask("How many hospitals have cardiology?")

# COMMAND ----------

print("TEST 2: SEMANTIC query")
ask("What services does Korle Bu Teaching Hospital offer?")

# COMMAND ----------

print("TEST 3: REASONING query")
ask("Which facilities have suspicious mismatches between their claimed procedures and listed equipment?")

# COMMAND ----------

print("TEST 4: DESERT query")
ask("Which regions in Ghana are medical deserts and where should the Virtue Foundation prioritize?")

# COMMAND ----------

print("TEST 5: LangGraph mode")
ask("Which region has the fewest hospitals?", use_langgraph=True)

# COMMAND ----------

print("=" * 60)
print("STEP 7 COMPLETE: Supervisor Agent + LangGraph")
print("=" * 60)
print(f"""
What we built:
  - Intent classifier (STRUCTURED / SEMANTIC / REASONING / DESERT)
  - Supervisor router with 4 sub-agents
  - LangGraph state machine for advanced orchestration
  - Unified ask() interface
  - MLflow logging of routing decisions + responses

Functions available:
  - ask(question)                      -- unified interface (uses supervisor)
  - ask(question, use_langgraph=True)  -- use LangGraph orchestration
  - supervisor_agent(question)         -- direct supervisor access
  - classify_query(question)           -- test intent classification

Sub-agents:
  - sql_sub_agent(question)      -- Text-to-SQL
  - rag_sub_agent(question)      -- Vector Search + RAG
  - reasoning_sub_agent(question) -- Medical reasoning
  - desert_sub_agent(question)   -- Medical desert analysis

Next: Run notebook 08_dashboard
""")