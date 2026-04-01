# Databricks notebook source
# MAGIC %md
# MAGIC # Step 5: SQL Agent for Structured Queries
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Builds a Text-to-SQL agent using Groq LLM
# MAGIC 2. Converts natural language questions into Spark SQL
# MAGIC 3. Executes queries safely against the enriched facilities table
# MAGIC 4. Includes pre-built anomaly detection queries
# MAGIC 5. Logs generated SQL and results with MLflow
# MAGIC
# MAGIC **Run notebooks 00, 01, 02, 03, 04 first!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5A. Install Packages

# COMMAND ----------

%pip install langchain langchain-groq mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5B. Configuration

# COMMAND ----------

import os
import json
import mlflow
from pyspark.sql import functions as F
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

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

# Load API key
try:
    GROQ_API_KEY = dbutils.secrets.get(scope="hackathon", key="GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=1024
)

# MLflow
experiment_name = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/vf_healthcare_agent"
mlflow.set_experiment(experiment_name)

print(f"Config: {TABLE_PREFIX}")
print(f"Enriched table: {ENRICHED_TABLE}")
print(f"Regional analysis: {DESERT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5C. Discover Table Schema

# COMMAND ----------

# Get the actual schema from the enriched table for the LLM prompt
schema_df = spark.sql(f"DESCRIBE TABLE {ENRICHED_TABLE}")
schema_info = schema_df.collect()

schema_text = "Table: " + ENRICHED_TABLE + "\nColumns:\n"
for row in schema_info:
    col_name = row["col_name"]
    data_type = row["data_type"]
    comment = row["comment"] if row["comment"] else ""
    schema_text += f"  - {col_name} ({data_type}) {comment}\n"

print(schema_text)

# COMMAND ----------

# Also get sample values for key columns to help the LLM
print("=== SAMPLE VALUES FOR KEY COLUMNS ===\n")

for col_name in ["facilityTypeId", "operatorTypeId", "address_stateOrRegion"]:
    try:
        values = spark.sql(f"""
            SELECT DISTINCT {col_name} FROM {ENRICHED_TABLE}
            WHERE {col_name} IS NOT NULL
            ORDER BY {col_name}
        """).collect()
        val_list = [str(r[0]) for r in values]
        print(f"{col_name}: {', '.join(val_list)}")
    except Exception as e:
        print(f"{col_name}: Error - {e}")

print("\nAnomaly flag columns:")
flag_cols = [r["col_name"] for r in schema_info if r["col_name"].startswith("flag_")]
for fc in flag_cols:
    count = spark.sql(f"SELECT COUNT(*) as c FROM {ENRICHED_TABLE} WHERE {fc} = true").collect()[0]["c"]
    print(f"  {fc}: {count} flagged")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5D. SQL Generation Prompt

# COMMAND ----------

# Build a comprehensive schema reference for the LLM
SCHEMA_REFERENCE = f"""
TABLE: {ENRICHED_TABLE}
This table contains deduplicated Ghana healthcare facility data with anomaly flags.

KEY COLUMNS:
  - pk_unique_id (string): Unique facility identifier
  - name (string): Facility name
  - facilityTypeId (string): Values: hospital, clinic, dentist, pharmacy
  - operatorTypeId (string): Values: public, private, null
  - address_city (string): City name
  - address_stateOrRegion (string): Ghana region (e.g., Greater Accra, Ashanti, Northern, etc.)
  - numberDoctors (int): Number of doctors (may be null)
  - capacity (int): Bed capacity (may be null)
  - specialties (string): JSON array of medical specialties (e.g., '["cardiology","surgery"]')
  - procedure (string): JSON array of procedures offered
  - equipment (string): JSON array of medical equipment
  - capability (string): JSON array of facility capabilities
  - description (string): Free-text description
  - search_text (string): Combined searchable text

COMPUTED COLUMNS:
  - num_procedures (int): Count of items in procedure array
  - num_equipment (int): Count of items in equipment array
  - num_capabilities (int): Count of items in capability array
  - num_specialties (int): Count of items in specialties array
  - source_count (int): Number of source URLs that mentioned this facility

ANOMALY FLAG COLUMNS (boolean):
  - flag_procedures_no_doctors: Many procedures claimed but no/few doctors listed
  - flag_capacity_no_equipment: Large bed capacity but no equipment listed
  - flag_clinic_claims_surgery: Facility type is clinic but claims surgery specialty
  - flag_too_many_specialties: Too many specialties for a small facility
  - flag_sparse_record: No procedure, equipment, capability, or description data

REGIONAL ANALYSIS TABLE: {DESERT_TABLE}
  - address_stateOrRegion, facility_count, total_doctors, total_beds
  - hospital_count, has_surgery, has_emergency, has_obstetrics, has_pediatrics
  - sparse_records, desert_score (higher = more underserved)

IMPORTANT NOTES:
  - specialties, procedure, equipment, capability are stored as JSON array STRINGS
  - To search inside them use: LOWER(specialties) LIKE '%cardiology%'
  - To count items use the pre-computed num_procedures, num_equipment, etc.
  - numberDoctors and capacity may be NULL, use COALESCE when summing
"""

SQL_GENERATION_PROMPT = ChatPromptTemplate.from_template("""You are a SQL expert for Databricks Spark SQL.
Convert the user's natural language question into a valid Spark SQL query.

{schema}

RULES:
1. Return ONLY the SQL query, no explanation, no markdown formatting, no backticks.
2. Use the exact table and column names from the schema above.
3. For specialty/procedure/equipment searches, use LOWER(column) LIKE '%term%'
4. Always handle NULLs: use COALESCE(numberDoctors, 0) when aggregating
5. NEVER use DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, or TRUNCATE.
6. Only use SELECT statements.
7. Limit results to 50 rows unless counting/aggregating.
8. Use descriptive aliases for computed columns.

USER QUESTION: {question}

SQL QUERY:""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5E. SQL Agent Function

# COMMAND ----------

BLOCKED_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]

def sql_agent(question: str, explain: bool = True):
    """
    Convert a natural language question to SQL, execute it, and optionally
    explain the results using the LLM.

    Returns: dict with keys: question, sql, result_df, result_text, explanation
    """
    with mlflow.start_run(nested=True):
        mlflow.log_param("agent", "sql_agent")
        mlflow.log_param("question", question[:250])

        # Step 1: Generate SQL
        chain = SQL_GENERATION_PROMPT | llm
        response = chain.invoke({
            "schema": SCHEMA_REFERENCE,
            "question": question
        })
        generated_sql = response.content.strip()

        # Clean up: remove markdown backticks if present
        if generated_sql.startswith("```"):
            lines = generated_sql.split("\n")
            generated_sql = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        generated_sql = generated_sql.strip().rstrip(";")

        mlflow.log_param("generated_sql", generated_sql[:500])
        print(f"Generated SQL:\n{generated_sql}\n")

        # Step 2: Safety check
        sql_upper = generated_sql.upper()
        for keyword in BLOCKED_KEYWORDS:
            if keyword in sql_upper:
                error_msg = f"BLOCKED: Generated SQL contains dangerous keyword '{keyword}'"
                print(error_msg)
                mlflow.log_param("status", "blocked")
                return {"question": question, "sql": generated_sql, "error": error_msg}

        # Step 3: Execute SQL
        try:
            result_df = spark.sql(generated_sql)
            result_rows = result_df.limit(50).collect()
            columns = result_df.columns

            # Format as text table
            result_text = " | ".join(columns) + "\n"
            result_text += "-" * len(result_text) + "\n"
            for row in result_rows:
                result_text += " | ".join(str(row[c]) for c in columns) + "\n"

            mlflow.log_metric("num_result_rows", len(result_rows))
            mlflow.log_text(result_text, "sql_results.txt")
            print(f"Results ({len(result_rows)} rows):")
            result_df.show(20, truncate=False)

        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}"
            print(error_msg)
            mlflow.log_param("status", "error")
            mlflow.log_param("error", str(e)[:500])
            return {"question": question, "sql": generated_sql, "error": error_msg}

        # Step 4: LLM explanation of results
        explanation = ""
        if explain and result_rows:
            explain_prompt = ChatPromptTemplate.from_template("""You are a healthcare data analyst for Ghana.
The user asked: {question}

We ran this SQL query:
{sql}

And got these results:
{results}

Provide a clear, concise answer to the user's question based on these results.
Include specific numbers and facility names where relevant.
If the data seems incomplete or has caveats, mention that.

ANSWER:""")
            explain_chain = explain_prompt | llm
            explain_response = explain_chain.invoke({
                "question": question,
                "sql": generated_sql,
                "results": result_text[:3000]
            })
            explanation = explain_response.content
            mlflow.log_text(explanation, "explanation.txt")

        mlflow.log_param("status", "success")
        return {
            "question": question,
            "sql": generated_sql,
            "result_df": result_df,
            "result_text": result_text,
            "explanation": explanation,
            "num_rows": len(result_rows)
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5F. Pre-Built Anomaly Detection Queries

# COMMAND ----------

ANOMALY_QUERIES = {
    "unrealistic_procedures": {
        "description": "Facilities claiming many procedures but with no/few doctors",
        "sql": f"""
            SELECT name, address_stateOrRegion, facilityTypeId,
                   numberDoctors, num_procedures, num_equipment, num_specialties
            FROM {ENRICHED_TABLE}
            WHERE num_procedures > 5
            AND (numberDoctors IS NULL OR numberDoctors < 2)
            ORDER BY num_procedures DESC
        """
    },
    "mismatched_capabilities": {
        "description": "Facilities claiming surgery specialty but with no equipment listed",
        "sql": f"""
            SELECT name, address_stateOrRegion, facilityTypeId,
                   specialties, num_equipment, numberDoctors
            FROM {ENRICHED_TABLE}
            WHERE LOWER(specialties) LIKE '%surgery%'
            AND num_equipment = 0
            ORDER BY name
        """
    },
    "clinic_claims_surgery": {
        "description": "Clinics (not hospitals) that claim surgical specialties",
        "sql": f"""
            SELECT name, address_city, address_stateOrRegion,
                   specialties, num_procedures, numberDoctors
            FROM {ENRICHED_TABLE}
            WHERE flag_clinic_claims_surgery = true
            ORDER BY num_procedures DESC
        """
    },
    "high_capacity_no_equipment": {
        "description": "Large facilities (50+ beds) with no equipment data",
        "sql": f"""
            SELECT name, address_stateOrRegion, facilityTypeId,
                   capacity, numberDoctors, num_equipment, num_procedures
            FROM {ENRICHED_TABLE}
            WHERE flag_capacity_no_equipment = true
            ORDER BY capacity DESC
        """
    },
    "too_many_specialties": {
        "description": "Facilities with 5+ specialties but fewer than 5 doctors",
        "sql": f"""
            SELECT name, address_stateOrRegion, facilityTypeId,
                   num_specialties, numberDoctors, capacity, specialties
            FROM {ENRICHED_TABLE}
            WHERE flag_too_many_specialties = true
            ORDER BY num_specialties DESC
        """
    },
    "correlated_features": {
        "description": "Facilities where expected correlated features don't match (Q4.7, Q4.9)",
        "sql": f"""
            SELECT name, address_stateOrRegion, facilityTypeId,
                   numberDoctors, capacity, num_procedures, num_equipment, num_specialties,
                   CASE
                       WHEN capacity > 100 AND num_equipment < 3 THEN 'Large capacity, minimal equipment'
                       WHEN num_specialties > 5 AND numberDoctors < 3 THEN 'Many specialties, few doctors'
                       WHEN num_procedures > 10 AND num_equipment < 2 THEN 'Many procedures, almost no equipment'
                       ELSE 'Other mismatch'
                   END as mismatch_type
            FROM {ENRICHED_TABLE}
            WHERE (capacity > 100 AND num_equipment < 3)
               OR (num_specialties > 5 AND (numberDoctors IS NULL OR numberDoctors < 3))
               OR (num_procedures > 10 AND num_equipment < 2)
            ORDER BY mismatch_type, name
        """
    },
    "single_facility_dependency": {
        "description": "Procedures that depend on only 1-2 facilities per region (Q7.5)",
        "sql": f"""
            SELECT address_stateOrRegion, facilityTypeId,
                   COUNT(*) as facility_count,
                   SUM(COALESCE(numberDoctors, 0)) as total_doctors
            FROM {ENRICHED_TABLE}
            WHERE LOWER(specialties) LIKE '%surgery%'
            GROUP BY address_stateOrRegion, facilityTypeId
            HAVING COUNT(*) <= 2
            ORDER BY facility_count ASC
        """
    }
}

# COMMAND ----------

def run_anomaly_query(query_name: str):
    """Run a pre-built anomaly detection query."""
    if query_name not in ANOMALY_QUERIES:
        print(f"Unknown query: {query_name}")
        print(f"Available: {', '.join(ANOMALY_QUERIES.keys())}")
        return None

    query = ANOMALY_QUERIES[query_name]
    print(f"=== {query['description']} ===\n")
    print(f"SQL:\n{query['sql']}\n")
    result = spark.sql(query["sql"])
    result.show(20, truncate=False)
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5G. Test SQL Agent

# COMMAND ----------

print("=" * 70)
print("TEST 1: How many hospitals have cardiology?")
print("=" * 70)
result = sql_agent("How many hospitals have cardiology?")
if "explanation" in result:
    print(f"\n{result['explanation']}")

# COMMAND ----------

print("=" * 70)
print("TEST 2: Which region has the most hospitals?")
print("=" * 70)
result = sql_agent("Which region has the most hospital-type facilities?")
if "explanation" in result:
    print(f"\n{result['explanation']}")

# COMMAND ----------

print("=" * 70)
print("TEST 3: Facilities with unrealistic procedure counts")
print("=" * 70)
result = sql_agent("Which facilities claim more than 10 procedures but have fewer than 3 doctors?")
if "explanation" in result:
    print(f"\n{result['explanation']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5H. Run Pre-Built Anomaly Queries

# COMMAND ----------

print("=" * 70)
print("ANOMALY QUERY: Unrealistic Procedures")
print("=" * 70)
run_anomaly_query("unrealistic_procedures")

# COMMAND ----------

print("=" * 70)
print("ANOMALY QUERY: Correlated Features Mismatch")
print("=" * 70)
run_anomaly_query("correlated_features")

# COMMAND ----------

print("=" * 70)
print("ANOMALY QUERY: Single Facility Dependencies")
print("=" * 70)
run_anomaly_query("single_facility_dependency")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5I. Quick SQL Helper

# COMMAND ----------

def sql_ask(question: str):
    """Quick helper to ask a natural language question and get SQL results + explanation."""
    print(f"Q: {question}")
    print("-" * 70)
    result = sql_agent(question)
    if "explanation" in result and result["explanation"]:
        print(f"\n{result['explanation']}")
    elif "error" in result:
        print(f"\nError: {result['error']}")
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Try It!
# MAGIC
# MAGIC ```python
# MAGIC sql_ask("What is the average number of doctors per hospital in each region?")
# MAGIC sql_ask("Which facilities have the most procedures listed?")
# MAGIC sql_ask("How many facilities are public vs private?")
# MAGIC sql_ask("Which regions have no hospital with surgery capability?")
# MAGIC ```

# COMMAND ----------

sql_ask("What are the top 5 regions by total bed capacity?")

# COMMAND ----------

print("=" * 60)
print("STEP 5 COMPLETE: SQL Agent")
print("=" * 60)
print(f"""
What we built:
  - Text-to-SQL agent using Groq LLM (Llama 3.3 70B)
  - Schema-aware prompt engineering for accurate SQL generation
  - Safety guardrails (blocked keywords, SELECT-only)
  - 7 pre-built anomaly detection queries
  - MLflow logging of every generated SQL + result
  - Handles Must-Have questions: 1.1, 1.2, 1.5, 4.4, 4.7, 4.8, 4.9, 7.5, 7.6

Functions available:
  - sql_agent(question)       -- full pipeline with explanation
  - sql_ask(question)         -- quick helper
  - run_anomaly_query(name)   -- pre-built anomaly queries

Next: Run notebook 06_reasoning_agent
""")
