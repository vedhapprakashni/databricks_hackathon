# Databricks notebook source
# MAGIC %md
# MAGIC # Step 6: Medical Reasoning Agent
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Builds a medical reasoning agent that cross-validates facility claims
# MAGIC 2. Detects anomalies with medical context (equipment ↔ procedures ↔ specialties)
# MAGIC 3. Performs cross-facility comparison within regions
# MAGIC 4. Provides medical desert reasoning with actionable insights
# MAGIC 5. Rates facility credibility (HIGH / MEDIUM / LOW)
# MAGIC
# MAGIC **Run notebooks 00–05 first!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6A. Install Packages

# COMMAND ----------

# MAGIC %pip install langchain langchain-community mlflow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6B. Configuration

# COMMAND ----------

import os
import json
import mlflow
from pyspark.sql import functions as F
from langchain_community.chat_models import ChatDatabricks
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
print(f"Enriched table: {ENRICHED_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6C. Load Facility Data

# COMMAND ----------

df_enriched = spark.table(ENRICHED_TABLE)
total = df_enriched.count()
print(f"Loaded {total} facilities from {ENRICHED_TABLE}")

# Load into a list of dicts for individual facility analysis
facilities_pdf = df_enriched.toPandas()
print(f"Converted to pandas: {len(facilities_pdf)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6D. Facility Anomaly Reasoning

# COMMAND ----------

ANOMALY_PROMPT = ChatPromptTemplate.from_template("""You are a medical facility auditor analyzing Ghana healthcare data for the Virtue Foundation.

FACILITY DATA:
  Name: {name}
  Location: {city}, {region}
  Type: {facility_type}
  Operator: {operator_type}
  Number of Doctors: {num_doctors}
  Bed Capacity: {capacity}
  Claimed Specialties: {specialties}
  Claimed Procedures: {procedures}
  Listed Equipment: {equipment}
  Capabilities: {capabilities}
  Description: {description}
  Number of source URLs: {source_count}

EXISTING FLAGS:
  {flags}

ANALYZE THIS FACILITY FOR:

1. **PROCEDURE-EQUIPMENT MISMATCHES**: Which claimed procedures lack the necessary equipment?
   - Example: Cataract surgery requires an operating microscope and phacoemulsification machine
   - Example: CT scans require a CT scanner
   - Example: MRI requires an MRI machine
   - Example: Surgery requires at minimum an operating theater, anesthesia equipment

2. **SCALE ISSUES**: Is the number of procedures/specialties realistic for the facility size?
   - A clinic with 2 doctors cannot realistically offer 15 specialties
   - A small clinic should not claim the same breadth as a teaching hospital

3. **TYPE MISMATCHES**: Are the claims consistent with the facility type?
   - A pharmacy claiming surgical procedures is suspicious
   - A dentist office claiming cardiac surgery is suspicious
   - A clinic claiming level-1 trauma care is unusual

4. **CREDIBILITY SIGNALS**: What gives confidence or doubt?
   - Multiple source URLs corroborating claims = positive
   - Very specific equipment lists = positive
   - Vague claims without specifics = negative

For EACH finding, explain WHY it is medically suspicious with specific reasoning.

OVERALL CREDIBILITY RATING: Rate as HIGH / MEDIUM / LOW with a one-line justification.

Format your response as:
## Findings
[numbered findings with medical reasoning]

## Credibility Rating
[HIGH/MEDIUM/LOW]: [justification]
""")

# COMMAND ----------

def analyze_facility(facility_row) -> dict:
    """Analyze a single facility for anomalies using medical reasoning."""
    # Extract fields, handling pandas NaN/None
    def safe_get(row, key, default="Not available"):
        val = row.get(key, default)
        if val is None or (isinstance(val, float) and str(val) == "nan"):
            return default
        return str(val)

    # Build flags text
    flag_cols = [c for c in facility_row.index if c.startswith("flag_") and facility_row.get(c) == True]
    flags_text = ", ".join(flag_cols) if flag_cols else "None flagged"

    response = llm.invoke(
        ANOMALY_PROMPT.format(
            name=safe_get(facility_row, "name"),
            city=safe_get(facility_row, "address_city"),
            region=safe_get(facility_row, "address_stateOrRegion"),
            facility_type=safe_get(facility_row, "facilityTypeId"),
            operator_type=safe_get(facility_row, "operatorTypeId"),
            num_doctors=safe_get(facility_row, "numberDoctors"),
            capacity=safe_get(facility_row, "capacity"),
            specialties=safe_get(facility_row, "specialties", "[]"),
            procedures=safe_get(facility_row, "procedure", "[]"),
            equipment=safe_get(facility_row, "equipment", "[]"),
            capabilities=safe_get(facility_row, "capability", "[]"),
            description=safe_get(facility_row, "description", "No description"),
            source_count=safe_get(facility_row, "source_count", "1"),
            flags=flags_text
        )
    )

    return {
        "name": safe_get(facility_row, "name"),
        "region": safe_get(facility_row, "address_stateOrRegion"),
        "analysis": response.content,
        "has_flags": len(flag_cols) > 0
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6E. Analyze Flagged Facilities

# COMMAND ----------

# Get facilities with at least one anomaly flag
flag_columns = [c for c in facilities_pdf.columns if c.startswith("flag_")]
flagged_mask = facilities_pdf[flag_columns].any(axis=1)
flagged_facilities = facilities_pdf[flagged_mask]
print(f"Facilities with at least one anomaly flag: {len(flagged_facilities)}")

# COMMAND ----------

# Analyze the top flagged facilities (limit to avoid API rate limits)
MAX_ANALYZE = 5
print(f"Analyzing top {MAX_ANALYZE} flagged facilities...\n")

analyses = []
for idx, (_, row) in enumerate(flagged_facilities.head(MAX_ANALYZE).iterrows()):
    print(f"{'=' * 70}")
    print(f"[{idx+1}/{MAX_ANALYZE}] Analyzing: {row.get('name', 'Unknown')}")
    print(f"{'=' * 70}")
    
    result = analyze_facility(row)
    analyses.append(result)
    print(result["analysis"])
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6F. Cross-Facility Regional Comparison

# COMMAND ----------

REGION_COMPARISON_PROMPT = ChatPromptTemplate.from_template("""You are a healthcare system analyst for Ghana, working with the Virtue Foundation.

REGION: {region}
TOTAL FACILITIES: {facility_count}
TOTAL DOCTORS: {total_doctors}
TOTAL BEDS: {total_beds}

FACILITIES IN THIS REGION:
{facility_details}

REGIONAL ANALYSIS:
{desert_info}

Perform a comprehensive regional healthcare assessment:

1. **SPECIALTY COVERAGE**: What medical specialties are available? What critical specialties are MISSING?
   - Consider: emergency medicine, surgery, obstetrics/gynecology, pediatrics, internal medicine, cardiology

2. **SINGLE-POINT DEPENDENCIES**: Which procedures or specialties depend on only 1 facility?
   If that facility closes, the entire region loses that capability.

3. **RESOURCE DISTRIBUTION**: Is the doctor/bed distribution balanced, or concentrated in few facilities?

4. **GAPS AND RECOMMENDATIONS**: What should the Virtue Foundation prioritize in this region?
   - What type of facility/service is most needed?
   - Where should volunteer doctors be sent?
   - What equipment donations would have the most impact?

Provide specific, actionable recommendations.
""")

# COMMAND ----------

def compare_region(region: str):
    """Perform cross-facility comparison for a given region."""
    region_facilities = facilities_pdf[facilities_pdf["address_stateOrRegion"] == region]

    if len(region_facilities) == 0:
        print(f"No facilities found for region: {region}")
        return None

    # Build facility details text
    details = []
    for _, row in region_facilities.iterrows():
        def safe(val, default="N/A"):
            if val is None or (isinstance(val, float) and str(val) == "nan"):
                return default
            return str(val)

        detail = (
            f"- {safe(row.get('name'))}"
            f" [{safe(row.get('facilityTypeId'))}]"
            f" Doctors: {safe(row.get('numberDoctors'))}"
            f" Beds: {safe(row.get('capacity'))}"
            f" Specialties: {safe(row.get('specialties'), '[]')}"
            f" Procedures: {safe(row.get('num_procedures'), '0')}"
            f" Equipment: {safe(row.get('num_equipment'), '0')}"
        )
        details.append(detail)
    facility_text = "\n".join(details[:30])  # Limit to avoid token overflow

    # Get desert info
    try:
        desert_rows = spark.sql(f"""
            SELECT * FROM {DESERT_TABLE}
            WHERE address_stateOrRegion = '{region}'
        """).collect()
        desert_info = ""
        if desert_rows:
            r = desert_rows[0]
            desert_info = (
                f"Desert Score: {r['desert_score']} (higher = more underserved)\n"
                f"Hospitals: {r['hospital_count']}\n"
                f"Has Surgery: {'Yes' if r['has_surgery'] > 0 else 'No'}\n"
                f"Has Emergency: {'Yes' if r['has_emergency'] > 0 else 'No'}\n"
                f"Has Obstetrics: {'Yes' if r['has_obstetrics'] > 0 else 'No'}\n"
                f"Has Pediatrics: {'Yes' if r['has_pediatrics'] > 0 else 'No'}\n"
                f"Sparse Records: {r['sparse_records']}"
            )
    except Exception:
        desert_info = "Desert analysis not available"

    total_doctors = region_facilities["numberDoctors"].fillna(0).astype(float).sum()
    total_beds = region_facilities["capacity"].fillna(0).astype(float).sum()

    response = llm.invoke(
        REGION_COMPARISON_PROMPT.format(
            region=region,
            facility_count=len(region_facilities),
            total_doctors=int(total_doctors),
            total_beds=int(total_beds),
            facility_details=facility_text,
            desert_info=desert_info
        )
    )

    return {
        "region": region,
        "facility_count": len(region_facilities),
        "analysis": response.content
    }

# COMMAND ----------

# Analyze the top 3 most underserved regions
print("Identifying most underserved regions...\n")

desert_df = spark.table(DESERT_TABLE).orderBy("desert_score", ascending=False).limit(3).collect()
for row in desert_df:
    region = row["address_stateOrRegion"]
    print(f"{'=' * 70}")
    print(f"REGION: {region} (Desert Score: {row['desert_score']})")
    print(f"{'=' * 70}")
    
    result = compare_region(region)
    if result:
        print(result["analysis"])
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6G. Medical Desert Deep-Dive Reasoning

# COMMAND ----------

DESERT_REASONING_PROMPT = ChatPromptTemplate.from_template("""You are an expert in healthcare access and medical deserts, advising the Virtue Foundation on Ghana.

REGIONAL HEALTHCARE DATA:
{regional_data}

CRITICAL QUESTIONS TO ANSWER:

1. **MEDICAL DESERTS**: Which regions qualify as medical deserts? A medical desert has:
   - Few or no hospitals
   - No surgical capability
   - No emergency medicine capability
   - Very few doctors relative to likely population
   - High concentration of sparse/unverified facility records

2. **EMERGENCY CARE GAPS**: Where would a person having a medical emergency (heart attack, trauma, difficult childbirth) have NO nearby facility capable of treating them?

3. **SURGICAL DESERTS**: Which regions have NO verified surgical capability? What would happen to a patient needing emergency surgery?

4. **PRIORITY RANKING**: Rank the regions from MOST to LEAST underserved, considering:
   - Facility count and type
   - Doctor availability
   - Specialty coverage (especially emergency, surgery, obstetrics)
   - Data quality (sparse records suggest even worse reality)

5. **ACTIONABLE RECOMMENDATIONS**: For the top 3 most underserved regions, recommend:
   - What type of medical mission would have the highest impact?
   - What equipment should be prioritized?
   - Should the focus be on building new facilities or strengthening existing ones?

Be specific and data-driven in every recommendation.
""")

# COMMAND ----------

def medical_desert_reasoning():
    """Perform comprehensive medical desert analysis with reasoning."""
    # Build regional summary
    desert_rows = spark.table(DESERT_TABLE).orderBy("desert_score", ascending=False).collect()

    regional_data = "Region | Facilities | Doctors | Beds | Hospitals | Surgery | Emergency | Obstetrics | Pediatrics | Desert Score\n"
    regional_data += "-" * 120 + "\n"
    for r in desert_rows:
        regional_data += (
            f"{r['address_stateOrRegion']} | "
            f"{r['facility_count']} | "
            f"{r['total_doctors']} | "
            f"{r['total_beds']} | "
            f"{r['hospital_count']} | "
            f"{'Yes' if r['has_surgery'] > 0 else 'NO'} | "
            f"{'Yes' if r['has_emergency'] > 0 else 'NO'} | "
            f"{'Yes' if r['has_obstetrics'] > 0 else 'NO'} | "
            f"{'Yes' if r['has_pediatrics'] > 0 else 'NO'} | "
            f"{r['desert_score']}\n"
        )

    response = llm.invoke(
        DESERT_REASONING_PROMPT.format(regional_data=regional_data)
    )

    return response.content

# COMMAND ----------

print("=" * 70)
print("MEDICAL DESERT DEEP-DIVE ANALYSIS")
print("=" * 70)
desert_analysis = medical_desert_reasoning()
print(desert_analysis)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6H. General Reasoning Agent Function

# COMMAND ----------

REASONING_PROMPT = ChatPromptTemplate.from_template("""You are a medical reasoning agent for the Virtue Foundation's Ghana healthcare intelligence system.
You have deep knowledge of medical facility requirements, equipment needs, and healthcare system analysis.

The user has a complex medical reasoning question. Use the facility data provided to give a thorough, evidence-based answer.

FACILITY DATA:
{facility_data}

REGIONAL SUMMARY:
{regional_summary}

USER QUESTION: {question}

Provide a detailed, medically-informed answer. For each claim, cite specific facilities or regions from the data.
If the data is insufficient to fully answer, state what additional information would be needed.

DETAILED ANSWER:""")

def reasoning_agent(question: str):
    """General-purpose medical reasoning agent that combines facility data with medical knowledge."""
    with mlflow.start_run(nested=True):
        mlflow.log_param("agent", "reasoning_agent")
        mlflow.log_param("question", question[:250])

        q_lower = question.lower()

        # Pull relevant facilities based on question keywords
        relevant = facilities_pdf.copy()

        # Filter by region if mentioned
        regions = facilities_pdf["address_stateOrRegion"].dropna().unique()
        for region in regions:
            if region and region.lower() in q_lower:
                relevant = relevant[relevant["address_stateOrRegion"] == region]
                break

        # Filter by type if mentioned
        type_map = {"hospital": "hospital", "clinic": "clinic", "pharmacy": "pharmacy", "dentist": "dentist"}
        for keyword, ftype in type_map.items():
            if keyword in q_lower:
                relevant = relevant[relevant["facilityTypeId"] == ftype]
                break

        # Limit context size
        if len(relevant) > 20:
            # Prioritize facilities with more data
            relevant = relevant.sort_values("source_count", ascending=False).head(20)

        # Format facility data
        fac_text = []
        for _, row in relevant.iterrows():
            def safe(val, default="N/A"):
                if val is None or (isinstance(val, float) and str(val) == "nan"):
                    return default
                return str(val)[:200]
            fac_text.append(
                f"- {safe(row.get('name'))} [{safe(row.get('facilityTypeId'))}] "
                f"in {safe(row.get('address_city'))}, {safe(row.get('address_stateOrRegion'))} | "
                f"Doctors: {safe(row.get('numberDoctors'))} | Beds: {safe(row.get('capacity'))} | "
                f"Specialties: {safe(row.get('specialties'), '[]')} | "
                f"Procedures: {safe(row.get('num_procedures'), '0')} | "
                f"Equipment: {safe(row.get('num_equipment'), '0')}"
            )
        facility_data = "\n".join(fac_text) if fac_text else "No matching facilities found."

        # Regional summary
        try:
            desert_rows = spark.table(DESERT_TABLE).collect()
            reg_text = []
            for r in desert_rows:
                reg_text.append(
                    f"{r['address_stateOrRegion']}: "
                    f"{r['facility_count']} facilities, {r['total_doctors']} doctors, "
                    f"Surgery: {'Yes' if r['has_surgery'] > 0 else 'No'}, "
                    f"Emergency: {'Yes' if r['has_emergency'] > 0 else 'No'}, "
                    f"Desert Score: {r['desert_score']}"
                )
            regional_summary = "\n".join(reg_text)
        except Exception:
            regional_summary = "Regional analysis not available."

        # Generate reasoning
        chain = REASONING_PROMPT | llm
        response = chain.invoke({
            "facility_data": facility_data,
            "regional_summary": regional_summary,
            "question": question
        })

        answer = response.content
        mlflow.log_text(answer, "reasoning_answer.txt")
        mlflow.log_param("num_facilities_used", len(relevant))

        return {
            "question": question,
            "answer": answer,
            "facilities_analyzed": len(relevant)
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6I. Test Medical Reasoning

# COMMAND ----------

print("=" * 70)
print("Q4.4: Facilities with unrealistic procedure claims")
print("=" * 70)
result = reasoning_agent("Which facilities claim an unrealistic number of procedures relative to their size, number of doctors, or facility type? Explain why each is suspicious.")
print(result["answer"])

# COMMAND ----------

print("=" * 70)
print("Q4.9: Things that shouldn't move together")
print("=" * 70)
result = reasoning_agent("Where do we see 'things that shouldn't move together'? For example, very large bed count but minimal surgical equipment, or highly specialized claims with no supporting signals.")
print(result["answer"])

# COMMAND ----------

print("=" * 70)
print("Q6.1: Where is the workforce for surgery practicing?")
print("=" * 70)
result = reasoning_agent("Where is the workforce for surgery actually practicing in Ghana? Which regions have surgical specialists and which regions have none?")
print(result["answer"])

# COMMAND ----------

print("=" * 70)
print("Q7.6: Oversupply vs scarcity of complex procedures")
print("=" * 70)
result = reasoning_agent("Where is there oversupply concentration where many facilities claim the same low-complexity procedure versus scarcity of high-complexity procedures?")
print(result["answer"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6J. Quick Reasoning Helper

# COMMAND ----------

def reason(question: str):
    """Quick helper to ask a medical reasoning question."""
    print(f"Q: {question}")
    print("-" * 70)
    result = reasoning_agent(question)
    print(result["answer"])
    print(f"\n[{result['facilities_analyzed']} facilities analyzed]")
    return result

# COMMAND ----------

print("=" * 60)
print("STEP 6 COMPLETE: Medical Reasoning Agent")
print("=" * 60)
print(f"""
What we built:
  - Facility anomaly analyzer with medical reasoning (procedure ↔ equipment)
  - Cross-facility regional comparison
  - Medical desert deep-dive reasoning
  - General-purpose reasoning agent
  - MLflow logging for all reasoning steps

Functions available:
  - analyze_facility(row)          -- analyze single facility
  - compare_region(region_name)    -- cross-facility analysis for a region
  - medical_desert_reasoning()     -- comprehensive desert analysis
  - reasoning_agent(question)      -- general medical reasoning
  - reason(question)               -- quick helper

Handles Must-Have questions:
  - 4.4: Unrealistic procedure claims
  - 4.7: Correlated feature analysis
  - 4.8: High breadth vs infrastructure
  - 4.9: Things that shouldn't move together
  - 6.1: Workforce distribution
  - 7.5: Single-facility dependencies
  - 7.6: Oversupply vs scarcity

Next: Run notebook 07_supervisor_agent
""")