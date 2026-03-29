# Bridging Medical Deserts: IDP Agent for the Virtue Foundation
## Databricks Accenture Hackathon -- Project Outline

---

## What This Project Is

An AI-powered healthcare intelligence system built on the Databricks platform that:
1. Ingests and deduplicates pre-parsed Ghana healthcare facility data from the Virtue Foundation
2. Enables natural language querying over structured and unstructured facility data
3. Detects anomalies and inconsistencies in facility claims
4. Identifies medical deserts -- areas where people cannot access critical healthcare
5. Helps NGO planners decide where to send doctors, equipment, and funding

---

## What We Are Working With

### The Data is Already Parsed

The Virtue Foundation already ran an IDP extraction pipeline (using Pydantic models + LLMs) to produce dataset.csv. The 4 Python files in prompts_and_pydantic_models/ show exactly how the data was created. Our job is NOT to re-extract from raw text -- it is to build an intelligent agent layer that reasons over this already-structured data.

See DATASET_ANALYSIS.md for full dataset analysis including data quality issues and strategic insights.

### Dataset Summary
- 1,003 rows, ~400-500 unique facilities (duplicates from multiple source URLs)
- 41 columns: structured (name, address, doctors, capacity) + free-text (procedure, equipment, capability)
- All facilities in Ghana
- Only ~20-30% have rich free-text data

### Agent Questions (59 Total)
The agent must answer 59 questions across 11 categories. About 16 are "Must Have" priority. See "Virtue Foundation Agent Questions - Hack Nation.md" for full list.

---

## Architecture (Databricks-Centric)

```
+------------------------------------------------------+
|              PRESENTATION LAYER                       |
|  Databricks SQL Dashboard / Lakeview / Streamlit      |
|  Natural language queries + Map visualization          |
+-------------------------+----------------------------+
                          |
+-------------------------v----------------------------+
|             SUPERVISOR AGENT (Router)                 |
|         Routes queries to the right sub-agent         |
|                                                      |
|  +----------+  +----------+  +--------------------+  |
|  | Genie    |  | Vector   |  | Medical Reasoning  |  |
|  | Text2SQL |  | Search   |  |     Agent           |  |
|  +----+-----+  +----+-----+  +---------+----------+  |
|       |              |                   |            |
|  Structured     Free-text          Complex reasoning  |
|  queries        semantic search    + anomaly detect   |
+---------|------------|-------------------|------------+
          |            |                   |
+---------v------------v-------------------v------------+
|              DATABRICKS LAKEHOUSE                     |
|                                                      |
|  +----------------+  +---------------------------+   |
|  | Delta Tables   |  | Databricks Vector Search  |   |
|  | (Deduplicated  |  | (Embeddings over          |   |
|  |  Facility Data)|  |  procedure/equip/capab)   |   |
|  +----------------+  +---------------------------+   |
|                                                      |
|  +----------------+  +---------------------------+   |
|  | Unity Catalog  |  | MLflow Experiment Tracking |   |
|  | (Governance)   |  | (Agent tracing, citations) |   |
|  +----------------+  +---------------------------+   |
+------------------------------------------------------+
```

---

## Implementation Plan (5 Days)

### Phase 1: Data Ingestion and Cleaning (Day 1 -- 3-4 hours)

#### Step 1.1 -- Databricks Workspace Setup
- Databricks Community Edition workspace
- Cluster configuration
- Groq API key stored as Databricks secret

#### Step 1.2 -- Data Cleaning and Deduplication
The dataset has duplicate rows for the same facility from different source URLs. This is the critical first step.

```python
# Group by pk_unique_id (unique facility identifier)
# Merge multiple rows into one consolidated record per facility
# Combine free-text fields from all source URLs
# Store as clean Delta table
df_clean = (
    spark.read.csv("/FileStore/dataset.csv", header=True)
    .groupBy("pk_unique_id")
    .agg(...)  # merge logic
)
df_clean.write.format("delta").saveAsTable("hackathon.vf.facilities_clean")
```

#### Step 1.3 -- Data Profiling
- Use Databricks Data Profile on the Delta table
- Identify nulls, distribution of specialties, regional coverage

---

### Phase 2: Genie / Text2SQL Agent (Day 2 -- 4-5 hours)

> Handles most "Must Have" questions -- the backbone of the system

This handles all structured queries mapped to "Genie Chat" in the questions doc:
- "How many hospitals have cardiology?"
- "Which region has the most [Type] hospitals?"
- "Which facilities claim unrealistic procedures relative to size?"
- Correlation queries between facility characteristics

#### Step 2.1 -- Set Up Genie Space
- Create a Genie space in Databricks SQL pointing to the facilities Delta table
- Configure column descriptions and table context
- Test with basic queries

#### Step 2.2 -- SQL-based Anomaly Detection
```sql
-- Example: Facilities with mismatched claims
SELECT name, address_stateOrRegion, numberDoctors, capacity,
       size(procedure) as num_procedures,
       size(equipment) as num_equipment
FROM hackathon.vf.facilities_clean
WHERE size(procedure) > 10 AND (numberDoctors IS NULL OR numberDoctors < 2)
```

---

### Phase 3: Vector Search for Free-Text Queries (Day 2-3 -- 4-5 hours)

> Handles semantic search over procedure, equipment, capability fields

#### Step 3.1 -- Create Combined Text Column
```python
# Create a searchable text column combining all free-text fields
df_with_text = df_clean.withColumn(
    "combined_text",
    concat_ws(" | ",
        col("name"), col("description"),
        col("procedure"), col("equipment"), col("capability"),
        col("specialties")
    )
)
```

#### Step 3.2 -- Databricks Vector Search Index
```python
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
vsc.create_endpoint(name="facility-search")
vsc.create_delta_sync_index(
    endpoint_name="facility-search",
    source_table_name="hackathon.vf.facilities_searchable",
    primary_key="pk_unique_id",
    embedding_source_column="combined_text",
    embedding_model_endpoint_name="databricks-bge-large-en"
)
```

#### Step 3.3 -- RAG Chain with Citations
```python
# Query vector index, pass results to LLM, return answer with source citations
results = index.similarity_search(query_text="hospitals with CT scanners", num_results=5)
# Each result includes pk_unique_id for row-level citation
```

---

### Phase 4: Medical Reasoning Agent (Day 3 -- 5-6 hours)

> Handles anomaly detection and complex medical inference

#### Step 4.1 -- Configure LLM Access
```python
# Register Groq (Llama 3.1 70B) as external model endpoint
import mlflow.deployments
client = mlflow.deployments.get_deploy_client("databricks")
client.create_endpoint(
    name="groq-llama",
    config={
        "served_entities": [{
            "external_model": {
                "name": "llama-3.1-70b-versatile",
                "provider": "custom",
                "task": "llm/v1/chat",
                "custom_provider_config": {
                    "api_base": "https://api.groq.com/openai/v1",
                    "api_key": "{{secrets/hackathon/GROQ_API_KEY}}"
                }
            }
        }]
    }
)
```

#### Step 4.2 -- Medical Reasoning Capabilities
The reasoning agent handles questions like:
- "Which facilities claim to offer [subspecialty] but lack required equipment?"
- "Which facilities have high bed-to-OR ratios indicative of misrepresentation?"
- "Where do we see things that should not move together?"

```python
# Example: Cross-validate procedure claims against equipment
medical_prompt = """
Given this facility data:
- Name: {name}
- Claimed procedures: {procedures}
- Listed equipment: {equipment}
- Capacity: {capacity} beds
- Doctors: {num_doctors}

Identify any anomalies or inconsistencies. Consider:
1. Are procedures claimed without necessary equipment?
2. Is the procedure count realistic for the facility size?
3. Are there mismatches between specialties and infrastructure?
"""
```

#### Step 4.3 -- MLflow Tracing for Citations
```python
import mlflow
mlflow.set_experiment("/hackathon/medical-reasoning")

# Each agent reasoning step is logged
# Provides step-level citations (stretch goal)
```

---

### Phase 5: Supervisor Agent + Integration (Day 4 -- 4-5 hours)

#### Step 5.1 -- Build Supervisor Router
```python
# Supervisor Agent: classifies query intent and routes to sub-agent
# Intent categories:
#   - STRUCTURED -> Genie (Text2SQL)
#   - SEMANTIC_SEARCH -> Vector Search
#   - REASONING -> Medical Reasoning Agent
#   - GEOSPATIAL -> Geospatial calculation + Genie
#   - MULTI_STEP -> Chain multiple sub-agents
```

#### Step 5.2 -- Agent Tools

| Tool | Maps To | Handles |
|---|---|---|
| text2sql_query | Genie Chat | Structured queries, counts, aggregations |
| semantic_search | Vector Search | Free-text facility lookups |
| medical_reason | Reasoning Agent | Anomaly detection, inference |
| geo_distance | Geospatial Calc | Distance and coverage queries |
| get_facility | Delta Table lookup | Single facility details |

---

### Phase 6: Medical Deserts + Visualization (Day 4-5 -- 3-4 hours)

#### Step 6.1 -- Medical Desert Analysis
```python
# Aggregate by region, identify gaps
spark.sql("""
    SELECT address_stateOrRegion AS region,
           COUNT(DISTINCT pk_unique_id) AS facility_count,
           SUM(numberDoctors) AS total_doctors,
           SUM(capacity) AS total_beds,
           COLLECT_SET(specialties) AS available_specialties
    FROM hackathon.vf.facilities_clean
    GROUP BY address_stateOrRegion
    ORDER BY facility_count ASC
""")
```

#### Step 6.2 -- Map Visualization
- Databricks SQL Dashboard with map widget or Folium map in notebook
- Color-coded markers by facility type and capability
- Highlighted desert regions

---

### Phase 7: Polish and Demo (Day 5 -- 3-4 hours)

- Build Databricks SQL Dashboard / Lakeview dashboard
- Finalize and test all Must-Have queries
- Record 5-minute demo video
- Write README with clear setup instructions
- Ensure GitHub repo has open-source license

---

## Databricks Features Utilization

| Databricks Feature | How We Use It | Evaluation Area |
|---|---|---|
| Delta Tables (Unity Catalog) | Store deduplicated facility data | Foundation |
| Genie (Text2SQL) | Natural language to SQL for structured queries | Technical Accuracy (35%) |
| Databricks Vector Search | Semantic search over free-text fields | IDP Innovation (30%) |
| External Model Endpoints | Groq/Llama 3.1 for reasoning agent | IDP Innovation (30%) |
| MLflow Tracking + Tracing | Agent step logging, citations | Citations (stretch) |
| Spark SQL | Geospatial aggregation and gap analysis | Social Impact (25%) |
| SQL Dashboards / Lakeview | Visualize medical deserts and coverage | User Experience (10%) |
| Databricks Secrets | Secure API key storage | Best practice |

---

## Prioritization

| Priority | Component | Evaluation Weight | Time |
|---|---|---|---|
| P0 | Data cleaning + deduplication into Delta Tables | Foundation | 3-4 hrs |
| P0 | Genie / Text2SQL for structured queries | 35% Technical Accuracy | 4-5 hrs |
| P0 | Vector Search for free-text queries | 30% IDP Innovation | 4-5 hrs |
| P1 | Medical Reasoning Agent (anomalies) | Part of 35% + 30% | 5-6 hrs |
| P1 | Supervisor Agent (routing) | Part of 35% | 2-3 hrs |
| P1 | Medical Desert analysis | 25% Social Impact | 3-4 hrs |
| P2 | Dashboard / Map visualization | 10% UX | 3-4 hrs |
| P2 | MLflow tracing for citations | Bonus | 2 hrs |
| P3 | Geospatial distance calculations | Bonus | 2 hrs |

---

## 5-Day Timeline

| Day | Focus | Deliverables |
|---|---|---|
| Day 1 (Mar 29) | Workspace setup + Data cleaning + dedup | Clean Delta tables ready |
| Day 2 (Mar 30) | Genie Text2SQL + Vector Search setup | Working structured + semantic queries |
| Day 3 (Mar 31) | Medical Reasoning Agent + anomaly detection | Complex query handling |
| Day 4 (Apr 1) | Supervisor Agent + Medical Deserts + Map | Full agent pipeline + visualization |
| Day 5 (Apr 2) | Dashboard + Testing + Demo Video | Submission-ready project |

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Platform | Databricks Community Edition | Required by challenge |
| Data Storage | Delta Tables in Unity Catalog | Deduplicated facility data |
| LLM | Groq (Llama 3.1 70B) via External Endpoint | Free, fast inference |
| Embeddings | Databricks BGE or HuggingFace all-MiniLM-L6-v2 | For vector search |
| Vector Search | Databricks Vector Search | Native RAG support |
| Text2SQL | Genie | Structured query backbone |
| Agent Framework | Mosaic AI Agent Framework / LangGraph | Orchestration |
| Experiment Tracking | MLflow (native) | Citations and tracing |
| Dashboards | Databricks SQL Dashboard / Lakeview | Visualization |

---

## Submission Checklist

- [ ] Built using required Databricks tools during hackathon period
- [ ] Clear text description explaining features and functionality
- [ ] Working demo link or test build with credentials for judges
- [ ] Public GitHub repository with open-source license
- [ ] Demo video (up to 5 minutes) showcasing solution and impact

---

## Project Structure

```
databricks-hackathon/
|-- README.md
|-- PROJECT_OUTLINE.md
|-- DATASET_ANALYSIS.md
|-- requirements.txt
|-- data/
|   +-- dataset.csv
|-- prompts_and_pydantic_models/        # VF extraction pipeline (reference)
|-- notebooks/                          # Databricks notebooks
|   |-- 01_data_cleaning.py             # Dedup + load into Delta tables
|   |-- 02_genie_setup.py               # Genie Text2SQL configuration
|   |-- 03_vector_search.py             # Vector Search index creation
|   |-- 04_reasoning_agent.py           # Medical Reasoning Agent
|   |-- 05_supervisor_agent.py          # Router + orchestration
|   |-- 06_medical_deserts.py           # Gap analysis
|   +-- 07_dashboard.py                 # Visualization
|-- src/
|   |-- agents/
|   |   |-- supervisor.py               # Query router
|   |   |-- medical_reasoner.py         # Medical reasoning
|   |   +-- tools.py                    # Agent tools
|   |-- analysis/
|   |   |-- anomaly_detector.py         # Anomaly detection logic
|   |   +-- medical_deserts.py          # Desert identification
|   +-- data/
|       +-- deduplicator.py             # Data cleaning logic
+-- app.py                               # Streamlit app (optional)
```

---

## Learning Resources

| Topic | Resource |
|---|---|
| Databricks Community Edition | https://community.cloud.databricks.com |
| Databricks Vector Search | https://docs.databricks.com/en/generative-ai/vector-search.html |
| Mosaic AI Agent Framework | https://docs.databricks.com/en/generative-ai/agent-framework/index.html |
| MLflow on Databricks | https://docs.databricks.com/en/mlflow/index.html |
| Genie (Text2SQL) | https://docs.databricks.com/en/genie/index.html |
| Delta Tables | https://docs.databricks.com/en/delta/index.html |
| Groq API | https://console.groq.com/docs/quickstart |
