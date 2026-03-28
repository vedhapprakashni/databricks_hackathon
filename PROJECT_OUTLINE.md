# Bridging Medical Deserts: IDP Agent for the Virtue Foundation
## Databricks Accenture Hackathon — Complete Project Outline

---

## What This Project Is

An AI-powered healthcare intelligence system built entirely on the Databricks platform that:
1. Reads messy, unstructured hospital data from Ghana (provided by the Virtue Foundation)
2. Extracts structured information using Intelligent Document Parsing (IDP)
3. Identifies "medical deserts" — areas where people cannot access critical healthcare
4. Helps NGO planners decide where to send doctors, equipment, and funding

---

## Why Databricks Is Central

This is a Databricks hackathon — every core component runs on or integrates with Databricks:

| Challenge Requirement | Databricks Solution |
|---|---|
| Data ingestion and storage | Unity Catalog + Delta Tables |
| Unstructured text parsing (IDP) | Databricks Foundation Model APIs (Llama 3.1 via pay-per-token) or External Model Endpoints (Groq) |
| Vector search for RAG | Databricks Vector Search (built into Unity Catalog) |
| Agentic orchestration | Mosaic AI Agent Framework |
| Experiment tracking and citations | MLflow (native to Databricks) |
| SQL querying over data | Databricks SQL + Genie (Text2SQL) |
| Serving the final agent | Databricks Model Serving |
| Dashboards and visualization | Databricks SQL Dashboards + Lakeview |

---

## Databricks-Centric Architecture

```
+------------------------------------------------------+
|              PRESENTATION LAYER                       |
|  Databricks SQL Dashboard / Lakeview / Streamlit      |
|  Natural language queries + Map visualization          |
+-------------------------+----------------------------+
                          |
+-------------------------v----------------------------+
|          MOSAIC AI AGENT FRAMEWORK                    |
|          (Agentic Orchestration Layer)                |
|                                                      |
|  +----------+  +----------+  +--------------------+  |
|  | IDP Agent|  | Analysis |  | Planning/Recommend |  |
|  | (Parser) |  |  Agent   |  |      Agent         |  |
|  +----+-----+  +----+-----+  +---------+----------+  |
+-------|--------------|-----------------------|-------+
        |              |                       |
+-------v--------------v-----------------------v-------+
|              DATABRICKS LAKEHOUSE                     |
|                                                      |
|  +----------------+  +---------------------------+   |
|  | Delta Tables   |  | Databricks Vector Search  |   |
|  | (Raw + Parsed  |  | (Embeddings for RAG)      |   |
|  |  Facility Data)|  |                           |   |
|  +----------------+  +---------------------------+   |
|                                                      |
|  +----------------+  +---------------------------+   |
|  | Unity Catalog  |  | MLflow Experiment Tracking |   |
|  | (Governance)   |  | (Agent tracing, citations) |   |
|  +----------------+  +---------------------------+   |
|                                                      |
|  +----------------+  +---------------------------+   |
|  | Genie          |  | Foundation Model APIs     |   |
|  | (Text2SQL)     |  | (LLM access for agents)   |   |
|  +----------------+  +---------------------------+   |
+------------------------------------------------------+
```

---

## Key Concepts

| Concept | What It Means | Databricks Feature |
|---|---|---|
| IDP (Intelligent Document Parsing) | Using AI to extract structured data from messy text | Foundation Model APIs + Notebooks |
| RAG (Retrieval-Augmented Generation) | Feeding relevant documents to an LLM for answers | Databricks Vector Search |
| Agentic AI | AI that plans, reasons, and takes multi-step actions | Mosaic AI Agent Framework |
| Vector Database | Stores text as numbers for semantic search | Databricks Vector Search in Unity Catalog |
| Text2SQL | Converting natural language to SQL queries | Genie |
| Experiment Tracking | Logging model inputs/outputs for transparency | MLflow (native in Databricks) |

---

## Step-by-Step Implementation Plan

### Phase 1: Databricks Setup and Data Ingestion (Day 1 — 3-4 hours)

#### Step 1.1 — Databricks Workspace Setup
- Create a Databricks Community Edition workspace at community.cloud.databricks.com
- Create a cluster (Community Edition gives a single-node cluster)
- Set up Unity Catalog (if available on Community Edition, otherwise use Hive metastore)
- Configure Groq API key as a Databricks secret:
  ```python
  # In Databricks notebook
  dbutils.secrets.createScope("hackathon")
  # Store GROQ_API_KEY in the secret scope
  ```

#### Step 1.2 — Ingest Ghana Dataset into Delta Tables
```python
# Load CSV into Databricks and store as Delta Table
df = spark.read.csv("/FileStore/ghana_facilities.csv", header=True, inferSchema=True)
df.write.format("delta").saveAsTable("hackathon.virtue_foundation.raw_facilities")

# Explore the data
display(df.describe())
display(df.select("procedure", "equipment", "capability").limit(10))
```

#### Step 1.3 — Data Profiling
- Use Databricks Data Profile feature on the Delta table
- Identify structured vs unstructured columns
- Assess data quality: nulls, inconsistencies, duplicates

---

### Phase 2: IDP Agent — Document Parsing on Databricks (Day 1-2 — 6-8 hours)

> Core of the project — 30% of evaluation (IDP Innovation)

#### Step 2.1 — Configure LLM Access in Databricks
Two options:
- Option A: Databricks Foundation Model APIs (built-in, pay-per-token)
- Option B: External Model Endpoint pointing to Groq API

```python
# Option B: Register Groq as an external model endpoint
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

#### Step 2.2 — Build IDP Extraction Pipeline
Use Databricks notebooks with Pydantic models for structured extraction:

```python
from pydantic import BaseModel
from typing import List

class FacilityExtraction(BaseModel):
    procedures: List[str]
    equipment: List[str]
    capabilities: List[str]
    specialties: List[str]
    has_emergency: bool
    has_surgery: bool
    has_imaging: bool
    anomalies: List[str]       # Suspicious or inconsistent claims

# Process each facility row through the LLM
# Parse free-text fields (procedure, equipment, capability)
# Store structured output back to a Delta table
```

#### Step 2.3 — Anomaly Detection
Flag facilities with suspicious data:
- Claims advanced equipment but has 0 doctors
- Says "Level II trauma center" but no emergency capability
- Has massive capacity but tiny facility area

#### Step 2.4 — Store Parsed Results in Delta Tables
```python
# Save structured extractions as a new Delta table
parsed_df.write.format("delta").saveAsTable("hackathon.virtue_foundation.parsed_facilities")
```

---

### Phase 3: RAG System with Databricks Vector Search (Day 2-3 — 4-5 hours)

> Powers the 35% Technical Accuracy score

#### Step 3.1 — Create Embeddings and Vector Search Index
```python
# Use Databricks Vector Search (built into Unity Catalog)
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

# Create a vector search endpoint
vsc.create_endpoint(name="facility-search-endpoint")

# Create a Delta Sync Index (auto-syncs with your Delta table)
vsc.create_delta_sync_index(
    endpoint_name="facility-search-endpoint",
    index_name="hackathon.virtue_foundation.facility_index",
    source_table_name="hackathon.virtue_foundation.parsed_facilities",
    pipeline_type="TRIGGERED",
    primary_key="facility_id",
    embedding_source_column="combined_text",
    embedding_model_endpoint_name="databricks-bge-large-en"
)
```

#### Step 3.2 — Build RAG Chain
```python
# Query the vector index for relevant facilities
results = vsc.get_index("hackathon.virtue_foundation.facility_index").similarity_search(
    query_text="hospitals with CT scanners in Northern Ghana",
    columns=["name", "address_city", "equipment", "capability"],
    num_results=5
)

# Feed results to LLM for natural language answer with citations
```

#### Step 3.3 — Add Citations
Each answer includes which facility records were used:
- Return source document IDs and row references
- Use MLflow tracing to log each reasoning step for agentic-step-level citations

---

### Phase 4: Agentic Orchestration with Mosaic AI (Day 3 — 5-6 hours)

#### Step 4.1 — Define Agent Tools

| Tool | Purpose |
|---|---|
| search_facilities | Databricks Vector Search query |
| get_facility_details | SQL query on Delta table |
| analyze_region | Aggregated stats via Spark SQL |
| identify_gaps | Find medical deserts using geospatial analysis |
| detect_anomalies | Flag suspicious facility claims |
| text2sql_query | Genie-powered natural language to SQL |

#### Step 4.2 — Build with Mosaic AI Agent Framework
```python
# Use Mosaic AI Agent Framework (or LangGraph running on Databricks)
from databricks.agents import Agent, Tool

agent = Agent(
    model_endpoint="groq-llama",
    tools=[
        search_facilities_tool,
        analyze_region_tool,
        identify_gaps_tool,
        detect_anomalies_tool,
    ],
    instructions="You are a healthcare intelligence agent..."
)

# Register and deploy the agent
mlflow.pyfunc.log_model(
    artifact_path="healthcare-agent",
    python_model=agent,
    registered_model_name="hackathon.virtue_foundation.idp_agent"
)
```

#### Step 4.3 — MLflow Tracing for Transparency
```python
import mlflow

# Enable automatic tracing
mlflow.set_experiment("/hackathon/idp-agent")

with mlflow.start_run():
    # Each agent step is automatically logged
    # Provides step-level citations for the stretch goal
    result = agent.invoke(user_query)
    mlflow.log_param("query", user_query)
    mlflow.log_metric("num_sources", len(result.sources))
```

#### Step 4.4 — Genie for Text2SQL
- Set up Genie space in Databricks SQL pointing to your Delta tables
- Allows non-technical users to query: "How many hospitals in Ashanti Region have more than 50 beds?"
- Genie translates to SQL automatically

---

### Phase 5: Medical Desert Identification (Day 4 — 3-4 hours)

> Worth 25% of evaluation (Social Impact)

#### Step 5.1 — Define Medical Desert Criteria
- No facility within X km has a specific capability (surgery, imaging)
- Doctor-to-population ratio is critically low
- No emergency services available

#### Step 5.2 — Geospatial Analysis on Databricks
```python
# Use Spark SQL for geographic analysis
spark.sql("""
    SELECT address_stateOrRegion AS region,
           COUNT(*) AS facility_count,
           SUM(numberDoctors) AS total_doctors,
           SUM(capacity) AS total_beds,
           SUM(CASE WHEN has_emergency THEN 1 ELSE 0 END) AS emergency_capable,
           SUM(CASE WHEN has_surgery THEN 1 ELSE 0 END) AS surgery_capable
    FROM hackathon.virtue_foundation.parsed_facilities
    GROUP BY address_stateOrRegion
    ORDER BY facility_count ASC
""")
```

#### Step 5.3 — Visualize with Map
- Use Databricks SQL Dashboard map widgets for in-platform visualization
- Alternatively, generate a Folium map in a notebook and export as HTML

---

### Phase 6: Dashboard and User Interface (Day 4-5 — 3-4 hours)

> Worth 10% but makes a huge impression on judges

#### Option A: Databricks SQL Dashboard (Recommended — stays in-platform)
- Create a Lakeview dashboard with:
  - Map widget showing facility locations color-coded by capability
  - Bar charts for regional statistics
  - Counter widgets for key metrics (total facilities, medical deserts, anomalies)
  - Filter widgets for region, specialty, capability

#### Option B: Streamlit App (For richer interactivity)
```python
import streamlit as st

st.title("Healthcare Intelligence Dashboard")
st.subheader("Virtue Foundation — Ghana")

query = st.text_input("Ask about healthcare facilities...")

tab1, tab2, tab3 = st.tabs(["Map", "Analysis", "Search"])

with tab1:
    # Interactive Folium map
    st_folium(map_object)

with tab2:
    # Charts from Databricks query results
    st.bar_chart(regional_stats)

with tab3:
    # RAG search results with citations
    if query:
        result = agent.invoke(query)
        st.write(result)
```

---

## Suggested Project Structure

```
databricks-hackathon/
|-- README.md
|-- PROJECT_OUTLINE.md
|-- requirements.txt
|-- data/
|   +-- ghana_facilities.csv
|-- notebooks/                         # Databricks notebooks
|   |-- 01_data_ingestion.py           # Load data into Delta tables
|   |-- 02_idp_extraction.py           # IDP agent for parsing
|   |-- 03_vector_search_setup.py      # Databricks Vector Search
|   |-- 04_rag_chain.py                # RAG pipeline
|   |-- 05_agent_orchestration.py      # Mosaic AI Agent
|   |-- 06_medical_deserts.py          # Gap analysis
|   +-- 07_exploration.ipynb           # Data exploration
|-- src/
|   |-- idp/
|   |   |-- extractor.py               # IDP extraction logic
|   |   |-- models.py                  # Pydantic models
|   |   +-- anomaly_detector.py        # Anomaly detection
|   |-- rag/
|   |   |-- vectorstore.py             # Vector Search integration
|   |   +-- chain.py                   # RAG chain
|   |-- agents/
|   |   |-- orchestrator.py            # Agent framework
|   |   +-- tools.py                   # Agent tools
|   +-- analysis/
|       |-- medical_deserts.py         # Desert identification
|       +-- gap_analysis.py            # Gap analysis
|-- app.py                              # Streamlit app (if used)
+-- tests/
    +-- test_extractor.py
```

---

## Databricks Features Utilization Map

| Databricks Feature | How We Use It | Challenge Requirement |
|---|---|---|
| Delta Tables (Unity Catalog) | Store raw and parsed facility data | Data management |
| Foundation Model APIs / External Endpoints | LLM access for IDP extraction | IDP Innovation (30%) |
| Databricks Vector Search | Semantic search over facility data for RAG | Technical Accuracy (35%) |
| Mosaic AI Agent Framework | Orchestrate multi-step reasoning agents | Technical Accuracy (35%) |
| MLflow Tracking + Tracing | Log agent steps, provide citations | Citations (stretch goal) |
| Genie (Text2SQL) | Natural language queries over structured data | User Experience (10%) |
| Databricks SQL Dashboards / Lakeview | Visualize medical deserts and facility coverage | Social Impact (25%) |
| Databricks Secrets | Store API keys securely (Groq) | Best practice |
| Spark SQL | Geospatial aggregation and gap analysis | Social Impact (25%) |

---

## Prioritization Strategy

| Priority | Component | Evaluation Weight | Time Estimate |
|---|---|---|---|
| P0 (Critical) | Data ingestion into Delta Tables | Foundation | 2 hrs |
| P0 (Critical) | IDP Extraction via Databricks LLM endpoints | 30% | 6-8 hrs |
| P0 (Critical) | RAG with Databricks Vector Search | 35% | 4-5 hrs |
| P1 (Important) | Medical Desert Analysis with Spark SQL | 25% | 3-4 hrs |
| P1 (Important) | Anomaly Detection | Part of 35% | 2-3 hrs |
| P2 (Nice to have) | Databricks SQL Dashboard + Map | 10% | 3-4 hrs |
| P2 (Nice to have) | MLflow Tracing for Citations | Bonus | 2 hrs |
| P3 (Stretch) | Genie Text2SQL Integration | Bonus | 2 hrs |
| P3 (Stretch) | Mosaic AI Agent Deployment | Bonus | 3 hrs |

---

## 5-Day Timeline

| Day | Focus | Databricks Features Used |
|---|---|---|
| Day 1 (Mar 29) | Workspace setup + Data ingestion into Delta Tables | Unity Catalog, Delta Tables, Clusters |
| Day 2 (Mar 30) | IDP Extraction Pipeline | Foundation Model APIs / External Endpoints, MLflow |
| Day 3 (Mar 31) | RAG + Agent Orchestration | Vector Search, Mosaic AI Agent Framework |
| Day 4 (Apr 1) | Medical Deserts + Visualization | Spark SQL, SQL Dashboards, Lakeview |
| Day 5 (Apr 2) | Polish, Testing, Demo Video | MLflow, Genie, Documentation |

---

## Tech Stack (All Databricks-Centered)

| Layer | Technology | Notes |
|---|---|---|
| Platform | Databricks Community Edition | Required by challenge |
| Data Storage | Delta Tables in Unity Catalog | Structured + parsed data |
| LLM Access | Groq (Llama 3.1 70B) via External Endpoint | Free tier, fast inference |
| Embeddings | Databricks BGE or HuggingFace all-MiniLM-L6-v2 | For vector search |
| Vector Search | Databricks Vector Search | Native RAG support |
| Agent Framework | Mosaic AI Agent Framework / LangGraph on Databricks | Agentic orchestration |
| Experiment Tracking | MLflow (native) | Citations and tracing |
| Text2SQL | Genie | Natural language querying |
| Dashboards | Databricks SQL Dashboard / Lakeview | In-platform visualization |
| Optional UI | Streamlit (deployed externally) | If richer interactivity needed |

---

## Submission Requirements Checklist

- [ ] Built using required Databricks tools during hackathon period
- [ ] Clear text description explaining features and functionality
- [ ] Working demo link or test build with credentials for judges
- [ ] Public GitHub repository with open-source license
- [ ] Demo video (up to 5 minutes) showcasing solution and impact

---

## Common Pitfalls to Avoid

- Do not build everything outside Databricks — judges expect Databricks usage
- Do not ignore the free-text fields — they are the core of IDP and worth 30%
- Do not skip citations — easy points, return source documents with MLflow tracing
- Do not forget the social impact story — judges care about "why it matters"
- Do not run expensive operations on all data first — test on 5-10 rows, then scale
- Do not build UI first — get the AI pipeline working on Databricks, then add the dashboard

---

## Learning Resources

| Topic | Resource |
|---|---|
| Databricks Community Edition | https://community.cloud.databricks.com |
| Databricks Vector Search | https://docs.databricks.com/en/generative-ai/vector-search.html |
| Mosaic AI Agent Framework | https://docs.databricks.com/en/generative-ai/agent-framework/index.html |
| MLflow on Databricks | https://docs.databricks.com/en/mlflow/index.html |
| Databricks SQL Dashboards | https://docs.databricks.com/en/dashboards/index.html |
| Genie (Text2SQL) | https://docs.databricks.com/en/genie/index.html |
| Delta Tables | https://docs.databricks.com/en/delta/index.html |
| Unity Catalog | https://docs.databricks.com/en/data-governance/unity-catalog/index.html |
| LangGraph on Databricks | https://docs.databricks.com/en/generative-ai/agent-framework/langgraph.html |
| Groq API | https://console.groq.com/docs/quickstart |
