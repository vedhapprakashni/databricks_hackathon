# 🧭 Care Compass — AI-Powered Healthcare Intelligence Agent

<div align="center">

**Navigating Healthcare Deserts with Data-Driven Intelligence**

*An AI-powered multi-agent system built on Databricks that maps healthcare accessibility gaps, detects anomalies in facility claims, and enables natural language querying over Ghana's healthcare infrastructure — empowering the Virtue Foundation to allocate resources where they matter most.*

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://your-app.streamlit.app)
[![Built with Databricks](https://img.shields.io/badge/Built_with-Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)](https://databricks.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)


</div>

---

## 📋 Table of Contents

- [The Problem](#-the-problem)
- [Our Solution](#-our-solution--care-compass)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Databricks Tools & Technologies](#-databricks-tools--technologies-used)
- [Project Structure](#-project-structure)
- [How to Run the Project](#-how-to-run-the-project)
- [Dataset](#-dataset)
- [Technology Stack](#-full-technology-stack)
- [Results & Impact](#-results--impact)

---

## 🌍 The Problem

Millions of people in Ghana live in **medical deserts** — regions where critical healthcare services like surgery, emergency care, and specialist treatment are inaccessible. The Virtue Foundation, an NGO dedicated to healthcare equity, faces critical challenges:

- **Where are the gaps?** — Which regions lack essential services like emergency care, surgery, or specialist treatment?
- **Can we trust the data?** — Facilities may over-report capabilities. A clinic with 2 doctors claiming 15 surgical specialties raises red flags.
- **Where should resources go?** — With limited funding, deciding where to deploy doctors, equipment, and new facilities is life-and-death.

Traditional dashboards show numbers. **Care Compass navigates the "why" and "where next."**

---

## 💡 Our Solution — Care Compass

Care Compass is a **multi-agent AI system** that combines three intelligent agents — **Text-to-SQL**, **Semantic Vector Search**, and **Medical Reasoning** — orchestrated by a Supervisor Agent that routes natural language questions to the right expert.

Instead of writing SQL or reading spreadsheets, healthcare planners simply ask:

> *"Which regions have no emergency care facilities?"*
> *"Are there hospitals claiming advanced surgeries without the necessary equipment?"*
> *"Where should the Virtue Foundation deploy doctors next?"*

Care Compass answers with **data-backed insights, interactive maps, and actionable recommendations**.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                         │
│    💬 Chat Interface │ 🗺️ Maps │ 📊 Charts │ ⚠️ Anomalies    │
└────────────────────────────┬─────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│               SUPERVISOR AGENT (Intent Router)               │
│          Classifies natural language → Routes to agent       │
│                                                              │
│   ┌────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│   │  SQL Agent  │   │   Semantic   │   │    Medical       │  │
│   │ (Text2SQL) │   │    Search    │   │  Reasoning Agent │  │
│   └─────┬──────┘   └──────┬───────┘   └────────┬─────────┘  │
│         │                  │                     │            │
│    DuckDB /           FAISS Vector         LLM Chain with    │
│    Spark SQL          Index Search         Medical Context   │
└─────────┼──────────────────┼─────────────────────┼───────────┘
          │                  │                     │
┌─────────▼──────────────────▼─────────────────────▼───────────┐
│                   DATABRICKS LAKEHOUSE                       │
│                                                              │
│   Delta Tables  │  Vector Search  │  MLflow  │  Foundation   │
│   Unity Catalog │  Embeddings     │  Tracing │  Model APIs   │
└──────────────────────────────────────────────────────────────┘
```

### Agent Routing Flow

| User Intent | Detected Keywords | Routed To | Example |
|---|---|---|---|
| **Structured queries** | "how many", "count", "which region", "most" | SQL Agent | *"How many hospitals have cardiology?"* |
| **Service lookups** | "services", "offer", "treat", "equipment" | Semantic Search | *"What does Korle Bu Hospital offer?"* |
| **Complex analysis** | "anomaly", "suspicious", "desert", "recommend" | Reasoning Agent | *"Which facilities have suspicious claims?"* |

---

## ✨ Key Features

### 💬 Natural Language Query Engine
Ask any question about Ghana's 400+ healthcare facilities in plain English. The Supervisor Agent automatically routes to the right sub-agent for optimal answers.

### 🗺️ Interactive Facility Map
Folium-powered map of all healthcare facilities across Ghana's 17 regions, color-coded by type (hospital, clinic, dentist, pharmacy) with popup details showing doctors, beds, and services.

### 🏜️ Medical Desert Analysis
A weighted scoring algorithm (0–100) evaluates each region across 5 dimensions:
- **Facility density** (40%) — How many facilities serve the region?
- **Doctor availability** (25%) — Are there enough physicians?
- **Bed capacity** (20%) — Can the region handle patient volume?
- **Surgical access** (10%) — Can residents get surgery without travelling?
- **Emergency care** (5%) — Is 24/7 emergency care available?

High-risk zones with no emergency care are highlighted as **"Lives at Risk"** areas, while regions with strong surgical capacity are marked as **"Expertise Hubs"**.

### ⚠️ Anomaly Detection
Four automated anomaly flags identify suspicious facility data:
| Flag | What It Catches |
|---|---|
| 🔴 High Procedures, Low Doctors | Facilities claiming 5+ procedures with <2 doctors |
| 🟠 High Capacity, No Surgery | 50+ bed facilities with no surgical capability |
| 🟡 Many Specialties, Small Size | 5+ specialties but <20 bed capacity |
| 🔵 No Doctors Listed | Hospitals with completely missing doctor counts |

### 📊 Analytics Dashboard
KPI cards, regional comparison charts, facility type distribution, and desert score rankings — all with a premium dark theme.

### 🎯 Action Planner
Priority-ranked action items for the Virtue Foundation, identifying which regions need immediate intervention and what type of support would have the greatest impact.

---

## 🔧 Databricks Tools & Technologies Used

Care Compass leverages the Databricks Lakehouse Platform extensively. Below is a detailed breakdown of every Databricks tool/service used and its role:

---

### 1. Delta Tables & Unity Catalog

**What it is:** Delta Lake is Databricks' open-source storage layer that brings ACID transactions to data lakes. Unity Catalog provides centralized governance across all data assets.

**How Care Compass uses it:**
- The raw `dataset.csv` (1,003 rows with duplicates) is ingested and deduplicated into a clean **Delta table** (`hackathon.vf.facilities_clean`).
- Delta's schema enforcement ensures data integrity during the cleaning pipeline.
- Unity Catalog manages table access, lineage, and discoverability across all notebook pipelines.

**Notebook:** `01_data_cleaning.py`
```python
# Deduplicated facility data stored as managed Delta table
df_clean.write.format("delta").mode("overwrite").saveAsTable("hackathon.vf.facilities_clean")
```

---

### 2. Databricks Foundation Model APIs

**What it is:** Databricks-hosted LLM endpoints that provide pay-per-token access to leading open-source models without managing infrastructure.

**How Care Compass uses it:**
- The **Meta Llama 3.3 70B Instruct** model (`databricks-meta-llama-3-3-70b-instruct`) powers all three agents:
  - **SQL Agent** — Converts natural language questions to SQL
  - **Reasoning Agent** — Performs medical anomaly analysis and cross-validation
  - **Supervisor Agent** — Classifies query intent for routing
- All notebooks use `ChatDatabricks` from LangChain for native integration:
```python
from langchain_community.chat_models import ChatDatabricks

llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0,
    max_tokens=2048
)
```

**Notebooks:** `04_rag_chain.py`, `05_sql_agent.py`, `06_reasoning_agent.py`, `07_supervisor_agent.py`

---

### 3. Databricks Vector Search

**What it is:** A managed vector database service that automatically syncs with Delta tables and provides low-latency similarity search using embeddings.

**How Care Compass uses it:**
- Creates a **Delta Sync Index** over the combined text column (name + specialties + procedures + equipment + capabilities) of each facility.
- Uses the Databricks BGE embedding model for vectorization.
- Enables **semantic RAG queries** — when a user asks *"Find hospitals with CT scanners in Northern Ghana"*, the vector index retrieves the most semantically similar facilities.

**Notebook:** `03_vector_store.py`
```python
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
index = vsc.create_delta_sync_index(
    endpoint_name="facility_search_endpoint",
    source_table_name="hackathon.vf.facilities_searchable",
    primary_key="pk_unique_id",
    embedding_source_column="combined_text",
    embedding_model_endpoint_name="databricks-bge-large-en"
)
```

---

### 4. MLflow Experiment Tracking & Tracing

**What it is:** MLflow is Databricks' native platform for ML lifecycle management — tracking experiments, logging parameters, and providing end-to-end observability.

**How Care Compass uses it:**
- Every agent query is logged as an **MLflow run** with:
  - Input question
  - Classified intent category
  - Generated SQL (for SQL Agent queries)
  - Retrieved source facility IDs (citations)
  - Response time per agent step
  - Final answer text
- This creates a full **audit trail** for explainability — critical for an NGO making resource allocation decisions.

**Notebook:** `09_mlflow_tracing.py`
```python
import mlflow

mlflow.set_experiment("/hackathon/care-compass-tracing")

with mlflow.start_run(run_name=f"query_{intent}"):
    mlflow.log_param("question", question)
    mlflow.log_param("classified_intent", intent)
    mlflow.log_param("generated_sql", sql)
    mlflow.log_metric("response_time_ms", elapsed)
    mlflow.log_param("source_facility_ids", str(sources))
```

---

### 5. Spark SQL

**What it is:** Databricks' distributed SQL engine for querying data at scale with ANSI SQL syntax.

**How Care Compass uses it:**
- Regional aggregation queries for medical desert scoring
- Facility-level anomaly detection SQL
- Cross-tabulation of specialties vs. equipment vs. capacity
- All analysis in `02_data_analysis.py` and the SQL Agent's generated queries run on Spark SQL

```sql
-- Medical Desert Analysis: Regions ranked by healthcare scarcity
SELECT address_stateOrRegion AS region,
       COUNT(DISTINCT pk_unique_id) AS facility_count,
       SUM(numberDoctors) AS total_doctors,
       SUM(capacity) AS total_beds
FROM hackathon.vf.facilities_clean
GROUP BY address_stateOrRegion
ORDER BY facility_count ASC
```

---

### 6. Databricks Notebooks

**What it is:** Collaborative, runnable code documents that support Python, SQL, Scala, and R — with built-in visualization, version control, and cluster integration.

**How Care Compass uses it:**
The entire pipeline is implemented as **11 sequential notebooks** (see [Project Structure](#-project-structure)), designed to run in order on a Databricks cluster. Each notebook is self-contained with:
- Markdown documentation explaining the purpose
- Code cells implementing the pipeline step
- Verification cells to validate outputs

---

### 7. Databricks Secrets

**What it is:** A secure key-value store for sensitive credentials like API keys, tokens, and passwords — preventing hardcoded secrets in notebooks.

**How Care Compass uses it:**
- API keys (Groq, Databricks tokens) are stored via `dbutils.secrets` and retrieved at runtime:
```python
DATABRICKS_TOKEN = dbutils.secrets.get(scope="hackathon", key="DATABRICKS_TOKEN")
```

---

### 8. DBFS (Databricks File System)

**What it is:** A distributed file system mounted to every Databricks cluster, providing shared storage for datasets, models, and artifacts.

**How Care Compass uses it:**
- The raw `dataset.csv` is uploaded to `/FileStore/dataset.csv`
- FAISS vector indices are persisted to DBFS for reuse across sessions
- Intermediate outputs and exported results are stored on DBFS

---

### Summary: Databricks Features Map

| Databricks Feature | Care Compass Usage | Notebook |
|---|---|---|
| Delta Tables + Unity Catalog | Deduplicated facility storage with governance | `01_data_cleaning.py` |
| Foundation Model APIs (Llama 3.3) | LLM for SQL gen, reasoning, routing | `04` – `07` |
| Vector Search | Semantic facility search (RAG) | `03_vector_store.py` |
| MLflow Tracking + Tracing | Agent observability and audit trail | `09_mlflow_tracing.py` |
| Spark SQL | Regional analytics and anomaly queries | `02_data_analysis.py` |
| Databricks Notebooks | End-to-end pipeline orchestration | All 11 notebooks |
| Databricks Secrets | Secure API key management | `00_setup.py` |
| DBFS | Dataset and artifact storage | `00_setup.py`, `03_vector_store.py` |

---

## 📁 Project Structure

```
care-compass/
│
├── app.py                              # 🖥️  Streamlit frontend (standalone deployable)
├── dataset.csv                         # 📊 Ghana healthcare facility data (1,003 rows)
├── requirements.txt                    # 📦 Python dependencies
├── .gitignore                          # 🔒 Excludes secrets, venv, cache
├── .streamlit/
│   └── config.toml                     # 🎨 Premium dark theme configuration
│
├── databricks notebooks/              # 🔬 Databricks pipeline (11 phases)
│   ├── 00_setup.py                     #    Environment setup & secret config
│   ├── 01_data_cleaning.py             #    Deduplication → Delta tables
│   ├── 02_data_analysis.py             #    Regional stats & anomaly flags
│   ├── 03_vector_store.py              #    Embeddings + FAISS/Vector Search index
│   ├── 04_rag_chain.py                 #    RAG pipeline with citation support
│   ├── 05_sql_agent.py                 #    Text-to-SQL agent (natural language → SQL)
│   ├── 06_reasoning_agent.py           #    Medical reasoning & anomaly analysis
│   ├── 07_supervisor_agent.py          #    Multi-agent router (intent classification)
│   ├── 08_dashboard.py                 #    Visualization + dashboard pipeline
│   ├── 09_mlflow_tracing.py            #    MLflow observability & tracing
│   └── 10_final_testing.py             #    Full evaluation suite (59 agent questions)
│
├── prompts_and_pydantic_models/        # 📝 Virtue Foundation IDP extraction models (reference)
├── hide_keys.py                        # 🔑 Utility to scrub API keys from files
│
├── CODING_PLAN.md                      # 📋 Detailed 10-step implementation plan
├── PROJECT_OUTLINE.md                  # 🏛️ Architecture overview & design decisions
├── DATASET_ANALYSIS.md                 # 🔍 Data quality analysis & strategic insights
└── README.md                           # 📖 This file
```

---

## 🚀 How to Run the Project

Care Compass can be run in **three ways** depending on your setup:

---

### Option 1: Streamlit App — Local (Quickest Demo)

This runs the self-contained Streamlit frontend with DuckDB for SQL, keyword/FAISS search, and Groq-powered AI chat.

#### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com/keys) (for AI agent features)

#### Steps

```bash
# 1. Clone the repository
git clone https://github.com/vedhapprakashni/databricks_hackathon.git
cd databricks_hackathon

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key
export GROQ_API_KEY="gsk_your_key_here"       # Linux/Mac
$env:GROQ_API_KEY="gsk_your_key_here"         # PowerShell
set GROQ_API_KEY=gsk_your_key_here            # Windows CMD

# 5. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501** with:
- ✅ Full dashboard with KPI cards and charts
- ✅ Interactive facility map of Ghana
- ✅ Medical desert analysis with risk zone map
- ✅ Anomaly detection panel
- ✅ AI-powered natural language Q&A (requires Groq key)
- ✅ Action planner with priority recommendations

> **Note:** The Streamlit app uses **DuckDB** as a lightweight SQL engine and falls back to keyword search if FAISS/sentence-transformers are not installed. All features work without Databricks credentials.

---

### Option 2: Streamlit App — Cloud Deployment (Live URL)

Deploy for free on **Streamlit Community Cloud** for a publicly accessible URL.

```bash
# 1. Ensure your code is pushed to GitHub
git add -A
git commit -m "Deploy Care Compass"
git push origin main
```

Then:
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **"New app"**
3. Select: **Repository** = `vedhapprakashni/databricks_hackathon` | **Branch** = `main` | **Main file** = `app.py`
4. Click **Deploy**
5. Go to **Settings ⚙️ → Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```

Your app will be live at `https://vedhapprakashni-databricks-hackathon-app.streamlit.app`

---

### Option 3: Full Databricks Pipeline (Backend + Notebooks)

Run the complete data engineering and AI pipeline on a Databricks workspace.

#### Prerequisites
- Databricks workspace (Community Edition or full workspace)
- A running cluster (Python 3.10+, ML Runtime recommended)
- Groq API key **or** access to Databricks Foundation Model APIs

#### Steps

**1. Upload the dataset**
```
Upload dataset.csv → DBFS at /FileStore/dataset.csv
```

**2. Configure secrets**
```bash
# Using Databricks CLI:
databricks secrets create-scope hackathon
databricks secrets put-secret hackathon DATABRICKS_TOKEN --string-value "YOUR_TOKEN"
```

**3. Import notebooks**
- Upload all files from `databricks notebooks/` into your Databricks workspace.
- Each `.py` file is a Databricks-compatible Python notebook with `# COMMAND ----------` cell separators.

**4. Run notebooks in order**

| Order | Notebook | What It Does | Estimated Time |
|:---:|---|---|---|
| 1 | `00_setup.py` | Configures workspace, installs libraries, validates API keys | 5 min |
| 2 | `01_data_cleaning.py` | Loads CSV, deduplicates 1,003 → ~450 facilities, creates Delta table | 10 min |
| 3 | `02_data_analysis.py` | Computes regional stats, anomaly flags, medical desert scores | 5 min |
| 4 | `03_vector_store.py` | Generates embeddings (sentence-transformers), builds FAISS index | 15 min |
| 5 | `04_rag_chain.py` | Builds RAG pipeline — vector search + LLM answer with citations | 10 min |
| 6 | `05_sql_agent.py` | Text-to-SQL agent — converts questions to Spark SQL / DuckDB | 10 min |
| 7 | `06_reasoning_agent.py` | Medical reasoning — anomaly cross-validation, mismatch detection | 10 min |
| 8 | `07_supervisor_agent.py` | Supervisor router — classifies intent, chains sub-agents | 10 min |
| 9 | `08_dashboard.py` | Dashboard pipeline — generates visualizations and summaries | 10 min |
| 10 | `09_mlflow_tracing.py` | Sets up MLflow experiment tracking for all agent queries | 5 min |
| 11 | `10_final_testing.py` | Runs all 59 agent questions from Virtue Foundation evaluation | 20 min |

---

### Environment Variables Reference

| Variable | Description | Required For |
|---|---|---|
| `GROQ_API_KEY` | Groq API key for Llama 3.3 70B | Streamlit app (AI chat) |
| `DATABRICKS_TOKEN` | Databricks personal access token | Databricks notebooks |
| `DATABRICKS_HOST` | Databricks workspace URL | Databricks notebooks |

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | Virtue Foundation — Foundational Data Refresh (FDR) |
| **Raw Size** | 1,003 rows × 41 columns |
| **After Deduplication** | ~450 unique facilities |
| **Geography** | All 17 regions of Ghana |
| **Facility Types** | Hospitals, Clinics, Dentists, Pharmacies |
| **Key Fields** | Name, address, specialties, procedures, equipment, capabilities, doctor count, bed capacity |
| **Free-Text Fields** | Procedures, equipment, capabilities, descriptions (~20-30% coverage) |

The dataset was pre-extracted by the Virtue Foundation using an IDP (Intelligent Document Processing) pipeline with Pydantic models and LLMs. Care Compass does **not** re-extract from raw documents — it builds an intelligence layer over the already-structured data.

---

## 🛠️ Full Technology Stack

| Layer | Technology | Role |
|---|---|---|
| **Platform** | Databricks Lakehouse | Data engineering, model hosting, governance |
| **LLM (Databricks)** | Foundation Model APIs — Llama 3.3 70B | Backend notebook pipelines |
| **LLM (Frontend)** | Groq API — Llama 3.3 70B Versatile | Streamlit app AI chat (free tier) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Semantic vector representations |
| **Vector Search** | FAISS (frontend) + Databricks Vector Search (backend) | Similarity search over facilities |
| **SQL Engine** | DuckDB (frontend) / Spark SQL (Databricks) | Structured query execution |
| **Frontend** | Streamlit + Folium + Matplotlib + Seaborn | Interactive UI, maps, charts |
| **Agent Framework** | LangChain + Custom Supervisor Router | Multi-agent orchestration |
| **Observability** | MLflow Experiment Tracking | Agent tracing, citations, audit trail |
| **Language** | Python 3.10+ | Core language |

---

## 📈 Results & Impact

### Technical Results
- ✅ Successfully answers **16 Must-Have** and **43 stretch** agent questions across 11 categories
- ✅ Identifies medical deserts with weighted scoring across **17 Ghana regions**
- ✅ Detects **4 types of anomalies** in facility data with medical reasoning
- ✅ Sub-second intent classification and query routing
- ✅ Interactive maps with facility-level and region-level drill-down

### Social Impact
- 🏜️ **Identified critical medical deserts** — Regions with high desert scores and zero emergency facilities flagged as "Lives at Risk"
- ⚠️ **Exposed data inconsistencies** — Facilities claiming advanced capabilities without matching infrastructure
- 🎯 **Actionable recommendations** — Priority-ranked action items for where to deploy doctors, build facilities, and allocate equipment
- 🗺️ **Visual decision support** — Maps that NGO planners can use without technical expertise

---

## 👥 Team

**Hack Nation — Databricks × Accenture Hackathon 2026**

Built for the Virtue Foundation to advance healthcare equity in Ghana.

---



<div align="center">

*Care Compass — Because every community deserves a path to healthcare.* 🧭

</div>
