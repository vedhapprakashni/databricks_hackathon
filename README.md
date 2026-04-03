# Bridging Medical Deserts: AI-Powered Healthcare Intelligence Agent

> **Databricks × Accenture Hackathon** — Virtue Foundation

An AI-powered healthcare intelligence system built on Databricks that identifies medical deserts, detects anomalies in facility claims, and enables natural language querying over Ghana healthcare facility data.

---

## 🌍 The Problem

Millions of people in Ghana live in **medical deserts** — regions where critical healthcare services like surgery, emergency care, and specialist treatment are inaccessible. NGO planners at the Virtue Foundation need data-driven tools to:

- Identify where healthcare gaps exist
- Verify facility claims vs. actual capability
- Decide where to allocate doctors, equipment, and funding

## 💡 Our Solution

A multi-agent AI system that combines **Text-to-SQL**, **semantic vector search**, and **medical reasoning** to answer complex healthcare questions in natural language.

### Architecture

```
┌──────────────────────────────────────────────────┐
│              STREAMLIT FRONTEND                  │
│   Chat Interface + Maps + Charts + Anomalies     │
└─────────────────────┬────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────┐
│            SUPERVISOR AGENT (Router)             │
│        Classifies intent → Routes to agent       │
│                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ SQL      │ │ Semantic │ │ Medical Reasoning│  │
│  │ Agent    │ │ Search   │ │ Agent            │  │
│  └────┬─────┘ └────┬─────┘ └────────┬─────────┘  │
│       │            │                │            │
│  DuckDB /     FAISS Vector    LLM Chain with     │
│  Spark SQL    Index           Medical Context    │
└───────┼────────────┼────────────────┼────────────┘
        │            │                │
┌───────▼────────────▼────────────────▼────────────┐
│              DATABRICKS LAKEHOUSE                │
│                                                  │
│  Delta Tables │ Vector Search │ MLflow Tracing   │
│  Unity Catalog│ Embeddings    │ Citations        │
└──────────────────────────────────────────────────┘
```

## 🚀 Live Demo

👉 **[Try the Live App](https://your-app.streamlit.app)** *(Update this link after deployment)*

## ✨ Key Features

| Feature | Description |
|---|---|
| 💬 Natural Language Queries | Ask any question about healthcare facilities in plain English |
| 🗺️ Interactive Maps | Folium maps showing facility distribution and medical desert heatmaps |
| 🏜️ Medical Desert Analysis | Weighted scoring system to identify underserved regions |
| ⚠️ Anomaly Detection | Flags facilities with inconsistent or suspicious data |
| 📊 Analytics Dashboard | KPIs, charts, and regional comparisons |
| 🔍 Multi-Agent Routing | Supervisor classifies intent and routes to SQL, Semantic, or Reasoning agent |
| 📝 MLflow Tracing | Full observability with step-level timing and citation provenance |

## 📁 Project Structure

```
├── app.py                          # Streamlit frontend (deployable)
├── dataset.csv                     # Ghana healthcare facility data
├── requirements.txt                # Python dependencies
├── .streamlit/config.toml          # Streamlit dark theme config
├── notebooks/                      # Databricks notebooks (11 phases)
│   ├── 00_setup.py                 # Environment setup
│   ├── 01_data_cleaning.py         # Deduplication + Delta tables
│   ├── 02_data_analysis.py         # Stats + anomaly flags
│   ├── 03_vector_store.py          # Embeddings + FAISS/VS index
│   ├── 04_rag_chain.py             # RAG pipeline
│   ├── 05_sql_agent.py             # Text-to-SQL agent
│   ├── 06_reasoning_agent.py       # Medical reasoning
│   ├── 07_supervisor_agent.py      # Multi-agent router
│   ├── 08_dashboard.py             # Visualization dashboard
│   ├── 09_mlflow_tracing.py        # MLflow observability
│   └── 10_final_testing.py         # Evaluation suite
├── hacakthon notebooks w outputs/  # Exported notebooks with outputs
├── CODING_PLAN.md                  # Detailed implementation plan
├── PROJECT_OUTLINE.md              # Architecture overview
└── DATASET_ANALYSIS.md             # Data quality analysis
```

## 🛠️ Setup

### Option 1: Run the Streamlit App Locally

```bash
# Clone the repository
git clone https://github.com/vedhapprakashni/databricks_hackathon.git
cd databricks_hackathon

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY="gsk_your_key_here"   # Linux/Mac
set GROQ_API_KEY=gsk_your_key_here        # Windows

# Run the app
streamlit run app.py
```

### Option 2: Run on Databricks

1. Import the notebooks from `hacakthon notebooks w outputs/` into your Databricks workspace
2. Set your Groq API key in Databricks Secrets or inline
3. Run notebooks 00 through 10 in order

## 🔑 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Groq API key for Llama 3.3 70B | Yes |

## 📊 Dataset

- **Source**: Virtue Foundation Foundational Data Refresh (FDR)
- **Size**: 1,003 rows → ~400-500 unique facilities after deduplication
- **Columns**: 41 (structured + free-text)
- **Geography**: All facilities in Ghana
- **Fields**: Name, address, specialties, procedures, equipment, capability, doctors, capacity

## 🧠 Technology Stack

| Component | Technology |
|---|---|
| Platform | Databricks (Unity Catalog, Delta Tables) |
| LLM | Groq API — Llama 3.3 70B Versatile |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (+ Databricks Vector Search) |
| SQL Engine | DuckDB (frontend) / Spark SQL (Databricks) |
| Frontend | Streamlit + Folium + Matplotlib |
| Observability | MLflow Experiment Tracking |
| Language | Python |

## 📈 Results

- Successfully answers **16 Must-Have** agent questions across 11 categories
- Identifies medical deserts with weighted scoring across **17 Ghana regions**
- Detects **4 types of anomalies** in facility data
- Sub-second query routing via intent classification
- Interactive maps with facility-level detail

## 📜 License

MIT License

---

*Built for the Databricks × Accenture Hackathon — powering healthcare intelligence for the Virtue Foundation* 🏥
