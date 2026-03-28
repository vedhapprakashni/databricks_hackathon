# Bridging Medical Deserts: IDP Agent for the Virtue Foundation
## Complete Project Outline and Implementation Approach

---

## What This Project Is (In Plain English)

You are building an AI-powered system that:
1. Reads messy, unstructured hospital data from Ghana (provided by the Virtue Foundation)
2. Extracts structured information (what equipment they have, what procedures they can do, how many doctors, etc.)
3. Identifies "medical deserts" — areas where people cannot access critical healthcare
4. Helps planners (NGOs, governments) decide where to send doctors, equipment, and funding

Think of it as: "A smart assistant that reads thousands of hospital reports and tells you where the healthcare gaps are."

---

## Key Concepts You Need to Learn

| Concept | What It Means | Why You Need It |
|---|---|---|
| IDP (Intelligent Document Parsing) | Using AI to extract structured data from messy text | Core of the challenge — parse hospital reports |
| RAG (Retrieval-Augmented Generation) | Feeding relevant documents to an LLM so it can answer questions | Lets users ask "Which hospitals in Northern Ghana have CT scanners?" |
| Agentic AI | AI that can plan, reason, and take multi-step actions | Your system needs to chain together: parse, analyze, recommend |
| LLM (Large Language Model) | Models like Llama that understand text | The "brain" that reads and understands hospital reports |
| Vector Database | Stores text as numbers for fast semantic search | Enables searching by meaning, not just keywords |
| LangGraph / CrewAI | Frameworks to build multi-step AI agents | Orchestrates the whole pipeline |

---

## Architecture Overview

```
+------------------------------------------------------+
|                    USER INTERFACE                     |
|         (Streamlit / Gradio Web Dashboard)            |
|   Natural language queries + Map visualization        |
+-------------------------+----------------------------+
                          |
+-------------------------v----------------------------+
|              AGENTIC ORCHESTRATOR                     |
|            (LangGraph / CrewAI)                       |
|                                                      |
|  +----------+  +----------+  +--------------------+  |
|  | IDP Agent|  | Analysis |  | Planning/Recommend |  |
|  | (Parser) |  |  Agent   |  |      Agent         |  |
|  +----+-----+  +----+-----+  +---------+----------+  |
+-------|--------------|-----------------------|-------+
        |              |                       |
+-------v--------------v-----------------------v-------+
|                   DATA LAYER                          |
|                                                      |
|  +------------+  +----------+  +-----------------+   |
|  | Raw CSV    |  | Vector   |  | Structured DB   |   |
|  | (Ghana     |  | Store    |  | (Parsed         |   |
|  |  Dataset)  |  | (FAISS)  |  |  Facilities)    |   |
|  +------------+  +----------+  +-----------------+   |
+------------------------------------------------------+
```

---

## Step-by-Step Implementation Plan

### Phase 1: Setup and Data Understanding (Day 1 — approx 3-4 hours)

#### Step 1.1 — Environment Setup
- Create a Databricks Free account at community.cloud.databricks.com
- Install core Python libraries locally for development:
  ```
  pip install langchain langgraph groq pandas faiss-cpu streamlit folium pydantic mlflow
  ```
- Set up a GitHub repo for your project

#### Step 1.2 — Download and Explore the Dataset
- Download the Virtue Foundation Ghana Dataset (linked in the challenge)
- Load it into a Pandas DataFrame and explore:
  - How many facilities?
  - What columns exist? (structured vs unstructured)
  - Look at procedure, equipment, capability columns — these are the messy free-text fields
  - Identify missing data, inconsistencies

#### Step 1.3 — Understand the Schema
- Read the schema documentation (provided in challenge description)
- Map each column to its meaning
- Identify which fields are structured (e.g., capacity, numberDoctors) vs unstructured (e.g., procedure, equipment, capability)

---

### Phase 2: IDP Agent — Document Parsing (Day 1-2 — approx 6-8 hours)

> This is the CORE of your project and worth 30% of evaluation

#### Step 2.1 — Build the Extraction Pipeline
Create a Python module that uses an LLM (Llama 3.1 via Groq) to extract structured data from free-text fields.

Approach:
```python
# Pseudocode for IDP extraction
from pydantic import BaseModel
from langchain_groq import ChatGroq

class FacilityExtraction(BaseModel):
    procedures: list[str]       # Extracted procedures
    equipment: list[str]        # Extracted equipment
    capabilities: list[str]     # Extracted capabilities
    specialties: list[str]      # Inferred specialties
    has_emergency: bool         # Can handle emergencies?
    has_surgery: bool           # Can perform surgery?
    has_imaging: bool           # Has imaging equipment?
    anomalies: list[str]        # Suspicious/inconsistent claims

# For each facility row:
# 1. Combine all free-text fields into one prompt
# 2. Send to LLM with structured output (Pydantic model)
# 3. Store the parsed result
```

#### Step 2.2 — Anomaly Detection
Flag facilities with suspicious data:
- Claims to have advanced equipment but has 0 doctors
- Says "Level II trauma center" but has no emergency capability
- Has massive capacity but tiny facility area

#### Step 2.3 — Store Parsed Results
- Save structured extractions to a clean DataFrame/database
- Create embeddings of facility descriptions for vector search

---

### Phase 3: RAG System — Intelligent Querying (Day 2-3 — approx 4-5 hours)

> This powers the 35% Technical Accuracy score

#### Step 3.1 — Build Vector Store
```python
# Pseudocode
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Create text chunks from facility data
texts = [f"Facility: {row['name']}, Location: {row['address_city']}, "
         f"Equipment: {row['equipment']}, Procedures: {row['procedure']}, "
         f"Capabilities: {row['capability']}"
         for _, row in df.iterrows()]

# Build FAISS index (using free HuggingFace embeddings)
vectorstore = FAISS.from_texts(texts, HuggingFaceEmbeddings())
```

#### Step 3.2 — Build the RAG Chain
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGroq(model="llama-3.1-70b-versatile"),
    retriever=vectorstore.as_retriever(),
    return_source_documents=True  # For citations!
)

# Now users can ask:
# "Which hospitals in Northern Ghana can perform cesarean sections?"
# "Find facilities with MRI machines within 100km of Tamale"
```

#### Step 3.3 — Add Citations (Stretch Goal)
Each answer should include which facility records were used:
- Return source_documents with row numbers
- For agentic-step citations, use MLflow tracing to log each reasoning step

---

### Phase 4: Agentic Orchestration (Day 3 — approx 5-6 hours)

#### Step 4.1 — Define Agent Tools
Create tools the agent can use:

| Tool | Purpose |
|---|---|
| search_facilities | RAG search across facility data |
| get_facility_details | Look up a specific facility by name |
| analyze_region | Get aggregated stats for a region |
| identify_gaps | Find medical deserts in a region |
| detect_anomalies | Flag suspicious facility claims |

#### Step 4.2 — Build with LangGraph
```python
from langgraph.graph import StateGraph

# Define the agent workflow
workflow = StateGraph(AgentState)
workflow.add_node("understand_query", understand_user_query)
workflow.add_node("search", search_relevant_data)
workflow.add_node("analyze", analyze_and_reason)
workflow.add_node("recommend", generate_recommendations)
workflow.add_node("respond", format_response)

# Connect the nodes
workflow.add_edge("understand_query", "search")
workflow.add_edge("search", "analyze")
workflow.add_edge("analyze", "recommend")
workflow.add_edge("recommend", "respond")
```

#### Step 4.3 — Planning System
Build a simple planning interface where a user can:
1. Select a region on the map
2. See current healthcare coverage
3. Get AI recommendations: "If you add 2 doctors and 1 CT scanner to facility X, you can serve 50,000 more people"

---

### Phase 5: Medical Desert Identification (Day 4 — approx 3-4 hours)

> Worth 25% of evaluation (Social Impact)

#### Step 5.1 — Define "Medical Desert" Criteria
A medical desert is an area where:
- No facility within X km has a specific capability (e.g., surgery, imaging)
- Doctor-to-population ratio is critically low
- No emergency services available

#### Step 5.2 — Geographic Analysis
```python
# Using facility coordinates (lat/long from dataset)
# Calculate coverage areas
# Identify gaps using spatial analysis
import folium

# Create a map centered on Ghana
m = folium.Map(location=[7.9465, -1.0232], zoom_start=7)

# Add facility markers (color-coded by capability)
for _, facility in df.iterrows():
    folium.CircleMarker(
        location=[facility['lat'], facility['lon']],
        radius=facility['capacity'] / 10,
        color='green' if facility['has_surgery'] else 'red',
        popup=facility['name']
    ).add_to(m)

# Highlight desert regions
```

---

### Phase 6: User Interface (Day 4-5 — approx 3-4 hours)

> Worth 10% but makes a huge impression

#### Step 6.1 — Build Streamlit Dashboard
```python
import streamlit as st

st.title("Healthcare Intelligence Dashboard")
st.subheader("Virtue Foundation — Ghana")

# Natural language query box
query = st.text_input("Ask about healthcare facilities...")

# Tabs
tab1, tab2, tab3 = st.tabs(["Map", "Analysis", "Search"])

with tab1:
    # Interactive map
    st_folium(map_object)

with tab2:
    # Charts showing facility distribution, gaps
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
|-- requirements.txt
|-- data/
|   +-- ghana_facilities.csv          # Dataset
|-- src/
|   |-- __init__.py
|   |-- idp/
|   |   |-- __init__.py
|   |   |-- extractor.py              # IDP extraction logic
|   |   |-- models.py                 # Pydantic models
|   |   +-- anomaly_detector.py       # Anomaly detection
|   |-- rag/
|   |   |-- __init__.py
|   |   |-- vectorstore.py            # FAISS setup
|   |   +-- chain.py                  # RAG chain
|   |-- agents/
|   |   |-- __init__.py
|   |   |-- orchestrator.py           # LangGraph agent
|   |   +-- tools.py                  # Agent tools
|   |-- analysis/
|   |   |-- __init__.py
|   |   |-- medical_deserts.py        # Desert identification
|   |   +-- gap_analysis.py           # Gap analysis
|   +-- visualization/
|       |-- __init__.py
|       +-- map_builder.py            # Folium map
|-- app.py                             # Streamlit entry point
|-- notebooks/
|   +-- exploration.ipynb              # Data exploration
+-- tests/
    +-- test_extractor.py
```

---

## Prioritization Strategy (What to Build First)

| Priority | Component | Evaluation Weight | Time Estimate |
|---|---|---|---|
| P0 (Critical) | IDP Extraction (parsing free-text) | 30% | 6-8 hrs |
| P0 (Critical) | RAG Search + Querying | 35% | 4-5 hrs |
| P1 (Important) | Medical Desert Identification | 25% | 3-4 hrs |
| P1 (Important) | Anomaly Detection | (part of 35%) | 2-3 hrs |
| P2 (Nice to have) | Streamlit UI + Map | 10% | 3-4 hrs |
| P2 (Nice to have) | Citations | Bonus | 2 hrs |
| P3 (Stretch) | Planning System | Bonus | 3 hrs |
| P3 (Stretch) | MLflow Tracing | Bonus | 2 hrs |

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | Llama 3.1 70B via Groq | Free, fast inference |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) | Free, no API key needed |
| Orchestration | LangGraph | Most flexible for agentic workflows |
| Vector Store | FAISS | Free, fast, local — works with free Databricks |
| Data Processing | Pandas + Pydantic | Clean data modeling |
| UI | Streamlit | Fastest to build, great for hackathons |
| Map | Folium | Interactive maps, easy to embed |
| Tracking | MLflow | Trace agent reasoning for citations |
| Platform | Databricks Free Edition | Required by challenge |

---

## 5-Day Timeline

| Day | Focus | Deliverables |
|---|---|---|
| Day 1 (Mar 28) | Setup + Data Exploration | Environment ready, dataset understood |
| Day 2 (Mar 29) | IDP Extraction Pipeline | Working extractor with Pydantic models |
| Day 3 (Mar 30) | RAG + Agentic Orchestration | Working query system with LangGraph |
| Day 4 (Mar 31) | Medical Deserts + Map | Gap analysis + Folium visualization |
| Day 5 (Apr 1) | UI + Polish + Testing | Streamlit dashboard, documentation, demo |

---

## Quick Start: What to Do First

1. Sign up for Databricks Community Edition
2. Get your Groq API key ready
3. Download the Ghana dataset from the challenge link
4. Create a Python virtual environment and install dependencies
5. Open the dataset in a notebook and start exploring
6. Start with the IDP extractor — it is the highest-value component

---

## Common Pitfalls to Avoid

- Do not over-engineer — Start simple, iterate
- Do not ignore the free-text fields — They are the core of IDP and worth 30%
- Do not skip citations — Easy points, just return source documents
- Do not forget the "why" — Judges care about social impact storytelling
- Do not run expensive operations on all data first — Test on 5-10 rows, then scale
- Do not build UI first — Get the AI pipeline working, then wrap it in Streamlit

---

## Learning Resources

| Topic | Resource |
|---|---|
| LangChain basics | https://python.langchain.com/docs/get_started/quickstart |
| RAG explained | https://python.langchain.com/docs/tutorials/rag/ |
| LangGraph agents | https://langchain-ai.github.io/langgraph/tutorials/ |
| FAISS | https://github.com/facebookresearch/faiss/wiki/Getting-started |
| Streamlit | https://docs.streamlit.io/get-started |
| Folium maps | https://python-visualization.github.io/folium/quickstart.html |
| Pydantic | https://docs.pydantic.dev/latest/ |
| MLflow | https://mlflow.org/docs/latest/getting-started/ |
| Groq API | https://console.groq.com/docs/quickstart |
