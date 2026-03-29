# Coding Plan -- Step by Step Implementation Guide

This document breaks down every coding step we will do together, in order. We will go step by step -- complete one step, verify it works, then move to the next.

---

## Important Note: Databricks Community Edition Limitations

Databricks Community Edition (free) does NOT include:
- Vector Search
- Genie (Text2SQL)
- Model Serving / External Endpoints
- Mosaic AI Agent Framework
- Unity Catalog

If the hackathon provides a full Databricks workspace, use that instead. If not, we will use Community Edition for data processing + notebooks, and build the agent stack using open-source tools (LangChain + LangGraph + FAISS + Groq API) that run in Databricks notebooks.

QUESTION FOR TEAM: Does the hackathon provide a full Databricks workspace? Or are we using Community Edition only? This affects which features we can use.

---

## STEP 0: Environment Setup
**Time: 30 minutes**

### 0A. Local Python Environment
```
pip install pandas langchain langchain-groq langchain-community langgraph faiss-cpu pydantic mlflow streamlit folium sentence-transformers
```

### 0B. Databricks Workspace
- Log in to Databricks (Community Edition or hackathon workspace)
- Create a cluster (if Community Edition: single node, Python 3.10+)
- Upload dataset.csv to DBFS at /FileStore/dataset.csv

### 0C. API Keys
- Groq API key from https://console.groq.com
- Store locally in .env file (do NOT commit to GitHub)

### Verify Step 0:
- [ ] Python environment created and libraries installed
- [ ] Databricks cluster running
- [ ] dataset.csv uploaded to DBFS
- [ ] Groq API key working (test with a simple API call)

---

## STEP 1: Data Cleaning and Deduplication
**Time: 2-3 hours**
**File: notebooks/01_data_cleaning.py**

### What We Build:
A notebook that loads the raw CSV, deduplicates facilities, merges info from multiple source URLs, and saves a clean table.

### 1A. Load and Explore Raw Data
```python
# Load CSV
df = spark.read.csv("/FileStore/dataset.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("raw_facilities")

# Basic stats
print(f"Total rows: {df.count()}")
print(f"Unique facilities (pk_unique_id): {df.select('pk_unique_id').distinct().count()}")
print(f"Organization types: {df.groupBy('organization_type').count().collect()}")
print(f"Facility types: {df.groupBy('facilityTypeId').count().collect()}")
print(f"Regions: {df.groupBy('address_stateOrRegion').count().orderBy('count', ascending=False).collect()}")
```

### 1B. Deduplication Logic
Group by pk_unique_id and merge data from multiple source rows:
- Take the first non-null value for single-value fields (name, email, capacity, etc.)
- Combine all values for list fields (procedure, equipment, capability, specialties)
- Collect all source URLs

### 1C. Save Clean Delta Table
```python
df_clean.write.format("delta").mode("overwrite").saveAsTable("facilities_clean")
# Or if no Unity Catalog:
df_clean.write.format("parquet").mode("overwrite").save("/FileStore/facilities_clean")
```

### 1D. Create Searchable Text Column
Combine all free-text fields into one text column for embedding/search later.

### Verify Step 1:
- [ ] Clean table has one row per unique facility
- [ ] No duplicate pk_unique_id values
- [ ] Free-text fields are properly merged (not lost)
- [ ] Row count is ~400-500 (down from 1003)
- [ ] Spot-check 3-4 facilities that had multiple source rows

---

## STEP 2: Data Analysis and Anomaly Flagging
**Time: 2 hours**
**File: notebooks/02_data_analysis.py**

### What We Build:
Analysis queries that profile the data and pre-compute anomaly flags.

### 2A. Regional Statistics
```python
# Facilities per region
# Doctors per region
# Specialties per region
# Facilities with vs without procedure/equipment data
```

### 2B. Anomaly Detection Flags
Add columns for suspicious patterns:
```python
# Flag: many procedures but no doctors listed
# Flag: large capacity but no equipment listed  
# Flag: claims advanced specialties but facility type is "clinic"
# Flag: has emergency medicine specialty but no emergency capability text
```

### 2C. Medical Desert Identification
```python
# Regions with fewest facilities
# Regions with no surgery capability
# Regions with no emergency services
# Regions with lowest doctor count
```

### 2D. Save Analysis Results
```python
# Save enriched table with anomaly flags
# Save regional summary table
# Save medical desert summary
```

### Verify Step 2:
- [ ] Regional stats table is populated
- [ ] Anomaly flags are added (check a few known anomalies manually)
- [ ] Medical desert regions identified
- [ ] All results saved as tables/files

---

## STEP 3: Embedding and Vector Store
**Time: 2-3 hours**
**File: notebooks/03_vector_store.py (or src/rag/vectorstore.py)**

### What We Build:
Create embeddings of facility data for semantic search (RAG).

### 3A. Generate Embeddings
```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model (runs locally, no API needed)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create text for each facility
texts = []
for row in facilities:
    text = f"Facility: {row['name']}. "
    text += f"Location: {row['address_city']}, {row['address_stateOrRegion']}. "
    text += f"Type: {row['facilityTypeId']}. "
    text += f"Specialties: {row['specialties']}. "
    text += f"Procedures: {row['procedure']}. "
    text += f"Equipment: {row['equipment']}. "
    text += f"Capabilities: {row['capability']}. "
    text += f"Description: {row['description']}."
    texts.append(text)

# Create embeddings
embeddings = model.encode(texts)

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype('float32'))
```

### 3B. Save Index for Reuse
```python
faiss.write_index(index, "/FileStore/facility_index.faiss")
# Also save the texts and metadata mapping
```

### 3C. Test Semantic Search
```python
query = "hospitals with CT scanners in Northern Ghana"
query_embedding = model.encode([query])
distances, indices = index.search(query_embedding.astype('float32'), k=5)
for idx in indices[0]:
    print(texts[idx])
```

### Verify Step 3:
- [ ] Embeddings generated for all facilities
- [ ] FAISS index saved
- [ ] Test query returns relevant results (manually check top 5)
- [ ] Query "hospitals with surgery in Ashanti" returns Ashanti hospitals with surgery
- [ ] Query "dental clinics in Accra" returns dental facilities in Accra

---

## STEP 4: RAG Chain (Query + LLM Answer)
**Time: 2-3 hours**
**File: notebooks/04_rag_chain.py (or src/rag/chain.py)**

### What We Build:
A RAG pipeline: user asks a question -> vector search retrieves relevant facilities -> LLM generates answer with citations.

### 4A. Build RAG Chain with LangChain + Groq
```python
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key="YOUR_GROQ_KEY",
    temperature=0
)

rag_prompt = ChatPromptTemplate.from_template("""
You are a healthcare facility analyst for Ghana.
Answer the question using ONLY the facility data provided below.
Include specific facility names and locations in your answer.
Cite which facilities support each claim.

FACILITY DATA:
{context}

QUESTION: {question}

ANSWER:
""")
```

### 4B. Build the Pipeline
```python
def query_facilities(question: str, k: int = 10):
    # 1. Embed the question
    query_emb = model.encode([question]).astype('float32')
    
    # 2. Search FAISS
    distances, indices = index.search(query_emb, k)
    
    # 3. Get matching facility texts
    context = "\n\n".join([texts[i] for i in indices[0]])
    
    # 4. Generate answer with LLM
    chain = rag_prompt | llm
    response = chain.invoke({"context": context, "question": question})
    
    # 5. Return answer + source facilities (citations)
    sources = [facility_ids[i] for i in indices[0]]
    return response.content, sources
```

### 4C. Test With Must-Have Questions
```python
# Test basic queries from the questions doc
test_questions = [
    "How many hospitals have cardiology?",
    "What services does Korle Bu Teaching Hospital offer?",
    "Are there any clinics in Tamale that do surgery?",
    "Which region has the most hospitals?",
]
for q in test_questions:
    answer, sources = query_facilities(q)
    print(f"Q: {q}\nA: {answer}\nSources: {sources}\n")
```

### Verify Step 4:
- [ ] RAG chain returns coherent answers
- [ ] Answers include facility names (citations)
- [ ] Test all 5 basic query Must-Haves from question 1.x
- [ ] Answers are factually grounded in the data (not hallucinated)

---

## STEP 5: SQL Agent for Structured Queries
**Time: 2-3 hours**
**File: notebooks/05_sql_agent.py (or src/agents/sql_agent.py)**

### What We Build:
An agent that converts natural language to SQL queries over the facility data.

### 5A. Text-to-SQL with LangChain
```python
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

# Connect to the Databricks table (or SQLite for local dev)
db = SQLDatabase.from_databricks(...)  # or SQLAlchemy for local

sql_chain = create_sql_query_chain(llm, db)
```

### 5B. Alternative: Direct Spark SQL Agent
```python
# If on Databricks, use Spark SQL directly
def sql_query_agent(question: str):
    # 1. LLM generates SQL from question
    sql_prompt = f"""Given this table schema:
    Table: facilities_clean
    Columns: name, pk_unique_id, specialties, procedure, equipment, capability,
             facilityTypeId, operatorTypeId, address_city, address_stateOrRegion,
             numberDoctors, capacity, ...
    
    Write a Spark SQL query to answer: {question}
    Return ONLY the SQL query, no explanation."""
    
    sql = llm.invoke(sql_prompt).content
    
    # 2. Execute SQL
    result = spark.sql(sql)
    
    # 3. LLM summarizes result
    return result, sql
```

### 5C. Handle Anomaly Detection Queries
```python
# Pre-built SQL for key anomaly questions:
anomaly_queries = {
    "unrealistic_procedures": """
        SELECT name, address_stateOrRegion, numberDoctors, capacity,
               size(split(procedure, ',')) as num_procedures
        FROM facilities_clean
        WHERE size(split(procedure, ',')) > 10 
        AND (numberDoctors IS NULL OR numberDoctors < 2)
    """,
    "mismatched_capabilities": """
        SELECT name, specialties, equipment, capability
        FROM facilities_clean
        WHERE specialties LIKE '%Surgery%' 
        AND (equipment IS NULL OR equipment = '[]')
    """,
    # ... more anomaly queries
}
```

### Verify Step 5:
- [ ] "How many hospitals have cardiology?" returns a correct count
- [ ] "Which region has the most hospitals?" returns correct region
- [ ] SQL generation does not hallucinate column names
- [ ] Anomaly queries return meaningful results

---

## STEP 6: Medical Reasoning Agent
**Time: 3-4 hours**
**File: notebooks/06_reasoning_agent.py (or src/agents/medical_reasoner.py)**

### What We Build:
An agent that performs complex medical reasoning -- cross-validating claims, detecting anomalies, and making inferences.

### 6A. Anomaly Reasoning
```python
def analyze_facility_anomalies(facility_data: dict) -> dict:
    prompt = f"""You are a medical facility auditor analyzing Ghana healthcare data.
    
    Facility: {facility_data['name']}
    Location: {facility_data['address_city']}, {facility_data['address_stateOrRegion']}
    Type: {facility_data['facilityTypeId']}
    Doctors: {facility_data['numberDoctors']}
    Bed Capacity: {facility_data['capacity']}
    Claimed Specialties: {facility_data['specialties']}
    Claimed Procedures: {facility_data['procedure']}
    Listed Equipment: {facility_data['equipment']}
    Capabilities: {facility_data['capability']}
    
    Analyze this facility for:
    1. MISMATCHES: Procedures claimed without necessary equipment
    2. SCALE ISSUES: Too many specialties/procedures for facility size
    3. CREDIBILITY: Claims that seem unrealistic for the facility type
    4. MISSING DATA: Critical information gaps
    
    For each finding, explain WHY it is suspicious with medical reasoning.
    Rate overall credibility: HIGH / MEDIUM / LOW
    """
    return llm.invoke(prompt).content
```

### 6B. Cross-Facility Comparison
```python
def compare_region_capabilities(region: str):
    # Get all facilities in region
    # Identify what specialties/procedures are available
    # Identify what is MISSING
    # Flag single-facility dependencies
    pass
```

### 6C. Medical Desert Reasoning
```python
def identify_medical_deserts():
    # For each region:
    #   - Count facilities, doctors, beds
    #   - Check for emergency capability
    #   - Check for surgical capability
    #   - Score as "desert" if below thresholds
    pass
```

### Verify Step 6:
- [ ] Anomaly analysis correctly identifies suspicious facilities
- [ ] Cross-facility comparison highlights gaps
- [ ] Medical desert analysis matches expectations from data exploration
- [ ] Reasoning agent provides explanations (not just flags)

---

## STEP 7: Supervisor Agent (Router)
**Time: 2-3 hours**
**File: notebooks/07_supervisor_agent.py (or src/agents/supervisor.py)**

### What We Build:
A supervisor that classifies the user's question and routes to the right sub-agent.

### 7A. Intent Classification
```python
def classify_query(question: str) -> str:
    prompt = f"""Classify this healthcare facility question into ONE category:

    STRUCTURED: Counting, aggregation, filtering by specific fields
      Examples: "How many hospitals have cardiology?", "Which region has most hospitals?"
    
    SEMANTIC: Searching for specific services, equipment, or capabilities
      Examples: "What services does X offer?", "Find clinics that do surgery"
    
    REASONING: Anomaly detection, cross-validation, medical inference
      Examples: "Which facilities have suspicious claims?", "What mismatches exist?"
    
    DESERT: Medical desert analysis, gap identification, resource distribution
      Examples: "Where are the biggest healthcare gaps?", "Which areas lack emergency care?"
    
    Question: {question}
    Category:"""
    
    return llm.invoke(prompt).content.strip()
```

### 7B. Router Logic
```python
def agent_query(question: str):
    category = classify_query(question)
    
    if category == "STRUCTURED":
        return sql_query_agent(question)
    elif category == "SEMANTIC":
        return query_facilities(question)  # RAG
    elif category == "REASONING":
        return reasoning_agent(question)
    elif category == "DESERT":
        return medical_desert_agent(question)
    else:
        return query_facilities(question)  # fallback to RAG
```

### 7C. LangGraph Version (Stretch)
```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)
workflow.add_node("classify", classify_node)
workflow.add_node("sql_agent", sql_node)
workflow.add_node("rag_agent", rag_node)
workflow.add_node("reasoning_agent", reasoning_node)
workflow.add_node("respond", response_node)

workflow.add_conditional_edges("classify", route_by_intent, {
    "STRUCTURED": "sql_agent",
    "SEMANTIC": "rag_agent", 
    "REASONING": "reasoning_agent",
})

for node in ["sql_agent", "rag_agent", "reasoning_agent"]:
    workflow.add_edge(node, "respond")

workflow.set_entry_point("classify")
agent = workflow.compile()
```

### Verify Step 7:
- [ ] Classification correctly routes basic queries to SQL agent
- [ ] Classification correctly routes semantic queries to RAG
- [ ] Classification correctly routes anomaly queries to reasoning agent
- [ ] End-to-end: ask a question, get a routed answer
- [ ] Test with 5+ questions from each category

---

## STEP 8: Visualization and Dashboard
**Time: 3-4 hours**
**File: app.py (Streamlit) or notebooks/08_dashboard.py**

### What We Build:
An interactive dashboard with natural language query, map, and analysis views.

### 8A. Streamlit App
```python
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Healthcare Intelligence Dashboard", layout="wide")
st.title("Healthcare Intelligence Dashboard")
st.subheader("Virtue Foundation -- Ghana")

# Sidebar
st.sidebar.header("Query the Agent")
query = st.sidebar.text_area("Ask a question about healthcare facilities...")
if st.sidebar.button("Search"):
    with st.spinner("Analyzing..."):
        answer, sources = agent_query(query)
    st.markdown(answer)
    st.markdown("**Sources:**")
    for s in sources:
        st.markdown(f"- {s}")

# Main area tabs
tab1, tab2, tab3, tab4 = st.tabs(["Map", "Regional Analysis", "Anomalies", "Medical Deserts"])
```

### 8B. Map with Folium
```python
# Ghana map with facility markers
# Color by: facility type, specialty, anomaly flag
# Popup with facility details
```

### 8C. Analysis Charts
```python
# Bar chart: facilities per region
# Pie chart: facility types
# Table: anomaly flagged facilities
# Heatmap: medical deserts
```

### Verify Step 8:
- [ ] Streamlit app launches without errors
- [ ] Query box works and returns answers
- [ ] Map displays with facility markers
- [ ] Charts render correctly
- [ ] Ask the team to test: is the UI intuitive for a non-technical NGO planner?

---

## STEP 9: MLflow Tracing and Citations
**Time: 2 hours (stretch goal)**
**File: src/agents/tracing.py**

### What We Build:
Log each agent reasoning step for transparency and citations.

### 9A. MLflow Experiment Tracking
```python
import mlflow

mlflow.set_experiment("vf-healthcare-agent")

def traced_agent_query(question):
    with mlflow.start_run():
        mlflow.log_param("question", question)
        
        # Step 1: Classification
        category = classify_query(question)
        mlflow.log_param("intent_category", category)
        
        # Step 2: Sub-agent execution
        if category == "STRUCTURED":
            sql, result = sql_query_agent(question)
            mlflow.log_param("generated_sql", sql)
        elif category == "SEMANTIC":
            answer, sources = query_facilities(question)
            mlflow.log_param("num_sources", len(sources))
            mlflow.log_param("source_ids", str(sources))
        
        # Step 3: Final answer
        mlflow.log_param("answer_length", len(answer))
        
        return answer
```

### Verify Step 9:
- [ ] MLflow experiments show up in Databricks UI
- [ ] Each query run has logged parameters
- [ ] Can trace which data was used at each step

---

## STEP 10: Final Polish and Submission Prep
**Time: 3-4 hours**

### 10A. Test All Must-Have Questions
Run through every Must-Have question from the agent questions doc:
- [ ] Basic Queries 1.1-1.5
- [ ] Geospatial 2.1, 2.3
- [ ] Anomaly Detection 4.4, 4.7, 4.8, 4.9
- [ ] Workforce 6.1
- [ ] Resource Gaps 7.5, 7.6
- [ ] NGO Analysis 8.3

### 10B. Write README.md
- Project description
- Setup instructions (how to run)
- Screenshots/demo
- Team information
- Open-source license (MIT or Apache 2.0)

### 10C. Record Demo Video (5 minutes)
- Show the problem (medical deserts)
- Show the agent answering questions
- Show the map visualization
- Show anomaly detection
- Explain social impact

### 10D. Final Git Push
```
git add .
git commit -m "Final submission: Healthcare IDP Agent"
git push origin main
```

### 10E. Submit
- GitHub repo URL
- Demo video link
- Project description

---

## Quick Reference: File Map

| Step | File | Description |
|---|---|---|
| 1 | notebooks/01_data_cleaning.py | Data dedup + Delta tables |
| 2 | notebooks/02_data_analysis.py | Stats + anomaly flags |
| 3 | notebooks/03_vector_store.py | Embeddings + FAISS index |
| 4 | notebooks/04_rag_chain.py | RAG pipeline |
| 5 | notebooks/05_sql_agent.py | Text-to-SQL agent |
| 6 | notebooks/06_reasoning_agent.py | Medical reasoning |
| 7 | notebooks/07_supervisor_agent.py | Query router |
| 8 | app.py | Streamlit dashboard |
| 9 | notebooks/09_mlflow_tracing.py | MLflow citations |
| 10 | README.md | Submission docs |

---

## Day-by-Day Schedule

| Day | Steps | Hours | Milestone |
|---|---|---|---|
| Day 1 | Step 0 + Step 1 + Step 2 | 5-6 hrs | Clean data + analysis done |
| Day 2 | Step 3 + Step 4 | 5-6 hrs | RAG pipeline working |
| Day 3 | Step 5 + Step 6 | 5-7 hrs | SQL + Reasoning agents working |
| Day 4 | Step 7 + Step 8 | 5-7 hrs | Full agent + dashboard |
| Day 5 | Step 9 + Step 10 | 5-6 hrs | Polish + demo + submit |
