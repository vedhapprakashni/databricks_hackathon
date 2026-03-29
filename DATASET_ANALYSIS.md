# Dataset and Documentation Analysis

## Dataset Overview (dataset.csv)

- 1,003 rows of Ghana healthcare facility data
- 41 columns covering structured fields + free-text fields
- Data has already been pre-parsed by the Virtue Foundation pipeline using the Pydantic models provided

### Key Metrics

| Metric | Value |
|---|---|
| Total rows | 1,003 |
| Unique facilities | ~400-500 (many appear multiple times from different source URLs) |
| Organization types | facility (majority), ngo (handful) |
| Country | All Ghana (GH) |
| Facility types | hospital, clinic, dentist, pharmacy |
| Operator types | public, private, null |

### Data Quality Issues

- Duplicate facilities: Same facility (same pk_unique_id) scraped from multiple source URLs (e.g., "1st Foundation Clinic" appears 4 times from different GhanaBusinessWeb pages)
- Sparse free-text fields: Many rows have empty procedure, equipment, capability arrays ([] or null)
- Rich facilities are rare: Only ~20-30% of facilities have detailed procedure/equipment/capability data
- Inconsistent naming: Some facilities have ALL CAPS names, some have location appended
- "internalMedicine" overuse: Many facilities tagged with internalMedicine as a default when no specific specialty info exists

---

## Agent Questions Summary (59 Total, 11 Categories)

### Must-Have Questions (MVP Targets)

| Category | Count | Key Questions |
|---|---|---|
| Basic Queries | 5 | "How many hospitals have cardiology?", "What services does [Facility] offer?" |
| Geospatial | 2 must-haves | "Hospitals within X km of [location]", "Geographic cold spots" |
| Anomaly Detection | 5 must-haves | Unrealistic procedures vs size, mismatched capabilities, correlated features |
| Workforce | 1 must-have | "Where is the workforce for [subspecialty] practicing?" |
| Resource Gaps | 2 must-haves | Single-facility dependency, oversupply vs scarcity |
| NGO Analysis | 1 must-have | "Gaps where no NGOs work despite need" |

### Architecture Components Needed (from questions doc)

| Component | Used By Questions | Priority |
|---|---|---|
| Genie Chat (Text2SQL) | Almost all questions | Critical |
| Vector Search with Filtering | 1.3, 1.4, 3.2, 5.1, 5.2, 6.5, 6.6 | Critical |
| Medical Reasoning Agent | 3.4, 4.x, 5.x, 6.x, 7.x, 8.x | Critical |
| Geospatial Calculation | 2.1, 2.3 | Important |
| Supervisor Agent (Router) | All (routes to sub-agents) | Important |
| External Data | 2.2, 2.4, 9.x, 10.x | Stretch |

---

## Pydantic Models Analysis

These 4 Python files are the extraction pipeline used by the Virtue Foundation to create the dataset:

| File | Purpose | System Prompt |
|---|---|---|
| organization_extraction.py | Classify text into NGOs vs Facilities | Detailed classification rules |
| facility_and_ngo_fields.py | Extract structured fields (name, address, phone, etc.) | Strict attribution rules |
| free_form.py | Extract procedure/equipment/capability facts | Detailed category definitions |
| medical_specialties.py | Map text to standardized specialty codes | Specialty hierarchy mapping |

IMPORTANT: The dataset is the OUTPUT of these extraction models. We do NOT need to re-run this extraction pipeline. Our job is to build an agent that queries and reasons over this already-parsed data.

---

## Critical Strategic Insights

### 1. The data is ALREADY parsed -- IDP innovation is in the QUERYING layer

The Virtue Foundation already ran their IDP pipeline (the Pydantic models) to create the dataset. Our IDP innovation should focus on:
- Synthesizing across multiple rows for the same facility (deduplication + merging)
- Reasoning over the free-text fields (procedure, equipment, capability) to answer complex questions
- Detecting anomalies in the already-parsed data (e.g., mismatched claims)

### 2. Genie (Text2SQL) is the backbone

The Agent Questions doc explicitly maps most questions to "Genie Chat" -- Databricks' Text2SQL engine. This should be the first and primary agent component.

### 3. Vector Search is critical for free-text fields

Questions like "What services does [Facility] offer?" and "Which facilities' language suggests they refer patients..." require semantic search over the procedure, equipment, and capability columns (which are free-text lists).

### 4. The Medical Reasoning Agent is what differentiates winners

Many "Should Have" and "Must Have" questions require a reasoning agent that understands medical context (e.g., "A facility claiming cataract surgery should have an operating microscope"). This is where Groq + Llama 3.1 comes in.

### 5. Deduplication is an immediate need

The same facility appears multiple times (different source URLs). A deduplication/merge step is needed before loading into Delta Tables.

---

## Revised Priority Order

1. Data Cleaning + Deduplication -> Load clean data into Delta Tables
2. Genie Chat / Text2SQL -> Handle all "Must Have" structured queries
3. Vector Search -> Handle free-text semantic queries
4. Medical Reasoning Agent -> Handle anomaly detection + complex reasoning
5. Supervisor Agent -> Route queries to the right sub-agent
6. Geospatial Analysis -> Medical desert identification
7. Dashboard / Map -> Visualization layer
