"""
Healthcare Intelligence Agent — Streamlit Frontend
Virtue Foundation × Databricks Hackathon

A multi-agent AI system for analyzing Ghana healthcare facility data,
identifying medical deserts, and detecting anomalies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
import duckdb
import folium
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import streamlit.components.v1 as components

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Intelligence Agent — Virtue Foundation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS — Premium Dark Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(52, 152, 219, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(52, 152, 219, 0.2);
    }
    .kpi-value {
        font-size: 42px;
        font-weight: 700;
        margin: 8px 0;
        background: linear-gradient(135deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label {
        font-size: 13px;
        color: #95a5a6;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 500;
    }

    /* Section headers */
    .section-header {
        font-size: 24px;
        font-weight: 700;
        color: #ecf0f1;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(52, 152, 219, 0.3);
    }

    /* Chat messages */
    .chat-user {
        background: linear-gradient(135deg, #2c3e50, #34495e);
        border-radius: 16px 16px 4px 16px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 3px solid #3498db;
    }
    .chat-agent {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px 16px 16px 4px;
        padding: 16px 20px;
        margin: 8px 0;
        border-left: 3px solid #2ecc71;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: 500;
    }

    /* Desert score badge */
    .desert-critical { color: #e74c3c; font-weight: 700; }
    .desert-high { color: #e67e22; font-weight: 700; }
    .desert-moderate { color: #f1c40f; font-weight: 600; }
    .desert-adequate { color: #2ecc71; font-weight: 600; }

    /* Anomaly badge */
    .anomaly-badge {
        display: inline-block;
        background: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }

    /* Title banner */
    .title-banner {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, rgba(52,152,219,0.1), rgba(46,204,113,0.1));
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(52,152,219,0.15);
    }
    .title-banner h1 {
        font-size: 32px;
        background: linear-gradient(135deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .title-banner p {
        color: #95a5a6;
        font-size: 15px;
        margin: 8px 0 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA LOADING & PROCESSING
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading healthcare facility data...")
def load_and_process_data():
    """Load dataset.csv, deduplicate, and enrich."""
    df = pd.read_csv("dataset.csv")

    # ── Deduplication: keep first row per pk_unique_id, merge text fields ──
    text_cols = ['specialties', 'procedure', 'equipment', 'capability', 'description']
    agg_dict = {}
    for col in df.columns:
        if col == 'pk_unique_id':
            continue
        if col in text_cols:
            agg_dict[col] = lambda x: ' | '.join([str(v) for v in x.dropna().unique()])
        else:
            agg_dict[col] = 'first'

    df_dedup = df.groupby('pk_unique_id', as_index=False).agg(agg_dict)

    # ── Parse JSON-like list columns ──
    def safe_parse_list(val):
        if pd.isna(val) or val in ('null', 'None', '', '[]', 'nan'):
            return []
        try:
            parsed = json.loads(str(val).replace("'", '"'))
            if isinstance(parsed, list):
                return [str(x) for x in parsed if x]
            return []
        except Exception:
            return [str(val)] if val else []

    for col in ['specialties', 'procedure', 'equipment']:
        df_dedup[f'{col}_list'] = df_dedup[col].apply(safe_parse_list)
        df_dedup[f'num_{col}'] = df_dedup[f'{col}_list'].apply(len)

    # ── Numeric cleaning ──
    for col in ['numberDoctors', 'capacity']:
        df_dedup[col] = pd.to_numeric(df_dedup[col], errors='coerce')

    # ── Create search text ──
    df_dedup['search_text'] = df_dedup.apply(
        lambda r: ' '.join(filter(None, [
            str(r.get('name', '')),
            str(r.get('description', '')),
            str(r.get('specialties', '')),
            str(r.get('procedure', '')),
            str(r.get('equipment', '')),
            str(r.get('capability', '')),
            str(r.get('address_city', '')),
            str(r.get('address_stateOrRegion', '')),
        ])), axis=1
    )

    # ── Anomaly Flags ──
    df_dedup['flag_high_procedures_low_doctors'] = (
        (df_dedup['num_procedure'] > 5) &
        (df_dedup['numberDoctors'].fillna(0) < 2)
    )
    df_dedup['flag_high_capacity_no_surgery'] = (
        (df_dedup['capacity'].fillna(0) > 50) &
        (~df_dedup['procedure'].fillna('').str.lower().str.contains('surg'))
    )
    df_dedup['flag_many_specialties_small_size'] = (
        (df_dedup['num_specialties'] > 5) &
        (df_dedup['capacity'].fillna(0) < 20) &
        (df_dedup['capacity'].fillna(0) > 0)
    )
    df_dedup['flag_no_doctors_listed'] = (
        (df_dedup['facilityTypeId'] == 'hospital') &
        (df_dedup['numberDoctors'].isna())
    )

    return df_dedup


@st.cache_data(show_spinner="Computing regional analysis...")
def compute_regional_analysis(df):
    """Compute medical desert scores by region."""
    # Filter out null regions
    valid = df[~df['address_stateOrRegion'].astype(str).str.lower().isin(
        ['null', 'none', 'nan', '']
    )]
    regional = valid.groupby('address_stateOrRegion').agg(
        facility_count=('pk_unique_id', 'count'),
        total_doctors=('numberDoctors', lambda x: x.fillna(0).sum()),
        total_beds=('capacity', lambda x: x.fillna(0).sum()),
        has_surgery=('procedure', lambda x: x.fillna('').str.lower().str.contains('surg').sum()),
        has_emergency=('capability', lambda x: x.fillna('').str.lower().str.contains('emergency|24').sum()),
        num_hospitals=('facilityTypeId', lambda x: (x == 'hospital').sum()),
        avg_specialties=('num_specialties', 'mean'),
    ).reset_index()

    # Desert score: higher = more underserved
    max_fac = regional['facility_count'].max() or 1
    max_doc = regional['total_doctors'].max() or 1
    max_beds = regional['total_beds'].max() or 1

    regional['desert_score'] = (
        40 * (1 - regional['facility_count'] / max_fac) +
        25 * (1 - regional['total_doctors'] / max_doc) +
        20 * (1 - regional['total_beds'] / max_beds) +
        10 * (1 - regional['has_surgery'].clip(0, 1)) +
        5 * (1 - regional['has_emergency'].clip(0, 1))
    ).round(1)

    return regional.sort_values('desert_score', ascending=False)


# ──────────────────────────────────────────────
# VECTOR INDEX (FAISS)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Building semantic search index...")
def build_faiss_index(texts):
    """Build FAISS index from search texts."""
    try:
        from sentence_transformers import SentenceTransformer
        import faiss

        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index, model
    except Exception as e:
        st.warning(f"FAISS index not available: {e}")
        return None, None


# ──────────────────────────────────────────────
# LLM SETUP
# ──────────────────────────────────────────────
def get_llm():
    """Initialize Databricks Foundation Model LLM."""
    db_host = os.environ.get("DATABRICKS_HOST")
    db_token = os.environ.get("DATABRICKS_TOKEN")

    # Try streamlit secrets only if secrets file exists
    if not (db_host and db_token):
        try:
            if hasattr(st, 'secrets') and len(st.secrets) > 0:
                db_host = st.secrets.get("DATABRICKS_HOST", db_host)
                db_token = st.secrets.get("DATABRICKS_TOKEN", db_token)
        except Exception:
            pass

    if not (db_host and db_token):
        return None

    try:
        from langchain_community.chat_models import ChatDatabricks
        os.environ["DATABRICKS_HOST"] = db_host
        os.environ["DATABRICKS_TOKEN"] = db_token
        return ChatDatabricks(
            endpoint="databricks-meta-llama-3-3-70b-instruct",
            temperature=0,
            max_tokens=2048
        )
    except Exception as e:
        st.error(f"Databricks LLM init error: {e}")
        return None


# ──────────────────────────────────────────────
# AGENT FUNCTIONS
# ──────────────────────────────────────────────
def semantic_search(query, df, index, model, top_k=5):
    """Search facilities using FAISS semantic similarity."""
    if index is None or model is None:
        # Fallback: keyword search
        query_lower = query.lower()
        mask = df['search_text'].str.lower().str.contains(
            '|'.join(query_lower.split()), na=False
        )
        results = df[mask].head(top_k)
        return results

    query_vec = model.encode([query]).astype('float32')
    import faiss
    faiss.normalize_L2(query_vec)
    scores, indices = index.search(query_vec, top_k)

    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]
    return results


def sql_agent(query, llm, schema_info):
    """Generate and execute SQL against DuckDB."""
    if llm is None:
        return "LLM not configured. Please set your Groq API key.", None

    prompt = f"""You are a SQL expert. Convert this natural language question to a DuckDB SQL query.

TABLE: facilities
COLUMNS AND TYPES:
{schema_info}

RULES:
- Use ONLY columns listed above
- Return ONLY the SQL query, nothing else
- Use ILIKE for case-insensitive text matching
- For JSON-like list columns (specialties, procedure, equipment), use ILIKE with wildcards
- Limit results to 20 rows unless asked for counts/aggregations
- Never use DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE

QUESTION: {query}

SQL:"""

    try:
        response = llm.invoke(prompt)
        sql = response.content.strip()
        # Clean markdown code fences
        sql = re.sub(r'^```\w*\n?', '', sql)
        sql = re.sub(r'\n?```$', '', sql)
        sql = sql.strip()

        # Safety check
        forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'CREATE']
        if any(kw in sql.upper() for kw in forbidden):
            return "⚠️ Query blocked — destructive SQL not allowed.", None

        return sql, None
    except Exception as e:
        return f"Error generating SQL: {e}", None


def reasoning_agent(query, context, llm):
    """Medical reasoning over facility data."""
    if llm is None:
        return "LLM not configured. Please set your Groq API key."

    prompt = f"""You are a medical intelligence analyst working for the Virtue Foundation, 
analyzing healthcare facility data from Ghana.

CONTEXT DATA:
{context[:3000]}

QUESTION: {query}

Provide a thorough, evidence-based analysis. Include:
1. Direct answer to the question
2. Supporting data points from the context
3. Implications for healthcare access
4. Recommendations for the Virtue Foundation

Be specific and reference actual facility names/regions when possible."""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error: {e}"


def classify_intent(query):
    """Classify query intent for routing."""
    q = query.lower()
    # SQL-oriented keywords
    sql_kw = ['how many', 'count', 'total', 'average', 'which region', 'most',
              'least', 'percentage', 'list all', 'number of', 'compare']
    # Semantic search keywords
    semantic_kw = ['services', 'offer', 'provide', 'treat', 'capable',
                   'equipment', 'procedure', 'what does', 'tell me about']
    # Reasoning keywords
    reasoning_kw = ['anomal', 'suspicious', 'mismatch', 'desert', 'underserved',
                    'unrealistic', 'cross-valid', 'inconsisten', 'flag', 'correlat',
                    'why', 'explain', 'analyze', 'recommend']

    scores = {
        'SQL': sum(1 for kw in sql_kw if kw in q),
        'SEMANTIC': sum(1 for kw in semantic_kw if kw in q),
        'REASONING': sum(1 for kw in reasoning_kw if kw in q),
    }
    return max(scores, key=scores.get) if max(scores.values()) > 0 else 'SEMANTIC'


# ──────────────────────────────────────────────
# GHANA COORDINATES
# ──────────────────────────────────────────────
GHANA_COORDS = {
    "Greater Accra": {"lat": 5.6037, "lon": -0.1870},
    "Ashanti": {"lat": 6.7470, "lon": -1.5209},
    "Western": {"lat": 5.3960, "lon": -2.1500},
    "Central": {"lat": 5.4510, "lon": -1.2000},
    "Eastern": {"lat": 6.2500, "lon": -0.4500},
    "Volta": {"lat": 6.6000, "lon": 0.4700},
    "Northern": {"lat": 9.4000, "lon": -1.0000},
    "Upper East": {"lat": 10.7500, "lon": -0.8500},
    "Upper West": {"lat": 10.2500, "lon": -2.1400},
    "Brong-Ahafo": {"lat": 7.9500, "lon": -1.6700},
    "Bono": {"lat": 7.9500, "lon": -2.3100},
    "Bono East": {"lat": 7.7500, "lon": -1.0500},
    "Ahafo": {"lat": 7.0000, "lon": -2.3500},
    "Western North": {"lat": 6.3000, "lon": -2.8000},
    "Oti": {"lat": 7.9000, "lon": 0.3000},
    "North East": {"lat": 10.5000, "lon": -0.2000},
    "Savannah": {"lat": 9.0000, "lon": -1.8000},
}

# Map district/metro names to parent region coords
DISTRICT_TO_REGION = {
    "accra": "Greater Accra", "tema": "Greater Accra", "ledzokuku": "Greater Accra",
    "ga ": "Greater Accra", "adentan": "Greater Accra", "kpeshie": "Greater Accra",
    "la dade": "Greater Accra", "ayawaso": "Greater Accra", "okaikoi": "Greater Accra",
    "weija": "Greater Accra", "ablekuma": "Greater Accra", "korle": "Greater Accra",
    "kumasi": "Ashanti", "obuasi": "Ashanti", "ejisu": "Ashanti", "mampong": "Ashanti",
    "bekwai": "Ashanti", "offinso": "Ashanti", "asokore": "Ashanti", "asutifi": "Ashanti",
    "atwima": "Ashanti", "bosomtwe": "Ashanti", "sekyere": "Ashanti", "afigya": "Ashanti",
    "amansie": "Ashanti", "adansi": "Ashanti", "ahafo ano": "Ashanti", "asante": "Ashanti",
    "takoradi": "Western", "sekondi": "Western", "tarkwa": "Western", "axim": "Western",
    "prestea": "Western", "shama": "Western", "ahanta": "Western", "nzema": "Western",
    "ellembelle": "Western", "jomoro": "Western", "wassa": "Western",
    "cape coast": "Central", "winneba": "Central", "mankessim": "Central",
    "agona": "Central", "gomoa": "Central", "mfantsiman": "Central", "keea": "Central",
    "awutu": "Central", "abura": "Central", "assin": "Central", "twifo": "Central",
    "koforidua": "Eastern", "nkawkaw": "Eastern", "nsawam": "Eastern", "suhum": "Eastern",
    "akim": "Eastern", "kwahu": "Eastern", "birim": "Eastern", "atiwa": "Eastern",
    "ho ": "Volta", "keta": "Volta", "hohoe": "Volta", "kpando": "Volta",
    "akatsi": "Volta", "tongu": "Volta", "anlo": "Volta", "south dayi": "Volta",
    "tamale": "Northern", "yendi": "Northern", "savelugu": "Northern",
    "tolon": "Northern", "sagnarigu": "Northern", "mion": "Northern",
    "bolgatanga": "Upper East", "bawku": "Upper East", "navrongo": "Upper East",
    "kassena": "Upper East", "builsa": "Upper East",
    "wa ": "Upper West", "tumu": "Upper West", "nadowli": "Upper West",
    "lawra": "Upper West", "jirapa": "Upper West", "sissala": "Upper West",
    "sunyani": "Bono", "berekum": "Bono", "dormaa": "Bono", "jaman": "Bono",
    "techiman": "Bono East", "atebubu": "Bono East", "kintampo": "Bono East",
    "nkoranza": "Bono East", "pru ": "Bono East",
    "goaso": "Ahafo", "tano": "Ahafo", "asunafo": "Ahafo", "bechem": "Ahafo",
    "bibiani": "Western North", "sefwi": "Western North", "juaboso": "Western North",
    "bole": "Savannah", "damongo": "Savannah", "sawla": "Savannah",
    "nalerigu": "North East", "gambaga": "North East",
    "kadjebi": "Oti", "nkwanta": "Oti", "krachi": "Oti",
    "brong": "Brong-Ahafo",
}


def get_region_coords(region_name):
    if not region_name or str(region_name).lower() in ('null', 'none', 'nan', ''):
        return None
    rn = str(region_name).lower().strip()

    # Direct match
    for key, coords in GHANA_COORDS.items():
        if key.lower() in rn or rn in key.lower():
            return coords

    # District-to-region mapping
    for district_kw, parent_region in DISTRICT_TO_REGION.items():
        if district_kw in rn:
            return GHANA_COORDS.get(parent_region)

    # Fallback: approximate center of Ghana with small random offset
    import random
    return {"lat": 7.5 + random.uniform(-0.5, 0.5), "lon": -1.5 + random.uniform(-0.5, 0.5)}


# ──────────────────────────────────────────────
# MAP BUILDERS
# ──────────────────────────────────────────────
def build_facility_map(df):
    """Interactive Folium map of facilities."""
    m = folium.Map(location=[7.9465, -1.0232], zoom_start=7, tiles="cartodbpositron")

    type_colors = {
        "hospital": "#e74c3c", "clinic": "#3498db",
        "dentist": "#2ecc71", "pharmacy": "#f39c12",
    }

    import random
    random.seed(42)

    for _, row in df.iterrows():
        coords = get_region_coords(row.get('address_stateOrRegion'))
        if not coords:
            continue

        offset_lat = random.uniform(-0.15, 0.15)
        offset_lon = random.uniform(-0.15, 0.15)
        ftype = str(row.get('facilityTypeId', 'unknown'))
        color = type_colors.get(ftype, '#95a5a6')
        name = str(row.get('name', 'Unknown'))

        popup = f"""<div style="min-width:180px;">
            <b style="color:{color}">{name}</b><br>
            Type: {ftype}<br>
            Region: {row.get('address_stateOrRegion','N/A')}<br>
            Doctors: {row.get('numberDoctors','N/A')}<br>
            Beds: {row.get('capacity','N/A')}
        </div>"""

        folium.CircleMarker(
            location=[coords['lat'] + offset_lat, coords['lon'] + offset_lon],
            radius=5 if ftype != 'hospital' else 8,
            color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(popup, max_width=250),
            tooltip=f"{name} ({ftype})"
        ).add_to(m)

    # Legend
    legend = """<div style="position:fixed;bottom:50px;left:50px;z-index:1000;
        background:white;padding:10px;border-radius:8px;border:1px solid #ccc;font-size:12px;">
        <b>Facility Types</b><br>
        <span style="color:#e74c3c">●</span> Hospital &nbsp;
        <span style="color:#3498db">●</span> Clinic &nbsp;
        <span style="color:#2ecc71">●</span> Dentist &nbsp;
        <span style="color:#f39c12">●</span> Pharmacy
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m


def build_desert_map(regional_df):
    """Medical desert and Risk Zone heatmap."""
    m = folium.Map(location=[7.9465, -1.0232], zoom_start=7, tiles="cartodbpositron")

    for _, row in regional_df.iterrows():
        coords = get_region_coords(row.get('address_stateOrRegion'))
        if not coords:
            continue

        score = row.get('desert_score', 0)
        has_emerg = row.get('has_emergency', 0)
        
        # Determine base color
        if score > 80:
            color, opacity = '#e74c3c', 0.8
        elif score > 50:
            color, opacity = '#e67e22', 0.6
        elif score > 20:
            color, opacity = '#f1c40f', 0.5
        else:
            color, opacity = '#2ecc71', 0.4

        radius = max(15, 50 - int(row.get('facility_count', 0)) * 2)

        # Build popup
        popup_html = f"""<div style="min-width:200px;color:white;background:rgba(0,0,0,0.85);
            padding:10px;border-radius:8px;">
            <h4 style="color:{color};margin:0">{row['address_stateOrRegion']}</h4>
            <hr style="border-color:{color};margin:5px 0">
            🏥 Facilities: <b>{int(row.get('facility_count',0))}</b><br>
            👨‍⚕️ Doctors: <b>{int(row.get('total_doctors',0))}</b><br>
            🚨 Emergency Units: <b>{int(has_emerg)}</b><br>
            📊 Desert Score: <b>{score}</b>"""
            
        # Add risk/expertise warnings to popup
        if score > 50 and has_emerg == 0:
             popup_html += "<br><br><b style='color:#e74c3c'>⚠️ LIVES AT RISK: No Emergency</b>"
        if row.get('has_surgery', 0) > 10:
             popup_html += "<br><br><b style='color:#3498db'>⭐ EXPERTISE HUB</b>"
             
        popup_html += "</div>"

        # Base circle
        folium.CircleMarker(
            location=[coords['lat'], coords['lon']],
            radius=radius, color=color, fill=True,
            fill_color=color, fill_opacity=opacity,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['address_stateOrRegion']} (Score: {score})"
        ).add_to(m)

        # Add Risk Zone Highlight (pulsating-like large circle)
        if score > 50 and has_emerg == 0:
            folium.Circle(
                location=[coords['lat'], coords['lon']],
                radius=40000, color='#c0392b', weight=2, fill=True,
                fill_color='#e74c3c', fill_opacity=0.1,
                tooltip="High Risk Zone: No Emergency Care"
            ).add_to(m)
            
        # Add Expertise Hub Highlight
        if row.get('has_surgery', 0) > 10:
            folium.Circle(
                location=[coords['lat'], coords['lon']],
                radius=30000, color='#2980b9', weight=2, fill=True,
                fill_color='#3498db', fill_opacity=0.1,
                tooltip="Expertise Hub: High Surgical Capacity"
            ).add_to(m)

    title = """<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
        z-index:1000;background:rgba(0,0,0,0.8);padding:10px 24px;border-radius:8px;
        color:white;font-size:16px;font-weight:bold;">
        🏜️ Ghana Medical Desert & Risk Zone Map</div>"""
    m.get_root().html.add_child(folium.Element(title))

    legend = """<div style="position:fixed;bottom:50px;left:50px;z-index:1000;
        background:rgba(0,0,0,0.85);padding:10px;border-radius:8px;
        border:1px solid #555;font-size:12px;color:white;">
        <b>Desert Score & Layers</b><br>
        <span style="color:#e74c3c">●</span> Critical (&gt;80)<br>
        <span style="color:#e67e22">●</span> High (50-80)<br>
        <span style="color:#f1c40f">●</span> Moderate (20-50)<br>
        <span style="color:#2ecc71">●</span> Adequate (&lt;20)<hr style="margin:4px 0;">
        <span style="color:#c0392b;border:1px solid #c0392b;border-radius:50%;padding:0 5px;">&nbsp;</span> Lives at Risk<br>
        <span style="color:#2980b9;border:1px solid #2980b9;border-radius:50%;padding:0 5px;">&nbsp;</span> Expertise Hub
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m


# ──────────────────────────────────────────────
# CHART BUILDERS
# ──────────────────────────────────────────────
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0e1117')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return img_b64


def chart_facilities_by_region(df):
    valid = df[~df['address_stateOrRegion'].astype(str).str.lower().isin(
        ['null', 'none', 'nan', ''])]
    counts = valid['address_stateOrRegion'].value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors = plt.cm.RdYlGn(counts.values / counts.values.max())
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.3)
    ax.set_xlabel("Facilities", color='white', fontsize=11)
    ax.set_title("Healthcare Facilities by Region", color='white', fontsize=15, fontweight='bold')
    ax.tick_params(colors='white', labelsize=9)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(val), va='center', color='white', fontsize=9)
    return fig


def chart_facility_types(df):
    valid = df[~df['facilityTypeId'].astype(str).str.lower().isin(
        ['null', 'none', 'nan', ''])]
    counts = valid['facilityTypeId'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
           colors=colors[:len(counts)], textprops={'color': 'white', 'fontsize': 10},
           wedgeprops={'edgecolor': '#0e1117', 'linewidth': 2})
    ax.set_title("Facility Type Distribution", color='white', fontsize=15, fontweight='bold')
    return fig


def chart_desert_scores(regional_df):
    sorted_df = regional_df.sort_values('desert_score', ascending=True)
    colors = []
    for s in sorted_df['desert_score'].values:
        if s > 80: colors.append('#e74c3c')
        elif s > 50: colors.append('#e67e22')
        elif s > 20: colors.append('#f1c40f')
        else: colors.append('#2ecc71')

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    bars = ax.barh(sorted_df['address_stateOrRegion'].values,
                   sorted_df['desert_score'].values,
                   color=colors, edgecolor='white', linewidth=0.3)
    ax.set_xlabel("Desert Score (Higher = More Underserved)", color='white', fontsize=11)
    ax.set_title("Medical Desert Score by Region", color='white', fontsize=15, fontweight='bold')
    ax.tick_params(colors='white', labelsize=9)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=50, color='#e67e22', linestyle='--', alpha=0.5)
    ax.axvline(x=80, color='#e74c3c', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, sorted_df['desert_score'].values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(int(val)), va='center', color='white', fontsize=9)
    return fig


def chart_anomaly_summary(df):
    flag_cols = [c for c in df.columns if c.startswith('flag_')]
    flag_counts = {}
    for col in flag_cols:
        label = col.replace('flag_', '').replace('_', ' ').title()
        flag_counts[label] = int(df[col].sum())

    sorted_flags = dict(sorted(flag_counts.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#9b59b6']
    bars = ax.bar(sorted_flags.keys(), sorted_flags.values(),
                  color=colors[:len(sorted_flags)], edgecolor='white', linewidth=0.3)
    ax.set_ylabel("Facilities Flagged", color='white', fontsize=11)
    ax.set_title("Anomaly Detection Summary", color='white', fontsize=15, fontweight='bold')
    ax.tick_params(colors='white', labelsize=9)
    plt.xticks(rotation=25, ha='right')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, sorted_flags.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', color='white', fontsize=10)
    return fig


# ══════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════
def main():
    # Load data
    df = load_and_process_data()
    regional = compute_regional_analysis(df)

    # Build FAISS index
    texts = df['search_text'].fillna('').tolist()
    faiss_index, embed_model = build_faiss_index(texts)

    # Register DuckDB table
    con = duckdb.connect()
    con.register('facilities', df)
    schema_info = """  - name (text) — facility name, e.g. 'Korle Bu Teaching Hospital'
  - pk_unique_id (int) — unique facility ID
  - specialties (text) — JSON-like list of medical specialties, e.g. contains 'cardiology', 'ophthalmology', 'pediatrics', 'gynecologyAndObstetrics', 'internalMedicine', 'generalSurgery', 'dentistry', 'psychiatry', 'emergencyMedicine', 'orthopedicSurgery'
  - procedure (text) — procedures offered, e.g. 'Performs cataract surgeries', 'Provides ultrasound'
  - equipment (text) — equipment available, e.g. 'X-ray', 'Ultrasound', 'CT scanner'
  - capability (text) — facility capabilities, e.g. '24-hour emergency', 'NHIS accredited'
  - facilityTypeId (text) — type: 'hospital', 'clinic', 'dentist', or null
  - operatorTypeId (text) — operator: 'private', 'public', or null
  - description (text) — free text description of the facility
  - numberDoctors (float) — number of doctors (often null)
  - capacity (float) — bed capacity (often null)
  - address_city (text) — city name, e.g. 'Accra', 'Kumasi', 'Tamale'
  - address_stateOrRegion (text) — region, e.g. 'Greater Accra', 'Ashanti', 'Western', 'Northern', 'Volta', 'Central', 'Eastern'
  - address_country (text) — always 'Ghana'
  - num_specialties (int) — count of specialties
  - num_procedure (int) — count of procedures
  - num_equipment (int) — count of equipment items"""

    # LLM
    llm = get_llm()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0;">
            <h2 style="margin:0;">🏥</h2>
            <h3 style="margin:4px 0;color:#3498db;">Healthcare<br>Intelligence Agent</h3>
            <p style="color:#95a5a6;font-size:12px;margin:4px 0;">Virtue Foundation</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        if llm is None:
            st.warning("⚠️ **Databricks Credentials not set**")
            with st.form("db_creds"):
                db_host = st.text_input("Enter Databricks Workspace URL:")
                db_token = st.text_input("Enter Databricks PAT:", type="password")
                submit = st.form_submit_button("Connect")
                if submit and db_host and db_token:
                    os.environ["DATABRICKS_HOST"] = db_host
                    os.environ["DATABRICKS_TOKEN"] = db_token
                    st.rerun()
        else:
            st.success("✅ LLM Connected")

        st.divider()
        st.markdown("#### 📊 Quick Stats")
        st.metric("Total Facilities", len(df))
        valid_regions = df[~df['address_stateOrRegion'].astype(str).str.lower().isin(
            ['null', 'none', 'nan', ''])]
        st.metric("Regions", valid_regions['address_stateOrRegion'].nunique())

        flag_cols = [c for c in df.columns if c.startswith('flag_')]
        anomaly_count = int(df[flag_cols].any(axis=1).sum()) if flag_cols else 0
        st.metric("Anomalies Flagged", anomaly_count)

        st.divider()
        st.markdown("""
        <div style="text-align:center;color:#7f8c8d;font-size:11px;">
            Built with Databricks + Groq<br>
            © 2026 Hackathon Submission
        </div>
        """, unsafe_allow_html=True)

    # ── Title Banner ──
    st.markdown("""
    <div class="title-banner">
        <h1>🏥 Healthcare Intelligence Agent</h1>
        <p>AI-Powered Medical Desert Analysis & Facility Intelligence — Virtue Foundation, Ghana</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", "💬 Ask the Agent", "🏜️ Medical Deserts", "⚠️ Anomalies", "🎯 Action Planner"
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1: OVERVIEW
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab1:
        # KPI Cards
        hospitals = len(df[df['facilityTypeId'] == 'hospital'])
        clinics = len(df[df['facilityTypeId'] == 'clinic'])
        total_docs = int(df['numberDoctors'].fillna(0).sum())
        total_beds = int(df['capacity'].fillna(0).sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        kpis = [
            (c1, "Total Facilities", len(df), "#3498db"),
            (c2, "Hospitals", hospitals, "#e74c3c"),
            (c3, "Clinics", clinics, "#2ecc71"),
            (c4, "Total Doctors", total_docs, "#f39c12"),
            (c5, "Total Beds", total_beds, "#9b59b6"),
        ]
        for col, label, val, color in kpis:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="background:linear-gradient(135deg, {color}, {color}88);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                        {val:,}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # Charts row
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown('<div class="section-header">📊 Facilities by Region</div>',
                        unsafe_allow_html=True)
            fig = chart_facilities_by_region(df)
            st.pyplot(fig, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">📊 Facility Types</div>',
                        unsafe_allow_html=True)
            fig = chart_facility_types(df)
            st.pyplot(fig, use_container_width=True)

        # Map
        st.markdown('<div class="section-header">🗺️ Facility Distribution Map</div>',
                    unsafe_allow_html=True)
        facility_map = build_facility_map(df)
        components.html(facility_map._repr_html_(), height=500)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2: ASK THE AGENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab2:
        st.markdown('<div class="section-header">💬 Ask the Healthcare Intelligence Agent</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        Ask any question about Ghana's healthcare facilities. Examples:
        - *"How many hospitals have cardiology?"*
        - *"Which region has the most hospitals?"*
        - *"What services does Korle Bu offer?"*
        - *"Which facilities claim unrealistic procedures relative to their size?"*
        - *"Where are the medical deserts in Ghana?"*
        """)

        # Chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                st.markdown(f'<div class="chat-user">🧑 <b>You:</b> {msg["content"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-agent">🤖 <b>Agent:</b><br>{msg["content"]}</div>',
                            unsafe_allow_html=True)
                if 'trace' in msg and msg['trace']:
                    with st.expander("🔍 View Agent Trace & Citations"):
                        st.markdown(msg['trace'])
                        if 'citations' in msg and isinstance(msg['citations'], pd.DataFrame) and not msg['citations'].empty:
                            st.markdown("**Cited Data Rows:**")
                            st.dataframe(msg['citations'], use_container_width=True)

        # Input
        query = st.chat_input("Ask a question about healthcare facilities...")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            st.markdown(f'<div class="chat-user">🧑 <b>You:</b> {query}</div>',
                        unsafe_allow_html=True)

            with st.spinner("🔍 Analyzing..."):
                intent = classify_intent(query)
                answer = ""
                trace_text = ""
                citations_df = pd.DataFrame()

                if intent == 'SQL':
                    sql, _ = sql_agent(query, llm, schema_info)
                    if sql and not sql.startswith("Error") and not sql.startswith("⚠️"):
                        try:
                            result = con.execute(sql).fetchdf()
                            trace_text = f"**Step 1:** Intent classified as SQL.\n**Step 2:** Generated DuckDB SQL Query:\n```sql\n{sql}\n```"
                            citations_df = result.head(20)
                            
                            if len(result) > 0:
                                # Summarize with LLM
                                if llm:
                                    summary_prompt = f"""Based on this SQL result for the question "{query}":\n\n{result.head(15).to_string()}\n\nProvide a clear, concise answer. Reference specific numbers and names."""
                                    summary = llm.invoke(summary_prompt)
                                    answer = f"**🔍 Route: SQL Agent**\n\n{summary.content}"
                                else:
                                    answer = f"**SQL Result:** Found {len(result)} rows."
                            else:
                                answer = f"No results found."
                        except Exception as e:
                            answer = f"SQL execution error: {e}"
                            trace_text = f"**Generated SQL:** `{sql}`"
                    else:
                        answer = sql or "Could not generate SQL."

                elif intent == 'SEMANTIC':
                    results = semantic_search(query, df, faiss_index, embed_model)
                    if len(results) > 0:
                        context = results[['name', 'address_stateOrRegion', 'facilityTypeId',
                                           'specialties', 'procedure', 'equipment',
                                           'capability', 'description']].to_string()
                        
                        trace_text = f"**Step 1:** Intent classified as SEMANTIC.\n**Step 2:** Retrieved {len(results)} facilities using FAISS Vector Search.\n**Step 3:** Analyzing context with LLM."
                        citations_df = results[['name', 'address_stateOrRegion', 'facilityTypeId', 'numberDoctors', 'capacity']].head(10)
                        
                        if llm:
                            answer = reasoning_agent(query, context, llm)
                            answer = f"**🔍 Route: Semantic Search + Reasoning**\n\n{answer}"
                        else:
                            answer = f"**Found {len(results)} relevant facilities.**"
                    else:
                        answer = "No matching facilities found."

                elif intent == 'REASONING':
                    # Get relevant context via search
                    results = semantic_search(query, df, faiss_index, embed_model, top_k=10)
                    context = ""
                    if len(results) > 0:
                        context = results[['name', 'address_stateOrRegion', 'facilityTypeId',
                                           'specialties', 'procedure', 'equipment',
                                           'capability', 'numberDoctors', 'capacity']].to_string()
                        citations_df = results[['name', 'address_stateOrRegion', 'facilityTypeId', 'numberDoctors', 'capacity']].head(10)

                    # Add regional data
                    context += f"\n\nREGIONAL DESERT SCORES:\n{regional.to_string()}"
                    trace_text = f"**Step 1:** Intent classified as REASONING.\n**Step 2:** Retrieved {len(results)} facilities via search.\n**Step 3:** Cross-referencing against Regional Desert Scores.\n**Step 4:** Executing validation & mismatch logic."

                    answer = reasoning_agent(query, context, llm)
                    answer = f"**🔍 Route: Medical Reasoning Agent**\n\n{answer}"

                st.session_state.messages.append({
                    "role": "agent", 
                    "content": answer,
                    "trace": trace_text,
                    "citations": citations_df
                })
                
                st.markdown(f'<div class="chat-agent">🤖 <b>Agent:</b><br>{answer}</div>', unsafe_allow_html=True)
                with st.expander("🔍 View Agent Trace & Citations"):
                    st.markdown(trace_text)
                    if not citations_df.empty:
                        st.markdown("**Cited Data Rows:**")
                        st.dataframe(citations_df, use_container_width=True)

        # Sample questions
        st.divider()
        st.markdown("**💡 Try these sample questions:**")
        sample_qs = [
            "How many hospitals have cardiology?",
            "Which region has the most hospitals?",
            "Which facilities claim unrealistic procedures relative to their size?",
            "Where are the medical deserts in Ghana?",
            "What services does Korle Bu Teaching Hospital offer?",
        ]
        cols = st.columns(len(sample_qs))
        for i, q in enumerate(sample_qs):
            with cols[i]:
                if st.button(q, key=f"sample_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3: MEDICAL DESERTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab3:
        st.markdown('<div class="section-header">🏜️ Medical Desert Analysis</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        A **medical desert** is a region where residents cannot access critical healthcare services.
        Our desert score weights: facility count (40%), doctors (25%), beds (20%), surgery (10%), emergency (5%).
        """)

        # Top critical deserts
        critical = regional[regional['desert_score'] > 50]
        if len(critical) > 0:
            st.error(f"🚨 **{len(critical)} regions** have HIGH or CRITICAL desert scores (>50)")
            cols = st.columns(min(4, len(critical)))
            for i, (_, row) in enumerate(critical.head(4).iterrows()):
                with cols[i]:
                    score = row['desert_score']
                    css_class = 'desert-critical' if score > 80 else 'desert-high'
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">{row['address_stateOrRegion']}</div>
                        <div class="kpi-value {css_class}" style="font-size:36px;
                            -webkit-text-fill-color:{'#e74c3c' if score>80 else '#e67e22'};">
                            {int(score)}
                        </div>
                        <div style="color:#95a5a6;font-size:12px;">
                            {int(row['facility_count'])} facilities · {int(row['total_doctors'])} doctors
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Desert map
        st.markdown('<div class="section-header">🗺️ Medical Desert Heatmap</div>',
                    unsafe_allow_html=True)
        desert_map = build_desert_map(regional)
        components.html(desert_map._repr_html_(), height=500)

        # Desert scores chart
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown('<div class="section-header">📊 Desert Scores</div>',
                        unsafe_allow_html=True)
            fig = chart_desert_scores(regional)
            st.pyplot(fig, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-header">📋 Regional Summary</div>',
                        unsafe_allow_html=True)
            display_cols = ['address_stateOrRegion', 'facility_count', 'total_doctors',
                            'total_beds', 'has_surgery', 'has_emergency', 'desert_score']
            st.dataframe(regional[display_cols].rename(columns={
                'address_stateOrRegion': 'Region',
                'facility_count': 'Facilities',
                'total_doctors': 'Doctors',
                'total_beds': 'Beds',
                'has_surgery': 'Surgery',
                'has_emergency': 'Emergency',
                'desert_score': 'Desert Score'
            }), use_container_width=True, height=400)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 4: ANOMALIES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab4:
        st.markdown('<div class="section-header">⚠️ Anomaly Detection</div>',
                    unsafe_allow_html=True)

        st.markdown("""
        Our system flags facilities with suspicious or inconsistent data patterns.
        These anomalies help the Virtue Foundation prioritize verification visits.
        """)

        # Summary chart
        fig = chart_anomaly_summary(df)
        st.pyplot(fig, use_container_width=True)

        # Flagged facilities table
        flag_cols = [c for c in df.columns if c.startswith('flag_')]
        flagged = df[df[flag_cols].any(axis=1)]

        st.markdown(f'<div class="section-header">🔍 Flagged Facilities ({len(flagged)} total)</div>',
                    unsafe_allow_html=True)

        if len(flagged) > 0:
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                selected_flag = st.selectbox(
                    "Filter by anomaly type:",
                    ["All"] + [c.replace('flag_', '').replace('_', ' ').title() for c in flag_cols]
                )
            with col2:
                selected_region = st.selectbox(
                    "Filter by region:",
                    ["All"] + sorted(flagged['address_stateOrRegion'].dropna().unique().tolist())
                )

            filtered = flagged.copy()
            if selected_flag != "All":
                col_name = 'flag_' + selected_flag.lower().replace(' ', '_')
                if col_name in filtered.columns:
                    filtered = filtered[filtered[col_name] == True]
            if selected_region != "All":
                filtered = filtered[filtered['address_stateOrRegion'] == selected_region]

            display = filtered[['name', 'address_stateOrRegion', 'facilityTypeId',
                                'numberDoctors', 'capacity', 'num_procedure',
                                'num_equipment', 'num_specialties'] + flag_cols].copy()

            # Rename for display
            display.columns = [c.replace('flag_', '⚠ ').replace('_', ' ').title()
                                if c.startswith('flag_') else c.replace('_', ' ').title()
                                for c in display.columns]

            st.dataframe(display, use_container_width=True, height=400)
        else:
            st.info("No anomalies detected in the current dataset.")


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 5: ACTION PLANNER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab5:
        st.markdown('<div class="section-header">🎯 Interactive Action Planner</div>', unsafe_allow_html=True)
        st.markdown("""
        Generate a dynamic **30-60-90 day intervention plan** for NGO deployment based on the latest AI analysis of medical deserts and facility anomalies.
        This provides a clear, actionable roadmap for the Virtue Foundation.
        """)
        
        if st.button("🚀 Generate Tactical Action Plan", type="primary"):
            if llm is None:
                st.error("Please connect your LLM (Databricks) in the sidebar to generate a plan.")
            else:
                with st.spinner("Analyzing regional needs and synthesizing deployment plan..."):
                    # Create context for planner
                    critical_deserts = regional[regional['desert_score'] > 50].head(5).to_string()
                    top_anomalies = flagged[['name', 'address_stateOrRegion'] + flag_cols].head(10).to_string()
                    
                    planning_prompt = f"""You are the Chief Medical Logistics Planner for the Virtue Foundation.
Based on the following critical data from Ghana, create a 30-60-90 day action plan for medical NGO deployment.

CRITICAL MEDICAL DESERTS:
{critical_deserts}

TOP FACILITY ANOMALIES (Verification Targets):
{top_anomalies}

Draft a clear, readable Kanban-style action plan consisting of:
1. **Immediate Actions (0-30 Days)**: Verification visits to the most suspicious facilities.
2. **Short-term Deployments (30-60 Days)**: Mobile clinics or targeted supply drops to the worst medical deserts.
3. **Long-term Strategy (60-90 Days)**: Infrastructure and major resource recommendations.

Format using Markdown headers, bullet points, and emojis. Be concise and reference specific facility names and regions."""
                    
                    try:
                        plan_result = llm.invoke(planning_prompt)
                        
                        st.success("Plan Generated Successfully!")
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(52,152,219,0.1), rgba(46,204,113,0.1)); padding: 24px; border-radius: 12px; border: 1px solid rgba(52,152,219,0.3);">
                            {plan_result.content}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating plan: {e}")

if __name__ == "__main__":
    main()
