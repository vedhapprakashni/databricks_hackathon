# Databricks notebook source
# MAGIC %md
# MAGIC # Step 0: Environment Setup (Full Databricks Platform)
# MAGIC
# MAGIC This notebook sets up the full Databricks environment:
# MAGIC - Installs required Python packages
# MAGIC - Configures Databricks Secrets for API keys
# MAGIC - Sets up Unity Catalog (catalog + schema)
# MAGIC - Verifies Databricks API, Vector Search endpoint, and MLflow
# MAGIC
# MAGIC **Run this notebook FIRST before any other notebook.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0A. Install Required Packages

# COMMAND ----------

# MAGIC %pip install langchain langchain-community langchain-community langgraph sentence-transformers pydantic mlflow databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0B. Configure Databricks Target using Databricks Secrets
# MAGIC
# MAGIC **First time only -- run these commands once in a separate cell or terminal:**
# MAGIC ```
# MAGIC databricks secrets create-scope hackathon
# MAGIC databricks secrets put-secret hackathon DATABRICKS_TOKEN --string-value "your-databricks-token-here"
# MAGIC ```
# MAGIC
# MAGIC Or use the Databricks UI: Workspace > Secrets > Create Scope > Add Secret
# MAGIC
# MAGIC **If secrets are not set up yet, set the key directly below (temporary).**

# COMMAND ----------

# databricks secrets create-scope hackathon
# databricks secrets put-secret hackathon DATABRICKS_TOKEN --string-value "YOUR_TOKEN_HERE"

# COMMAND ----------

import os

# Try to get from Databricks Secrets first, fall back to direct setting
try:
    DATABRICKS_TOKEN = dbutils.secrets.get(scope="hackathon", key="DATABRICKS_TOKEN")
    print("Databricks API key loaded from Databricks Secrets")
except Exception:
    # ============================================================
    # FALLBACK: Set your Databricks API key directly here
    # ============================================================
    DATABRICKS_TOKEN = "YOUR_TOKEN_HERE"  # <-- Replace this!
    print("WARNING: Using hardcoded API key. Set up Databricks Secrets for production.")

os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0C. Set Up Unity Catalog

# COMMAND ----------

# Create catalog and schema in Unity Catalog
# NOTE: If you don't have permission to create a catalog, use the default catalog
CATALOG_NAME = "hackathon_vf"
SCHEMA_NAME = "healthcare"

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG_NAME}")
    spark.sql(f"USE CATALOG {CATALOG_NAME}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
    spark.sql(f"USE SCHEMA {SCHEMA_NAME}")
    print(f"Unity Catalog ready: {CATALOG_NAME}.{SCHEMA_NAME}")
except Exception as e:
    # Fallback: use default catalog with hive_metastore
    CATALOG_NAME = "hive_metastore"
    SCHEMA_NAME = "hackathon"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {SCHEMA_NAME}")
    spark.sql(f"USE {SCHEMA_NAME}")
    print(f"Fallback: using {CATALOG_NAME}.{SCHEMA_NAME}")
    print(f"  (Unity Catalog not available: {e})")

TABLE_PREFIX = f"{CATALOG_NAME}.{SCHEMA_NAME}"
print(f"Table prefix: {TABLE_PREFIX}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0D. Upload Dataset
# MAGIC
# MAGIC Upload `dataset.csv` to Databricks:
# MAGIC 1. Click "Data" in the left sidebar
# MAGIC 2. Click your catalog > schema > "Add Data" or use "Upload to Volume"
# MAGIC 3. Or upload to DBFS via "Add Data" > "Upload File"

# COMMAND ----------

# Check for dataset in common locations
DATASET_PATH = None
possible_paths = [
    "/FileStore/tables/dataset.csv",
    "/FileStore/dataset.csv",
    "/FileStore/tables/dataset-1.csv",
    "/Workspace/Users/vm8810@srmist.edu.in/dataset.csv",
    f"/Workspace/Users/vm8810@srmist.edu.in/trial2/dataset.csv",
]

for p in possible_paths:
    try:
        dbutils.fs.ls(p)
        DATASET_PATH = p
        print(f"Dataset found at: {p}")
        break
    except Exception:
        continue

if not DATASET_PATH:
    print("Dataset NOT found. Please upload dataset.csv to one of:")
    for p in possible_paths:
        print(f"  {p}")

# COMMAND ----------

import pandas as pd

# 1. Read using Pandas (Pandas CAN read workspace files on Serverless)
PANDAS_PATH = "/Workspace/Users/vm8810@srmist.edu.in/trial2/dataset.csv"
pdf_raw = pd.read_csv(PANDAS_PATH)

# 2. Convert to Spark DataFrame
df_raw = spark.createDataFrame(pdf_raw)

print(f"Total rows loaded: {df_raw.count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 0E. Verify All Imports

# COMMAND ----------

try:
    import pandas as pd
    print("pandas OK")

    from langchain_community.chat_models import ChatDatabricks
    print("langchain-community OK")

    from langchain_core.prompts import ChatPromptTemplate
    print("langchain OK")

    from sentence_transformers import SentenceTransformer
    print("sentence-transformers OK")

    from databricks.vector_search.client import VectorSearchClient
    print("databricks-vectorsearch OK")

    import mlflow
    print("mlflow OK")

    import json, numpy as np
    print("numpy OK")

    print("\nAll imports successful!")
except ImportError as e:
    print(f"ERROR: Missing package - {e}")
    print("Re-run the pip install cell above.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0F. Test Databricks API Connection

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

llm = ChatDatabricks(
   
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    
    temperature=0
)

response = llm.invoke("Say 'Databricks connection successful!' and nothing else.")
print(response.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0G. Set Up Vector Search Endpoint
# MAGIC
# MAGIC Creates a Vector Search endpoint that will be used in notebook 03.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

VS_ENDPOINT_NAME = "vf_facility_search"

# Check if endpoint exists, create if not
try:
    endpoint = vsc.get_endpoint(VS_ENDPOINT_NAME)
    print(f"Vector Search endpoint '{VS_ENDPOINT_NAME}' already exists. Status: {endpoint.get('endpoint_status', {}).get('state', 'unknown')}")
except Exception:
    try:
        vsc.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")
        print(f"Vector Search endpoint '{VS_ENDPOINT_NAME}' created! It may take a few minutes to become ready.")
    except Exception as e:
        print(f"Could not create Vector Search endpoint: {e}")
        print("You may need to create it manually via the Databricks UI: Compute > Vector Search > Create")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0H. Set Up MLflow Experiment

# COMMAND ----------

import mlflow

experiment_name = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/vf_healthcare_agent"
mlflow.set_experiment(experiment_name)
print(f"MLflow experiment: {experiment_name}")

# COMMAND ----------

print("=" * 60)
print("SETUP COMPLETE!")
print("=" * 60)
print(f"""
Configuration:
  Catalog:         {CATALOG_NAME}
  Schema:          {SCHEMA_NAME}
  Table prefix:    {TABLE_PREFIX}
  Dataset:         {DATASET_PATH}
  VS Endpoint:     {VS_ENDPOINT_NAME}
  LLM:             Databricks (llama-3.1-70b-versatile)
  Databricks Target:    {'Set via Secrets' if 'YOUR' not in DATABRICKS_TOKEN else 'SET MANUALLY (update!)'}

Next: Run notebook 01_data_cleaning
""")