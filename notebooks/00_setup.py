# Databricks notebook source
# MAGIC %md
# MAGIC # Step 0: Environment Setup
# MAGIC
# MAGIC This notebook installs all required Python packages on the Databricks cluster
# MAGIC and verifies everything is working.
# MAGIC
# MAGIC **Run this notebook FIRST before any other notebook.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0A. Install Required Packages

# COMMAND ----------

# Install all required packages
%pip install langchain langchain-groq langchain-community langgraph faiss-cpu sentence-transformers pydantic mlflow

# COMMAND ----------

# Restart Python after pip install (required in Databricks)
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0B. Set Your Groq API Key
# MAGIC
# MAGIC Get your free API key from https://console.groq.com
# MAGIC
# MAGIC **Option 1 (Quick):** Set it directly below (do NOT commit this to GitHub)
# MAGIC
# MAGIC **Option 2 (Secure):** Use Databricks Secrets (only on paid tiers)

# COMMAND ----------

import os

# ============================================================
# SET YOUR GROQ API KEY HERE
# ============================================================
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"  # <-- Replace this!
# ============================================================

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0C. Upload Dataset
# MAGIC
# MAGIC Before running this, upload `dataset.csv` to Databricks:
# MAGIC 1. Click "Data" in the left sidebar
# MAGIC 2. Click "Create Table" or "Add Data"
# MAGIC 3. Upload `dataset.csv`
# MAGIC 4. It will be stored at `/FileStore/tables/dataset.csv` (or similar path)
# MAGIC
# MAGIC Update the path below if needed.

# COMMAND ----------

# Verify dataset is uploaded
DATASET_PATH = "/FileStore/tables/dataset.csv"

try:
    dbutils.fs.ls(DATASET_PATH)
    print(f"Dataset found at {DATASET_PATH}")
except Exception:
    # Try alternative paths
    alt_paths = [
        "/FileStore/dataset.csv",
        "/FileStore/tables/dataset-1.csv",
    ]
    for p in alt_paths:
        try:
            dbutils.fs.ls(p)
            DATASET_PATH = p
            print(f"Dataset found at {p}")
            break
        except Exception:
            continue
    else:
        print("ERROR: dataset.csv not found! Please upload it to DBFS.")
        print("Go to Data > Add Data > Upload File > Select dataset.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0D. Verify All Imports Work

# COMMAND ----------

# Test all critical imports
try:
    import pandas as pd
    print("pandas OK")

    from langchain_groq import ChatGroq
    print("langchain-groq OK")

    from langchain.prompts import ChatPromptTemplate
    print("langchain OK")

    from sentence_transformers import SentenceTransformer
    print("sentence-transformers OK")

    import faiss
    print("faiss OK")

    import mlflow
    print("mlflow OK")

    import json
    import numpy as np
    print("numpy OK")

    print("\n All imports successful!")
except ImportError as e:
    print(f"ERROR: Missing package - {e}")
    print("Re-run the pip install cell above.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0E. Test Groq API Connection

# COMMAND ----------

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0
)

response = llm.invoke("Say 'Groq connection successful!' and nothing else.")
print(response.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0F. Save Config for Other Notebooks

# COMMAND ----------

# Save configuration so other notebooks can use it
spark.sql("CREATE DATABASE IF NOT EXISTS hackathon")

# Store the dataset path as a Spark config
spark.conf.set("hackathon.dataset_path", DATASET_PATH)

print("Setup complete! You can now run notebook 01_data_cleaning.")
