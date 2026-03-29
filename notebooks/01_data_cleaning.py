# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1: Data Cleaning and Deduplication
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads the raw dataset.csv
# MAGIC 2. Deduplicates facilities (same facility appears from multiple source URLs)
# MAGIC 3. Merges data from multiple rows into one consolidated record per facility
# MAGIC 4. Creates a search_text column for embeddings
# MAGIC 5. Saves clean data as a managed Delta table in Unity Catalog
# MAGIC 6. Enables Change Data Feed for Vector Search sync
# MAGIC
# MAGIC **Run notebook 00_setup first!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1A. Load Raw Data

# COMMAND ----------

import json
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType

# ============================================================
# CONFIG: Must match what was set in 00_setup
# ============================================================
CATALOG = "hackathon_vf"
SCHEMA = "healthcare"
DATASET_PATH = "file:/Workspace/Users/vm8810@srmist.edu.in/dataset.csv"
TABLE_PREFIX = f"{CATALOG}.{SCHEMA}"

try:
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql(f"USE SCHEMA {SCHEMA}")
except Exception:
    CATALOG = "hive_metastore"
    SCHEMA = "hackathon"
    TABLE_PREFIX = f"{CATALOG}.{SCHEMA}"
    spark.sql(f"USE {SCHEMA}")

print(f"Using: {TABLE_PREFIX}")
print(f"Dataset: {DATASET_PATH}")

# COMMAND ----------

# Load the CSV
df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "true")
    .option("escape", '"')
    .csv(DATASET_PATH)
)

print(f"Total rows loaded: {df_raw.count()}")
print(f"Total columns: {len(df_raw.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1B. Explore the Raw Data

# COMMAND ----------

print("Columns:")
for i, col in enumerate(df_raw.columns):
    print(f"  {i+1}. {col}")

# COMMAND ----------

total_rows = df_raw.count()
unique_facilities = df_raw.select("pk_unique_id").distinct().count()
org_types = df_raw.groupBy("organization_type").count().collect()

print(f"Total rows: {total_rows}")
print(f"Unique facilities (pk_unique_id): {unique_facilities}")
print(f"Duplicate rows: {total_rows - unique_facilities}")
print(f"\nOrganization types:")
for row in org_types:
    print(f"  {row['organization_type']}: {row['count']}")

# COMMAND ----------

print("Facility types:")
df_raw.groupBy("facilityTypeId").count().orderBy("count", ascending=False).show(10, truncate=False)

print("\nFacilities by region:")
df_raw.groupBy("address_stateOrRegion").count().orderBy("count", ascending=False).show(20, truncate=False)

# COMMAND ----------

# How many rows have actual free-text data?
has_procedure = df_raw.filter(
    (F.col("procedure").isNotNull()) & (F.col("procedure") != "null") & (F.col("procedure") != "[]")
).count()
has_equipment = df_raw.filter(
    (F.col("equipment").isNotNull()) & (F.col("equipment") != "null") & (F.col("equipment") != "[]")
).count()
has_capability = df_raw.filter(
    (F.col("capability").isNotNull()) & (F.col("capability") != "null") & (F.col("capability") != "[]")
).count()

print(f"Rows with procedure data: {has_procedure} / {total_rows} ({100*has_procedure//total_rows}%)")
print(f"Rows with equipment data: {has_equipment} / {total_rows} ({100*has_equipment//total_rows}%)")
print(f"Rows with capability data: {has_capability} / {total_rows} ({100*has_capability//total_rows}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1C. Deduplication Logic
# MAGIC
# MAGIC Group by `pk_unique_id` and merge:
# MAGIC - Single-value fields: first non-null
# MAGIC - Array/text fields: combine all unique values
# MAGIC - Collect all source URLs for citation

# COMMAND ----------

# UDF to merge collected JSON array strings into one deduplicated list
@F.udf(StringType())
def merge_collected_arrays(arr_of_strings):
    """Takes an array of JSON-encoded array strings and merges all items."""
    merged = set()
    if not arr_of_strings:
        return "[]"
    for arr_str in arr_of_strings:
        if arr_str and arr_str not in ("null", "[]", "[\"\"]\n", "[\"\"]", ""):
            try:
                items = json.loads(arr_str)
                if isinstance(items, list):
                    for item in items:
                        cleaned = str(item).strip().strip('"').strip()
                        if cleaned and cleaned != "":
                            merged.add(cleaned)
                else:
                    cleaned = str(items).strip()
                    if cleaned:
                        merged.add(cleaned)
            except (json.JSONDecodeError, TypeError):
                cleaned = str(arr_str).strip()
                if cleaned and cleaned not in ("null", "[]"):
                    merged.add(cleaned)
    if not merged:
        return "[]"
    return json.dumps(sorted(list(merged)))

# COMMAND ----------

# Define field categories
single_fields = [
    "name", "organization_type", "email", "officialWebsite", "officialPhone",
    "yearEstablished", "acceptsVolunteers", "facebookLink", "twitterLink",
    "linkedinLink", "instagramLink", "logo",
    "address_line1", "address_line2", "address_line3",
    "address_city", "address_stateOrRegion", "address_zipOrPostcode",
    "address_country", "address_countryCode",
    "facilityTypeId", "operatorTypeId", "description",
    "area", "numberDoctors", "capacity",
    "missionStatement", "missionStatementLink", "organizationDescription",
    "mongo DB"
]

merge_fields = [
    "specialties", "procedure", "equipment", "capability",
    "phone_numbers", "websites", "affiliationTypeIds", "countries"
]

# Build aggregation expressions
agg_exprs = []

for field in single_fields:
    if field in df_raw.columns:
        # Delta tables do not allow spaces in column names
        safe_alias = field.replace(" ", "_").replace(".", "")
        agg_exprs.append(F.first(F.col(field), ignorenulls=True).alias(safe_alias))

agg_exprs.append(F.collect_set("source_url").cast("string").alias("source_urls"))
agg_exprs.append(F.count("*").alias("source_count"))

for field in merge_fields:
    if field in df_raw.columns:
        agg_exprs.append(F.collect_list(field).alias(f"_{field}_raw"))

# Group and aggregate
df_grouped = df_raw.groupBy("pk_unique_id").agg(*agg_exprs)
print(f"After grouping: {df_grouped.count()} unique facilities")

# COMMAND ----------

# Merge the collected array fields
df_clean = df_grouped
for field in merge_fields:
    raw_col = f"_{field}_raw"
    if raw_col in df_clean.columns:
        df_clean = df_clean.withColumn(field, merge_collected_arrays(F.col(raw_col)))
        df_clean = df_clean.drop(raw_col)

print(f"Clean dataframe: {df_clean.count()} rows, {len(df_clean.columns)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1D. Create Searchable Text Column

# COMMAND ----------

@F.udf(StringType())
def build_search_text(name, description, specialties, procedure, equipment, capability,
                       facility_type, city, region, num_doctors, bed_capacity):
    """Build a comprehensive search text for embedding and vector search."""
    parts = []

    if name:
        parts.append(f"Facility Name: {name}")
    if facility_type:
        parts.append(f"Type: {facility_type}")
    if city:
        parts.append(f"City: {city}")
    if region:
        parts.append(f"Region: {region}")
    if num_doctors:
        parts.append(f"Number of doctors: {num_doctors}")
    if bed_capacity:
        parts.append(f"Bed capacity: {bed_capacity}")
    if description:
        parts.append(f"Description: {description}")

    for label, field in [("Specialties", specialties),
                          ("Procedures", procedure),
                          ("Equipment", equipment),
                          ("Capabilities", capability)]:
        if field and field not in ("null", "[]"):
            try:
                items = json.loads(field)
                if isinstance(items, list) and items:
                    parts.append(f"{label}: {', '.join(str(i) for i in items)}")
            except (json.JSONDecodeError, TypeError):
                if str(field).strip():
                    parts.append(f"{label}: {field}")

    return " | ".join(parts) if parts else ""

df_clean = df_clean.withColumn(
    "search_text",
    build_search_text(
        F.col("name"), F.col("description"), F.col("specialties"),
        F.col("procedure"), F.col("equipment"), F.col("capability"),
        F.col("facilityTypeId"), F.col("address_city"),
        F.col("address_stateOrRegion"),
        F.col("numberDoctors"), F.col("capacity")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1E. Save as Managed Delta Table with Change Data Feed
# MAGIC
# MAGIC Change Data Feed (CDF) is required for Databricks Vector Search to sync automatically.

# COMMAND ----------

# Save as managed Delta table in Unity Catalog
TABLE_NAME = f"{TABLE_PREFIX}.facilities_clean"

(
    df_clean.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

# Enable Change Data Feed for Vector Search sync
spark.sql(f"ALTER TABLE {TABLE_NAME} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"Saved {df_clean.count()} clean facilities to {TABLE_NAME}")
print("Change Data Feed enabled for Vector Search sync")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1F. Verify

# COMMAND ----------

df_verify = spark.table(TABLE_NAME)
dups = df_verify.count() - df_verify.select("pk_unique_id").distinct().count()

print(f"Total clean facilities: {df_verify.count()}")
print(f"Duplicate pk_unique_id: {dups}")
assert dups == 0, "ERROR: Duplicates still present!"

print("\nSample facilities with rich data:")
df_verify.filter(
    (F.col("procedure") != "[]") & (F.col("equipment") != "[]")
).select("name", "address_city", "address_stateOrRegion", "facilityTypeId", "numberDoctors", "capacity"
).show(10, truncate=False)

# COMMAND ----------

print(f"Step 1 complete! Clean data at {TABLE_NAME}")
print("Next: Run notebook 02_data_analysis")
