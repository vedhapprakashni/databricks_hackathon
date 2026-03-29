# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1: Data Cleaning and Deduplication
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Loads the raw dataset.csv
# MAGIC 2. Deduplicates facilities (same facility appears multiple times from different source URLs)
# MAGIC 3. Merges data from multiple rows into one consolidated record per facility
# MAGIC 4. Creates a clean Delta table ready for querying
# MAGIC
# MAGIC **Run notebook 00_setup first!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1A. Load Raw Data

# COMMAND ----------

import json
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, IntegerType

# Load the CSV
DATASET_PATH = "/FileStore/tables/dataset.csv"

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

# Show column names
print("Columns:")
for i, col in enumerate(df_raw.columns):
    print(f"  {i+1}. {col}")

# COMMAND ----------

# Basic stats
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

# Facility type distribution
print("Facility types:")
df_raw.groupBy("facilityTypeId").count().orderBy("count", ascending=False).show(10, truncate=False)

# COMMAND ----------

# Regional distribution
print("Facilities by region:")
df_raw.groupBy("address_stateOrRegion").count().orderBy("count", ascending=False).show(20, truncate=False)

# COMMAND ----------

# How many rows have actual procedure/equipment/capability data?
has_procedure = df_raw.filter(
    (F.col("procedure").isNotNull()) & 
    (F.col("procedure") != "null") & 
    (F.col("procedure") != "[]")
).count()

has_equipment = df_raw.filter(
    (F.col("equipment").isNotNull()) & 
    (F.col("equipment") != "null") & 
    (F.col("equipment") != "[]")
).count()

has_capability = df_raw.filter(
    (F.col("capability").isNotNull()) & 
    (F.col("capability") != "null") & 
    (F.col("capability") != "[]")
).count()

print(f"Rows with procedure data: {has_procedure} / {total_rows} ({100*has_procedure//total_rows}%)")
print(f"Rows with equipment data: {has_equipment} / {total_rows} ({100*has_equipment//total_rows}%)")
print(f"Rows with capability data: {has_capability} / {total_rows} ({100*has_capability//total_rows}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1C. Deduplication Logic
# MAGIC
# MAGIC The same facility (same `pk_unique_id`) can appear multiple times because it was
# MAGIC scraped from different websites. We need to:
# MAGIC 1. Group by `pk_unique_id`
# MAGIC 2. For single-value fields: take the first non-null value
# MAGIC 3. For list/text fields: collect and combine all unique values
# MAGIC 4. Collect all source URLs for citation

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

# Helper: UDF to merge JSON array strings
# e.g. merge '["a","b"]' and '["b","c"]' into '["a","b","c"]'
@F.udf(StringType())
def merge_json_arrays(*arrays):
    """Merge multiple JSON array strings into one deduplicated list."""
    merged = set()
    for arr_str in arrays:
        if arr_str and arr_str not in ("null", "[]", "[\"\"]\n", "[\"\"]"):
            try:
                items = json.loads(arr_str)
                if isinstance(items, list):
                    for item in items:
                        if item and str(item).strip():
                            merged.add(str(item).strip())
            except (json.JSONDecodeError, TypeError):
                # If it's not valid JSON, treat as plain text
                if str(arr_str).strip():
                    merged.add(str(arr_str).strip())
    if not merged:
        return "[]"
    return json.dumps(sorted(list(merged)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deduplication: Group by pk_unique_id and merge

# COMMAND ----------

# For single-value fields, take first non-null
# For array fields, collect all and merge

# Single-value fields (take first non-null)
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

# Array/text fields (merge all values)
merge_fields = [
    "specialties", "procedure", "equipment", "capability",
    "phone_numbers", "websites", "affiliationTypeIds", "countries"
]

# Build aggregation expressions
agg_exprs = []

# Single fields: first non-null
for field in single_fields:
    if field in df_raw.columns:
        agg_exprs.append(F.first(F.col(field), ignorenulls=True).alias(field))

# Collect source URLs
agg_exprs.append(
    F.collect_set("source_url").cast("string").alias("source_urls")
)

# Count how many source pages this facility appeared on
agg_exprs.append(
    F.count("*").alias("source_count")
)

# For merge fields, collect all values
for field in merge_fields:
    if field in df_raw.columns:
        agg_exprs.append(
            F.collect_list(field).alias(f"_{field}_raw")
        )

# Group by pk_unique_id and aggregate
df_grouped = df_raw.groupBy("pk_unique_id").agg(*agg_exprs)

print(f"After dedup: {df_grouped.count()} unique facilities")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge the collected array fields

# COMMAND ----------

# Now merge the collected arrays for each merge field
# We need a UDF that takes an array of JSON-array-strings and merges them
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

# Apply the merge UDF to each collected field
df_clean = df_grouped
for field in merge_fields:
    raw_col = f"_{field}_raw"
    if raw_col in df_clean.columns:
        df_clean = df_clean.withColumn(field, merge_collected_arrays(F.col(raw_col)))
        df_clean = df_clean.drop(raw_col)

print(f"Clean dataframe columns: {len(df_clean.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1D. Create Searchable Text Column
# MAGIC
# MAGIC Combine all free-text fields into one text column for embedding/search.

# COMMAND ----------

@F.udf(StringType())
def build_search_text(name, description, specialties, procedure, equipment, capability, 
                       facility_type, city, region, num_doctors, bed_capacity):
    """Build a comprehensive search text for each facility."""
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
    
    # Parse JSON arrays into readable text
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

# Apply
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
# MAGIC ## 1E. Save Clean Data

# COMMAND ----------

# Create database
spark.sql("CREATE DATABASE IF NOT EXISTS hackathon")

# Save as Delta table
df_clean.write.format("delta").mode("overwrite").saveAsTable("hackathon.facilities_clean")

print(f"Saved {df_clean.count()} clean facility records to hackathon.facilities_clean")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1F. Verify the Clean Data

# COMMAND ----------

# Reload and verify
df_verify = spark.table("hackathon.facilities_clean")

print(f"Total clean facilities: {df_verify.count()}")
print(f"Any duplicate pk_unique_id: {df_verify.count() - df_verify.select('pk_unique_id').distinct().count()}")

# COMMAND ----------

# Show a sample of well-populated facilities
print("Sample facilities with rich data:")
df_verify.filter(
    (F.col("procedure") != "[]") & 
    (F.col("equipment") != "[]")
).select(
    "name", "address_city", "address_stateOrRegion", 
    "facilityTypeId", "numberDoctors", "capacity"
).show(10, truncate=False)

# COMMAND ----------

# Check the search_text column
df_verify.select("name", "search_text").filter(
    F.length("search_text") > 100
).show(5, truncate=100)

# COMMAND ----------

print("Step 1 complete! Clean data is ready in hackathon.facilities_clean")
print(f"Next: Run notebook 02_data_analysis.py")
