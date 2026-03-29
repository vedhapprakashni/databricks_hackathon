# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2: Data Analysis and Anomaly Flagging
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Profiles the clean data (regional stats, specialty distribution)
# MAGIC 2. Adds anomaly detection flags to each facility
# MAGIC 3. Identifies medical deserts (underserved regions)
# MAGIC 4. Saves enriched data with flags to Unity Catalog Delta tables
# MAGIC
# MAGIC **Run notebooks 00, 01 first!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2A. Load Clean Data

# COMMAND ----------

import json
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, BooleanType, StringType

# ============================================================
# CONFIG: Must match what was set in 00_setup
# ============================================================
CATALOG = "hackathon_vf"
SCHEMA = "healthcare"
TABLE_PREFIX = f"{CATALOG}.{SCHEMA}"

try:
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql(f"USE SCHEMA {SCHEMA}")
except Exception:
    CATALOG = "hive_metastore"
    SCHEMA = "hackathon"
    TABLE_PREFIX = f"{CATALOG}.{SCHEMA}"
    spark.sql(f"USE {SCHEMA}")

df = spark.table(f"{TABLE_PREFIX}.facilities_clean")
total = df.count()
print(f"Loaded {total} clean facilities from {TABLE_PREFIX}.facilities_clean")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2B. Regional Statistics

# COMMAND ----------

print("=== FACILITIES PER REGION ===")
region_stats = df.groupBy("address_stateOrRegion").agg(
    F.count("*").alias("facility_count"),
    F.sum(F.when(F.col("numberDoctors").isNotNull(), F.col("numberDoctors")).otherwise(0)).alias("total_doctors"),
    F.sum(F.when(F.col("capacity").isNotNull(), F.col("capacity")).otherwise(0)).alias("total_beds"),
    F.countDistinct("facilityTypeId").alias("facility_types"),
).orderBy("facility_count", ascending=True)
region_stats.show(20, truncate=False)

# COMMAND ----------

# UDFs for specialty analysis
@F.udf(IntegerType())
def count_json_items(json_str):
    if not json_str or json_str in ("null", "[]"):
        return 0
    try:
        items = json.loads(json_str)
        return len(items) if isinstance(items, list) else 0
    except (json.JSONDecodeError, TypeError):
        return 0

# Specialty coverage by region
df_specs = df.withColumn("specialty_count", count_json_items(F.col("specialties")))
print("=== SPECIALTY COVERAGE BY REGION ===")
df_specs.groupBy("address_stateOrRegion").agg(
    F.sum("specialty_count").alias("total_specialty_slots"),
    F.avg("specialty_count").alias("avg_specialties_per_facility"),
).orderBy("total_specialty_slots", ascending=True).show(20, truncate=False)

# COMMAND ----------

print("=== FACILITY TYPE DISTRIBUTION ===")
df.groupBy("facilityTypeId").agg(
    F.count("*").alias("count"),
    F.avg("numberDoctors").alias("avg_doctors"),
    F.avg("capacity").alias("avg_beds"),
).show(10, truncate=False)

print("=== PUBLIC VS PRIVATE ===")
df.groupBy("operatorTypeId").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2C. Anomaly Detection Flags

# COMMAND ----------

# Add count columns
df_enriched = (
    df
    .withColumn("num_procedures", count_json_items(F.col("procedure")))
    .withColumn("num_equipment", count_json_items(F.col("equipment")))
    .withColumn("num_capabilities", count_json_items(F.col("capability")))
    .withColumn("num_specialties", count_json_items(F.col("specialties")))
)

# FLAG 1: Many procedures but no/few doctors
df_enriched = df_enriched.withColumn("flag_procedures_no_doctors",
    F.when((F.col("num_procedures") > 5) &
           ((F.col("numberDoctors").isNull()) | (F.col("numberDoctors") < 2)), True
    ).otherwise(False))

# FLAG 2: Large capacity but no equipment listed
df_enriched = df_enriched.withColumn("flag_capacity_no_equipment",
    F.when((F.col("capacity").isNotNull()) & (F.col("capacity") > 50) &
           (F.col("num_equipment") == 0), True
    ).otherwise(False))

# FLAG 3: Clinic claims surgery specialty
df_enriched = df_enriched.withColumn("flag_clinic_claims_surgery",
    F.when((F.col("facilityTypeId") == "clinic") &
           (F.lower(F.col("specialties")).contains("surgery")), True
    ).otherwise(False))

# FLAG 4: Too many specialties for a small facility
df_enriched = df_enriched.withColumn("flag_too_many_specialties",
    F.when((F.col("num_specialties") > 5) &
           ((F.col("numberDoctors").isNull()) | (F.col("numberDoctors") < 5)), True
    ).otherwise(False))

# FLAG 5: Sparse record (no data at all)
df_enriched = df_enriched.withColumn("flag_sparse_record",
    F.when((F.col("num_procedures") == 0) & (F.col("num_equipment") == 0) &
           (F.col("num_capabilities") == 0) & (F.col("description").isNull()), True
    ).otherwise(False))

# COMMAND ----------

print("=== ANOMALY FLAG SUMMARY ===")
for flag_col in [c for c in df_enriched.columns if c.startswith("flag_")]:
    count = df_enriched.filter(F.col(flag_col) == True).count()
    print(f"  {flag_col}: {count} facilities flagged")

print("\n=== FACILITIES WITH PROCEDURE/DOCTOR MISMATCH ===")
df_enriched.filter(F.col("flag_procedures_no_doctors") == True).select(
    "name", "address_stateOrRegion", "facilityTypeId",
    "numberDoctors", "num_procedures", "num_equipment"
).show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2D. Medical Desert Identification

# COMMAND ----------

desert_analysis = df_enriched.groupBy("address_stateOrRegion").agg(
    F.count("*").alias("facility_count"),
    F.sum(F.when(F.col("numberDoctors").isNotNull(), F.col("numberDoctors")).otherwise(0)).alias("total_doctors"),
    F.sum(F.when(F.col("capacity").isNotNull(), F.col("capacity")).otherwise(0)).alias("total_beds"),
    F.sum(F.when(F.col("facilityTypeId") == "hospital", 1).otherwise(0)).alias("hospital_count"),
    F.sum(F.when(F.lower(F.col("specialties")).contains("surgery"), 1).otherwise(0)).alias("has_surgery"),
    F.sum(F.when(F.lower(F.col("specialties")).contains("emergency"), 1).otherwise(0)).alias("has_emergency"),
    F.sum(F.when(F.lower(F.col("specialties")).contains("obstetrics"), 1).otherwise(0)).alias("has_obstetrics"),
    F.sum(F.when(F.lower(F.col("specialties")).contains("pediatrics"), 1).otherwise(0)).alias("has_pediatrics"),
    F.sum(F.when(F.col("flag_sparse_record") == True, 1).otherwise(0)).alias("sparse_records"),
)

desert_analysis = desert_analysis.withColumn("desert_score",
    F.lit(100)
    - (F.col("facility_count") * 2)
    - (F.col("total_doctors") * 1)
    - (F.col("has_surgery") * 10)
    - (F.col("has_emergency") * 10)
    - (F.col("has_obstetrics") * 5)
)

print("=== MEDICAL DESERT ANALYSIS (higher score = more underserved) ===")
desert_analysis.orderBy("desert_score", ascending=False).show(20, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2E. Save Enriched Data to Unity Catalog

# COMMAND ----------

ENRICHED_TABLE = f"{TABLE_PREFIX}.facilities_enriched"
DESERT_TABLE = f"{TABLE_PREFIX}.regional_analysis"

df_enriched.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(ENRICHED_TABLE)
spark.sql(f"ALTER TABLE {ENRICHED_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print(f"Saved {df_enriched.count()} enriched facilities to {ENRICHED_TABLE}")

desert_analysis.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(DESERT_TABLE)
print(f"Saved regional analysis to {DESERT_TABLE}")

# COMMAND ----------

print("=" * 60)
print("DATA ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nTotal clean facilities: {df_enriched.count()}")
print(f"\nAnomalies found:")
for flag_col in [c for c in df_enriched.columns if c.startswith("flag_")]:
    count = df_enriched.filter(F.col(flag_col) == True).count()
    print(f"  {flag_col}: {count}")
print(f"\nTop medical deserts:")
for row in desert_analysis.orderBy("desert_score", ascending=False).limit(5).collect():
    print(f"  {row['address_stateOrRegion']}: score={row['desert_score']}, facilities={row['facility_count']}, doctors={row['total_doctors']}")
print(f"\nTables: {ENRICHED_TABLE}, {DESERT_TABLE}")
print("Next: Run notebook 03_vector_store")
