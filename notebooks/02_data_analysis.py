# Databricks notebook source
# MAGIC %md
# MAGIC # Step 2: Data Analysis and Anomaly Flagging
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Profiles the clean data (regional stats, specialty distribution)
# MAGIC 2. Adds anomaly detection flags to each facility
# MAGIC 3. Identifies medical deserts (underserved regions)
# MAGIC 4. Saves enriched data with flags back to Delta tables
# MAGIC
# MAGIC **Run notebook 01_data_cleaning first!**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2A. Load Clean Data

# COMMAND ----------

import json
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, IntegerType, BooleanType

df = spark.table("hackathon.facilities_clean")
total = df.count()
print(f"Loaded {total} clean facilities")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2B. Regional Statistics

# COMMAND ----------

# Facilities per region
print("=== FACILITIES PER REGION ===")
region_stats = df.groupBy("address_stateOrRegion").agg(
    F.count("*").alias("facility_count"),
    F.sum(F.when(F.col("numberDoctors").isNotNull(), F.col("numberDoctors")).otherwise(0)).alias("total_doctors"),
    F.sum(F.when(F.col("capacity").isNotNull(), F.col("capacity")).otherwise(0)).alias("total_beds"),
    F.countDistinct("facilityTypeId").alias("facility_types"),
).orderBy("facility_count", ascending=True)

region_stats.show(20, truncate=False)

# COMMAND ----------

# Specialties per region
@F.udf(IntegerType())
def count_specialties(spec_str):
    """Count number of specialties from JSON array string."""
    if not spec_str or spec_str in ("null", "[]"):
        return 0
    try:
        items = json.loads(spec_str)
        return len(items) if isinstance(items, list) else 0
    except (json.JSONDecodeError, TypeError):
        return 0

@F.udf(StringType())
def extract_specialties_flat(spec_str):
    """Extract specialties as comma-separated string."""
    if not spec_str or spec_str in ("null", "[]"):
        return ""
    try:
        items = json.loads(spec_str)
        return ", ".join(items) if isinstance(items, list) else ""
    except (json.JSONDecodeError, TypeError):
        return ""

# COMMAND ----------

# Which specialties exist in each region?
df_with_specs = df.withColumn("specialty_count", count_specialties(F.col("specialties")))

print("=== SPECIALTY COVERAGE BY REGION ===")
df_with_specs.groupBy("address_stateOrRegion").agg(
    F.sum("specialty_count").alias("total_specialty_slots"),
    F.avg("specialty_count").alias("avg_specialties_per_facility"),
).orderBy("total_specialty_slots", ascending=True).show(20, truncate=False)

# COMMAND ----------

# Facility type distribution
print("=== FACILITY TYPE DISTRIBUTION ===")
df.groupBy("facilityTypeId").agg(
    F.count("*").alias("count"),
    F.avg("numberDoctors").alias("avg_doctors"),
    F.avg("capacity").alias("avg_beds"),
).show(10, truncate=False)

# COMMAND ----------

# Operator type (public vs private)
print("=== PUBLIC VS PRIVATE ===")
df.groupBy("operatorTypeId").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2C. Anomaly Detection Flags
# MAGIC
# MAGIC Add boolean flags for suspicious patterns that the agent can use for
# MAGIC anomaly detection queries.

# COMMAND ----------

@F.udf(IntegerType())
def count_json_items(json_str):
    """Count items in a JSON array string."""
    if not json_str or json_str in ("null", "[]"):
        return 0
    try:
        items = json.loads(json_str)
        return len(items) if isinstance(items, list) else 0
    except (json.JSONDecodeError, TypeError):
        return 0

@F.udf(BooleanType())
def has_keyword(json_str, keyword):
    """Check if a JSON array string contains a keyword (case-insensitive)."""
    if not json_str or json_str in ("null", "[]"):
        return False
    try:
        text = json_str.lower()
        return keyword.lower() in text
    except (TypeError, AttributeError):
        return False

# COMMAND ----------

# Add count columns for analysis
df_enriched = df.withColumn("num_procedures", count_json_items(F.col("procedure"))) \
    .withColumn("num_equipment", count_json_items(F.col("equipment"))) \
    .withColumn("num_capabilities", count_json_items(F.col("capability"))) \
    .withColumn("num_specialties", count_json_items(F.col("specialties")))

# COMMAND ----------

# FLAG 1: Many procedures but no/few doctors
# Suspicious if a facility claims >5 procedures but has 0-1 doctors
df_enriched = df_enriched.withColumn(
    "flag_procedures_no_doctors",
    F.when(
        (F.col("num_procedures") > 5) & 
        ((F.col("numberDoctors").isNull()) | (F.col("numberDoctors") < 2)),
        True
    ).otherwise(False)
)

# FLAG 2: Large capacity but no equipment listed
df_enriched = df_enriched.withColumn(
    "flag_capacity_no_equipment",
    F.when(
        (F.col("capacity").isNotNull()) & 
        (F.col("capacity") > 50) & 
        (F.col("num_equipment") == 0),
        True
    ).otherwise(False)
)

# FLAG 3: Claims surgery specialty but is just a "clinic"
df_enriched = df_enriched.withColumn(
    "flag_clinic_claims_surgery",
    F.when(
        (F.col("facilityTypeId") == "clinic") &
        (F.lower(F.col("specialties")).contains("surgery")),
        True
    ).otherwise(False)
)

# FLAG 4: Has many specialties for a small facility
df_enriched = df_enriched.withColumn(
    "flag_too_many_specialties",
    F.when(
        (F.col("num_specialties") > 5) & 
        ((F.col("numberDoctors").isNull()) | (F.col("numberDoctors") < 5)),
        True
    ).otherwise(False)
)

# FLAG 5: No data at all (sparse record)
df_enriched = df_enriched.withColumn(
    "flag_sparse_record",
    F.when(
        (F.col("num_procedures") == 0) & 
        (F.col("num_equipment") == 0) & 
        (F.col("num_capabilities") == 0) &
        (F.col("description").isNull()),
        True
    ).otherwise(False)
)

# COMMAND ----------

# Count anomalies
print("=== ANOMALY FLAG SUMMARY ===")
for flag_col in [c for c in df_enriched.columns if c.startswith("flag_")]:
    count = df_enriched.filter(F.col(flag_col) == True).count()
    print(f"  {flag_col}: {count} facilities flagged")

# COMMAND ----------

# Show flagged facilities
print("\n=== FACILITIES WITH PROCEDURE/DOCTOR MISMATCH ===")
df_enriched.filter(F.col("flag_procedures_no_doctors") == True).select(
    "name", "address_stateOrRegion", "facilityTypeId", 
    "numberDoctors", "num_procedures", "num_equipment"
).show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2D. Medical Desert Identification

# COMMAND ----------

# Compute desert score per region
# A region is a "medical desert" if it has:
#   - Few facilities relative to other regions
#   - Low doctor count
#   - Missing critical specialties (surgery, emergency, obstetrics)

@F.udf(BooleanType())
def has_specialty_in_region(specialties_list, target):
    """Check if any facility in region has target specialty."""
    if not specialties_list:
        return False
    for s in specialties_list:
        if s and target.lower() in s.lower():
            return True
    return False

desert_analysis = df_enriched.groupBy("address_stateOrRegion").agg(
    F.count("*").alias("facility_count"),
    F.sum(F.when(F.col("numberDoctors").isNotNull(), F.col("numberDoctors")).otherwise(0)).alias("total_doctors"),
    F.sum(F.when(F.col("capacity").isNotNull(), F.col("capacity")).otherwise(0)).alias("total_beds"),
    F.sum(F.when(F.col("facilityTypeId") == "hospital", 1).otherwise(0)).alias("hospital_count"),
    
    # Check for critical specialties
    F.sum(F.when(F.lower(F.col("specialties")).contains("surgery"), 1).otherwise(0)).alias("has_surgery"),
    F.sum(F.when(F.lower(F.col("specialties")).contains("emergency"), 1).otherwise(0)).alias("has_emergency"),
    F.sum(F.when(F.lower(F.col("specialties")).contains("obstetrics"), 1).otherwise(0)).alias("has_obstetrics"),
    F.sum(F.when(F.lower(F.col("specialties")).contains("pediatrics"), 1).otherwise(0)).alias("has_pediatrics"),
    
    # Data quality
    F.sum(F.when(F.col("flag_sparse_record") == True, 1).otherwise(0)).alias("sparse_records"),
)

# Calculate desert score (higher = more underserved)
desert_analysis = desert_analysis.withColumn(
    "desert_score",
    F.lit(100) 
    - (F.col("facility_count") * 2)  # More facilities = lower score
    - (F.col("total_doctors") * 1)   # More doctors = lower score
    - (F.col("has_surgery") * 10)    # Surgery = lower score
    - (F.col("has_emergency") * 10)  # Emergency = lower score
    - (F.col("has_obstetrics") * 5)  # Obstetrics = lower score
)

print("=== MEDICAL DESERT ANALYSIS (higher score = more underserved) ===")
desert_analysis.orderBy("desert_score", ascending=False).show(20, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2E. Save Enriched Data

# COMMAND ----------

# Save enriched facilities with anomaly flags
df_enriched.write.format("delta").mode("overwrite").saveAsTable("hackathon.facilities_enriched")
print(f"Saved {df_enriched.count()} enriched facilities to hackathon.facilities_enriched")

# Save regional desert analysis
desert_analysis.write.format("delta").mode("overwrite").saveAsTable("hackathon.regional_analysis")
print(f"Saved regional analysis to hackathon.regional_analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2F. Summary View

# COMMAND ----------

# Quick summary for the team
print("=" * 60)
print("DATA ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nTotal clean facilities: {df_enriched.count()}")
print(f"\nAnomalies found:")
for flag_col in [c for c in df_enriched.columns if c.startswith("flag_")]:
    count = df_enriched.filter(F.col(flag_col) == True).count()
    print(f"  {flag_col}: {count}")
print(f"\nRegions analyzed: {desert_analysis.count()}")
print(f"Top medical deserts (highest desert_score):")
top_deserts = desert_analysis.orderBy("desert_score", ascending=False).limit(5).collect()
for row in top_deserts:
    print(f"  {row['address_stateOrRegion']}: score={row['desert_score']}, facilities={row['facility_count']}, doctors={row['total_doctors']}")

print(f"\nTables created:")
print(f"  hackathon.facilities_enriched (with anomaly flags)")
print(f"  hackathon.regional_analysis (desert scores)")
print(f"\nNext: Run notebook 03_vector_store.py")
