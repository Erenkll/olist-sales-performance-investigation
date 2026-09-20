# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 07 — Executive Findings and Recommendations
# MAGIC
# MAGIC This notebook consolidates the time-series, GMV decomposition and segment analyses to answer the main business question:
# MAGIC
# MAGIC > Why did sales revenue decline in the final months of the observation period?
# MAGIC
# MAGIC The analysis separates changes in order volume from changes in average order value and identifies the customer, product and seller segments most associated with the decline.

# COMMAND ----------

from pyspark.sql import functions as F

monthly_kpis = spark.table("workspace.olist.monthly_kpis")
monthly_decomposition = spark.table("workspace.olist.monthly_gmv_decomposition")
customer_state_contribution = spark.table(
    "workspace.olist.customer_state_contribution"
)
category_contribution = spark.table(
    "workspace.olist.category_contribution"
)
seller_state_contribution = spark.table(
    "workspace.olist.seller_state_contribution"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Executive KPI Summary
# MAGIC
# MAGIC May and July 2018 are compared because both are complete months. August is evaluated separately using active-day-normalized metrics because it contains only 29 observed days.

# COMMAND ----------

may_date = F.lit("2018-05-01").cast("date")
july_date = F.lit("2018-07-01").cast("date")

may_july_kpis = monthly_kpis.agg(
    F.max(F.when(F.col("purchase_month") == may_date, F.col("booked_gmv"))).alias("may_gmv"),
    F.max(F.when(F.col("purchase_month") == july_date, F.col("booked_gmv"))).alias("july_gmv"),
    F.max(F.when(F.col("purchase_month") == may_date, F.col("order_count"))).alias("may_orders"),
    F.max(F.when(F.col("purchase_month") == july_date, F.col("order_count"))).alias("july_orders"),
    F.max(F.when(F.col("purchase_month") == may_date, F.col("aov"))).alias("may_aov"),
    F.max(F.when(F.col("purchase_month") == july_date, F.col("aov"))).alias("july_aov")
)

executive_summary = (
    may_july_kpis
    .withColumn("gmv_change", F.col("july_gmv") - F.col("may_gmv"))
    .withColumn("gmv_change_pct", F.col("gmv_change") / F.col("may_gmv") * 100)
    .withColumn("order_change", F.col("july_orders") - F.col("may_orders"))
    .withColumn("order_change_pct", F.col("order_change") / F.col("may_orders") * 100)
    .withColumn("aov_change", F.col("july_aov") - F.col("may_aov"))
    .withColumn("aov_change_pct", F.col("aov_change") / F.col("may_aov") * 100)
    .withColumn(
        "volume_effect",
        F.col("order_change") * ((F.col("may_aov") + F.col("july_aov")) / 2)
    )
    .withColumn(
        "aov_effect",
        F.col("aov_change") * ((F.col("may_orders") + F.col("july_orders")) / 2)
    )
)

display(
    executive_summary.select(
        F.round("may_gmv", 2).alias("may_gmv"),
        F.round("july_gmv", 2).alias("july_gmv"),
        F.round("gmv_change", 2).alias("gmv_change"),
        F.round("gmv_change_pct", 2).alias("gmv_change_pct"),
        "may_orders",
        "july_orders",
        "order_change",
        F.round("order_change_pct", 2).alias("order_change_pct"),
        F.round("may_aov", 2).alias("may_aov"),
        F.round("july_aov", 2).alias("july_aov"),
        F.round("aov_change_pct", 2).alias("aov_change_pct"),
        F.round("volume_effect", 2).alias("volume_effect"),
        F.round("aov_effect", 2).alias("aov_effect")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. August Partial-Month Assessment
# MAGIC
# MAGIC August contains 29 active days and should not be compared with complete months using raw monthly GMV. Daily GMV, orders per active day and AOV are used instead.

# COMMAND ----------

august_date = F.lit("2018-08-01").cast("date")

july_august_kpis = monthly_kpis.agg(
    F.max(F.when(F.col("purchase_month") == july_date, F.col("gmv_per_active_day"))).alias("july_daily_gmv"),
    F.max(F.when(F.col("purchase_month") == august_date, F.col("gmv_per_active_day"))).alias("august_daily_gmv"),
    F.max(F.when(F.col("purchase_month") == july_date, F.col("orders_per_active_day"))).alias("july_orders_per_day"),
    F.max(F.when(F.col("purchase_month") == august_date, F.col("orders_per_active_day"))).alias("august_orders_per_day"),
    F.max(F.when(F.col("purchase_month") == july_date, F.col("aov"))).alias("july_aov"),
    F.max(F.when(F.col("purchase_month") == august_date, F.col("aov"))).alias("august_aov"),
    F.max(F.when(F.col("purchase_month") == august_date, F.col("active_days"))).alias("august_active_days"),
    F.max(F.when(F.col("purchase_month") == august_date, F.col("coverage_pct"))).alias("august_coverage_pct")
)

august_summary = (
    july_august_kpis
    .withColumn(
        "daily_gmv_change_pct",
        (F.col("august_daily_gmv") / F.col("july_daily_gmv") - 1) * 100
    )
    .withColumn(
        "orders_per_day_change_pct",
        (F.col("august_orders_per_day") / F.col("july_orders_per_day") - 1) * 100
    )
    .withColumn(
        "aov_change_pct",
        (F.col("august_aov") / F.col("july_aov") - 1) * 100
    )
)

display(
    august_summary.select(
        "august_active_days",
        F.round("august_coverage_pct", 2).alias("august_coverage_pct"),
        F.round("july_daily_gmv", 2).alias("july_daily_gmv"),
        F.round("august_daily_gmv", 2).alias("august_daily_gmv"),
        F.round("daily_gmv_change_pct", 2).alias("daily_gmv_change_pct"),
        F.round("july_orders_per_day", 2).alias("july_orders_per_day"),
        F.round("august_orders_per_day", 2).alias("august_orders_per_day"),
        F.round("orders_per_day_change_pct", 2).alias("orders_per_day_change_pct"),
        F.round("july_aov", 2).alias("july_aov"),
        F.round("august_aov", 2).alias("august_aov"),
        F.round("aov_change_pct", 2).alias("aov_change_pct")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Primary Segment Contributors
# MAGIC
# MAGIC The table below summarizes the segments with the largest negative GMV changes between May and July 2018.
# MAGIC
# MAGIC Each dimension is an alternative view of the same GMV decline. Customer, product and seller contributions overlap and must not be added together.

# COMMAND ----------

top_customer_states = (
    customer_state_contribution
    .filter(F.col("gmv_change") < 0)
    .orderBy("gmv_change")
    .limit(3)
    .select(
        F.lit(1).alias("dimension_order"),
        F.lit("Customer state").alias("dimension"),
        F.col("customer_state").alias("segment"),
        "gmv_change",
        "order_change",
        "net_change_contribution_pct"
    )
)

top_categories = (
    category_contribution
    .filter(F.col("gmv_change") < 0)
    .orderBy("gmv_change")
    .limit(5)
    .select(
        F.lit(2).alias("dimension_order"),
        F.lit("Product category").alias("dimension"),
        F.col("product_category_name").alias("segment"),
        "gmv_change",
        "order_change",
        "net_change_contribution_pct"
    )
)

top_seller_states = (
    seller_state_contribution
    .filter(F.col("gmv_change") < 0)
    .orderBy("gmv_change")
    .limit(3)
    .select(
        F.lit(3).alias("dimension_order"),
        F.lit("Seller state").alias("dimension"),
        F.col("seller_state").alias("segment"),
        "gmv_change",
        "order_change",
        "net_change_contribution_pct"
    )
)

segment_evidence = (
    top_customer_states
    .unionByName(top_categories)
    .unionByName(top_seller_states)
    .orderBy("dimension_order", "gmv_change")
)

display(
    segment_evidence.select(
        "dimension",
        "segment",
        F.round("gmv_change", 2).alias("gmv_change"),
        "order_change",
        F.round(
            "net_change_contribution_pct", 2
        ).alias("net_change_contribution_pct")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Executive Conclusion
# MAGIC
# MAGIC ### Why did sales revenue decline?
# MAGIC
# MAGIC Booked GMV decreased from **R$1.146M in May to R$1.040M in July 2018**, representing a decline of approximately **R$105.9K (-9.24%)**.
# MAGIC
# MAGIC The primary driver was lower order volume:
# MAGIC
# MAGIC - Order count decreased by **600 orders (-8.78%)**.
# MAGIC - The volume effect accounted for approximately **R$100.2K**, or **94.6%**, of the total GMV decline.
# MAGIC - The AOV effect accounted for approximately **R$5.7K**, or **5.4%**, of the decline.
# MAGIC
# MAGIC The reduction was concentrated in several overlapping segments:
# MAGIC
# MAGIC - **SP customers:** GMV decreased by approximately **R$113.2K**.
# MAGIC - **SP-based sellers:** GMV decreased by approximately **R$132.5K**.
# MAGIC - The largest category declines occurred in **watches and gifts, bed and bath products, toys, baby products and garden tools**.
# MAGIC
# MAGIC These results indicate that the May–July decline was primarily a **broad order-volume contraction concentrated around the São Paulo market**, rather than a decline caused mainly by lower order value or a single product category.
# MAGIC
# MAGIC August shows a different pattern. After adjusting for its partial coverage:
# MAGIC
# MAGIC - Daily order volume increased.
# MAGIC - Daily GMV increased by approximately **2.48%**.
# MAGIC - AOV decreased by approximately **6.94%**.
# MAGIC
# MAGIC Therefore, transaction activity began to recover in August, but customers were placing lower-value orders.
# MAGIC
# MAGIC > This analysis identifies where the decline occurred and how it was distributed. It does not establish that customer location, seller location or product category causally produced the decline. These dimensions overlap and must not be added together.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Recommended Actions
# MAGIC
# MAGIC | Priority | Action | Business objective |
# MAGIC |---|---|---|
# MAGIC | 1 | Investigate the SP customer funnel: traffic, conversion, repeat purchasing and campaign activity | Determine why order volume declined in the largest customer market |
# MAGIC | 2 | Review availability, pricing and promotions for the leading declining categories | Recover lost category-level demand |
# MAGIC | 3 | Examine active sellers, inventory availability, cancellations and delivery performance among SP sellers | Determine whether seller-side constraints contributed to the decline |
# MAGIC | 4 | Test bundles, cross-sell offers and free-shipping thresholds | Recover AOV without reducing the recovered order volume |
# MAGIC | 5 | Monitor daily GMV, orders per active day and AOV separately | Distinguish volume recovery from basket-value deterioration |

# COMMAND ----------

