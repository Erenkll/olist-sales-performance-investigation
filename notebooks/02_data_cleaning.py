# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 02 — Data Cleaning
# MAGIC
# MAGIC Apply the business-scope and data-quality decisions established during the data audit.
# MAGIC
# MAGIC The primary KPI population represents **booked GMV**: orders with a valid commercial status and a positive payment value during the complete analysis period.

# COMMAND ----------

from pyspark.sql import functions as F

RAW_PATH = "/Volumes/workspace/olist/raw"

# COMMAND ----------

orders = spark.read.csv(
    f"{RAW_PATH}/olist_orders_dataset.csv",
    header=True,
    inferSchema=True
)

order_items = spark.read.csv(
    f"{RAW_PATH}/olist_order_items_dataset.csv",
    header=True,
    inferSchema=True
)

order_payments = spark.read.csv(
    f"{RAW_PATH}/olist_order_payments_dataset.csv",
    header=True,
    inferSchema=True
)

customers = spark.read.csv(
    f"{RAW_PATH}/olist_customers_dataset.csv",
    header=True,
    inferSchema=True
)

products = spark.read.csv(
    f"{RAW_PATH}/olist_products_dataset.csv",
    header=True,
    inferSchema=True
)

sellers = spark.read.csv(
    f"{RAW_PATH}/olist_sellers_dataset.csv",
    header=True,
    inferSchema=True
)

# COMMAND ----------

ANALYSIS_START = "2017-01-01"
ANALYSIS_END = "2018-09-01"

VALID_STATUSES = [
    "delivered",
    "shipped",
    "invoiced",
    "processing",
    "approved"
]

# COMMAND ----------

orders_clean = (
    orders
    .filter(
        F.col("order_purchase_timestamp") >=
        F.lit(ANALYSIS_START).cast("timestamp")
    )
    .filter(
        F.col("order_purchase_timestamp") <
        F.lit(ANALYSIS_END).cast("timestamp")
    )
    .filter(
        F.col("order_status").isin(VALID_STATUSES)
    )
    .select(
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at"
    )
)

# COMMAND ----------

display(
    orders_clean.agg(
        F.count("*").alias("rows"),
        F.countDistinct("order_id").alias("unique_order_ids"),
        F.min("order_purchase_timestamp").alias("first_purchase"),
        F.max("order_purchase_timestamp").alias("last_purchase")
    )
)

display(
    orders_clean
    .groupBy("order_status")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean payments
# MAGIC
# MAGIC Keep payment records belonging to the scoped order population and retain only positive, non-null payment values.

# COMMAND ----------

order_payments_clean = (
    order_payments
    .join(
        orders_clean.select("order_id"),
        on="order_id",
        how="inner"
    )
    .filter(
        F.col("payment_value").isNotNull()
    )
    .filter(
        F.col("payment_value") > 0
    )
    .select(
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value"
    )
)

# COMMAND ----------

display(
    order_payments_clean.agg(
        F.count("*").alias("payment_rows"),
        
        F.countDistinct(
            "order_id",
            "payment_sequential"
        ).alias("unique_payment_keys"),
        
        F.countDistinct(
            "order_id"
        ).alias("orders_with_payment"),
        
        F.sum(
            F.col("payment_value").isNull().cast("int")
        ).alias("missing_payment_value"),
        
        F.sum(
            (F.col("payment_value") <= 0).cast("int")
        ).alias("non_positive_payment")
    )
)

# COMMAND ----------

payment_sequence_issues = (
    order_payments_clean
    .groupBy("order_id")
    .agg(
        F.count("*").alias("payment_row_count"),
        F.min("payment_sequential").alias("min_payment_sequence"),
        F.max("payment_sequential").alias("max_payment_sequence")
    )
    .filter(
        (F.col("min_payment_sequence") != 1) |
        (
            F.col("max_payment_sequence") !=
            F.col("payment_row_count")
        )
    )
)

display(payment_sequence_issues)

# COMMAND ----------

display(
    payment_sequence_issues.agg(
        F.count("*").alias("orders_with_sequence_issue")
    )
)

# COMMAND ----------

sequence_issue_examples = (
    order_payments
    .join(
        payment_sequence_issues.select("order_id"),
        on="order_id",
        how="inner"
    )
    .select(
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_installments",
        "payment_value"
    )
    .orderBy(
        "order_id",
        "payment_sequential"
    )
)

display(
    sequence_issue_examples.limit(50)
)

# COMMAND ----------

sequence_issue_payment_totals = (
    order_payments
    .join(
        payment_sequence_issues.select("order_id"),
        on="order_id",
        how="inner"
    )
    .groupBy("order_id")
    .agg(
        F.sum("payment_value").alias("payment_total")
    )
)

# COMMAND ----------

sequence_issue_item_totals = (
    order_items
    .join(
        payment_sequence_issues.select("order_id"),
        on="order_id",
        how="inner"
    )
    .groupBy("order_id")
    .agg(
        F.sum(
            F.col("price") +
            F.col("freight_value")
        ).alias("item_plus_freight_total")
    )
)

# COMMAND ----------

payment_sequence_reconciliation = (
    payment_sequence_issues
    .select("order_id")
    .join(
        sequence_issue_payment_totals,
        on="order_id",
        how="left"
    )
    .join(
        sequence_issue_item_totals,
        on="order_id",
        how="left"
    )
    .withColumn(
        "difference",
        F.round(
            F.col("payment_total") -
            F.col("item_plus_freight_total"),
            2
        )
    )
)

# COMMAND ----------

display(
    payment_sequence_reconciliation.agg(
        F.count("*").alias("sequence_issue_orders"),
        
        F.sum(
            (F.col("difference") != 0).cast("int")
        ).alias("orders_with_value_difference"),
        
        F.max(
            F.abs("difference")
        ).alias("max_absolute_difference")
    )
)

# COMMAND ----------

display(
    payment_sequence_reconciliation
    .filter(
        F.col("difference") != 0
    )
    .orderBy(
        F.desc(F.abs("difference"))
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Payment sequence audit decision
# MAGIC
# MAGIC - 80 orders have non-contiguous payment sequence numbers.
# MAGIC - The anomaly already exists in the raw source data.
# MAGIC - Payment totals reconcile with item plus freight totals.
# MAGIC - Only 2 orders have a 0.01 BRL rounding difference.
# MAGIC - No orders are excluded and payment sequence values are not modified.

# COMMAND ----------

order_items_clean = (
    order_items
    .join(
        orders_clean.select("order_id"),
        on="order_id",
        how="inner"
    )
    .filter(
        F.col("product_id").isNotNull()
    )
    .filter(
        F.col("seller_id").isNotNull()
    )
    .filter(
        F.col("price") >= 0
    )
    .filter(
        F.col("freight_value") >= 0
    )
    .select(
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "shipping_limit_date",
        "price",
        "freight_value"
    )
)

# COMMAND ----------

display(
    order_items_clean.agg(
        F.count("*").alias("item_rows"),
        F.countDistinct(
            "order_id",
            "order_item_id"
        ).alias("unique_item_keys"),
        F.countDistinct(
            "order_id"
        ).alias("orders_with_items"),
        F.sum("price").alias("merchandise_value"),
        F.sum("freight_value").alias("freight_value")
    )
)

# COMMAND ----------

products_clean = (
    products
    .select(
        "product_id",
        "product_category_name"
    )
    .withColumn(
        "product_category_name",
        F.when(
            F.col("product_category_name").isNull() |
            (F.trim(F.col("product_category_name")) == ""),
            F.lit("unknown")
        ).otherwise(
            F.trim(F.col("product_category_name"))
        )
    )
)

# COMMAND ----------

display(
    products_clean.agg(
        F.count("*").alias("product_rows"),
        F.countDistinct("product_id").alias("unique_product_ids"),
        F.sum(
            (F.col("product_category_name") == "unknown")
            .cast("int")
        ).alias("products_with_unknown_category")
    )
)

# COMMAND ----------

order_items_enriched = (
    order_items_clean
    .join(
        products_clean,
        on="product_id",
        how="left"
    )
    .withColumn(
        "product_category_name",
        F.coalesce(
            F.col("product_category_name"),
            F.lit("unknown")
        )
    )
)

# COMMAND ----------

display(
    order_items_enriched.agg(
        F.count("*").alias("item_rows_after_join"),
        F.countDistinct(
            "order_id",
            "order_item_id"
        ).alias("unique_item_keys"),
        F.sum(
            (F.col("product_category_name") == "unknown")
            .cast("int")
        ).alias("unknown_category_items")
    )
)

# COMMAND ----------

customers_clean = (
    customers
    .join(
        orders_clean
        .select("customer_id")
        .distinct(),
        on="customer_id",
        how="inner"
    )
    .select(
        "customer_id",
        "customer_unique_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state"
    )
    .withColumn(
        "customer_city",
        F.lower(F.trim("customer_city"))
    )
    .withColumn(
        "customer_state",
        F.upper(F.trim("customer_state"))
    )
)

# COMMAND ----------

sellers_clean = (
    sellers
    .join(
        order_items_clean
        .select("seller_id")
        .distinct(),
        on="seller_id",
        how="inner"
    )
    .select(
        "seller_id",
        "seller_zip_code_prefix",
        "seller_city",
        "seller_state"
    )
    .withColumn(
        "seller_city",
        F.lower(F.trim("seller_city"))
    )
    .withColumn(
        "seller_state",
        F.upper(F.trim("seller_state"))
    )
)

# COMMAND ----------

display(
    customers_clean.agg(
        F.count("*").alias("customer_rows"),
        F.countDistinct("customer_id").alias("unique_customer_ids"),
        F.sum(
            F.col("customer_state")
            .isNull()
            .cast("int")
        ).alias("missing_customer_state")
    )
)

# COMMAND ----------

display(
    sellers_clean.agg(
        F.count("*").alias("seller_rows"),
        F.countDistinct("seller_id").alias("unique_seller_ids"),
        F.sum(
            F.col("seller_state")
            .isNull()
            .cast("int")
        ).alias("missing_seller_state")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist clean source tables
# MAGIC
# MAGIC The cleaned source-level datasets are saved as Delta tables.
# MAGIC
# MAGIC Currency values are kept at their original precision. Rounding will only be applied when presenting analytical results.

# COMMAND ----------

clean_tables = {
    "clean_orders": orders_clean,
    "clean_order_payments": order_payments_clean,
    "clean_order_items": order_items_clean,
    "clean_products": products_clean,
    "clean_customers": customers_clean,
    "clean_sellers": sellers_clean
}

for table_name, dataframe in clean_tables.items():
    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(
            f"workspace.olist.{table_name}"
        )
    )

# COMMAND ----------

display(
    spark.sql(
        "SHOW TABLES IN workspace.olist"
    )
    .filter(
        F.col("tableName").startswith("clean_")
    )
    .orderBy("tableName")
)