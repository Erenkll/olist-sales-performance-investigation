# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 01 — Data Audit
# MAGIC
# MAGIC Confirm schemas, observation levels, keys, date coverage, and incomplete periods before cleaning or feature derivation.

# COMMAND ----------

from pyspark.sql import functions as F

RAW_PATH = "/Volumes/workspace/olist/raw"

# COMMAND ----------

customers = spark.read.csv(f"{RAW_PATH}/olist_customers_dataset.csv", header=True, inferSchema=True)
geo_locations = spark.read.csv(f"{RAW_PATH}/olist_geolocation_dataset.csv", header=True, inferSchema=True)
orders = spark.read.csv(f"{RAW_PATH}/olist_orders_dataset.csv", header=True, inferSchema=True)
order_items = spark.read.csv(f"{RAW_PATH}/olist_order_items_dataset.csv", header=True, inferSchema=True)
order_payments = spark.read.csv(f"{RAW_PATH}/olist_order_payments_dataset.csv", header=True, inferSchema=True)
order_reviews = spark.read.csv(f"{RAW_PATH}/olist_order_reviews_dataset.csv", header=True, inferSchema=True)
products = spark.read.csv(f"{RAW_PATH}/olist_products_dataset.csv", header=True, inferSchema=True)
sellers = spark.read.csv(f"{RAW_PATH}/olist_sellers_dataset.csv", header=True, inferSchema=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Schema inspection

# COMMAND ----------

orders.printSchema()
order_items.printSchema()
order_payments.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Grain and key checks

# COMMAND ----------

display(orders.agg(F.count("*").alias("rows"), F.countDistinct("order_id").alias("unique_order_ids")))
display(order_items.agg(F.count("*").alias("rows"), F.countDistinct("order_id", "order_item_id").alias("unique_order_item_keys")))
display(order_payments.agg(F.count("*").alias("rows"), F.countDistinct("order_id", "payment_sequential").alias("unique_payment_keys")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monthly coverage audit
# MAGIC
# MAGIC The last calendar month may be incomplete. It must not automatically be interpreted as a business decline.

# COMMAND ----------

order_time_audit = (
    orders
    .withColumn("purchase_month", F.to_date(F.date_trunc("month", "order_purchase_timestamp")))
    .groupBy("purchase_month")
    .agg(
        F.countDistinct("order_id").alias("order_count"),
        F.sum(F.when(F.col("order_status").isin("canceled", "unavailable"), 1).otherwise(0)).alias("canceled_or_unavailable"),
        F.min("order_purchase_timestamp").alias("first_purchase"),
        F.max("order_purchase_timestamp").alias("last_purchase")
    )
    .orderBy("purchase_month")
)

display(order_time_audit)

# COMMAND ----------

last_months = order_time_audit.orderBy(F.desc("purchase_month")).limit(8)
display(last_months.orderBy("purchase_month"))

# COMMAND ----------

status_audit = (orders.groupBy("order_status").agg(F.count("*").alias("order_count"), F.sum(F.col("order_approved_at").isNull().cast("int")).alias("missing_approved_at")).orderBy(F.desc("order_count")))
display(status_audit)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Payment coverage audit
# MAGIC
# MAGIC Check whether valid orders have payment records and whether payment values are usable for GMV calculation.

# COMMAND ----------

payment_by_order = (
    order_payments
    .groupBy("order_id")
    .agg(
        F.sum("payment_value").alias("order_payment_value"),
        F.count("*").alias("payment_row_count"),
        F.max("payment_sequential").alias("max_payment_sequence")
    )
)

# COMMAND ----------

payment_coverage_audit = (
    orders
    .join(payment_by_order, "order_id", "left")
    .groupBy("order_status")
    .agg(
        F.count("*").alias("order_count"),
        
        F.sum(
            F.col("order_payment_value").isNull().cast("int")
        ).alias("orders_without_payment"),
        
        F.sum(
            (F.col("order_payment_value") <= 0).cast("int")
        ).alias("non_positive_payment")
    )
    .orderBy(F.desc("order_count"))
)

display(payment_coverage_audit)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Order item coverage audit
# MAGIC
# MAGIC Check whether orders have corresponding order-item records required for category and seller analysis.

# COMMAND ----------

orders_with_items = (
    order_items
    .select("order_id")
    .distinct()
    .withColumn("has_items", F.lit(1))
)

# COMMAND ----------

item_coverage_audit = (
    orders
    .join(
        orders_with_items,
        on="order_id",
        how="left"
    )
    .groupBy("order_status")
    .agg(
        F.count("*").alias("order_count"),
        F.sum(
            F.col("has_items").isNull().cast("int")
        ).alias("orders_without_items")
    )
    .orderBy(F.desc("order_count"))
)

display(item_coverage_audit)

# COMMAND ----------

valid_statuses = [
    "delivered",
    "shipped",
    "invoiced",
    "processing",
    "approved"
]

# COMMAND ----------

valid_missing_item_orders = (
    orders
    .join(
        orders_with_items,
        on="order_id",
        how="left"
    )
    .join(
        payment_by_order,
        on="order_id",
        how="left"
    )
    .filter(
        (F.col("order_purchase_timestamp") >= "2017-01-01") &
        (F.col("order_purchase_timestamp") < "2018-09-01")
    )
    .filter(
        F.col("order_status").isin(valid_statuses)
    )
    .filter(
        F.col("has_items").isNull()
    )
    .select(
        "order_id",
        "order_status",
        "order_purchase_timestamp",
        "order_payment_value"
    )
)

display(valid_missing_item_orders)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dimension key checks
# MAGIC
# MAGIC Verify that customer, product, and seller identifiers are unique before relationship joins.

# COMMAND ----------

display(
    customers.agg(
        F.count("*").alias("rows"),
        F.countDistinct("customer_id").alias("unique_customer_ids")
    )
)

display(
    products.agg(
        F.count("*").alias("rows"),
        F.countDistinct("product_id").alias("unique_product_ids")
    )
)

display(
    sellers.agg(
        F.count("*").alias("rows"),
        F.countDistinct("seller_id").alias("unique_seller_ids")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dimension relationship coverage
# MAGIC
# MAGIC Check whether customer, product, and seller identifiers in the analysis population have matching dimension records.

# COMMAND ----------

analysis_orders_audit = (
    orders
    .join(
        payment_by_order,
        on="order_id",
        how="left"
    )
    .filter(
        (F.col("order_purchase_timestamp") >= "2017-01-01") &
        (F.col("order_purchase_timestamp") < "2018-09-01")
    )
    .filter(
        F.col("order_status").isin(valid_statuses)
    )
    .filter(
        F.col("order_payment_value") > 0
    )
)

# COMMAND ----------

customer_lookup = (
    customers
    .select(
        "customer_id",
        "customer_state"
    )
    .withColumn(
        "customer_match",
        F.lit(1)
    )
)

customer_coverage_audit = (
    analysis_orders_audit
    .join(
        customer_lookup,
        on="customer_id",
        how="left"
    )
    .agg(
        F.count("*").alias("analysis_orders"),
        F.sum(
            F.col("customer_match").isNull().cast("int")
        ).alias("orders_without_customer"),
        F.sum(
            F.col("customer_state").isNull().cast("int")
        ).alias("orders_without_customer_state")
    )
)

display(customer_coverage_audit)

# COMMAND ----------

analysis_items_audit = (
    order_items
    .join(
        analysis_orders_audit.select("order_id"),
        on="order_id",
        how="inner"
    )
)

# COMMAND ----------

product_lookup = (
    products
    .select(
        "product_id",
        "product_category_name"
    )
    .withColumn(
        "product_match",
        F.lit(1)
    )
)

seller_lookup = (
    sellers
    .select("seller_id")
    .withColumn(
        "seller_match",
        F.lit(1)
    )
)

# COMMAND ----------

item_dimension_coverage_audit = (
    analysis_items_audit
    .join(
        product_lookup,
        on="product_id",
        how="left"
    )
    .join(
        seller_lookup,
        on="seller_id",
        how="left"
    )
    .agg(
        F.count("*").alias("analysis_item_rows"),
        
        F.sum(
            F.col("product_id").isNull().cast("int")
        ).alias("missing_product_id"),
        
        F.sum(
            F.col("product_match").isNull().cast("int")
        ).alias("items_without_product_match"),
        
        F.sum(
            F.col("product_category_name").isNull().cast("int")
        ).alias("items_without_category"),
        
        F.sum(
            F.col("seller_id").isNull().cast("int")
        ).alias("missing_seller_id"),
        
        F.sum(
            F.col("seller_match").isNull().cast("int")
        ).alias("items_without_seller_match"),
        
        F.sum(
            (
                F.col("price").isNull() |
                (F.col("price") <= 0)
            ).cast("int")
        ).alias("invalid_price"),
        
        F.sum(
            (
                F.col("freight_value").isNull() |
                (F.col("freight_value") < 0)
            ).cast("int")
        ).alias("invalid_freight")
    )
)

display(item_dimension_coverage_audit)

# COMMAND ----------

display(
    item_dimension_coverage_audit.select(
        "invalid_freight"
    )
)

# COMMAND ----------

category_missing_impact = (
    analysis_items_audit
    .join(
        product_lookup,
        on="product_id",
        how="left"
    )
    .agg(
        F.count("*").alias("total_item_count"),
        
        F.sum(
            F.col("product_category_name").isNull().cast("int")
        ).alias("missing_category_item_count"),
        
        F.sum("price").alias("total_merchandise_value"),
        
        F.sum(
            F.when(
                F.col("product_category_name").isNull(),
                F.col("price")
            ).otherwise(0)
        ).alias("missing_category_merchandise_value")
    )
    .withColumn(
        "missing_category_item_pct",
        F.round(
            F.col("missing_category_item_count") /
            F.col("total_item_count") * 100,
            2
        )
    )
    .withColumn(
        "missing_category_value_pct",
        F.round(
            F.col("missing_category_merchandise_value") /
            F.col("total_merchandise_value") * 100,
            2
        )
    )
)

display(category_missing_impact)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Audit decisions
# MAGIC
# MAGIC ### Analysis population
# MAGIC
# MAGIC - Analysis period: `2017-01-01` inclusive to `2018-09-01` exclusive.
# MAGIC - 2016 is excluded because transaction volume is sparse and discontinuous.
# MAGIC - September and October 2018 are excluded because they are incomplete residual periods.
# MAGIC
# MAGIC ### Included order statuses
# MAGIC
# MAGIC - `delivered`
# MAGIC - `shipped`
# MAGIC - `invoiced`
# MAGIC - `processing`
# MAGIC - `approved`
# MAGIC
# MAGIC ### Excluded order statuses
# MAGIC
# MAGIC - `created`
# MAGIC - `canceled`
# MAGIC - `unavailable`
# MAGIC
# MAGIC ### Payment rules
# MAGIC
# MAGIC - Orders must have a positive `order_payment_value`.
# MAGIC - One delivered order without a payment record exists in the raw data, but it falls outside the selected analysis period.
# MAGIC - Missing `order_approved_at` is not used as an exclusion rule when the order has a valid status and positive payment.
# MAGIC
# MAGIC ### Relationship and data-quality findings
# MAGIC
# MAGIC - Final audit population: 97,905 orders and 111,752 order-item rows.
# MAGIC - Order, order-item, and payment composite keys are unique.
# MAGIC - All analysis orders have matching customer and customer-state records.
# MAGIC - All analysis-period orders have corresponding order-item records.
# MAGIC - All product and seller identifiers match their dimension tables.
# MAGIC - No invalid or missing item prices were found.
# MAGIC - 1,587 item rows have a missing product category.
# MAGIC - Missing-category items represent 1.42% of item rows and 1.33% of merchandise value.
# MAGIC - Missing product categories will be retained as the `unknown` segment.

# COMMAND ----------

