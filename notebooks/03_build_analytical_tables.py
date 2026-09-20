# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Build Analytical Tables
# MAGIC
# MAGIC This notebook creates two analytical datasets with different observation levels:
# MAGIC
# MAGIC 1. **order_sales** — one row per order  
# MAGIC    Used for GMV, order count, AOV, monthly trend and customer-state analysis.
# MAGIC
# MAGIC 2. **item_sales** — one row per order item  
# MAGIC    Used for product-category and seller contribution analysis.
# MAGIC
# MAGIC Payment and item tables are not joined at their raw grains because this would create a many-to-many row explosion.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

orders_clean = spark.table(
    "workspace.olist.clean_orders"
)

order_payments_clean = spark.table(
    "workspace.olist.clean_order_payments"
)

order_items_clean = spark.table(
    "workspace.olist.clean_order_items"
)

products_clean = spark.table(
    "workspace.olist.clean_products"
)

customers_clean = spark.table(
    "workspace.olist.clean_customers"
)

sellers_clean = spark.table(
    "workspace.olist.clean_sellers"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Aggregate payments to order level
# MAGIC
# MAGIC The payment table has multiple rows per order because an order may use multiple payment records.
# MAGIC
# MAGIC Before joining payments with orders, payments are aggregated to one row per `order_id`.
# MAGIC
# MAGIC `booked_gmv` represents the total amount paid by the customer, including freight. It is not Olist's net accounting revenue.

# COMMAND ----------

payment_by_order = (
    order_payments_clean
    .groupBy("order_id")
    .agg(
        F.sum(
            F.col("payment_value")
            .cast("decimal(18, 2)")
        ).alias("booked_gmv"),

        F.count("*").alias("payment_record_count"),

        F.countDistinct(
            "payment_type"
        ).alias("payment_type_count"),

        F.max(
            "payment_installments"
        ).alias("max_installments")
    )
)

# COMMAND ----------

display(
    payment_by_order.agg(
        F.count("*").alias("rows"),
        F.countDistinct(
            "order_id"
        ).alias("unique_order_ids"),

        F.sum(
            F.col("booked_gmv").isNull().cast("int")
        ).alias("missing_gmv"),

        F.sum(
            (F.col("booked_gmv") <= 0).cast("int")
        ).alias("non_positive_gmv"),

        F.sum(
            "booked_gmv"
        ).alias("total_booked_gmv")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build the order-level sales table
# MAGIC
# MAGIC Order items are aggregated to one row per order before joining with orders and payments.
# MAGIC
# MAGIC This provides order-level item count, merchandise value, freight value and product/seller diversity without changing the order grain.

# COMMAND ----------

item_by_order = (
    order_items_clean
    .groupBy("order_id")
    .agg(
        F.count("*").alias("item_count"),

        F.countDistinct(
            "product_id"
        ).alias("product_count"),

        F.countDistinct(
            "seller_id"
        ).alias("seller_count"),

        F.sum(
            F.col("price")
            .cast("decimal(18, 2)")
        ).alias("merchandise_value"),

        F.sum(
            F.col("freight_value")
            .cast("decimal(18, 2)")
        ).alias("freight_value")
    )
    .withColumn(
        "item_gross_value",
        F.col("merchandise_value") +
        F.col("freight_value")
    )
)

# COMMAND ----------

display(
    item_by_order.agg(
        F.count("*").alias("rows"),

        F.countDistinct(
            "order_id"
        ).alias("unique_order_ids"),

        F.sum(
            "item_count"
        ).alias("total_item_count"),

        F.sum(
            "merchandise_value"
        ).alias("total_merchandise_value"),

        F.sum(
            "freight_value"
        ).alias("total_freight_value")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Order-Level Sales Table Construction
# MAGIC
# MAGIC The order, payment, item, and customer datasets are joined after each source has been reduced to the order grain.
# MAGIC
# MAGIC The resulting `order_sales` dataset contains exactly one row per order and supports GMV, order count, AOV, time-trend, and customer-state analysis.

# COMMAND ----------

order_sales = (
    orders_clean
    .join(
        payment_by_order,
        on="order_id",
        how="left"
    )
    .join(
        item_by_order,
        on="order_id",
        how="left"
    )
    .join(
        customers_clean.select(
            "customer_id",
            "customer_unique_id",
            "customer_city",
            "customer_state"
        ),
        on="customer_id",
        how="left"
    )
    .withColumn(
        "purchase_date",
        F.to_date(
            "order_purchase_timestamp"
        )
    )
    .withColumn(
        "purchase_month",
        F.to_date(
            F.date_trunc(
                "month",
                F.col("order_purchase_timestamp")
            )
        )
    )
    .select(
        "order_id",
        "customer_id",
        "customer_unique_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "purchase_date",
        "purchase_month",
        "customer_city",
        "customer_state",
        "booked_gmv",
        "payment_record_count",
        "payment_type_count",
        "max_installments",
        "item_count",
        "product_count",
        "seller_count",
        "merchandise_value",
        "freight_value",
        "item_gross_value"
    )
)

# COMMAND ----------

display(
    order_sales.agg(
        F.count("*").alias("rows"),

        F.countDistinct(
            "order_id"
        ).alias("unique_order_ids"),

        F.sum(
            F.col("booked_gmv")
            .isNull()
            .cast("int")
        ).alias("missing_gmv"),

        F.sum(
            F.col("item_count")
            .isNull()
            .cast("int")
        ).alias("missing_items"),

        F.sum(
            F.col("customer_state")
            .isNull()
            .cast("int")
        ).alias("missing_customer_state"),

        F.sum(
            "booked_gmv"
        ).alias("total_booked_gmv"),

        F.sum(
            "item_gross_value"
        ).alias("total_item_gross_value")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Payment-to-Item Value Reconciliation
# MAGIC
# MAGIC `booked_gmv` is compared with merchandise plus freight value at the order level.
# MAGIC
# MAGIC A tolerance of 0.01 BRL is used to distinguish material differences from minor rounding differences.

# COMMAND ----------

order_value_reconciliation = (
    order_sales
    .withColumn(
        "value_difference",
        F.col("booked_gmv") -
        F.col("item_gross_value")
    )
    .withColumn(
        "absolute_value_difference",
        F.abs(
            F.col("value_difference")
        )
    )
)

# COMMAND ----------

value_reconciliation_summary = (
    order_value_reconciliation
    .agg(
        F.count("*").alias("total_orders"),

        F.sum(
            (
                F.col("absolute_value_difference") <= 0.01
            ).cast("int")
        ).alias("orders_within_tolerance"),

        F.sum(
            (
                F.col("absolute_value_difference") > 0.01
            ).cast("int")
        ).alias("orders_with_value_difference"),

        F.sum(
            (
                F.col("value_difference") > 0.01
            ).cast("int")
        ).alias("payment_above_item_total"),

        F.sum(
            (
                F.col("value_difference") < -0.01
            ).cast("int")
        ).alias("payment_below_item_total"),

        F.sum(
            "value_difference"
        ).alias("net_value_difference"),

        F.sum(
            "absolute_value_difference"
        ).alias("total_absolute_difference"),

        F.max(
            "absolute_value_difference"
        ).alias("max_absolute_difference")
    )
)

display(value_reconciliation_summary)

# COMMAND ----------

display(
    order_value_reconciliation
    .filter(
        F.col("absolute_value_difference") > 0.01
    )
    .select(
        "order_id",
        "order_status",
        "booked_gmv",
        "merchandise_value",
        "freight_value",
        "item_gross_value",
        "value_difference",
        "item_count",
        "payment_record_count"
    )
    .orderBy(
        F.desc("absolute_value_difference")
    )
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reconciliation Decision
# MAGIC
# MAGIC - 99.70% of orders are within the 0.01 BRL reconciliation tolerance.
# MAGIC - 296 orders have a payment-to-item value difference above the tolerance.
# MAGIC - The net difference represents approximately 0.018% of total booked GMV.
# MAGIC - No orders are excluded.
# MAGIC - `booked_gmv` remains the authoritative monetary KPI.
# MAGIC - For category and seller analysis, booked GMV is allocated proportionally according to each item's price plus freight value.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Item-Level Sales Table Construction
# MAGIC
# MAGIC The `item_sales` dataset contains one row per order item.
# MAGIC
# MAGIC Order-level booked GMV is allocated across items according to each item's share of the order's total item gross value. This ensures that category and seller contributions reconcile with total booked GMV.

# COMMAND ----------

item_sales = (
    order_items_clean
    .join(
        products_clean.select(
            "product_id",
            "product_category_name"
        ),
        on="product_id",
        how="left"
    )
    .join(
        sellers_clean.select(
            "seller_id",
            "seller_city",
            "seller_state"
        ),
        on="seller_id",
        how="left"
    )
    .join(
        order_sales.select(
            "order_id",
            "order_status",
            "purchase_date",
            "purchase_month",
            "customer_state",
            "booked_gmv",
            F.col("item_gross_value")
            .alias("order_item_gross_value")
        ),
        on="order_id",
        how="left"
    )
    .withColumn(
        "line_gross_value",
        F.col("price") +
        F.col("freight_value")
    )
    .withColumn(
        "allocated_booked_gmv",
        F.when(
            F.col("order_item_gross_value") > 0,
            (
                F.col("booked_gmv").cast("double") *
                F.col("line_gross_value").cast("double")
            ) /
            F.col("order_item_gross_value").cast("double")
        )
    )
    .select(
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "order_status",
        "purchase_date",
        "purchase_month",
        "customer_state",
        "seller_city",
        "seller_state",
        "product_category_name",
        "price",
        "freight_value",
        "line_gross_value",
        "order_item_gross_value",
        "booked_gmv",
        "allocated_booked_gmv"
    )
)

# COMMAND ----------

display(
    item_sales.agg(
        F.count("*").alias("rows"),

        F.countDistinct(
            "order_id",
            "order_item_id"
        ).alias("unique_item_keys"),

        F.countDistinct(
            "order_id"
        ).alias("unique_order_ids"),

        F.sum(
            F.col("product_category_name")
            .isNull()
            .cast("int")
        ).alias("missing_category"),

        F.sum(
            F.col("seller_state")
            .isNull()
            .cast("int")
        ).alias("missing_seller_state"),

        F.sum(
            "line_gross_value"
        ).alias("total_line_gross_value"),

        F.sum(
            "allocated_booked_gmv"
        ).alias("total_allocated_booked_gmv")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Post-Construction Quality Validation
# MAGIC
# MAGIC Allocated booked GMV is aggregated back to the order level and compared with the original order-level booked GMV.
# MAGIC
# MAGIC This validation confirms that the item-level allocation preserves the authoritative monetary KPI without introducing join-related duplication or value loss.

# COMMAND ----------

allocated_gmv_by_order = (
    item_sales
    .groupBy("order_id")
    .agg(
        F.sum(
            "allocated_booked_gmv"
        ).alias("reconstructed_booked_gmv")
    )
)

# COMMAND ----------

allocation_reconciliation = (
    order_sales
    .select(
        "order_id",
        "booked_gmv"
    )
    .join(
        allocated_gmv_by_order,
        on="order_id",
        how="left"
    )
    .withColumn(
        "allocation_difference",
        F.col("reconstructed_booked_gmv") -
        F.col("booked_gmv").cast("double")
    )
    .withColumn(
        "absolute_allocation_difference",
        F.abs(
            F.col("allocation_difference")
        )
    )
)

# COMMAND ----------

display(
    allocation_reconciliation.agg(
        F.count("*").alias("total_orders"),

        F.sum(
            F.col("reconstructed_booked_gmv")
            .isNull()
            .cast("int")
        ).alias("orders_without_allocation"),

        F.sum(
            (
                F.col("absolute_allocation_difference") > 0.01
            ).cast("int")
        ).alias("orders_above_tolerance"),

        F.max(
            "absolute_allocation_difference"
        ).alias("max_absolute_difference"),

        F.sum(
            "allocation_difference"
        ).alias("total_allocation_difference")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Analytical Table Persistence
# MAGIC
# MAGIC The validated analytical datasets are saved as Delta tables for use in the KPI, decomposition, and segment analysis notebooks.

# COMMAND ----------

analytical_tables = {
    "order_sales": order_sales,
    "item_sales": item_sales
}

for table_name, dataframe in analytical_tables.items():
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
        F.col("tableName").isin(
            "order_sales",
            "item_sales"
        )
    )
    .orderBy(
        "tableName"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook Outcome
# MAGIC
# MAGIC Two validated analytical datasets were created:
# MAGIC
# MAGIC ### `order_sales`
# MAGIC
# MAGIC - Observation level: one row per order
# MAGIC - Rows: 97,905
# MAGIC - Primary uses: booked GMV, order count, AOV, monthly trends, and customer-state analysis
# MAGIC
# MAGIC ### `item_sales`
# MAGIC
# MAGIC - Observation level: one row per order item
# MAGIC - Rows: 111,752
# MAGIC - Primary uses: product-category and seller contribution analysis
# MAGIC - Booked GMV is proportionally allocated across order items
# MAGIC
# MAGIC Post-construction validation confirmed:
# MAGIC
# MAGIC - No row loss
# MAGIC - No join-induced duplication
# MAGIC - No missing monetary allocation
# MAGIC - No allocation differences above the 0.01 BRL tolerance