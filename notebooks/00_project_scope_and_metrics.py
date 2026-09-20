# Databricks notebook source
# MAGIC %md
# MAGIC # Olist Sales Performance Investigation
# MAGIC
# MAGIC ## Business question
# MAGIC
# MAGIC > **Son aylarda satış gelirimiz düşüyor. Neden?**
# MAGIC
# MAGIC This project applies the common diagnostic analytics core:
# MAGIC
# MAGIC 1. KPI / Metric Analysis
# MAGIC 2. Metric Decomposition / Driver Analysis
# MAGIC 3. Trend Analysis
# MAGIC 4. Segment / Breakdown Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metric contract
# MAGIC
# MAGIC | Metric | Definition |
# MAGIC |---|---|
# MAGIC | GMV | Sum of order-level `payment_value` |
# MAGIC | Order Count | Count of distinct valid `order_id` values |
# MAGIC | AOV | `GMV / Order Count` |
# MAGIC | Time axis | Purchase month from `order_purchase_timestamp` |
# MAGIC | Valid sale | Approved order whose status is not `canceled` or `unavailable` |
# MAGIC
# MAGIC `payment_value` represents marketplace transaction value. Therefore, the primary monetary KPI is called **GMV**, not Olist company revenue.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analytical grains
# MAGIC
# MAGIC - **`order_sales` — one row per order:** GMV, Order Count, AOV, time, and customer-state analysis.
# MAGIC - **`item_sales` — one row per order item:** product-category and seller analysis.
# MAGIC
# MAGIC Payment data will first be aggregated to `order_id` before any join to order items. This prevents duplicated GMV caused by one-to-many joins.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Diagnostic path
# MAGIC
# MAGIC ```text
# MAGIC GMV decreased
# MAGIC ├── Did Order Count decrease?
# MAGIC ├── Did AOV decrease?
# MAGIC ├── When did the change begin?
# MAGIC └── Which categories, states, and sellers contributed most?
# MAGIC ```