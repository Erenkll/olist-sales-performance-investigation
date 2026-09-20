# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 06 — Segment Contribution Analysis
# MAGIC
# MAGIC ## Objective
# MAGIC
# MAGIC Identify which customer states, product categories and seller states contributed most to the May–July 2018 booked GMV decline.
# MAGIC
# MAGIC Different analytical grains are used:
# MAGIC
# MAGIC - **Customer state:** `order_sales`, because each order belongs to one customer state.
# MAGIC - **Product category and seller state:** `item_sales`, using allocated booked GMV to prevent order-level GMV duplication.
# MAGIC
# MAGIC May 2018 is the recent peak and July 2018 is the latest complete comparison month.

# COMMAND ----------

from pyspark.sql import functions as F

order_sales = spark.table("workspace.olist.order_sales")
item_sales = spark.table("workspace.olist.item_sales")

may_date = "2018-05-01"
july_date = "2018-07-01"

# COMMAND ----------

customer_state_contribution = (
    order_sales
    .filter(F.col("purchase_month").isin(may_date, july_date))
    .groupBy("customer_state")
    .agg(
        F.sum(F.when(F.col("purchase_month") == may_date, F.col("booked_gmv"))
              .otherwise(0)).alias("may_gmv"),
        F.sum(F.when(F.col("purchase_month") == july_date, F.col("booked_gmv"))
              .otherwise(0)).alias("july_gmv"),
        F.countDistinct(F.when(F.col("purchase_month") == may_date, F.col("order_id")))
         .alias("may_orders"),
        F.countDistinct(F.when(F.col("purchase_month") == july_date, F.col("order_id")))
         .alias("july_orders")
    )
    .withColumn("gmv_change", F.col("july_gmv") - F.col("may_gmv"))
    .withColumn("order_change", F.col("july_orders") - F.col("may_orders"))
    .withColumn("may_aov", F.col("may_gmv") / F.col("may_orders"))
    .withColumn("july_aov", F.col("july_gmv") / F.col("july_orders"))
    .withColumn("aov_change", F.col("july_aov") - F.col("may_aov"))
)

# COMMAND ----------

total_gmv_change = (
    customer_state_contribution
    .agg(F.sum("gmv_change").alias("total_gmv_change"))
    .first()["total_gmv_change"]
)

customer_state_contribution = (
    customer_state_contribution
    .withColumn(
        "net_change_contribution_pct",
        F.col("gmv_change") / F.lit(total_gmv_change) * 100
    )
)

# COMMAND ----------

display(
    customer_state_contribution
    .select(
        "customer_state",
        F.round("may_gmv", 2).alias("may_gmv"),
        F.round("july_gmv", 2).alias("july_gmv"),
        F.round("gmv_change", 2).alias("gmv_change"),
        "may_orders", "july_orders", "order_change",
        F.round("aov_change", 2).alias("aov_change"),
        F.round("net_change_contribution_pct", 2).alias("net_change_contribution_pct")
    )
    .orderBy("gmv_change")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Product category contribution
# MAGIC
# MAGIC Order-level booked GMV is allocated across order items. Therefore, category contributions can be aggregated without duplicating the GMV of multi-item orders.
# MAGIC
# MAGIC Item count and allocated GMV per item are included to distinguish category-volume changes from changes in item value.

# COMMAND ----------

category_contribution = (
    item_sales
    .filter(F.col("purchase_month").isin(may_date, july_date))
    .groupBy("product_category_name")
    .agg(
        F.sum(F.when(F.col("purchase_month") == may_date, F.col("allocated_booked_gmv"))
              .otherwise(0)).alias("may_gmv"),
        F.sum(F.when(F.col("purchase_month") == july_date, F.col("allocated_booked_gmv"))
              .otherwise(0)).alias("july_gmv"),
        F.count(F.when(F.col("purchase_month") == may_date, 1)).alias("may_items"),
        F.count(F.when(F.col("purchase_month") == july_date, 1)).alias("july_items"),
        F.countDistinct(F.when(F.col("purchase_month") == may_date, F.col("order_id")))
         .alias("may_orders"),
        F.countDistinct(F.when(F.col("purchase_month") == july_date, F.col("order_id")))
         .alias("july_orders")
    )
    .withColumn("gmv_change", F.col("july_gmv") - F.col("may_gmv"))
    .withColumn("item_change", F.col("july_items") - F.col("may_items"))
    .withColumn("order_change", F.col("july_orders") - F.col("may_orders"))
    .withColumn("may_gmv_per_item", F.col("may_gmv") / F.col("may_items"))
    .withColumn("july_gmv_per_item", F.col("july_gmv") / F.col("july_items"))
    .withColumn(
        "gmv_per_item_change",
        F.col("july_gmv_per_item") - F.col("may_gmv_per_item")
    )
)

# COMMAND ----------

total_category_change = (
    category_contribution
    .agg(F.sum("gmv_change").alias("total_gmv_change"))
    .first()["total_gmv_change"]
)

category_contribution = (
    category_contribution
    .withColumn(
        "net_change_contribution_pct",
        F.col("gmv_change") / F.lit(total_category_change) * 100
    )
)

# COMMAND ----------

display(
    category_contribution
    .withColumn(
        "may_gmv_per_item",
        F.try_divide(F.col("may_gmv"), F.col("may_items"))
    )
    .withColumn(
        "july_gmv_per_item",
        F.try_divide(F.col("july_gmv"), F.col("july_items"))
    )
    .withColumn(
        "gmv_per_item_change",
        F.col("july_gmv_per_item") - F.col("may_gmv_per_item")
    )
    .select(
        "product_category_name",
        F.round("may_gmv", 2).alias("may_gmv"),
        F.round("july_gmv", 2).alias("july_gmv"),
        F.round("gmv_change", 2).alias("gmv_change"),
        "may_items", "july_items", "item_change",
        "may_orders", "july_orders", "order_change",
        F.round("gmv_per_item_change", 2).alias("gmv_per_item_change"),
        F.round("net_change_contribution_pct", 2).alias("net_change_contribution_pct")
    )
    .orderBy("gmv_change")
    .limit(15)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seller state contribution
# MAGIC
# MAGIC Allocated booked GMV is aggregated by seller state to identify whether the decline was concentrated in specific supply-side regions.

# COMMAND ----------

seller_state_contribution = (
    item_sales
    .filter(F.col("purchase_month").isin(may_date, july_date))
    .groupBy("seller_state")
    .agg(
        F.sum(F.when(F.col("purchase_month") == may_date, F.col("allocated_booked_gmv"))
              .otherwise(0)).alias("may_gmv"),
        F.sum(F.when(F.col("purchase_month") == july_date, F.col("allocated_booked_gmv"))
              .otherwise(0)).alias("july_gmv"),
        F.count(F.when(F.col("purchase_month") == may_date, 1)).alias("may_items"),
        F.count(F.when(F.col("purchase_month") == july_date, 1)).alias("july_items"),
        F.countDistinct(F.when(F.col("purchase_month") == may_date, F.col("order_id")))
         .alias("may_orders"),
        F.countDistinct(F.when(F.col("purchase_month") == july_date, F.col("order_id")))
         .alias("july_orders")
    )
    .withColumn("gmv_change", F.col("july_gmv") - F.col("may_gmv"))
    .withColumn("item_change", F.col("july_items") - F.col("may_items"))
    .withColumn("order_change", F.col("july_orders") - F.col("may_orders"))
    .withColumn("may_gmv_per_item", F.try_divide(F.col("may_gmv"), F.col("may_items")))
    .withColumn("july_gmv_per_item", F.try_divide(F.col("july_gmv"), F.col("july_items")))
    .withColumn(
        "gmv_per_item_change",
        F.col("july_gmv_per_item") - F.col("may_gmv_per_item")
    )
)

# COMMAND ----------

total_seller_state_change = (
    seller_state_contribution
    .agg(F.sum("gmv_change").alias("total_gmv_change"))
    .first()["total_gmv_change"]
)

seller_state_contribution = (
    seller_state_contribution
    .withColumn(
        "net_change_contribution_pct",
        F.try_divide(
            F.col("gmv_change"),
            F.lit(total_seller_state_change)
        ) * 100
    )
)

# COMMAND ----------

display(
    seller_state_contribution
    .select(
        "seller_state",
        F.round("may_gmv", 2).alias("may_gmv"),
        F.round("july_gmv", 2).alias("july_gmv"),
        F.round("gmv_change", 2).alias("gmv_change"),
        "may_items", "july_items", "item_change",
        "may_orders", "july_orders", "order_change",
        F.round("gmv_per_item_change", 2).alias("gmv_per_item_change"),
        F.round("net_change_contribution_pct", 2).alias("net_change_contribution_pct")
    )
    .orderBy("gmv_change")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top declining segments
# MAGIC
# MAGIC The three segment dimensions represent alternative views of the same GMV change. Their contributions should be interpreted separately and must not be added together.

# COMMAND ----------

customer_plot = (
    customer_state_contribution
    .filter(F.col("gmv_change") < 0)
    .orderBy("gmv_change")
    .limit(8)
    .select("customer_state", "gmv_change")
    .toPandas()
)

category_plot = (
    category_contribution
    .filter(F.col("gmv_change") < 0)
    .orderBy("gmv_change")
    .limit(10)
    .select("product_category_name", "gmv_change")
    .toPandas()
)

seller_plot = (
    seller_state_contribution
    .filter(F.col("gmv_change") < 0)
    .orderBy("gmv_change")
    .limit(8)
    .select("seller_state", "gmv_change")
    .toPandas()
)

for data in [customer_plot, category_plot, seller_plot]:
    data["gmv_change"] = data["gmv_change"].astype(float)

category_plot["product_category_name"] = (
    category_plot["product_category_name"]
    .str.replace("_", " ")
    .str.title()
)

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from pathlib import Path

current_dir = Path.cwd()
repo_root = current_dir.parent if current_dir.name == "notebooks" else current_dir

figure_dir = repo_root / "reports" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

print(figure_dir)

def brl_compact(value, position):
    if abs(value) >= 1_000:
        return f"R$ {value / 1_000:.0f}k"
    return f"R$ {value:.0f}"

def plot_segment_decline(ax, data, segment_column, title):
    bars = ax.barh(
        data[segment_column],
        data["gmv_change"],
        color="#E15759"
    )

    max_value = data["gmv_change"].abs().max()

    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(-max_value * 1.10, max_value * 0.04)
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(brl_compact))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, data["gmv_change"]):
        large_bar = abs(value) >= max_value * 0.18

        if large_bar:
            label_x = value / 2
            label_color = "white"
            alignment = "center"
        else:
            label_x = value - max_value * 0.02
            label_color = "black"
            alignment = "right"

        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"R$ {value:,.0f}",
            ha=alignment,
            va="center",
            color=label_color,
            fontsize=9,
            fontweight="bold"
        )

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6.5),
    gridspec_kw={"width_ratios": [0.9, 1.35, 0.9]},
    constrained_layout=True
)

plot_segment_decline(
    axes[0],
    customer_plot,
    "customer_state",
    "Customer States"
)

plot_segment_decline(
    axes[1],
    category_plot,
    "product_category_name",
    "Product Categories"
)

plot_segment_decline(
    axes[2],
    seller_plot,
    "seller_state",
    "Seller States"
)

fig.suptitle(
    "Top Contributors to the May–July 2018 GMV Decline",
    fontsize=16,
    fontweight="bold"
)

fig.savefig(
    figure_dir / "segment_contributors.png",
    dpi=180,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Segment contribution findings
# MAGIC
# MAGIC - The decline was geographically concentrated in São Paulo. Customer GMV from SP decreased by approximately R$113K between May and July 2018.
# MAGIC - On the seller side, SP-based sellers recorded the largest decline, approximately R$132K.
# MAGIC - The product decline was distributed across several categories, led by watches and gifts, bed and bath products, toys, baby products, and garden tools.
# MAGIC - The category results indicate a broad reduction in order and item volume rather than a decline caused by a single product category.
# MAGIC - Customer state, product category, and seller state are alternative views of the same GMV change. Their contributions overlap and must not be added together.

# COMMAND ----------

ratio_columns = [
    "may_gmv_per_item",
    "july_gmv_per_item",
    "gmv_per_item_change"
]

category_contribution = (
    category_contribution
    .drop(*ratio_columns)
    .withColumn(
        "may_gmv_per_item",
        F.try_divide(F.col("may_gmv"), F.col("may_items"))
    )
    .withColumn(
        "july_gmv_per_item",
        F.try_divide(F.col("july_gmv"), F.col("july_items"))
    )
    .withColumn(
        "gmv_per_item_change",
        F.col("july_gmv_per_item") - F.col("may_gmv_per_item")
    )
)

seller_state_contribution = (
    seller_state_contribution
    .drop(*ratio_columns)
    .withColumn(
        "may_gmv_per_item",
        F.try_divide(F.col("may_gmv"), F.col("may_items"))
    )
    .withColumn(
        "july_gmv_per_item",
        F.try_divide(F.col("july_gmv"), F.col("july_items"))
    )
    .withColumn(
        "gmv_per_item_change",
        F.col("july_gmv_per_item") - F.col("may_gmv_per_item")
    )
)

# COMMAND ----------

customer_state_contribution.write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable("workspace.olist.customer_state_contribution")

category_contribution.write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable("workspace.olist.category_contribution")

seller_state_contribution.write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable("workspace.olist.seller_state_contribution")

# COMMAND ----------

from pathlib import Path
import shutil

current_dir = Path.cwd()

source_root = (
    current_dir.parent
    if current_dir.name == "notebooks"
    else current_dir
)

git_root = (
    source_root.parent
    / "olist-sales-performance-investigation-git"
)

items_to_copy = ["notebooks", "reports", ".gitignore"]

print("Source:", source_root)
print("Target:", git_root)

if not git_root.exists():
    raise FileNotFoundError(f"Git folder not found: {git_root}")

for item_name in items_to_copy:
    source = source_root / item_name
    destination = git_root / item_name

    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            dirs_exist_ok=True
        )
        print(f"Copied folder: {item_name}")
    elif source.is_file():
        shutil.copy2(source, destination)
        print(f"Copied file: {item_name}")
    else:
        print(f"Not found: {source}")

print("\nGit folder contents:")

for item in sorted(git_root.iterdir()):
    print(item.name)

# COMMAND ----------

