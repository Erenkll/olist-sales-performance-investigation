# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # KPI and Trend Analysis
# MAGIC
# MAGIC ## Business Question
# MAGIC
# MAGIC > Sales performance appears to be declining in recent months. When did the decline begin, and was it driven by order volume, average order value, or both?
# MAGIC
# MAGIC ## Primary Metrics
# MAGIC
# MAGIC - **Booked GMV:** Total customer payment value
# MAGIC - **Order Count:** Number of valid orders
# MAGIC - **AOV:** Booked GMV divided by order count
# MAGIC - **Items per Order:** Average number of order items
# MAGIC - **GMV per Active Day:** Monthly GMV normalized by observed sales days

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

order_sales = spark.table(
    "workspace.olist.order_sales"
)

# COMMAND ----------

monthly_kpis_base = (
    order_sales
    .groupBy(
        "purchase_month"
    )
    .agg(
        F.sum(
            "booked_gmv"
        ).alias("booked_gmv"),

        F.count(
            "*"
        ).alias("order_count"),

        F.sum(
            "item_count"
        ).alias("item_count"),

        F.countDistinct(
            "purchase_date"
        ).alias("active_days"),

        F.min(
            "purchase_date"
        ).alias("first_purchase_date"),

        F.max(
            "purchase_date"
        ).alias("last_purchase_date")
    )
    .withColumn(
        "aov",
        F.col("booked_gmv") /
        F.col("order_count")
    )
    .withColumn(
        "items_per_order",
        F.col("item_count") /
        F.col("order_count")
    )
    .withColumn(
        "calendar_days",
        F.dayofmonth(
            F.last_day("purchase_month")
        )
    )
    .withColumn(
        "coverage_pct",
        (
            F.col("active_days") /
            F.col("calendar_days")
        ) * 100
    )
    .withColumn(
        "gmv_per_active_day",
        F.col("booked_gmv") /
        F.col("active_days")
    )
)

# COMMAND ----------

month_window = Window.orderBy(
    "purchase_month"
)

monthly_kpis = (
    monthly_kpis_base
    .withColumn(
        "previous_month_gmv",
        F.lag(
            "booked_gmv"
        ).over(month_window)
    )
    .withColumn(
        "previous_month_order_count",
        F.lag(
            "order_count"
        ).over(month_window)
    )
    .withColumn(
        "previous_month_aov",
        F.lag(
            "aov"
        ).over(month_window)
    )
    .withColumn(
        "gmv_mom_pct",
        (
            (
                F.col("booked_gmv") /
                F.col("previous_month_gmv")
            ) - 1
        ) * 100
    )
    .withColumn(
        "order_count_mom_pct",
        (
            (
                F.col("order_count") /
                F.col("previous_month_order_count")
            ) - 1
        ) * 100
    )
    .withColumn(
        "aov_mom_pct",
        (
            (
                F.col("aov") /
                F.col("previous_month_aov")
            ) - 1
        ) * 100
    )
    .orderBy(
        "purchase_month"
    )
)

# COMMAND ----------

display(
    monthly_kpis.select(
        "purchase_month",
        F.round(
            "booked_gmv",
            2
        ).alias("booked_gmv"),
        "order_count",
        F.round(
            "aov",
            2
        ).alias("aov"),
        F.round(
            "items_per_order",
            3
        ).alias("items_per_order"),
        "active_days",
        "calendar_days",
        F.round(
            "coverage_pct",
            2
        ).alias("coverage_pct"),
        F.round(
            "gmv_per_active_day",
            2
        ).alias("gmv_per_active_day"),
        F.round(
            "gmv_mom_pct",
            2
        ).alias("gmv_mom_pct"),
        F.round(
            "order_count_mom_pct",
            2
        ).alias("order_count_mom_pct"),
        F.round(
            "aov_mom_pct",
            2
        ).alias("aov_mom_pct")
    )
)

# COMMAND ----------

year_window = Window.orderBy(
    "purchase_month"
)

monthly_kpis_extended = (
    monthly_kpis
    .withColumn(
        "previous_year_gmv",
        F.lag(
            "booked_gmv",
            12
        ).over(year_window)
    )
    .withColumn(
        "gmv_yoy_pct",
        (
            (
                F.col("booked_gmv") /
                F.col("previous_year_gmv")
            ) - 1
        ) * 100
    )
)

# COMMAND ----------

display(
    monthly_kpis_extended
    .filter(F.col("purchase_month") >= "2018-01-01")
    .select(
        "purchase_month",
        F.round("booked_gmv", 2).alias("booked_gmv"),
        "order_count",
        F.round("aov", 2).alias("aov"),
        F.round("gmv_mom_pct", 2).alias("gmv_mom_pct"),
        F.round("order_count_mom_pct", 2).alias("order_count_mom_pct"),
        F.round("aov_mom_pct", 2).alias("aov_mom_pct"),
        F.round("gmv_yoy_pct", 2).alias("gmv_yoy_pct"),
        F.round("coverage_pct", 2).alias("coverage_pct")
    )
    .orderBy("purchase_month")
)

# COMMAND ----------

recent_daily_kpis = (
    monthly_kpis_extended
    .withColumn(
        "orders_per_active_day",
        F.col("order_count") / F.col("active_days")
    )
    .withColumn(
        "previous_daily_gmv",
        F.lag("gmv_per_active_day").over(month_window)
    )
    .withColumn(
        "daily_gmv_mom_pct",
        (
            F.col("gmv_per_active_day") /
            F.col("previous_daily_gmv") - 1
        ) * 100
    )
)

display(
    recent_daily_kpis
    .filter(F.col("purchase_month") >= "2018-05-01")
    .select(
        "purchase_month",
        "active_days",
        F.round("coverage_pct", 2).alias("coverage_pct"),
        F.round("booked_gmv", 2).alias("booked_gmv"),
        F.round("gmv_per_active_day", 2).alias("gmv_per_active_day"),
        F.round("daily_gmv_mom_pct", 2).alias("daily_gmv_mom_pct"),
        F.round("orders_per_active_day", 2).alias("orders_per_active_day"),
        F.round("aov", 2).alias("aov")
    )
    .orderBy("purchase_month")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preliminary KPI Diagnosis
# MAGIC
# MAGIC - The comparable full-month decline begins in June 2018.
# MAGIC - From May to July, booked GMV declined by approximately 9.24%.
# MAGIC - Order count declined by approximately 8.78%, while AOV declined by only 0.51%.
# MAGIC - Therefore, the May-to-July GMV decline was primarily volume-driven.
# MAGIC - August contains only 29 active days and should not be compared using monthly totals alone.
# MAGIC - August daily GMV increased by 2.48% compared with July.
# MAGIC - Order velocity recovered, but AOV declined from 166.83 BRL to 155.26 BRL.
# MAGIC - The recent performance pattern therefore consists of an initial order-volume contraction followed by an AOV deterioration.

# COMMAND ----------

rolling_window = (
    Window
    .orderBy("purchase_month")
    .rowsBetween(-2, 0)
)

monthly_kpi_trend = (
    recent_daily_kpis
    .withColumn(
        "gmv_3m_avg",
        F.avg("booked_gmv").over(rolling_window)
    )
    .withColumn(
        "order_count_3m_avg",
        F.avg("order_count").over(rolling_window)
    )
    .withColumn(
        "aov_3m_avg",
        F.avg("aov").over(rolling_window)
    )
)

display(
    monthly_kpi_trend
    .filter(F.col("purchase_month") >= "2018-01-01")
    .select(
        "purchase_month",
        F.round("booked_gmv", 2).alias("booked_gmv"),
        F.round("gmv_3m_avg", 2).alias("gmv_3m_avg"),
        "order_count",
        F.round("order_count_3m_avg", 2).alias("order_count_3m_avg"),
        F.round("aov", 2).alias("aov"),
        F.round("aov_3m_avg", 2).alias("aov_3m_avg")
    )
    .orderBy("purchase_month")
)

# COMMAND ----------

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"WARN WindowExpression: No Partition Defined.*",
    category=UserWarning
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trend visualizations
# MAGIC
# MAGIC Monthly GMV is compared with its three-month moving average to distinguish short-term volatility from the underlying sales trend. August 2018 is marked as a partial month because it contains only 29 active days.

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from pathlib import Path

current_dir = Path.cwd()
repo_root = current_dir.parent if current_dir.name == "notebooks" else current_dir

figure_dir = repo_root / "reports" / "figures"
figure_dir.mkdir(parents=True, exist_ok=True)

print(figure_dir)

trend_pd = (
    monthly_kpi_trend
    .orderBy("purchase_month")
    .select("purchase_month", "booked_gmv", "gmv_3m_avg")
    .toPandas()
)

trend_pd["purchase_month"] = pd.to_datetime(trend_pd["purchase_month"])
trend_pd["booked_gmv"] = trend_pd["booked_gmv"].astype(float)
trend_pd["gmv_3m_avg"] = trend_pd["gmv_3m_avg"].astype(float)

fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(
    trend_pd["purchase_month"], trend_pd["booked_gmv"],
    marker="o", linewidth=1.8, label="Monthly booked GMV"
)

ax.plot(
    trend_pd["purchase_month"], trend_pd["gmv_3m_avg"],
    linewidth=3, label="3-month moving average"
)

ax.axvspan(
    pd.Timestamp("2018-08-01"), pd.Timestamp("2018-09-01"),
    color="orange", alpha=0.15, label="August: partial month"
)

ax.set_title("Monthly Booked GMV and 3-Month Moving Average")
ax.set_xlabel("Purchase month")
ax.set_ylabel("Booked GMV (BRL)")
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("R$ {x:,.0f}"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()
fig.savefig(
    figure_dir / "monthly_gmv_trend.png",
    dpi=180,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Order volume and average order value
# MAGIC
# MAGIC The following charts separate the two components of booked GMV: order volume and average order value. Separate panels are used because the metrics have different units and scales.

# COMMAND ----------

drivers_pd = (
    monthly_kpi_trend
    .orderBy("purchase_month")
    .select(
        "purchase_month",
        "order_count", "order_count_3m_avg",
        "aov", "aov_3m_avg"
    )
    .toPandas()
)

drivers_pd["purchase_month"] = pd.to_datetime(drivers_pd["purchase_month"])

numeric_columns = [
    "order_count", "order_count_3m_avg",
    "aov", "aov_3m_avg"
]

drivers_pd[numeric_columns] = drivers_pd[numeric_columns].astype(float)

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

axes[0].plot(
    drivers_pd["purchase_month"], drivers_pd["order_count"],
    marker="o", linewidth=1.8, label="Monthly order count"
)
axes[0].plot(
    drivers_pd["purchase_month"], drivers_pd["order_count_3m_avg"],
    linewidth=3, label="3-month moving average"
)
axes[0].set_title("Monthly Order Volume")
axes[0].set_ylabel("Orders")
axes[0].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))

axes[1].plot(
    drivers_pd["purchase_month"], drivers_pd["aov"],
    marker="o", linewidth=1.8, label="Monthly AOV"
)
axes[1].plot(
    drivers_pd["purchase_month"], drivers_pd["aov_3m_avg"],
    linewidth=3, label="3-month moving average"
)
axes[1].set_title("Average Order Value")
axes[1].set_xlabel("Purchase month")
axes[1].set_ylabel("AOV (BRL)")
axes[1].yaxis.set_major_formatter(mtick.StrMethodFormatter("R$ {x:,.0f}"))

for ax in axes:
    ax.axvspan(
        pd.Timestamp("2018-08-01"), pd.Timestamp("2018-09-01"),
        color="orange", alpha=0.15, label="August: partial month"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

fig.autofmt_xdate()
plt.tight_layout()
plt.show()

fig.savefig(
    figure_dir / "order_volume_and_aov.png",
    dpi=180,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recent daily performance
# MAGIC
# MAGIC Daily-normalized metrics are used for the most recent months so that August 2018 can be compared fairly despite having only 29 active days.

# COMMAND ----------

recent_pd = (
    monthly_kpi_trend
    .filter(F.col("purchase_month") >= "2018-05-01")
    .orderBy("purchase_month")
    .select(
        "purchase_month",
        "gmv_per_active_day",
        "orders_per_active_day",
        "aov"
    )
    .toPandas()
)

recent_pd["purchase_month"] = pd.to_datetime(recent_pd["purchase_month"])
recent_pd["month"] = recent_pd["purchase_month"].dt.strftime("%b %Y")

metrics = [
    ("gmv_per_active_day", "Booked GMV per Active Day", "R$ {:,.0f}"),
    ("orders_per_active_day", "Orders per Active Day", "{:,.0f}"),
    ("aov", "Average Order Value", "R$ {:,.0f}")
]

fig, axes = plt.subplots(3, 1, figsize=(11, 11))

for ax, (column, title, label_format) in zip(axes, metrics):
    values = recent_pd[column].astype(float)
    colors = ["#4C78A8"] * (len(recent_pd) - 1) + ["#F2A541"]

    bars = ax.bar(recent_pd["month"], values, color=colors)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label_format.format(value),
            ha="center", va="bottom"
        )

axes[0].yaxis.set_major_formatter(mtick.StrMethodFormatter("R$ {x:,.0f}"))
axes[1].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
axes[2].yaxis.set_major_formatter(mtick.StrMethodFormatter("R$ {x:,.0f}"))
axes[2].set_xlabel("Purchase month")

plt.tight_layout()
plt.show()

fig.savefig(
    figure_dir / "recent_daily_kpis.png",
    dpi=180,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## KPI trend findings
# MAGIC
# MAGIC - Booked GMV did not decline continuously; it reached a recent peak in May 2018.
# MAGIC - Between May and July, two comparable full months, booked GMV declined by 9.24%.
# MAGIC - During the same period, order count declined by 8.78%, while AOV declined by only 0.51%.
# MAGIC - Therefore, the May-to-July GMV decline was primarily driven by lower order volume.
# MAGIC - August contains only 29 active days, so its monthly total is not directly comparable with complete months.
# MAGIC - On a daily basis, August showed a volume recovery: orders per active day increased from 201 to 221.
# MAGIC - However, AOV declined from R$166.83 to R$155.26, limiting the GMV recovery.
# MAGIC
# MAGIC **Preliminary diagnosis:** The recent sales weakness changed over time. June and July were affected mainly by lower order volume, while August showed recovering order activity but weaker basket value.

# COMMAND ----------

(
    monthly_kpi_trend.write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("workspace.olist.monthly_kpis")
)

# COMMAND ----------

