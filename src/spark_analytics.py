"""PySpark analytics for the synthetic e-commerce Parquet datasets."""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


LOGGER: logging.Logger = logging.getLogger(__name__)


class SalesAnalytics:
    """Run sales analytics using a locally configured Spark session."""

    def __init__(self, app_name: str = "EcommerceSalesAnalytics") -> None:
        """Initialize analytics with the Spark application name."""
        self.app_name = app_name
        self.spark: SparkSession | None = None

    def create_spark_session(self) -> SparkSession:
        """Create a local Spark session with memory and performance settings."""
        self.spark = (
            SparkSession.builder.appName(self.app_name).master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        LOGGER.info("Created local Spark session with adaptive execution and Kryo serialization")
        return self.spark

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet file or directory into a Spark DataFrame."""
        if self.spark is None:
            raise RuntimeError("Create a Spark session before loading data")
        parquet_path = Path(path)
        LOGGER.info("Loading Parquet data from %s", parquet_path)
        return self.spark.read.parquet(str(parquet_path))

    @staticmethod
    def _sales_facts(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Join orders to products and calculate revenue for each order line."""
        return (
            orders_df.join(products_df.select("product_id", "price"), "product_id", "inner")
            .withColumn("revenue", F.col("quantity") * F.col("price"))
        )

    def top_customers_by_revenue(self, orders_df: DataFrame, products_df: DataFrame, n: int = 10) -> DataFrame:
        """Return the top ``n`` customers ranked by total product revenue."""
        if n < 1:
            raise ValueError("n must be greater than zero")
        return (
            self._sales_facts(orders_df, products_df).groupBy("customer_id")
            .agg(F.round(F.sum("revenue"), 2).alias("total_revenue"), F.countDistinct("order_id").alias("order_count"))
            .orderBy(F.desc("total_revenue")).limit(n)
        )

    def sales_by_category(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return total revenue and units sold grouped by product category."""
        return (
            self._sales_facts(orders_df, products_df)
            .join(products_df.select("product_id", "category"), "product_id", "inner")
            .groupBy("category")
            .agg(F.round(F.sum("revenue"), 2).alias("total_revenue"), F.sum("quantity").alias("units_sold"))
            .orderBy(F.desc("total_revenue"))
        )

    def monthly_trends(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Return monthly revenue and month-over-month growth percentages."""
        monthly_revenue = (
            self._sales_facts(orders_df, products_df)
            .withColumn("month", F.date_format(F.to_date("order_date"), "yyyy-MM"))
            .groupBy("month").agg(F.round(F.sum("revenue"), 2).alias("monthly_revenue"))
        )
        month_window = Window.orderBy("month")
        previous_revenue = F.lag("monthly_revenue").over(month_window)
        return (
            monthly_revenue.withColumn("previous_month_revenue", previous_revenue)
            .withColumn(
                "mom_growth_percent",
                F.when(F.col("previous_month_revenue").isNull() | (F.col("previous_month_revenue") == 0), F.lit(None).cast("double"))
                .otherwise(F.round((F.col("monthly_revenue") - F.col("previous_month_revenue")) / F.col("previous_month_revenue") * 100, 2)),
            )
            .orderBy("month")
        )

    def close(self) -> None:
        """Stop the Spark session if one is active."""
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
            LOGGER.info("Stopped Spark session")


def main() -> None:
    """Run the analytics methods against generated raw Parquet files."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    analytics = SalesAnalytics()
    try:
        analytics.create_spark_session()
        orders = analytics.load_parquet("data/raw/orders.parquet")
        products = analytics.load_parquet("data/raw/products.parquet")
        analytics.top_customers_by_revenue(orders, products).show()
        analytics.sales_by_category(orders, products).show()
        analytics.monthly_trends(orders, products).show()
    finally:
        analytics.close()


if __name__ == "__main__":
    main()