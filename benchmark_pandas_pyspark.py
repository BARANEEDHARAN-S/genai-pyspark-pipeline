"""Compare Pandas and PySpark performance for the one-million-row order data."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, TypeVar

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src import config


T = TypeVar("T")


@dataclass(frozen=True)
class Timing:
    """Elapsed time for one named benchmark operation."""

    engine: str
    operation: str
    seconds: float


def timed(engine: str, operation: str, action: Callable[[], T]) -> tuple[T, Timing]:
    """Run an operation and return its value together with elapsed time."""
    started_at = perf_counter()
    value = action()
    return value, Timing(engine, operation, perf_counter() - started_at)


def run_pandas(orders_path: str, products_path: str) -> tuple[pd.DataFrame, list[Timing]]:
    """Run the benchmark operations with Pandas."""
    timings: list[Timing] = []
    orders, timing = timed("Pandas", "load orders", lambda: pd.read_parquet(orders_path))
    timings.append(timing)
    products, timing = timed("Pandas", "load products", lambda: pd.read_parquet(products_path))
    timings.append(timing)

    joined, timing = timed(
        "Pandas",
        "join",
        lambda: orders.merge(products[["product_id", "price"]], on="product_id", how="inner"),
    )
    timings.append(timing)
    with_revenue, timing = timed(
        "Pandas", "calculate revenue", lambda: joined.assign(revenue=joined["quantity"] * joined["price"])
    )
    timings.append(timing)
    grouped, timing = timed(
        "Pandas",
        "group by customer_id",
        lambda: with_revenue.groupby("customer_id", as_index=False)["revenue"].sum(),
    )
    timings.append(timing)
    top_10, timing = timed(
        "Pandas",
        "top 10",
        lambda: grouped.nlargest(10, "revenue").reset_index(drop=True),
    )
    timings.append(timing)
    return top_10.rename(columns={"revenue": "total_revenue"}), timings


def run_spark(spark: SparkSession, orders_path: str, products_path: str) -> tuple[DataFrame, list[Timing]]:
    """Run the benchmark operations with PySpark, materializing each stage."""
    timings: list[Timing] = []

    def materialize(dataframe: DataFrame) -> DataFrame:
        """Cache a Spark stage and force its execution before stopping the timer."""
        cached = dataframe.cache()
        cached.count()
        return cached

    orders, timing = timed("PySpark", "load orders", lambda: materialize(spark.read.parquet(orders_path)))
    timings.append(timing)
    products, timing = timed("PySpark", "load products", lambda: materialize(spark.read.parquet(products_path)))
    timings.append(timing)

    joined, timing = timed(
        "PySpark",
        "join",
        lambda: materialize(orders.join(products.select("product_id", "price"), "product_id", "inner")),
    )
    timings.append(timing)
    with_revenue, timing = timed(
        "PySpark",
        "calculate revenue",
        lambda: materialize(joined.withColumn("revenue", F.col("quantity") * F.col("price"))),
    )
    timings.append(timing)
    grouped, timing = timed(
        "PySpark",
        "group by customer_id",
        lambda: materialize(with_revenue.groupBy("customer_id").agg(F.sum("revenue").alias("revenue"))),
    )
    timings.append(timing)
    top_10, timing = timed(
        "PySpark",
        "top 10",
        lambda: materialize(grouped.orderBy(F.desc("revenue")).limit(10)),
    )
    timings.append(timing)
    return top_10.withColumnRenamed("revenue", "total_revenue"), timings


def print_comparison(timings: list[Timing]) -> None:
    """Print one comparison row per engine and operation."""
    comparison = pd.DataFrame(
        [{"engine": item.engine, "operation": item.operation, "seconds": item.seconds} for item in timings]
    )
    print("\n=== Pandas vs PySpark Performance ===")
    print(comparison.to_string(index=False, formatters={"seconds": "{:.4f}".format}))
    totals = comparison.groupby("engine", as_index=False)["seconds"].sum().rename(columns={"seconds": "total_seconds"})
    print("\n=== Total elapsed time ===")
    print(totals.to_string(index=False, formatters={"total_seconds": "{:.4f}".format}))


def main() -> None:
    """Benchmark both engines using the generated one-million-row Parquet data."""
    orders_path = str(config.RAW_DATA_DIR / "orders.parquet")
    products_path = str(config.RAW_DATA_DIR / "products.parquet")

    pandas_top_10, pandas_timings = run_pandas(orders_path, products_path)
    spark = (
        SparkSession.builder.appName("PandasVsPySparkBenchmark")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    try:
        spark_top_10, spark_timings = run_spark(spark, orders_path, products_path)
        print_comparison(pandas_timings + spark_timings)
        print("\n=== Pandas top 10 ===")
        print(pandas_top_10.to_string(index=False))
        print("\n=== PySpark top 10 ===")
        spark_top_10.show(truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()