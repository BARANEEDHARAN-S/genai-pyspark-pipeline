"""Run the e-commerce PySpark analytics and display their results."""

from __future__ import annotations

import logging
from time import perf_counter

from pyspark.sql import DataFrame

from src import config
from src.spark_analytics import SalesAnalytics


LOGGER: logging.Logger = logging.getLogger(__name__)


def load_with_timing(analytics: SalesAnalytics, name: str, path: str) -> DataFrame:
    """Load one Parquet dataset and print its execution time."""
    started_at = perf_counter()
    dataframe = analytics.load_parquet(path)
    elapsed_seconds = perf_counter() - started_at
    print(f"Loaded {name} in {elapsed_seconds:.2f} seconds")
    return dataframe


def run_analytics() -> int:
    """Run all sales analytics, display results, and stop Spark afterward."""
    analytics = SalesAnalytics()
    try:
        started_at = perf_counter()
        analytics.create_spark_session()
        print(f"Created Spark session in {perf_counter() - started_at:.2f} seconds")

        raw_dir = config.RAW_DATA_DIR
        customers = load_with_timing(analytics, "customers", str(raw_dir / "customers.parquet"))
        products = load_with_timing(analytics, "products", str(raw_dir / "products.parquet"))
        orders = load_with_timing(analytics, "orders", str(raw_dir / "orders.parquet"))
        LOGGER.info("Loaded customers with %d columns", len(customers.columns))

        started_at = perf_counter()
        top_customers = analytics.top_customers_by_revenue(orders, products)
        print("\n=== Top 10 Customers by Revenue ===")
        top_customers.show(10, truncate=False)
        print(f"Top customers analysis completed in {perf_counter() - started_at:.2f} seconds")

        started_at = perf_counter()
        category_sales = analytics.sales_by_category(orders, products)
        print("\n=== Sales by Category ===")
        category_sales.show(truncate=False)
        print(f"Sales by category analysis completed in {perf_counter() - started_at:.2f} seconds")

        started_at = perf_counter()
        monthly_sales = analytics.monthly_trends(orders, products)
        print("\n=== Monthly Revenue Trends ===")
        monthly_sales.show(truncate=False)
        print(f"Monthly trends analysis completed in {perf_counter() - started_at:.2f} seconds")
        return 0
    except Exception:
        LOGGER.exception("Analytics execution failed")
        return 1
    finally:
        analytics.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(run_analytics())
