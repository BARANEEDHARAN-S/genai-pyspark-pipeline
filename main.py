"""Generate and persist the synthetic e-commerce datasets."""

from __future__ import annotations

import logging
import sys
from time import perf_counter
from pathlib import Path

from src import config
from src.data_generator import SyntheticDataGenerator


LOGGER: logging.Logger = logging.getLogger(__name__)


def format_file_size(size_bytes: int) -> str:
    """Convert a byte count into a human-readable file size."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def save_datasets() -> None:
    """Generate all datasets and save them as Parquet files in the raw-data directory."""
    started_at = perf_counter()
    output_dir: Path = config.RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = SyntheticDataGenerator(
        customer_count=config.CUSTOMER_COUNT,
        product_count=config.PRODUCT_COUNT,
        order_count=config.ORDER_COUNT,
        seed=config.RANDOM_SEED,
        show_progress=True,
    )
    customers, products, orders = generator.generate_all()

    datasets = {
        "customers": customers,
        "products": products,
        "orders": orders,
    }
    for name, dataframe in datasets.items():
        output_path = output_dir / f"{name}.parquet"
        dataframe.to_parquet(output_path, index=False)
        LOGGER.info("Saved %s: %s", output_path, format_file_size(output_path.stat().st_size))

    elapsed_seconds = perf_counter() - started_at
    LOGGER.info("Generated and saved all datasets in %.2f seconds", elapsed_seconds)


def main() -> int:
    """Run dataset generation and return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        save_datasets()
    except Exception:
        LOGGER.exception("Data generation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())