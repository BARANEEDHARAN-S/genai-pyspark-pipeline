"""Configuration values for the e-commerce data pipeline."""

import os
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Path = Path(os.getenv("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DATA_DIR: Path = Path(
    os.getenv("PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed")
)
CUSTOMER_COUNT: int = int(os.getenv("CUSTOMER_COUNT", "100000"))
PRODUCT_COUNT: int = int(os.getenv("PRODUCT_COUNT", "10000"))
ORDER_COUNT: int = int(os.getenv("ORDER_COUNT", "1000000"))
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
