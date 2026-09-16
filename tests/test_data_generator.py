"""Tests for synthetic e-commerce data generation."""

import csv
from pathlib import Path

from src.data_generator import generate_all_data


def test_generate_all_data_creates_expected_files(tmp_path: Path) -> None:
    """Ensure generation creates all datasets with the requested row counts."""
    output_dir = tmp_path / "raw"
    paths = generate_all_data(output_dir, customer_count=3, product_count=2, order_count=5)

    assert set(paths) == {"customers", "products", "orders"}
    for path, expected_rows in ((paths["customers"], 3), (paths["products"], 2), (paths["orders"], 5)):
        with path.open(newline="", encoding="utf-8") as file:
            assert len(list(csv.DictReader(file))) == expected_rows