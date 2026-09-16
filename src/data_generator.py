"""Generate large, reproducible synthetic e-commerce datasets."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Final

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm


LOGGER: logging.Logger = logging.getLogger(__name__)
CUSTOMER_COLUMNS: Final[tuple[str, ...]] = ("customer_id", "name", "email", "age", "city", "country", "registration_date")
PRODUCT_COLUMNS: Final[tuple[str, ...]] = ("product_id", "name", "category", "price", "stock", "rating")
ORDER_COLUMNS: Final[tuple[str, ...]] = ("order_id", "customer_id", "product_id", "quantity", "order_date")
CATEGORIES: Final[tuple[str, ...]] = ("Electronics", "Clothing", "Home", "Sports", "Books")


class SyntheticDataGenerator:
    """Generate synthetic customers, products, and orders as pandas DataFrames."""

    def __init__(self, customer_count: int = 100_000, product_count: int = 10_000, order_count: int = 1_000_000, seed: int = 42, show_progress: bool = True) -> None:
        """Initialize the generator with dataset sizes, seed, and progress settings."""
        for name, value in (("customer_count", customer_count), ("product_count", product_count), ("order_count", order_count)):
            if value < 1:
                raise ValueError(f"{name} must be greater than zero")
        self.customer_count = customer_count
        self.product_count = product_count
        self.order_count = order_count
        self.show_progress = show_progress
        self.fake = Faker()
        self.fake.seed_instance(seed)
        self.rng = np.random.default_rng(seed)

    def generate_customers(self) -> pd.DataFrame:
        """Generate customers with ages normally distributed around 35."""
        LOGGER.info("Generating %d customers", self.customer_count)
        ages = np.clip(np.rint(self.rng.normal(35, 10, self.customer_count)), 18, 90).astype(int)
        start_date = date.today() - timedelta(days=365 * 5)
        registration_days = self.rng.integers(0, 365 * 5 + 1, self.customer_count)
        rows = ({"customer_id": f"C{index:06d}", "name": self.fake.name(), "email": self.fake.unique.email(), "age": int(ages[index - 1]), "city": self.fake.city(), "country": self.fake.country(), "registration_date": (start_date + timedelta(days=int(registration_days[index - 1]))).isoformat()} for index in tqdm(range(1, self.customer_count + 1), desc="Generating customers", unit="customer", disable=not self.show_progress))
        return pd.DataFrame(rows, columns=CUSTOMER_COLUMNS)

    def generate_products(self) -> pd.DataFrame:
        """Generate products with categories, prices, stock, and ratings."""
        LOGGER.info("Generating %d products", self.product_count)
        categories = self.rng.choice(CATEGORIES, self.product_count)
        product_numbers = tqdm(
            range(1, self.product_count + 1),
            desc="Generating products",
            unit="product",
            disable=not self.show_progress,
        )
        product_ids = [f"P{index:06d}" for index in product_numbers]
        product_names = [
            f"{category} Product {index}"
            for index, category in enumerate(categories, 1)
        ]
        products = pd.DataFrame(
            {
                "product_id": product_ids,
                "name": product_names,
                "category": categories,
                "price": np.round(self.rng.uniform(10, 500, self.product_count), 2),
                "stock": self.rng.integers(0, 1_001, self.product_count),
                "rating": np.round(self.rng.uniform(1, 5, self.product_count), 1),
            },
            columns=PRODUCT_COLUMNS,
        )
        LOGGER.info("Generated products with shape %s", products.shape)
        return products

    def generate_orders(self) -> pd.DataFrame:
        """Generate orders where 80% come from the top 20% of customers."""
        LOGGER.info("Generating %d orders with an 80/20 customer distribution", self.order_count)
        top_count = max(1, int(np.ceil(self.customer_count * 0.20)))
        top_ids = np.arange(1, top_count + 1)
        other_ids = np.arange(top_count + 1, self.customer_count + 1)
        concentrated_count = int(self.order_count * 0.80)
        pareto_values = self.rng.pareto(a=1.5, size=top_count) + 1.0
        top_probabilities = pareto_values / pareto_values.sum()
        customer_numbers = np.concatenate([
            self.rng.choice(top_ids, concentrated_count, replace=True, p=top_probabilities),
            self.rng.choice(other_ids if len(other_ids) else top_ids, self.order_count - concentrated_count, replace=True),
        ])
        self.rng.shuffle(customer_numbers)
        start_date = date.today() - timedelta(days=365)
        order_days = self.rng.integers(0, 366, self.order_count)
        return pd.DataFrame({"order_id": [f"O{index:07d}" for index in range(1, self.order_count + 1)], "customer_id": [f"C{number:06d}" for number in customer_numbers], "product_id": [f"P{number:06d}" for number in self.rng.integers(1, self.product_count + 1, self.order_count)], "quantity": self.rng.integers(1, 11, self.order_count), "order_date": [(start_date + timedelta(days=int(day))).isoformat() for day in tqdm(order_days, desc="Generating orders", unit="order", disable=not self.show_progress)]}, columns=ORDER_COLUMNS)

    def generate_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate and return customers, products, and orders in that order."""
        return self.generate_customers(), self.generate_products(), self.generate_orders()


def generate_all_data(customer_count: int = 100_000, product_count: int = 10_000, order_count: int = 1_000_000, seed: int = 42, show_progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate all datasets using ``SyntheticDataGenerator``."""
    return SyntheticDataGenerator(customer_count, product_count, order_count, seed, show_progress).generate_all()


def main() -> None:
    """Configure logging and generate the default in-memory datasets."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    generate_all_data()


if __name__ == "__main__":
    main()