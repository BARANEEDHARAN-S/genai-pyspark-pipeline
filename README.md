# E-commerce PySpark Data Pipeline

A small, testable e-commerce data pipeline that generates synthetic customers,
products, and orders, then uses PySpark to produce business insights.

## Project structure

```text
genai-pyspark-pipeline/
├── data/
│   ├── processed/              # PySpark analysis results
│   └── raw/                    # Generated CSV input data
├── notebooks/                  # Jupyter notebooks for exploration
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
├── tests/
│   └── test_data_generator.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Setup

Python 3.10 or newer and Java 17 or newer are recommended. PySpark requires a
working Java runtime; set `JAVA_HOME` before running the analytics module.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, for example:

```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-17"
```

## Run the pipeline

Generate 100,000 customers, 10,000 products, and 1,000,000 orders as Parquet files:

```bash
python main.py
python -m src.spark_analytics
```

The generated input files are written to `data/raw/`. Spark writes the
following result directories beneath `data/processed/`:

- `revenue_by_category/`
- `monthly_revenue/`
- `top_customers/`
- `order_status_summary/`

Spark output paths are directories containing part files, which makes the
results suitable for larger datasets as well as local testing.

## Run tests

```bash
pytest
```

## Compare Pandas and PySpark

Benchmark loading, joining, revenue calculation, customer aggregation, and
the top 10 result against the one-million-row orders dataset:

```bash
python benchmark_pandas_pyspark.py
```

The benchmark materializes each PySpark stage so the comparison includes the
actual execution time for every operation. Results vary with local hardware,
Java configuration, and whether Spark has already started.

## Configuration

Override the default paths and record counts with environment variables:

```text
RAW_DATA_DIR, PROCESSED_DATA_DIR, CUSTOMER_COUNT, PRODUCT_COUNT, ORDER_COUNT
```
