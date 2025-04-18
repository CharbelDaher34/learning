from metaflow import FlowSpec, step, Parameter, conda_base, schedule
import polars as pl
from datetime import datetime
import re
import os
from typing import Dict, List
import psycopg2
from psycopg2.extras import execute_batch


@schedule(hourly=1)  # Run every hour
@conda_base(python="3.10.0")
class AGNewsETLFlow(FlowSpec):
    """ETL flow for processing AG News dataset."""

    db_connection_string = Parameter(
        "db_connection_string",
        default=os.getenv(
            "DATABASE_URL", "postgresql://mluser:mlpassword@localhost:5432/mlpipeline"
        ),
    )

    @step
    def start(self):
        """Start the flow."""
        self.splits = {"train": "train.jsonl", "test": "test.jsonl"}
        self.next(self.extract_data)

    @step
    def extract_data(self):
        """Extract data from Hugging Face dataset."""
        try:
            self.df = pl.read_ndjson(
                "hf://datasets/sh0416/ag_news/" + self.splits["train"]
            )
            self.next(self.transform_data)
        except Exception as e:
            print(f"Error extracting data: {e}")
            raise

    @step
    def transform_data(self):
        """Clean and transform the text data."""

        def clean_text(text: str) -> str:
            # Remove HTML tags
            text = re.sub(r"<[^>]+>", "", text)
            # Convert to lowercase
            text = text.lower()
            # Remove special characters
            text = re.sub(r"[^\w\s]", "", text)
            return text.strip()

        try:
            # Combine title and description
            self.df = self.df.with_columns(
                [
                    pl.col("title").map_elements(clean_text).alias("cleaned_title"),
                    pl.col("description")
                    .map_elements(clean_text)
                    .alias("cleaned_description"),
                    pl.lit(datetime.now()).alias("processed_at"),
                ]
            )

            self.df = self.df.with_columns(
                [
                    (
                        pl.col("cleaned_title") + " " + pl.col("cleaned_description")
                    ).alias("text")
                ]
            )

            self.next(self.load_data)
        except Exception as e:
            print(f"Error transforming data: {e}")
            raise

    @step
    def load_data(self):
        """Load data into PostgreSQL database."""
        try:
            # Create table if not exists
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS ag_news (
                id SERIAL PRIMARY KEY,
                label INTEGER,
                text TEXT,
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """

            insert_sql = """
            INSERT INTO ag_news (label, text, processed_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE
            SET text = EXCLUDED.text,
                processed_at = EXCLUDED.processed_at;
            """

            with psycopg2.connect(self.db_connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_sql)

                    # Prepare data for insertion
                    data = [
                        (row["label"], row["text"], row["processed_at"])
                        for row in self.df.to_dicts()
                    ]

                    # Batch insert
                    execute_batch(cur, insert_sql, data, page_size=1000)
                conn.commit()

            self.next(self.end)
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    @step
    def end(self):
        """End the flow."""
        print("ETL pipeline completed successfully!")


if __name__ == "__main__":
    AGNewsETLFlow()
