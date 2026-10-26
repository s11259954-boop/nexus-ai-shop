"""
Data Preprocessing Module.
Handles cleaning, text normalization, missing values, and saves clean dataset.
"""

import os
import re
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PRODUCTS_PATH = os.path.join(BASE_DIR, "data", "raw", "products.csv")
RAW_INTERACTIONS_PATH = os.path.join(BASE_DIR, "data", "raw", "interactions.csv")
PROCESSED_PRODUCTS_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_products.csv")


def clean_text(text: str) -> str:
    """Cleans and standardizes raw text."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special punctuation but keep alphanumeric and spaces
    text = re.sub(r"[^\w\s-]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_products(input_path: str = RAW_PRODUCTS_PATH, output_path: str = PROCESSED_PRODUCTS_PATH) -> pd.DataFrame:
    """Reads raw products, standardizes attributes, builds combined soup, and saves processed CSV."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw products file not found at: {input_path}")

    df = pd.read_csv(input_path)

    # Impute missing values
    df["rating"] = df["rating"].fillna(0.0).astype(float)
    df["review_count"] = df["review_count"].fillna(0).astype(int)
    df["discount_percent"] = df["discount_percent"].fillna(0).astype(int)
    df["popularity_score"] = df["popularity_score"].fillna(50.0).astype(float)
    df["description"] = df["description"].fillna("")
    df["brand"] = df["brand"].fillna("Generic")
    df["subcategory"] = df["subcategory"].fillna("General")
    df["category"] = df["category"].fillna("General")

    # Clean text columns
    df["clean_name"] = df["product_name"].apply(clean_text)
    df["clean_description"] = df["description"].apply(clean_text)
    df["clean_brand"] = df["brand"].apply(clean_text)
    df["clean_category"] = df["category"].apply(clean_text)
    df["clean_subcategory"] = df["subcategory"].apply(clean_text)

    # Build rich metadata content soup for TF-IDF / NLP representations
    # Emphasize category, subcategory and brand by repeating them
    df["content_soup"] = (
        df["clean_name"] + " " +
        df["clean_category"] + " " + df["clean_category"] + " " +
        df["clean_subcategory"] + " " + df["clean_subcategory"] + " " +
        df["clean_brand"] + " " + df["clean_brand"] + " " +
        df["clean_description"]
    )

    # Normalized rating (0 to 1) and popularity (0 to 1) for downstream ranking
    max_rating = 5.0
    df["norm_rating"] = df["rating"] / max_rating

    pop_min = df["popularity_score"].min()
    pop_max = df["popularity_score"].max()
    if pop_max > pop_min:
        df["norm_popularity"] = (df["popularity_score"] - pop_min) / (pop_max - pop_min)
    else:
        df["norm_popularity"] = 0.5

    # Original price before discount
    df["original_price"] = df.apply(
        lambda row: round(row["price"] / (1 - (row["discount_percent"] / 100)), 2)
        if row["discount_percent"] > 0 else row["price"],
        axis=1
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed {len(df)} products saved to {output_path}")
    return df


if __name__ == "__main__":
    preprocess_products()
