"""
Data Preprocessing Module.
Handles cleaning, text normalization, missing values, and saves clean dataset.
"""

import os
import sys
import re
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

RAW_PRODUCTS_PATH = os.path.join(BASE_DIR, "data", "raw", "products.csv")
RAW_INTERACTIONS_PATH = os.path.join(BASE_DIR, "data", "raw", "interactions.csv")
RAW_USERS_PATH = os.path.join(BASE_DIR, "data", "raw", "users.csv")
PROCESSED_PRODUCTS_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_products.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


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


def ensure_dataset_exists():
    """
    Guarantees that raw datasets, processed files, and trained model matrices
    exist. If any are missing, automatically and safely bootstraps them.
    """
    # 1. Check raw datasets
    if not os.path.exists(RAW_PRODUCTS_PATH) or not os.path.exists(RAW_INTERACTIONS_PATH):
        print("⚠️ Raw datasets missing. Auto-generating catalog and seed interactions...")
        try:
            from data.seed_data import seed_database_and_export_csv
            seed_database_and_export_csv()
        except Exception as e:
            print(f"Error during auto-seeding: {e}")

    # 2. Check processed clean products
    if not os.path.exists(PROCESSED_PRODUCTS_PATH):
        print("⚠️ Processed products dataset missing. Running preprocessing...")
        preprocess_products()

    # 3. Check trained model files
    sim_path = os.path.join(MODELS_DIR, "similarity_matrix.pkl")
    uim_path = os.path.join(MODELS_DIR, "user_item_matrix.pkl")
    if not os.path.exists(sim_path) or not os.path.exists(uim_path):
        print("⚠️ Feature matrices missing. Running feature engineering pipeline...")
        try:
            from src.feature_engineering import run_feature_pipeline
            run_feature_pipeline()
        except Exception as e:
            print(f"Error running feature pipeline: {e}")

    return PROCESSED_PRODUCTS_PATH


def preprocess_products(input_path: str = RAW_PRODUCTS_PATH, output_path: str = PROCESSED_PRODUCTS_PATH) -> pd.DataFrame:
    """Reads raw products, standardizes attributes, builds combined soup, and saves processed CSV."""
    if not os.path.exists(input_path):
        print(f"⚠️ Raw products not found at {input_path}. Auto-seeding...")
        try:
            from data.seed_data import seed_database_and_export_csv
            seed_database_and_export_csv()
        except Exception as e:
            raise FileNotFoundError(f"Raw products file not found at: {input_path} and auto-seeding failed: {e}")

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
