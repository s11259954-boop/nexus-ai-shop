"""
Feature Engineering Module.
Extracts TF-IDF features, computes cosine similarity matrices,
and constructs user-item interaction matrices with explicit weightings.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PRODUCTS_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_products.csv")
RAW_INTERACTIONS_PATH = os.path.join(BASE_DIR, "data", "raw", "interactions.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Domain weight hierarchy for implicit feedback
INTERACTION_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "like": 3.0,
    "wishlist": 3.5,
    "cart": 4.0,
    "purchase": 5.0
}


def build_tfidf_features(df_products: pd.DataFrame):
    """Fits TF-IDF vectorizer on content soup and calculates cosine similarity matrix."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(df_products["content_soup"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Save vectorizer & matrices
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"), "wb") as f:
        pickle.dump(tfidf_matrix, f)

    with open(os.path.join(MODELS_DIR, "similarity_matrix.pkl"), "wb") as f:
        pickle.dump(cosine_sim, f)

    print(f"✅ TF-IDF Matrix shape: {tfidf_matrix.shape}")
    print(f"✅ Cosine Similarity Matrix shape: {cosine_sim.shape}")
    return vectorizer, tfidf_matrix, cosine_sim


def build_user_item_matrix(df_interactions: pd.DataFrame, df_products: pd.DataFrame) -> pd.DataFrame:
    """Builds user-item matrix weighted by interaction type."""
    df_copy = df_interactions.copy()
    df_copy["weight"] = df_copy["interaction_type"].map(lambda t: INTERACTION_WEIGHTS.get(t, 1.0))

    # Aggregate weights for each (user_id, product_id) pair
    user_item_grouped = df_copy.groupby(["user_id", "product_id"])["weight"].sum().reset_index()

    # Pivot to create (user x item) matrix
    user_item_matrix = user_item_grouped.pivot(index="user_id", columns="product_id", values="weight").fillna(0.0)

    # Ensure all product columns exist even if no interactions yet
    for pid in df_products["product_id"]:
        if pid not in user_item_matrix.columns:
            user_item_matrix[pid] = 0.0

    user_item_matrix = user_item_matrix[sorted(user_item_matrix.columns)]

    with open(os.path.join(MODELS_DIR, "user_item_matrix.pkl"), "wb") as f:
        pickle.dump(user_item_matrix, f)

    print(f"✅ User-Item Matrix shape: {user_item_matrix.shape}")
    return user_item_matrix


def build_co_occurrence_matrix(df_interactions: pd.DataFrame) -> pd.DataFrame:
    """Builds co-purchase / co-interaction matrix for Frequently Bought Together feature."""
    # Look at sessions or same user purchasing/carting together
    relevant = df_interactions[df_interactions["interaction_type"].isin(["cart", "purchase", "wishlist"])]
    user_products = relevant.groupby("user_id")["product_id"].unique().to_dict()

    co_counts = defaultdict(lambda: defaultdict(int))
    for u, pids in user_products.items():
        for i in range(len(pids)):
            for j in range(len(pids)):
                if i != j:
                    co_counts[pids[i]][pids[j]] += 1

    df_co = pd.DataFrame(co_counts).fillna(0)
    with open(os.path.join(MODELS_DIR, "co_occurrence_matrix.pkl"), "wb") as f:
        pickle.dump(df_co, f)

    print(f"✅ Co-occurrence Matrix shape: {df_co.shape}")
    return df_co


def run_feature_pipeline():
    """Runs end-to-end feature engineering pipeline."""
    df_products = pd.read_csv(PROCESSED_PRODUCTS_PATH)
    df_interactions = pd.read_csv(RAW_INTERACTIONS_PATH)

    build_tfidf_features(df_products)
    build_user_item_matrix(df_interactions, df_products)
    build_co_occurrence_matrix(df_interactions)
    print("🎯 Feature Engineering Pipeline completed successfully!")


if __name__ == "__main__":
    from preprocessing import preprocess_products
    preprocess_products()
    run_feature_pipeline()
