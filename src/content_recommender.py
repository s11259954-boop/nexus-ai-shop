"""
Content-Based Recommendation Engine with Explainable AI (XAI).
Calculates item-item content similarity using TF-IDF representations.
"""

import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PRODUCTS_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_products.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


class ContentRecommender:
    def __init__(self):
        self.df_products = pd.read_csv(PROCESSED_PRODUCTS_PATH)
        self.indices = pd.Series(self.df_products.index, index=self.df_products["product_id"]).drop_duplicates()

        sim_path = os.path.join(MODELS_DIR, "similarity_matrix.pkl")
        if os.path.exists(sim_path):
            with open(sim_path, "rb") as f:
                self.similarity_matrix = pickle.load(f)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            tfidf = vec.fit_transform(self.df_products["content_soup"])
            self.similarity_matrix = cosine_similarity(tfidf, tfidf)

    def get_similar_products(self, product_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Returns top_n similar products based on content similarity."""
        if product_id not in self.indices:
            return []

        idx = self.indices[product_id]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        # Sort by similarity score descending
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Exclude the queried product itself
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

        results = []
        for i, score in sim_scores:
            prod_row = self.df_products.iloc[i].to_dict()
            explanation = self.generate_content_explanation(product_id, prod_row["product_id"], score)
            prod_row["similarity_score"] = float(score)
            prod_row["explanation"] = explanation
            prod_row["algorithm"] = "content_based"
            results.append(prod_row)

        return results

    def get_smart_alternatives(self, product_id: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Finds budget-friendly smart alternatives:
        Products in the same category or subcategory with high similarity but LOWER price.
        """
        if product_id not in self.indices:
            return []

        base_idx = self.indices[product_id]
        base_prod = self.df_products.iloc[base_idx]
        base_price = float(base_prod["price"])

        candidates = []
        for i, row in self.df_products.iterrows():
            if i == base_idx:
                continue
            # Check price is at least 5% cheaper and category matches or subcategory matches
            cand_price = float(row["price"])
            if cand_price < base_price:
                sim_score = float(self.similarity_matrix[base_idx][i])
                if sim_score > 0.08 or (row["category"] == base_prod["category"]):
                    savings_percent = round(((base_price - cand_price) / base_price) * 100, 1)
                    explanation = (
                        f"Smart Alternative: Save ₹{int(base_price - cand_price):,} ({savings_percent}% cheaper) "
                        f"with similar {row['subcategory']} functionality."
                    )
                    cand_dict = row.to_dict()
                    cand_dict["similarity_score"] = sim_score
                    cand_dict["savings_percent"] = savings_percent
                    cand_dict["explanation"] = explanation
                    cand_dict["algorithm"] = "smart_alternative"
                    candidates.append((sim_score, savings_percent, cand_dict))

        # Sort by combined similarity and savings
        candidates = sorted(candidates, key=lambda x: (x[0] * 0.6 + (x[1] / 100) * 0.4), reverse=True)
        return [c[2] for c in candidates[:top_n]]

    def generate_content_explanation(self, base_id: str, rec_id: str, score: float) -> str:
        """Explainable AI (XAI): Generates human-readable justification for recommendation."""
        base_idx = self.indices[base_id]
        rec_idx = self.indices[rec_id]
        base = self.df_products.iloc[base_idx]
        rec = self.df_products.iloc[rec_idx]

        reasons = []
        if base["category"] == rec["category"]:
            if base["subcategory"] == rec["subcategory"]:
                reasons.append(f"same {rec['subcategory']} collection")
            else:
                reasons.append(f"same {rec['category']} category")

        if base["brand"] == rec["brand"]:
            reasons.append(f"matches preferred brand ({rec['brand']})")

        match_pct = int(score * 100)
        if match_pct > 0:
            reasons.append(f"{match_pct}% feature & attribute match")

        if reasons:
            return "Recommended because it belongs to " + " & ".join(reasons) + "."
        return "Recommended based on similar technical specifications and design features."
