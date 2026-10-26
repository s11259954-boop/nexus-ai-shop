"""
Smart Ranking & Re-ranking Module.
Combines multiple signals (similarity, user preference, ratings, popularity, recency)
and applies Maximal Marginal Relevance (MMR) for diversity.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PRODUCTS_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_products.csv")


class SmartRankingEngine:
    def __init__(
        self,
        weight_similarity: float = 0.40,
        weight_user_pref: float = 0.30,
        weight_rating: float = 0.15,
        weight_popularity: float = 0.10,
        weight_recency: float = 0.05
    ):
        self.w_sim = weight_similarity
        self.w_pref = weight_user_pref
        self.w_rating = weight_rating
        self.w_pop = weight_popularity
        self.w_rec = weight_recency

        self.df_products = pd.read_csv(PROCESSED_PRODUCTS_PATH)
        self.products_dict = self.df_products.set_index("product_id").to_dict("index")

    def update_weights(
        self,
        w_sim: Optional[float] = None,
        w_pref: Optional[float] = None,
        w_rating: Optional[float] = None,
        w_pop: Optional[float] = None,
        w_rec: Optional[float] = None
    ):
        """Allows real-time customization of ranking weights."""
        if w_sim is not None: self.w_sim = w_sim
        if w_pref is not None: self.w_pref = w_pref
        if w_rating is not None: self.w_rating = w_rating
        if w_pop is not None: self.w_pop = w_pop
        if w_rec is not None: self.w_rec = w_rec

        # Normalize weights so they sum to 1.0
        total = self.w_sim + self.w_pref + self.w_rating + self.w_pop + self.w_rec
        if total > 0:
            self.w_sim /= total
            self.w_pref /= total
            self.w_rating /= total
            self.w_pop /= total
            self.w_rec /= total

    def rank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        user_preferred_category: Optional[str] = None,
        user_preferred_brand: Optional[str] = None,
        apply_diversity: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Calculates composite score for each candidate and sorts them.
        Formula:
        Final Score =
            w_sim * Similarity_Score
          + w_pref * User_Preference_Score
          + w_rating * Normalized_Rating
          + w_pop * Normalized_Popularity
          + w_rec * Recency_Score
        """
        if not candidates:
            return []

        ranked_items = []
        for item in candidates:
            pid = item.get("product_id")
            prod_meta = self.products_dict.get(pid, item)

            sim_score = float(item.get("similarity_score", item.get("hybrid_score", 0.5)))

            # User preference alignment
            pref_score = 0.0
            if user_preferred_category and prod_meta.get("category") == user_preferred_category:
                pref_score += 0.7
            if user_preferred_brand and prod_meta.get("brand") == user_preferred_brand:
                pref_score += 0.3

            # Normalized rating (0 to 1)
            norm_rating = float(prod_meta.get("norm_rating", float(prod_meta.get("rating", 4.0)) / 5.0))

            # Normalized popularity (0 to 1)
            norm_pop = float(prod_meta.get("norm_popularity", 0.5))

            # Recency score (simulated between 0.5 and 1.0)
            rec_score = float(item.get("recency_score", 0.8))

            # Multi-factor smart ranking score
            composite_score = (
                (self.w_sim * sim_score) +
                (self.w_pref * pref_score) +
                (self.w_rating * norm_rating) +
                (self.w_pop * norm_pop) +
                (self.w_rec * rec_score)
            )

            item_copy = dict(item)
            item_copy["smart_rank_score"] = round(float(composite_score), 4)
            item_copy["factor_breakdown"] = {
                "similarity_component": round(float(self.w_sim * sim_score), 3),
                "preference_component": round(float(self.w_pref * pref_score), 3),
                "rating_component": round(float(self.w_rating * norm_rating), 3),
                "popularity_component": round(float(self.w_pop * norm_pop), 3),
                "recency_component": round(float(self.w_rec * rec_score), 3),
            }
            ranked_items.append(item_copy)

        # Sort descending by composite score
        ranked_items = sorted(ranked_items, key=lambda x: x["smart_rank_score"], reverse=True)

        if apply_diversity:
            ranked_items = self._diversify_list(ranked_items)

        return ranked_items

    def _diversify_list(self, items: List[Dict[str, Any]], penalty_rate: float = 0.05) -> List[Dict[str, Any]]:
        """
        Soft MMR penalty: prevents monopolization by a single subcategory
        by applying small marginal penalties to subsequent items with the same subcategory.
        """
        seen_subcategories = {}
        diversified = []

        for item in items:
            subcat = item.get("subcategory", "General")
            count = seen_subcategories.get(subcat, 0)
            seen_subcategories[subcat] = count + 1

            # Marginal penalty for repeated categories to boost diversity
            penalty = count * penalty_rate
            item["smart_rank_score"] = max(0.0, item["smart_rank_score"] - penalty)
            diversified.append(item)

        # Re-sort after diversity adjustments
        return sorted(diversified, key=lambda x: x["smart_rank_score"], reverse=True)
