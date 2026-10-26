"""
Hybrid Recommendation Engine with Cold-Start Handling and Multi-Type Recommendations.
Combines Content-Based and Collaborative signals with graceful degradation.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.content_recommender import ContentRecommender
from src.collaborative_recommender import CollaborativeRecommender

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PRODUCTS_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_products.csv")
RAW_INTERACTIONS_PATH = os.path.join(BASE_DIR, "data", "raw", "interactions.csv")
RAW_USERS_PATH = os.path.join(BASE_DIR, "data", "raw", "users.csv")


class HybridRecommender:
    def __init__(self, content_weight: float = 0.5, collab_weight: float = 0.5):
        self.content_weight = content_weight
        self.collab_weight = collab_weight

        self.content_rec = ContentRecommender()
        self.collab_rec = CollaborativeRecommender()

        self.df_products = pd.read_csv(PROCESSED_PRODUCTS_PATH)
        self.products_dict = self.df_products.set_index("product_id").to_dict("index")

        if os.path.exists(RAW_INTERACTIONS_PATH):
            self.df_interactions = pd.read_csv(RAW_INTERACTIONS_PATH)
        else:
            self.df_interactions = pd.DataFrame(columns=["user_id", "product_id", "interaction_type", "timestamp"])

        if os.path.exists(RAW_USERS_PATH):
            self.df_users = pd.read_csv(RAW_USERS_PATH)
        else:
            self.df_users = pd.DataFrame(columns=["user_id", "username", "email", "preferred_category"])

    def reload_interactions(self, new_df: Optional[pd.DataFrame] = None):
        """Refreshes interactions dataframe when user takes actions in real-time."""
        if new_df is not None:
            self.df_interactions = new_df
        elif os.path.exists(RAW_INTERACTIONS_PATH):
            self.df_interactions = pd.read_csv(RAW_INTERACTIONS_PATH)

    def get_user_interactions(self, user_id: str) -> pd.DataFrame:
        """Returns all interaction rows for user."""
        if self.df_interactions.empty:
            return pd.DataFrame()
        return self.df_interactions[self.df_interactions["user_id"] == user_id]

    def is_cold_start_user(self, user_id: str) -> bool:
        """Determines if user is in cold-start state (<= 1 interaction)."""
        user_history = self.get_user_interactions(user_id)
        return len(user_history) <= 1

    def get_trending_products(self, top_n: int = 8, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Returns trending products based on interaction volume, popularity, and rating.
        Can optionally filter by category.
        """
        df = self.df_products.copy()
        if category and category != "All":
            df = df[df["category"] == category]

        # Calculate trending score: Popularity * 0.6 + (Rating / 5) * 40
        df["trending_score"] = (df["popularity_score"] * 0.6) + ((df["rating"] / 5.0) * 40.0)
        df_sorted = df.sort_values(by="trending_score", ascending=False).head(top_n)

        results = []
        for _, row in df_sorted.iterrows():
            prod = row.to_dict()
            prod["explanation"] = f"🔥 Trending now: High user engagement and {prod['rating']}★ rating."
            prod["algorithm"] = "trending"
            results.append(prod)
        return results

    def get_new_arrivals(self, top_n: int = 6) -> List[Dict[str, Any]]:
        """Returns new arrival products."""
        # Using product list reverse index or recent IDs
        df_sorted = self.df_products.iloc[::-1].head(top_n)
        results = []
        for _, row in df_sorted.iterrows():
            prod = row.to_dict()
            prod["explanation"] = "✨ New Arrival: Freshly added to our catalog."
            prod["algorithm"] = "new_arrival"
            results.append(prod)
        return results

    def get_recently_viewed(self, user_id: str, top_n: int = 6) -> List[Dict[str, Any]]:
        """Returns most recently viewed products for the active user."""
        user_history = self.get_user_interactions(user_id)
        if user_history.empty:
            return []

        views = user_history[user_history["interaction_type"].isin(["view", "click"])]
        if views.empty:
            return []

        # Sort by timestamp descending
        recent_pids = views.sort_values("timestamp", ascending=False)["product_id"].unique()[:top_n]

        results = []
        for pid in recent_pids:
            if pid in self.products_dict:
                prod = dict(self.products_dict[pid])
                prod["product_id"] = pid
                prod["explanation"] = "🕘 Based on your recent browsing activity."
                prod["algorithm"] = "recently_viewed"
                results.append(prod)
        return results

    def get_cold_start_recommendations(self, user_id: str, top_n: int = 6) -> List[Dict[str, Any]]:
        """
        Cold-Start Solution:
        Combines top-rated, trending products, and user's preferred onboarding category if set.
        """
        pref_category = None
        user_match = self.df_users[self.df_users["user_id"] == user_id]
        if not user_match.empty and pd.notna(user_match.iloc[0]["preferred_category"]):
            pref_category = user_match.iloc[0]["preferred_category"]

        if pref_category and pref_category in self.df_products["category"].values:
            candidates = self.df_products[self.df_products["category"] == pref_category].copy()
            candidates["cold_score"] = (candidates["rating"] / 5.0) * 0.5 + (candidates["norm_popularity"]) * 0.5
            sorted_candidates = candidates.sort_values("cold_score", ascending=False).head(top_n)

            results = []
            for _, row in sorted_candidates.iterrows():
                prod = row.to_dict()
                prod["explanation"] = f"⭐ Curated for you: Top rated & trending in your chosen interest '{pref_category}'."
                prod["algorithm"] = "cold_start_category"
                results.append(prod)
            return results

        # General cold-start: blend highest rated and most popular
        df = self.df_products.copy()
        df["cold_score"] = (df["rating"] / 5.0) * 0.5 + (df["norm_popularity"]) * 0.5
        top_picks = df.sort_values("cold_score", ascending=False).head(top_n)

        results = []
        for _, row in top_picks.iterrows():
            prod = row.to_dict()
            prod["explanation"] = f"⭐ Welcome to our store! Popular trending product loved by our community ({prod['rating']}★)."
            prod["algorithm"] = "cold_start_popular"
            results.append(prod)
        return results

    def get_personalized_recommendations(
        self,
        user_id: str,
        top_n: int = 6,
        content_weight: Optional[float] = None,
        collab_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Main Hybrid Recommender:
        1. Checks for Cold Start -> Returns curated trending/category picks.
        2. Blends Content-Based recommendations from recently interacted items with Collaborative Filtering predictions.
        3. Generates transparent Explainable AI justifications.
        """
        c_weight = content_weight if content_weight is not None else self.content_weight
        cf_weight = collab_weight if collab_weight is not None else self.collab_weight

        # 1. Cold start check
        if self.is_cold_start_user(user_id):
            return self.get_cold_start_recommendations(user_id, top_n=top_n)

        # 2. Extract user interaction profile for Content Recommender
        user_history = self.get_user_interactions(user_id)
        recent_pids = user_history.sort_values("timestamp", ascending=False)["product_id"].unique()[:4]

        content_candidates: Dict[str, float] = {}
        for pid in recent_pids:
            sims = self.content_rec.get_similar_products(pid, top_n=8)
            for s in sims:
                spid = s["product_id"]
                score = s["similarity_score"]
                content_candidates[spid] = max(content_candidates.get(spid, 0.0), score)

        # 3. Extract Collaborative candidates
        collab_recs = self.collab_rec.get_collaborative_recommendations(user_id, top_n=10)
        collab_candidates = {r["product_id"]: r["similarity_score"] for r in collab_recs}

        # 4. Merge candidates and compute hybrid scores
        all_candidate_pids = set(content_candidates.keys()).union(set(collab_candidates.keys()))

        # Remove products already heavily purchased by user
        purchased_pids = set(user_history[user_history["interaction_type"] == "purchase"]["product_id"])
        all_candidate_pids = [p for p in all_candidate_pids if p not in purchased_pids]

        scored_list = []
        for pid in all_candidate_pids:
            if pid not in self.products_dict:
                continue

            c_score = content_candidates.get(pid, 0.0)
            cf_score = collab_candidates.get(pid, 0.0)

            # If item has only content or only collaborative, smooth it
            hybrid_score = (c_score * c_weight) + (cf_score * cf_weight)

            # Explainable AI justification
            prod_info = self.products_dict[pid]
            if c_score > 0.1 and cf_score > 0.1:
                explanation = (
                    f"Recommended because you explored {prod_info['category']} ({prod_info['brand']}) "
                    f"and shoppers with similar interests also loved this."
                )
            elif c_score >= cf_score:
                explanation = f"Recommended based on your interest in {prod_info['category']} ({prod_info['subcategory']})."
            else:
                explanation = "Recommended because customers with similar tastes frequently purchased this."

            prod_dict = dict(prod_info)
            prod_dict["product_id"] = pid
            prod_dict["hybrid_score"] = round(float(hybrid_score), 4)
            prod_dict["content_score"] = round(float(c_score), 4)
            prod_dict["collab_score"] = round(float(cf_score), 4)
            prod_dict["explanation"] = explanation
            prod_dict["algorithm"] = "hybrid"
            scored_list.append(prod_dict)

        # Fallback if candidates were too few
        if len(scored_list) < top_n:
            trendings = self.get_trending_products(top_n=top_n)
            existing_pids = {p["product_id"] for p in scored_list}
            for t in trendings:
                if t["product_id"] not in existing_pids:
                    t["hybrid_score"] = 0.25
                    scored_list.append(t)
                    if len(scored_list) >= top_n:
                        break

        scored_list = sorted(scored_list, key=lambda x: x["hybrid_score"], reverse=True)
        return scored_list[:top_n]
