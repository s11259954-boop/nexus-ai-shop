"""
Collaborative Filtering Recommendation Engine with Matrix Factorization.
Provides personalized recommendations based on collective user behavior,
as well as Frequently Bought Together associations.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.preprocessing import ensure_dataset_exists, PROCESSED_PRODUCTS_PATH
RAW_INTERACTIONS_PATH = os.path.join(BASE_DIR, "data", "raw", "interactions.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


class CollaborativeRecommender:
    def __init__(self, n_components: int = 10):
        ensure_dataset_exists()
        self.df_products = pd.read_csv(PROCESSED_PRODUCTS_PATH)
        self.products_dict = self.df_products.set_index("product_id").to_dict("index")
        self.n_components = n_components
        self.predicted_matrix = pd.DataFrame()
        self.item_sim_matrix = pd.DataFrame()

        # Load matrices
        uim_path = os.path.join(MODELS_DIR, "user_item_matrix.pkl")
        if os.path.exists(uim_path):
            with open(uim_path, "rb") as f:
                self.user_item_matrix = pickle.load(f)
        else:
            self.user_item_matrix = pd.DataFrame()

        co_path = os.path.join(MODELS_DIR, "co_occurrence_matrix.pkl")
        if os.path.exists(co_path):
            with open(co_path, "rb") as f:
                self.co_occurrence_matrix = pickle.load(f)
        else:
            self.co_occurrence_matrix = pd.DataFrame()

        self._fit_svd_model()

    def _fit_svd_model(self):
        """Fits TruncatedSVD on the User-Item matrix to perform matrix factorization."""
        if self.user_item_matrix.empty or len(self.user_item_matrix.columns) == 0:
            self.svd = None
            self.predicted_matrix = pd.DataFrame()
            self.item_sim_matrix = pd.DataFrame()
            return

        # SVD components cannot exceed min(n_samples, n_features) - 1
        n_comp = min(self.n_components, min(self.user_item_matrix.shape) - 1)
        if n_comp < 2:
            n_comp = 2

        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        user_factors = self.svd.fit_transform(self.user_item_matrix.values)
        item_factors = self.svd.components_

        # Reconstructed predicted interaction score matrix
        preds = np.dot(user_factors, item_factors)
        self.predicted_matrix = pd.DataFrame(
            preds,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns
        )

        # Item-Item collaborative cosine similarity matrix
        self.item_sim_matrix = pd.DataFrame(
            cosine_similarity(self.user_item_matrix.T.values),
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )

    def get_collaborative_recommendations(self, user_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Generates collaborative filtering recommendations for a specific user."""
        if self.predicted_matrix.empty or user_id not in self.predicted_matrix.index:
            # User not in matrix (Cold start)
            return []

        user_scores = self.predicted_matrix.loc[user_id].copy()
        # Find products user has already interacted heavily with to avoid recommending exact same purchases
        actual_interactions = self.user_item_matrix.loc[user_id]
        already_purchased = actual_interactions[actual_interactions >= 4.0].index.tolist()

        # Penalize already purchased items slightly to encourage discovery
        for pid in already_purchased:
            if pid in user_scores:
                user_scores[pid] *= 0.1

        top_pids = user_scores.sort_values(ascending=False).head(top_n).index.tolist()

        recommendations = []
        for pid in top_pids:
            if pid in self.products_dict:
                prod = dict(self.products_dict[pid])
                prod["product_id"] = pid
                score = float(user_scores[pid])
                # Normalize score roughly 0 to 1
                norm_score = round(min(1.0, max(0.0, score / 15.0)), 4)
                prod["similarity_score"] = norm_score
                prod["explanation"] = "Recommended because shoppers with similar purchase and browsing patterns frequently buy this."
                prod["algorithm"] = "collaborative_filtering"
                recommendations.append(prod)

        return recommendations

    def get_frequently_bought_together(self, product_id: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """Returns items frequently purchased or carted in the same user journey."""
        if self.co_occurrence_matrix.empty or product_id not in self.co_occurrence_matrix.columns:
            return []

        co_series = self.co_occurrence_matrix[product_id].sort_values(ascending=False)
        co_series = co_series[co_series.index != product_id]
        top_co = co_series.head(top_n)

        results = []
        for pid, count in top_co.items():
            if pid in self.products_dict and count > 0:
                prod = dict(self.products_dict[pid])
                prod["product_id"] = pid
                prod["co_count"] = int(count)
                prod["explanation"] = f"Frequently bought together: {int(count)} other customers purchased both in the same shopping cycle."
                prod["algorithm"] = "frequently_bought_together"
                results.append(prod)

        return results
