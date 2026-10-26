"""
Unit Tests for Recommendation Algorithms.
"""

import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.content_recommender import ContentRecommender
from src.collaborative_recommender import CollaborativeRecommender
from src.hybrid_recommender import HybridRecommender
from src.ranking import SmartRankingEngine
from src.evaluation import RecommenderEvaluator


def test_content_recommender():
    recommender = ContentRecommender()
    similar = recommender.get_similar_products("PROD_ELEC_001", top_n=3)
    assert len(similar) == 3
    # Check similarity score is between 0 and 1
    assert 0.0 <= similar[0]["similarity_score"] <= 1.0
    # Check explanation is provided
    assert "explanation" in similar[0]
    assert len(similar[0]["explanation"]) > 5


def test_smart_alternatives():
    recommender = ContentRecommender()
    # iPhone 15 is ₹69,999 - smart alternatives should be cheaper
    alts = recommender.get_smart_alternatives("PROD_ELEC_001", top_n=3)
    assert len(alts) > 0
    for alt in alts:
        assert alt["price"] < 69999.0
        assert "savings_percent" in alt
        assert alt["savings_percent"] > 0


def test_collaborative_recommender():
    recommender = CollaborativeRecommender()
    # Test for existing user
    recs = recommender.get_collaborative_recommendations("USER_TECH_001", top_n=3)
    assert len(recs) > 0
    assert "explanation" in recs[0]

    # Test frequently bought together
    fbt = recommender.get_frequently_bought_together("PROD_ELEC_001", top_n=2)
    assert isinstance(fbt, list)


def test_cold_start_handling():
    recommender = HybridRecommender()
    # Cold start user with no previous history
    is_cold = recommender.is_cold_start_user("USER_COLD_006")
    assert is_cold is True

    recs = recommender.get_personalized_recommendations("USER_COLD_006", top_n=5)
    assert len(recs) == 5
    # Check that cold start explanation was provided
    assert "cold_start" in recs[0].get("algorithm", "") or "Trending" in recs[0].get("explanation", "") or "Welcome" in recs[0].get("explanation", "")


def test_smart_ranking():
    engine = SmartRankingEngine(
        weight_similarity=0.4,
        weight_user_pref=0.3,
        weight_rating=0.15,
        weight_popularity=0.10,
        weight_recency=0.05
    )
    mock_candidates = [
        {"product_id": "PROD_ELEC_001", "similarity_score": 0.8, "category": "Electronics", "brand": "Apple"},
        {"product_id": "PROD_FASH_001", "similarity_score": 0.3, "category": "Fashion", "brand": "Nike"}
    ]
    ranked = engine.rank_candidates(mock_candidates, user_preferred_category="Electronics")
    assert len(ranked) == 2
    # PROD_ELEC_001 should rank first due to higher similarity & category match
    assert ranked[0]["product_id"] == "PROD_ELEC_001"
    assert "smart_rank_score" in ranked[0]
    assert "factor_breakdown" in ranked[0]


def test_evaluation_metrics():
    evaluator = RecommenderEvaluator(k=5)
    metrics = evaluator.evaluate_all_metrics()
    assert "precision_at_k" in metrics
    assert "recall_at_k" in metrics
    assert "hit_rate" in metrics
    assert "catalog_coverage_pct" in metrics
    assert "diversity_score" in metrics
    assert metrics["catalog_coverage_pct"] > 50.0
