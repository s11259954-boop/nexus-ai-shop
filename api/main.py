"""
FastAPI Backend Application for AI-Powered Product Recommendation & Analytics Platform.
Provides RESTful endpoints for catalog browsing, multi-algorithm recommendations,
explainable AI metrics, interaction tracking, and executive analytics.
"""

import os
import sys
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root directory is on PYTHONPATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from database.database import get_db, SessionLocal, Product, User, Interaction, Wishlist, Rating, log_interaction
from src.hybrid_recommender import HybridRecommender
from src.content_recommender import ContentRecommender
from src.collaborative_recommender import CollaborativeRecommender
from src.ranking import SmartRankingEngine
from src.evaluation import RecommenderEvaluator

app = FastAPI(
    title="AI-Powered Personalized Product Recommendation & Analytics API",
    description="Production-ready REST API featuring Content-Based, Collaborative Filtering, Hybrid Recommendations, Explainable AI (XAI), and Real-Time Event Tracking.",
    version="1.0.0"
)

# Enable CORS for local Streamlit frontend and external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Recommender singletons
hybrid_rec = HybridRecommender()
content_rec = ContentRecommender()
collab_rec = CollaborativeRecommender()
ranking_engine = SmartRankingEngine()
evaluator = RecommenderEvaluator(k=5)


# ========================================================
# Pydantic Schemas
# ========================================================

class InteractionCreate(BaseModel):
    user_id: str = Field(..., example="USER_TECH_001")
    product_id: str = Field(..., example="PROD_ELEC_001")
    interaction_type: str = Field(..., example="view", description="view, click, like, wishlist, cart, purchase")


class WishlistToggle(BaseModel):
    user_id: str
    product_id: str


class RankingWeightUpdate(BaseModel):
    w_similarity: Optional[float] = 0.40
    w_user_pref: Optional[float] = 0.30
    w_rating: Optional[float] = 0.15
    w_popularity: Optional[float] = 0.10
    w_recency: Optional[float] = 0.05


# ========================================================
# Endpoints
# ========================================================

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": "AI Product Recommendation Engine",
        "docs_url": "/docs",
        "algorithms": ["content_based", "collaborative_filtering", "hybrid", "smart_alternatives", "frequently_bought_together"]
    }


@app.get("/products", tags=["Products"])
def get_products(
    category: Optional[str] = None,
    brand: Optional[str] = None,
    search: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_price: Optional[float] = None,
    sort_by: Optional[str] = "popularity",  # popularity, price_asc, price_desc, rating
    limit: int = 50,
    offset: int = 0
):
    """Retrieves product catalog with comprehensive filtering and sorting."""
    db = SessionLocal()
    try:
        query = db.query(Product)

        if category and category != "All":
            query = query.filter(Product.category == category)
        if brand and brand != "All":
            query = query.filter(Product.brand == brand)
        if min_rating is not None:
            query = query.filter(Product.rating >= min_rating)
        if max_price is not None:
            query = query.filter(Product.price <= max_price)
        if search:
            search_term = f"%{search.lower()}%"
            query = query.filter(
                (Product.product_name.ilike(search_term)) |
                (Product.description.ilike(search_term)) |
                (Product.brand.ilike(search_term)) |
                (Product.subcategory.ilike(search_term))
            )

        if sort_by == "price_asc":
            query = query.order_by(Product.price.asc())
        elif sort_by == "price_desc":
            query = query.order_by(Product.price.desc())
        elif sort_by == "rating":
            query = query.order_by(Product.rating.desc())
        else:
            query = query.order_by(Product.popularity_score.desc())

        total = query.count()
        products = query.offset(offset).limit(limit).all()

        return {
            "total": total,
            "count": len(products),
            "products": [
                {
                    "product_id": p.product_id,
                    "product_name": p.product_name,
                    "category": p.category,
                    "subcategory": p.subcategory,
                    "brand": p.brand,
                    "description": p.description,
                    "price": p.price,
                    "rating": p.rating,
                    "review_count": p.review_count,
                    "discount_percent": p.discount_percent,
                    "popularity_score": p.popularity_score,
                    "image_url": p.image_url
                }
                for p in products
            ]
        }
    finally:
        db.close()


@app.get("/products/{product_id}", tags=["Products"])
def get_product_details(product_id: str):
    """Fetches details of a single product."""
    db = SessionLocal()
    try:
        product = db.query(Product).filter(Product.product_id == product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return {
            "product_id": product.product_id,
            "product_name": product.product_name,
            "category": product.category,
            "subcategory": product.subcategory,
            "brand": product.brand,
            "description": product.description,
            "price": product.price,
            "rating": product.rating,
            "review_count": product.review_count,
            "discount_percent": product.discount_percent,
            "popularity_score": product.popularity_score,
            "image_url": product.image_url,
            "stock": product.stock
        }
    finally:
        db.close()


@app.get("/recommendations/similar/{product_id}", tags=["Recommendations"])
def get_similar_products(product_id: str, limit: int = 5):
    """Content-Based: Returns similar products based on NLP TF-IDF feature similarity."""
    recs = content_rec.get_similar_products(product_id, top_n=limit)
    return {
        "product_id": product_id,
        "algorithm": "Content-Based Filtering (TF-IDF + Cosine Similarity)",
        "recommendations": recs
    }


@app.get("/recommendations/smart-alternatives/{product_id}", tags=["Recommendations"])
def get_smart_alternatives(product_id: str, limit: int = 3):
    """Smart Alternatives: Lower-priced products in same category with high feature overlap."""
    alts = content_rec.get_smart_alternatives(product_id, top_n=limit)
    return {
        "product_id": product_id,
        "algorithm": "Smart Cost-Effective Alternatives",
        "recommendations": alts
    }


@app.get("/recommendations/frequently-bought/{product_id}", tags=["Recommendations"])
def get_frequently_bought_together(product_id: str, limit: int = 3):
    """Market Basket / Co-occurrence: Products frequently carted/purchased together."""
    fbt = collab_rec.get_frequently_bought_together(product_id, top_n=limit)
    return {
        "product_id": product_id,
        "algorithm": "Co-occurrence Association Mining",
        "recommendations": fbt
    }


@app.get("/recommendations/personalized/{user_id}", tags=["Recommendations"])
def get_personalized_recommendations(
    user_id: str,
    limit: int = 6,
    content_weight: Optional[float] = Query(0.5, ge=0.0, le=1.0),
    collab_weight: Optional[float] = Query(0.5, ge=0.0, le=1.0),
    enable_smart_ranking: bool = True
):
    """
    Main Hybrid Recommender:
    Combines Content + Collaborative Filtering with Cold-Start handling and Smart Ranking.
    """
    # Check if cold start
    is_cold = hybrid_rec.is_cold_start_user(user_id)

    candidates = hybrid_rec.get_personalized_recommendations(
        user_id=user_id,
        top_n=limit * 2,  # Fetch pool for ranking
        content_weight=content_weight,
        collab_weight=collab_weight
    )

    if enable_smart_ranking and candidates:
        db = SessionLocal()
        user = db.query(User).filter(User.user_id == user_id).first()
        pref_cat = user.preferred_category if user else None
        db.close()

        final_recs = ranking_engine.rank_candidates(
            candidates=candidates,
            user_preferred_category=pref_cat,
            apply_diversity=True
        )[:limit]
    else:
        final_recs = candidates[:limit]

    return {
        "user_id": user_id,
        "is_cold_start": is_cold,
        "algorithm": "Hybrid (Content + Collaborative + Smart Multi-factor Ranking)",
        "content_weight": content_weight,
        "collab_weight": collab_weight,
        "recommendations": final_recs
    }


@app.get("/recommendations/trending", tags=["Recommendations"])
def get_trending(limit: int = 6, category: Optional[str] = None):
    """Trending products based on recent interactions and popularity velocity."""
    items = hybrid_rec.get_trending_products(top_n=limit, category=category)
    return {"algorithm": "Popularity & Interaction Velocity", "trending_products": items}


@app.get("/recommendations/new-arrivals", tags=["Recommendations"])
def get_new_arrivals(limit: int = 6):
    """New catalog items."""
    items = hybrid_rec.get_new_arrivals(top_n=limit)
    return {"algorithm": "Catalog Recency", "new_arrivals": items}


@app.get("/recommendations/recently-viewed/{user_id}", tags=["Recommendations"])
def get_recently_viewed(user_id: str, limit: int = 6):
    """Returns products recently browsed by user."""
    items = hybrid_rec.get_recently_viewed(user_id, top_n=limit)
    return {"user_id": user_id, "recently_viewed": items}


@app.post("/interactions", tags=["User Behavior Tracking"])
def record_interaction(event: InteractionCreate):
    """Records real-time user event (view, click, like, wishlist, cart, purchase)."""
    db = SessionLocal()
    try:
        interaction = log_interaction(
            db=db,
            user_id=event.user_id,
            product_id=event.product_id,
            interaction_type=event.interaction_type
        )
        return {
            "status": "success",
            "interaction_id": interaction.interaction_id,
            "user_id": interaction.user_id,
            "product_id": interaction.product_id,
            "interaction_type": interaction.interaction_type,
            "timestamp": interaction.timestamp
        }
    finally:
        db.close()


@app.post("/wishlist/toggle", tags=["User Behavior Tracking"])
def toggle_wishlist(data: WishlistToggle):
    """Adds or removes an item from user's wishlist."""
    db = SessionLocal()
    try:
        item = db.query(Wishlist).filter(
            Wishlist.user_id == data.user_id,
            Wishlist.product_id == data.product_id
        ).first()

        if item:
            db.delete(item)
            db.commit()
            return {"status": "removed", "user_id": data.user_id, "product_id": data.product_id}
        else:
            new_item = Wishlist(user_id=data.user_id, product_id=data.product_id)
            db.add(new_item)
            # Log as wishlist interaction
            log_interaction(db, data.user_id, data.product_id, "wishlist")
            db.commit()
            return {"status": "added", "user_id": data.user_id, "product_id": data.product_id}
    finally:
        db.close()


@app.get("/users", tags=["Users"])
def get_users():
    """Lists registered test personas and users."""
    db = SessionLocal()
    try:
        users = db.query(User).all()
        return [
            {
                "user_id": u.user_id,
                "username": u.username,
                "email": u.email,
                "preferred_category": u.preferred_category
            }
            for u in users
        ]
    finally:
        db.close()


@app.get("/analytics/metrics", tags=["Analytics"])
def get_analytics_metrics():
    """Returns high-level business KPIs and interaction metrics for Admin Dashboard."""
    db = SessionLocal()
    try:
        total_users = db.query(User).count()
        total_products = db.query(Product).count()
        total_interactions = db.query(Interaction).count()

        views = db.query(Interaction).filter(Interaction.interaction_type == "view").count()
        clicks = db.query(Interaction).filter(Interaction.interaction_type == "click").count()
        carts = db.query(Interaction).filter(Interaction.interaction_type == "cart").count()
        purchases = db.query(Interaction).filter(Interaction.interaction_type == "purchase").count()

        ctr = round((clicks / views * 100), 2) if views > 0 else 0.0
        conversion_rate = round((purchases / clicks * 100), 2) if clicks > 0 else 0.0

        return {
            "kpis": {
                "total_users": total_users,
                "total_products": total_products,
                "total_interactions": total_interactions,
                "total_views": views,
                "total_clicks": clicks,
                "total_carts": carts,
                "total_purchases": purchases,
                "ctr_pct": ctr,
                "conversion_rate_pct": conversion_rate
            }
        }
    finally:
        db.close()


@app.get("/evaluation", tags=["Evaluation & A/B Testing"])
def get_evaluation_and_ab_test():
    """Returns formal recommender metrics (Precision@K, Recall@K, Hit Rate, Diversity, Coverage) and A/B Test Results."""
    metrics = evaluator.evaluate_all_metrics()
    ab_test = evaluator.run_ab_test_simulation()
    return {
        "evaluation_metrics": metrics,
        "ab_test_experiment": ab_test
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
