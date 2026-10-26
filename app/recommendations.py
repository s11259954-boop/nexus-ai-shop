"""
Streamlit AI Recommendations Playground & Explainable Ranking Lab.
Provides interactive controls to adjust ranking weights in real-time,
compare Content-Based vs Collaborative vs Hybrid, and view factor breakdowns.
"""

import os
import sys
import streamlit as st
import pandas as pd
from typing import Callable, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


def render_recommendations_lab(
    hybrid_rec,
    content_rec,
    collab_rec,
    ranking_engine,
    active_user: Dict[str, Any],
    on_select_product: Callable[[str], None],
    on_log_interaction: Callable[[str, str], None]
):
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 5px;">🧠 AI Recommendation Lab & Ranking Tuner</h1>
            <p style="color: #64748b; font-size: 1rem;">
                Experiment with recommendation algorithms, inspect formula calculations, and tune smart ranking weights live.
            </p>
        </div>
    """, unsafe_allow_html=True)

    uid = active_user["user_id"]
    user_name = active_user["username"]

    st.markdown(f"**Target User:** `{user_name} ({uid})` | **Preferred Category:** `{active_user.get('preferred_category', 'All')}`")

    # Formula Display
    st.markdown("""
        <div style="background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="font-size: 0.85rem; font-weight: 700; color: #475569; margin-bottom: 4px;">SMART MULTI-FACTOR RANKING FORMULA:</div>
            <code style="font-size: 1rem; color: #0f172a; font-weight: 600;">
                Final Score = (w_sim × Similarity) + (w_pref × User_Preference) + (w_rating × Rating_Norm) + (w_pop × Popularity_Norm) + (w_rec × Recency)
            </code>
        </div>
    """, unsafe_allow_html=True)

    # Interactive Sliders in 5 Columns
    st.markdown("### 🎛️ Tune Ranking Weights in Real-Time")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        w_sim = st.slider("Similarity (w_sim)", 0.0, 1.0, 0.40, 0.05)
    with c2:
        w_pref = st.slider("User Pref (w_pref)", 0.0, 1.0, 0.30, 0.05)
    with c3:
        w_rating = st.slider("Rating (w_rating)", 0.0, 1.0, 0.15, 0.05)
    with c4:
        w_pop = st.slider("Popularity (w_pop)", 0.0, 1.0, 0.10, 0.05)
    with c5:
        w_rec = st.slider("Recency (w_rec)", 0.0, 1.0, 0.05, 0.05)

    apply_diversity = st.checkbox("Apply Diversity Re-ranking (Penalize repetitive subcategories)", value=True)

    # Update ranking engine
    ranking_engine.update_weights(
        w_sim=w_sim,
        w_pref=w_pref,
        w_rating=w_rating,
        w_pop=w_pop,
        w_rec=w_rec
    )

    st.write("---")

    # Algorithm Comparison Tabs
    tab_hybrid, tab_collab, tab_content = st.tabs([
        "⚡ Hybrid + Smart Multi-Factor Ranking (Recommended)",
        "🤝 Pure Collaborative Filtering",
        "📄 Pure Content-Based Filtering"
    ])

    with tab_hybrid:
        st.markdown("#### Top-N Ranked Recommendations for Current Persona")
        candidates = hybrid_rec.get_personalized_recommendations(uid, top_n=10)
        ranked = ranking_engine.rank_candidates(
            candidates=candidates,
            user_preferred_category=active_user.get("preferred_category"),
            apply_diversity=apply_diversity
        )[:6]

        if not ranked:
            st.info("No recommendations generated.")
        else:
            cols = st.columns(3)
            for i, p in enumerate(ranked):
                with cols[i % 3]:
                    _render_lab_card(p, on_select_product, on_log_interaction)

            # Mathematical Factor Breakdown Table
            st.write("")
            st.markdown("#### 🔬 Detailed Score Breakdown Matrix (Explainable AI)")
            breakdown_rows = []
            for item in ranked:
                breakdown = item.get("factor_breakdown", {})
                breakdown_rows.append({
                    "Product ID": item["product_id"],
                    "Product Name": item["product_name"][:30],
                    "Category": item["category"],
                    "Final Smart Score": round(item.get("smart_rank_score", 0), 4),
                    "Sim Contribution": breakdown.get("similarity_component", 0),
                    "Pref Contribution": breakdown.get("preference_component", 0),
                    "Rating Contribution": breakdown.get("rating_component", 0),
                    "Pop Contribution": breakdown.get("popularity_component", 0),
                    "Recency Contribution": breakdown.get("recency_component", 0),
                })
            st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True)

    with tab_collab:
        st.markdown("#### Pure Collaborative Filtering (User-Item Matrix Factorization)")
        collab_items = collab_rec.get_collaborative_recommendations(uid, top_n=6)
        if not collab_items:
            st.info("No collaborative signals available for this user (e.g. Cold start).")
        else:
            cols = st.columns(3)
            for i, p in enumerate(collab_items):
                with cols[i % 3]:
                    _render_lab_card(p, on_select_product, on_log_interaction)

    with tab_content:
        st.markdown("#### Pure Content-Based Filtering (TF-IDF Cosine Similarity)")
        # Use first product in user preference or catalog
        prod_seed = "PROD_ELEC_001"
        sim_items = content_rec.get_similar_products(prod_seed, top_n=6)
        cols = st.columns(3)
        for i, p in enumerate(sim_items):
            with cols[i % 3]:
                _render_lab_card(p, on_select_product, on_log_interaction)


def _render_lab_card(item: Dict[str, Any], on_select_product: Callable[[str], None], on_log_interaction: Callable[[str, str], None]):
    pid = item["product_id"]
    price = f"₹{int(item['price']):,}"
    score = item.get("smart_rank_score", item.get("similarity_score", 0.5))

    st.markdown(f"""
        <div style="border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; background: white; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.75rem; font-weight: 700; color: #6366f1;">{item.get('brand', '')}</span>
                <span style="background: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 4px; font-weight: 800; font-size: 0.75rem;">
                    Score: {round(float(score), 3)}
                </span>
            </div>
            <img src="{item.get('image_url', '')}" style="width: 100%; height: 130px; object-fit: cover; border-radius: 6px; margin-bottom: 8px;" />
            <div style="font-weight: 700; font-size: 0.95rem; height: 38px; overflow: hidden; line-height: 1.3;">
                {item['product_name']}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin: 6px 0;">
                <span style="font-size: 1.1rem; font-weight: 900; color: #0f172a;">{price}</span>
                <span style="font-size: 0.8rem; color: #b45309; font-weight: 700;">⭐ {item.get('rating', 4.0)}</span>
            </div>
            <div style="font-size: 0.75rem; color: #166534; background: #f0fdf4; padding: 4px 6px; border-radius: 4px; margin-bottom: 8px;">
                💡 {item.get('explanation', 'Recommended by AI Engine')[:75]}...
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Inspect Product", key=f"lab_inspect_{item.get('algorithm', '')}_{pid}", use_container_width=True):
        on_log_interaction(pid, "click")
        on_select_product(pid)
