"""
Main Streamlit Application for AI-Powered Personalized Product Recommendation & Analytics Platform.
Coordinates navigation between Home, Catalog & Details, AI Lab, User Profile, and Admin Dashboard.
"""

import os
import sys
import requests
import streamlit as st
import pandas as pd

# Add workspace root to PYTHONPATH and ensure 'app' resolves to package, not app.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))

while APP_DIR in sys.path:
    sys.path.remove(APP_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
else:
    sys.path.remove(BASE_DIR)
    sys.path.insert(0, BASE_DIR)

if "app" in sys.modules and getattr(sys.modules["app"], "__file__", "").endswith("app.py"):
    del sys.modules["app"]

from database.database import SessionLocal, Product, User, log_interaction
from src.hybrid_recommender import HybridRecommender
from src.content_recommender import ContentRecommender
from src.collaborative_recommender import CollaborativeRecommender
from src.ranking import SmartRankingEngine

try:
    from app.home import render_home_page
    from app.products import render_catalog_and_details_page
    from app.profile import render_profile_page
    from app.recommendations import render_recommendations_lab
    from app.admin import render_admin_dashboard
except (ImportError, ModuleNotFoundError):
    from home import render_home_page
    from products import render_catalog_and_details_page
    from profile import render_profile_page
    from recommendations import render_recommendations_lab
    from admin import render_admin_dashboard

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI E-Commerce Recommendation & Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Global CSS for modern UI design
st.markdown("""
<style>
    /* Global styling */
    .main {
        background-color: #fcfcfd;
    }
    /* Buttons styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    /* Sidebar header */
    .sidebar-header {
        font-size: 1.25rem;
        font-weight: 800;
        color: #1e1b4b;
        margin-bottom: 20px;
    }
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ml_engines():
    """Initializes and caches recommendation models."""
    from src.preprocessing import ensure_dataset_exists
    ensure_dataset_exists()
    hybrid = HybridRecommender()
    content = ContentRecommender()
    collab = CollaborativeRecommender()
    ranking = SmartRankingEngine()
    return hybrid, content, collab, ranking


@st.cache_data(ttl=60)
def load_catalog_and_users():
    """Loads product catalog and registered users from SQLite/MySQL."""
    from database.database import ensure_db_initialized
    ensure_db_initialized()
    db = SessionLocal()
    try:
        products = db.query(Product).all()
        users = db.query(User).all()

        if not products or not users:
            from data.seed_data import seed_database_and_export_csv
            seed_database_and_export_csv()
            products = db.query(Product).all()
            users = db.query(User).all()

        prod_dicts = [
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
                "image_url": p.image_url,
                "stock": p.stock,
                "original_price": round(p.price / (1 - (p.discount_percent / 100)), 2) if p.discount_percent > 0 else p.price
            }
            for p in products
        ]

        user_dicts = [
            {
                "user_id": u.user_id,
                "username": u.username,
                "email": u.email,
                "preferred_category": u.preferred_category
            }
            for u in users
        ]
        return pd.DataFrame(prod_dicts), user_dicts
    finally:
        db.close()


def main():
    hybrid_rec, content_rec, collab_rec, ranking_engine = load_ml_engines()
    df_products, all_users = load_catalog_and_users()

    # Session State Initialization
    if "active_user_id" not in st.session_state:
        st.session_state["active_user_id"] = "USER_TECH_001"  # Default: Arjun (Tech Lover)
    if "selected_product_id" not in st.session_state:
        st.session_state["selected_product_id"] = None
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = "Home"

    # Find active user object
    user_match = [u for u in all_users if u["user_id"] == st.session_state["active_user_id"]]
    active_user = user_match[0] if user_match else all_users[0]

    # Interaction Logger Callback
    def handle_log_interaction(product_id: str, interaction_type: str):
        db = SessionLocal()
        try:
            log_interaction(
                db=db,
                user_id=active_user["user_id"],
                product_id=product_id,
                interaction_type=interaction_type
            )
            # Notify recommender of real-time update
            hybrid_rec.reload_interactions()
        finally:
            db.close()

    def handle_select_product(product_id: str):
        st.session_state["selected_product_id"] = product_id
        st.session_state["nav_selection"] = "Search & Catalog"
        st.rerun()

    def handle_clear_selection():
        st.session_state["selected_product_id"] = None
        st.rerun()

    def handle_switch_user(new_uid: str):
        st.session_state["active_user_id"] = new_uid
        st.toast(f"Switched persona to {new_uid}")

    # ========================================================
    # Sidebar Navigation & Persona Switcher
    # ========================================================
    with st.sidebar:
        st.markdown("""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                <span style="font-size: 2rem;">🛍️</span>
                <div>
                    <div style="font-weight: 800; font-size: 1.15rem; color: #1e1b4b; line-height: 1.2;">NEXUS AI SHOP</div>
                    <div style="font-size: 0.75rem; color: #6366f1; font-weight: 700;">RECOMMENDER PLATFORM</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        nav_options = [
            "🏠 Home",
            "🛍️ Search & Catalog",
            "🧠 AI Recommendation Lab",
            "👤 User Profile & History",
            "📊 Admin Dashboard"
        ]

        # Match navigation selection
        current_nav_index = 0
        nav_clean_map = {
            "🏠 Home": "Home",
            "🛍️ Search & Catalog": "Search & Catalog",
            "🧠 AI Recommendation Lab": "AI Recommendation Lab",
            "👤 User Profile & History": "User Profile & History",
            "📊 Admin Dashboard": "Admin Dashboard"
        }
        reverse_map = {v: k for k, v in nav_clean_map.items()}
        current_label = reverse_map.get(st.session_state["nav_selection"], "🏠 Home")
        if current_label in nav_options:
            current_nav_index = nav_options.index(current_label)

        chosen_nav = st.radio("Navigation", nav_options, index=current_nav_index)
        st.session_state["nav_selection"] = nav_clean_map[chosen_nav]

        st.markdown("---")

        # Active Persona Quick-Switcher in Sidebar
        st.markdown("##### 👤 Active Persona")
        persona_dict = {
            u["user_id"]: f"{u['username'].split()[0]} ({u.get('preferred_category', 'Cold-Start') or 'Cold-Start'})"
            for u in all_users[:6]
        }
        quick_uid = st.selectbox(
            "Switch User:",
            options=list(persona_dict.keys()),
            format_func=lambda x: persona_dict[x],
            index=list(persona_dict.keys()).index(st.session_state["active_user_id"]) if st.session_state["active_user_id"] in persona_dict else 0,
            key="sb_user_switch"
        )
        if quick_uid != st.session_state["active_user_id"]:
            st.session_state["active_user_id"] = quick_uid
            st.rerun()

        # Check FastAPI status
        st.markdown("---")
        try:
            res = requests.get("http://localhost:8000/", timeout=0.8)
            if res.status_code == 200:
                st.markdown("🟢 **FastAPI Backend:** `Online`")
            else:
                st.markdown("🟡 **FastAPI Backend:** `Responding`")
        except Exception:
            st.markdown("⚪ **FastAPI Backend:** `Offline (Direct Mode)`")
            st.caption("Start with: `uvicorn api.main:app --reload`")

    # ========================================================
    # Main Page Router
    # ========================================================
    nav = st.session_state["nav_selection"]

    if nav == "Home":
        render_home_page(
            hybrid_rec=hybrid_rec,
            content_rec=content_rec,
            active_user=active_user,
            on_select_product=handle_select_product,
            on_log_interaction=handle_log_interaction
        )

    elif nav == "Search & Catalog":
        render_catalog_and_details_page(
            df_products=df_products,
            content_rec=content_rec,
            collab_rec=collab_rec,
            active_user=active_user,
            selected_product_id=st.session_state["selected_product_id"],
            on_select_product=handle_select_product,
            on_clear_selection=handle_clear_selection,
            on_log_interaction=handle_log_interaction
        )

    elif nav == "AI Recommendation Lab":
        render_recommendations_lab(
            hybrid_rec=hybrid_rec,
            content_rec=content_rec,
            collab_rec=collab_rec,
            ranking_engine=ranking_engine,
            active_user=active_user,
            on_select_product=handle_select_product,
            on_log_interaction=handle_log_interaction
        )

    elif nav == "User Profile & History":
        render_profile_page(
            active_user=active_user,
            all_users=all_users,
            df_products=df_products,
            on_switch_user=handle_switch_user,
            on_select_product=handle_select_product,
            on_log_interaction=handle_log_interaction
        )

    elif nav == "Admin Dashboard":
        render_admin_dashboard()


if __name__ == "__main__":
    main()
