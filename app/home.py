"""
Streamlit Home Page Module.
Features Hero banner, Category shortcuts, Trending products carousel,
Personalized "Recommended For You" grid, and New Arrivals.
"""

import streamlit as st
from typing import Callable, Dict, Any


def render_home_page(
    hybrid_rec,
    content_rec,
    active_user: Dict[str, Any],
    on_select_product: Callable[[str], None],
    on_log_interaction: Callable[[str, str], None]
):
    # Hero Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%); 
                padding: 40px; border-radius: 16px; color: white; margin-bottom: 30px; box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);">
        <div style="display: inline-block; background: rgba(255,255,255,0.15); padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin-bottom: 12px; letter-spacing: 0.5px;">
            ⚡ AI-POWERED PERSONALIZED SHOPPING
        </div>
        <h1 style="font-size: 2.5rem; font-weight: 800; margin: 0 0 12px 0; color: #ffffff;">
            Discover Products Curated <span style="color: #67e8f9;">Just For You</span>
        </h1>
        <p style="font-size: 1.1rem; opacity: 0.9; max-width: 650px; margin-bottom: 20px; line-height: 1.5;">
            Our hybrid intelligence engine learns from your tastes, browsing behavior, and real-time interactions 
            to surface exact matches, smart budget alternatives, and trending gems.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # User Context Banner
    user_name = active_user.get("username", "Guest Shopper")
    is_cold = hybrid_rec.is_cold_start_user(active_user.get("user_id", ""))
    pref_cat = active_user.get("preferred_category", "All Categories")

    col_u1, col_u2 = st.columns([3, 1])
    with col_u1:
        if is_cold:
            st.info(f"👋 **Welcome {user_name}!** You are in **Cold-Start Mode** (New User). We'll surface our highest-rated and trending products until you explore a few items.")
        else:
            st.success(f"🎯 **Shopping as {user_name}** | Primary Interest: **{pref_cat}** | Dynamic AI personalization is active.")

    st.write("")

    # Category Quick Filter Pills
    st.markdown("### 🏷️ Explore Popular Categories")
    categories = ["All", "Electronics", "Fashion", "Home & Kitchen", "Sports & Fitness", "Beauty & Personal Care", "Books & Stationery"]
    cols = st.columns(len(categories))
    selected_cat = "All"
    for idx, cat in enumerate(categories):
        if cols[idx].button(cat, key=f"cat_btn_{cat}", use_container_width=True):
            st.session_state["filter_category"] = cat
            st.session_state["nav_selection"] = "Search & Catalog"
            st.rerun()

    st.write("---")

    # Section 1: Recommended For You (Personalized Hybrid Feed)
    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px;">
            <h2 style="margin: 0; font-size: 1.6rem; font-weight: 700;">⭐ Recommended For You</h2>
            <span style="color: #6366f1; font-weight: 600; font-size: 0.9rem;">Hybrid AI Engine</span>
        </div>
    """, unsafe_allow_html=True)

    recs = hybrid_rec.get_personalized_recommendations(active_user["user_id"], top_n=4)
    if recs:
        p_cols = st.columns(4)
        for i, prod in enumerate(recs):
            with p_cols[i]:
                _render_product_card(prod, on_select_product, on_log_interaction, key_prefix="rec")
    else:
        st.write("No recommendations yet.")

    st.write("---")

    # Section 2: Trending Products
    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px;">
            <h2 style="margin: 0; font-size: 1.6rem; font-weight: 700;">🔥 Trending Products</h2>
            <span style="color: #ef4444; font-weight: 600; font-size: 0.9rem;">High Engagement Velocity</span>
        </div>
    """, unsafe_allow_html=True)

    trending = hybrid_rec.get_trending_products(top_n=4)
    t_cols = st.columns(4)
    for i, prod in enumerate(trending):
        with t_cols[i]:
            _render_product_card(prod, on_select_product, on_log_interaction, key_prefix="trend")

    st.write("---")

    # Section 3: New Arrivals
    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px;">
            <h2 style="margin: 0; font-size: 1.6rem; font-weight: 700;">🆕 New Arrivals</h2>
            <span style="color: #10b981; font-weight: 600; font-size: 0.9rem;">Fresh In Catalog</span>
        </div>
    """, unsafe_allow_html=True)

    new_arrivals = hybrid_rec.get_new_arrivals(top_n=4)
    n_cols = st.columns(4)
    for i, prod in enumerate(new_arrivals):
        with n_cols[i]:
            _render_product_card(prod, on_select_product, on_log_interaction, key_prefix="new")


def _render_product_card(
    prod: Dict[str, Any],
    on_select_product: Callable[[str], None],
    on_log_interaction: Callable[[str, str], None],
    key_prefix: str = ""
):
    """Renders a responsive, modern e-commerce product card."""
    pid = prod["product_id"]
    price = f"₹{int(prod['price']):,}"
    rating = prod.get("rating", 4.5)
    reviews = prod.get("review_count", 100)
    brand = prod.get("brand", "")
    discount = prod.get("discount_percent", 0)
    img_url = prod.get("image_url", "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600")

    # Container card
    with st.container():
        st.markdown(f"""
            <div style="border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; background: white; margin-bottom: 10px; transition: transform 0.2s;">
                <div style="position: relative; width: 100%; height: 160px; overflow: hidden; border-radius: 8px; margin-bottom: 10px; background: #f8fafc;">
                    <img src="{img_url}" style="width: 100%; height: 100%; object-fit: cover;" alt="{prod['product_name']}" />
                    {f'<div style="position: absolute; top: 8px; left: 8px; background: #ef4444; color: white; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700;">-{discount}% OFF</div>' if discount > 0 else ''}
                </div>
                <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">{brand}</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #1e293b; height: 42px; overflow: hidden; line-height: 1.3; margin: 4px 0 6px 0;">{prod['product_name']}</div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <div style="font-size: 1.1rem; font-weight: 800; color: #0f172a;">{price}</div>
                    <div style="font-size: 0.8rem; background: #fef3c7; color: #b45309; padding: 2px 6px; border-radius: 4px; font-weight: 700;">⭐ {rating}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Explainable AI Pill
        if "explanation" in prod and prod["explanation"]:
            st.caption(f"💡 *{prod['explanation'][:65]}...*")

        col_b1, col_b2 = st.columns([2, 1])
        if col_b1.button("View Details", key=f"{key_prefix}_view_{pid}", use_container_width=True):
            on_log_interaction(pid, "click")
            on_select_product(pid)
        if col_b2.button("❤️", key=f"{key_prefix}_wish_{pid}", use_container_width=True, help="Add to Wishlist"):
            on_log_interaction(pid, "wishlist")
            st.toast(f"Added {prod['product_name'][:18]}... to Wishlist!")
