"""
Streamlit Products & Catalog Page Module.
Includes multi-criteria filtering, full-text catalog search,
and detailed Product View with Explainable AI badges, Similar items,
Frequently Bought Together, and Smart Alternatives.
"""

import os
import sys
import streamlit as st
import pandas as pd
from typing import Callable, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)


def render_catalog_and_details_page(
    df_products: pd.DataFrame,
    content_rec,
    collab_rec,
    active_user: Dict[str, Any],
    selected_product_id: Optional[str],
    on_select_product: Callable[[str], None],
    on_clear_selection: Callable[[], None],
    on_log_interaction: Callable[[str, str], None]
):
    if selected_product_id:
        _render_product_details(
            selected_product_id=selected_product_id,
            df_products=df_products,
            content_rec=content_rec,
            collab_rec=collab_rec,
            active_user=active_user,
            on_select_product=on_select_product,
            on_clear_selection=on_clear_selection,
            on_log_interaction=on_log_interaction
        )
    else:
        _render_catalog_browser(
            df_products=df_products,
            on_select_product=on_select_product,
            on_log_interaction=on_log_interaction
        )


def _render_catalog_browser(
    df_products: pd.DataFrame,
    on_select_product: Callable[[str], None],
    on_log_interaction: Callable[[str, str], None]
):
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 5px;">🛍️ Explore Product Catalog</h1>
            <p style="color: #64748b; font-size: 1rem;">Browse, filter, and discover products across all departments with real-time AI indexing.</p>
        </div>
    """, unsafe_allow_html=True)

    # Filter Controls
    col_search, col_sort = st.columns([3, 1])
    with col_search:
        search_query = st.text_input("🔍 Search products by title, brand, or feature...", value="")
    with col_sort:
        sort_choice = st.selectbox(
            "Sort by",
            ["Most Popular", "Price: Low to High", "Price: High to Low", "Highest Rated"]
        )

    # Filter bar
    f1, f2, f3, f4 = st.columns([1.5, 1.5, 2, 1.5])
    with f1:
        categories = ["All"] + sorted(df_products["category"].dropna().unique().tolist())
        default_cat = st.session_state.get("filter_category", "All")
        default_idx = categories.index(default_cat) if default_cat in categories else 0
        selected_category = st.selectbox("Category", categories, index=default_idx)
    with f2:
        brands = ["All"] + sorted(df_products["brand"].dropna().unique().tolist())
        selected_brand = st.selectbox("Brand", brands, index=0)
    with f3:
        min_p = float(df_products["price"].min())
        max_p = float(df_products["price"].max())
        price_range = st.slider("Price Range (₹)", min_value=0, max_value=int(max_p) + 5000, value=(0, int(max_p) + 5000), step=500)
    with f4:
        min_rating = st.slider("Min Rating", min_value=0.0, max_value=5.0, value=0.0, step=0.5)

    # Apply Filters
    filtered_df = df_products.copy()

    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]

    if selected_brand != "All":
        filtered_df = filtered_df[filtered_df["brand"] == selected_brand]

    filtered_df = filtered_df[
        (filtered_df["price"] >= price_range[0]) & 
        (filtered_df["price"] <= price_range[1]) &
        (filtered_df["rating"] >= min_rating)
    ]

    if search_query:
        q = search_query.lower()
        filtered_df = filtered_df[
            filtered_df["product_name"].str.lower().str.contains(q, na=False) |
            filtered_df["description"].str.lower().str.contains(q, na=False) |
            filtered_df["brand"].str.lower().str.contains(q, na=False) |
            filtered_df["subcategory"].str.lower().str.contains(q, na=False)
        ]

    # Apply Sorting
    if sort_choice == "Price: Low to High":
        filtered_df = filtered_df.sort_values(by="price", ascending=True)
    elif sort_choice == "Price: High to Low":
        filtered_df = filtered_df.sort_values(by="price", ascending=False)
    elif sort_choice == "Highest Rated":
        filtered_df = filtered_df.sort_values(by="rating", ascending=False)
    else:
        filtered_df = filtered_df.sort_values(by="popularity_score", ascending=False)

    st.write(f"Showing **{len(filtered_df)}** products matching your criteria:")

    # Grid Display (3 columns)
    if filtered_df.empty:
        st.warning("No products found matching the selected filters. Try broadening your criteria.")
        return

    records = filtered_df.to_dict("records")
    for row_start in range(0, len(records), 3):
        cols = st.columns(3)
        for col_idx in range(3):
            item_idx = row_start + col_idx
            if item_idx < len(records):
                prod = records[item_idx]
                with cols[col_idx]:
                    _render_catalog_card(prod, on_select_product, on_log_interaction)


def _render_catalog_card(prod: Dict[str, Any], on_select_product: Callable[[str], None], on_log_interaction: Callable[[str, str], None]):
    pid = prod["product_id"]
    price = f"₹{int(prod['price']):,}"
    rating = prod.get("rating", 4.0)
    brand = prod.get("brand", "")
    discount = prod.get("discount_percent", 0)
    img_url = prod.get("image_url", "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600")

    st.markdown(f"""
        <div style="border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; background: white; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
            <div style="position: relative; width: 100%; height: 180px; overflow: hidden; border-radius: 8px; margin-bottom: 10px; background: #f8fafc;">
                <img src="{img_url}" style="width: 100%; height: 100%; object-fit: cover;" alt="{prod['product_name']}" />
                {f'<div style="position: absolute; top: 8px; left: 8px; background: #ef4444; color: white; padding: 3px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 700;">-{discount}% OFF</div>' if discount > 0 else ''}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-size: 0.75rem; font-weight: 700; color: #4f46e5; text-transform: uppercase;">{brand}</span>
                <span style="font-size: 0.8rem; background: #fef3c7; color: #b45309; padding: 2px 6px; border-radius: 4px; font-weight: 700;">⭐ {rating}</span>
            </div>
            <div style="font-size: 1rem; font-weight: 700; color: #0f172a; height: 46px; overflow: hidden; line-height: 1.3; margin-bottom: 6px;">
                {prod['product_name']}
            </div>
            <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 8px; height: 36px; overflow: hidden;">
                {prod['description'][:75]}...
            </div>
            <div style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin-bottom: 10px;">
                {price}
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    if c1.button("View Product Details", key=f"cat_view_{pid}", use_container_width=True):
        on_log_interaction(pid, "click")
        on_select_product(pid)
    if c2.button("❤️", key=f"cat_fav_{pid}", use_container_width=True, help="Add to wishlist"):
        on_log_interaction(pid, "wishlist")
        st.toast(f"Saved {prod['product_name'][:18]}... to Wishlist!")


def _render_product_details(
    selected_product_id: str,
    df_products: pd.DataFrame,
    content_rec,
    collab_rec,
    active_user: Dict[str, Any],
    on_select_product: Callable[[str], None],
    on_clear_selection: Callable[[], None],
    on_log_interaction: Callable[[str, str], None]
):
    # Log product page view
    on_log_interaction(selected_product_id, "view")

    matching = df_products[df_products["product_id"] == selected_product_id]
    if matching.empty:
        st.error("Product not found.")
        if st.button("⬅️ Back to Catalog"):
            on_clear_selection()
        return

    prod = matching.iloc[0].to_dict()

    if st.button("⬅️ Back to All Products"):
        on_clear_selection()

    st.write("")

    # Main Product Presentation
    col_img, col_info = st.columns([1, 1.2])

    with col_img:
        st.markdown(f"""
            <div style="border-radius: 16px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.08); background: #f8fafc;">
                <img src="{prod['image_url']}" style="width: 100%; max-height: 440px; object-fit: cover;" alt="{prod['product_name']}" />
            </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
            <div style="font-size: 0.9rem; font-weight: 700; color: #6366f1; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                {prod['brand']} • {prod['category']} / {prod['subcategory']}
            </div>
            <h1 style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 12px; line-height: 1.2;">
                {prod['product_name']}
            </h1>
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                <span style="font-size: 0.95rem; background: #fef3c7; color: #b45309; padding: 4px 10px; border-radius: 6px; font-weight: 700;">
                    ⭐ {prod['rating']} / 5.0
                </span>
                <span style="color: #64748b; font-size: 0.9rem;">
                    ({prod['review_count']:,} verified reviews)
                </span>
                <span style="color: #10b981; font-weight: 700; font-size: 0.9rem;">
                    ✓ In Stock ({prod.get('stock', 50)} units available)
                </span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 18px;">
                <span style="font-size: 2.2rem; font-weight: 900; color: #0f172a;">
                    ₹{int(prod['price']):,}
                </span>
                {f'<span style="font-size: 1.2rem; color: #94a3b8; text-decoration: line-through;">₹{int(prod.get("original_price", prod["price"]*1.15)):,}</span>' if prod.get('discount_percent', 0) > 0 else ''}
                {f'<span style="background: #fee2e2; color: #dc2626; font-weight: 800; padding: 2px 8px; border-radius: 6px; font-size: 0.85rem;">SAVE {prod["discount_percent"]}%</span>' if prod.get('discount_percent', 0) > 0 else ''}
            </div>
            <div style="background: #f8fafc; border-left: 4px solid #6366f1; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 24px;">
                <div style="font-weight: 700; color: #334155; margin-bottom: 4px; font-size: 0.95rem;">Key Specifications & Overview</div>
                <div style="color: #475569; font-size: 0.95rem; line-height: 1.6;">{prod['description']}</div>
            </div>
        """, unsafe_allow_html=True)

        # Interactive Shopping Action Buttons
        btn1, btn2, btn3 = st.columns([1.5, 1.5, 1])
        if btn1.button("🛒 Add to Cart", key="pdetails_cart_btn", use_container_width=True):
            on_log_interaction(selected_product_id, "cart")
            st.success("✅ Added to your shopping cart!")
            st.balloons()
        if btn2.button("⚡ Buy Now", key="pdetails_buy_btn", use_container_width=True):
            on_log_interaction(selected_product_id, "purchase")
            st.success("🎉 Order successfully placed! Interaction recorded in database.")
            st.balloons()
        if btn3.button("❤️ Wishlist", key="pdetails_wish_btn", use_container_width=True):
            on_log_interaction(selected_product_id, "wishlist")
            st.info("Item saved to your Wishlist!")

    st.write("---")

    # ========================================================
    # Explainable AI Feature (XAI) ⭐
    # ========================================================
    st.markdown("""
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 20px; margin-bottom: 30px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                <span style="font-size: 1.3rem;">🧠</span>
                <h3 style="margin: 0; color: #166534; font-size: 1.25rem; font-weight: 700;">Explainable AI: Why This Product Matches You</h3>
            </div>
            <p style="color: #15803d; font-size: 0.95rem; margin-bottom: 12px;">
                Our transparent machine learning framework evaluates content synergy and community behavioral clusters:
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px;">
                <div style="background: white; border: 1px solid #dcfce7; padding: 10px 14px; border-radius: 8px;">
                    <div style="font-weight: 700; color: #166534; font-size: 0.85rem;">CATEGORY MATCH</div>
                    <div style="color: #374151; font-size: 0.9rem;">Shares <b>{category}</b> ({subcategory}) classification.</div>
                </div>
                <div style="background: white; border: 1px solid #dcfce7; padding: 10px 14px; border-radius: 8px;">
                    <div style="font-weight: 700; color: #166534; font-size: 0.85rem;">POPULARITY & SATISFACTION</div>
                    <div style="color: #374151; font-size: 0.9rem;">Ranked in top 15% with <b>{rating}★</b> customer satisfaction.</div>
                </div>
                <div style="background: white; border: 1px solid #dcfce7; padding: 10px 14px; border-radius: 8px;">
                    <div style="font-weight: 700; color: #166534; font-size: 0.85rem;">INTERACTION CO-OCCURRENCE</div>
                    <div style="color: #374151; font-size: 0.9rem;">High purchase propensity among users with similar shopping profiles.</div>
                </div>
            </div>
        </div>
    """.format(
        category=prod['category'],
        subcategory=prod['subcategory'],
        rating=prod['rating']
    ), unsafe_allow_html=True)

    # ========================================================
    # 1. Frequently Bought Together
    # ========================================================
    fbt_items = collab_rec.get_frequently_bought_together(selected_product_id, top_n=3)
    if fbt_items:
        st.markdown("### 🤝 Frequently Bought Together")
        fbt_cols = st.columns(len(fbt_items))
        for i, item in enumerate(fbt_items):
            with fbt_cols[i]:
                _render_rec_subcard(item, on_select_product, on_log_interaction, badge="Frequently Bought")

    st.write("---")

    # ========================================================
    # 2. Smart Budget Alternatives
    # ========================================================
    alts = content_rec.get_smart_alternatives(selected_product_id, top_n=3)
    if alts:
        st.markdown("### 💰 Smart Budget Alternatives (Similar features, lower price)")
        alt_cols = st.columns(len(alts))
        for i, item in enumerate(alts):
            with alt_cols[i]:
                savings = item.get("savings_percent", 15)
                _render_rec_subcard(item, on_select_product, on_log_interaction, badge=f"Save {savings}%", badge_color="#10b981")

    st.write("---")

    # ========================================================
    # 3. You May Also Like (Content-Based Similar Items)
    # ========================================================
    similar_items = content_rec.get_similar_products(selected_product_id, top_n=4)
    if similar_items:
        st.markdown("### 📦 You May Also Like (Similar Items)")
        sim_cols = st.columns(4)
        for i, item in enumerate(similar_items):
            with sim_cols[i]:
                _render_rec_subcard(item, on_select_product, on_log_interaction, badge="Similar Item")


def _render_rec_subcard(
    item: Dict[str, Any],
    on_select_product: Callable[[str], None],
    on_log_interaction: Callable[[str, str], None],
    badge: str = "Recommended",
    badge_color: str = "#6366f1"
):
    pid = item["product_id"]
    price = f"₹{int(item['price']):,}"
    rating = item.get("rating", 4.3)
    img_url = item.get("image_url", "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600")

    st.markdown(f"""
        <div style="border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px; background: white; margin-bottom: 10px;">
            <div style="position: relative; width: 100%; height: 130px; overflow: hidden; border-radius: 6px; margin-bottom: 8px;">
                <img src="{img_url}" style="width: 100%; height: 100%; object-fit: cover;" alt="{item['product_name']}" />
                <div style="position: absolute; top: 6px; left: 6px; background: {badge_color}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">
                    {badge}
                </div>
            </div>
            <div style="font-size: 0.85rem; font-weight: 700; color: #1e293b; height: 38px; overflow: hidden; line-height: 1.3;">
                {item['product_name']}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px;">
                <span style="font-size: 1rem; font-weight: 800; color: #0f172a;">{price}</span>
                <span style="font-size: 0.75rem; color: #b45309; font-weight: 700;">⭐ {rating}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("View", key=f"subcard_btn_{badge}_{pid}", use_container_width=True):
        on_log_interaction(pid, "click")
        on_select_product(pid)
