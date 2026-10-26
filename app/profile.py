"""
Streamlit User Profile & Persona Switcher Page Module.
Allows testing different user personas (Tech, Fashion, Cold-Start new user),
viewing browsing history, managing wishlist, and inspecting preference vectors.
"""

import streamlit as st
import pandas as pd
from typing import Callable, Dict, Any, List


def render_profile_page(
    active_user: Dict[str, Any],
    all_users: List[Dict[str, Any]],
    df_products: pd.DataFrame,
    on_switch_user: Callable[[str], None],
    on_select_product: Callable[[str], None],
    on_log_interaction: Callable[[str, str], None]
):
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h1 style="font-size: 2.2rem; font-weight: 800; margin-bottom: 5px;">👤 User Profile & AI Persona Switcher</h1>
            <p style="color: #64748b; font-size: 1rem;">Switch between test buyer personas or test the Cold-Start new user journey in real-time.</p>
        </div>
    """, unsafe_allow_html=True)

    # Persona Switcher Card
    st.markdown("### 🔄 Active Persona")
    col_p1, col_p2 = st.columns([2, 1])

    with col_p1:
        user_options = {u["user_id"]: f"{u['username']} ({u.get('preferred_category', 'No Category') or 'New User - Cold Start'})" for u in all_users}
        current_uid = active_user.get("user_id", "USER_TECH_001")

        selected_uid = st.selectbox(
            "Select Persona to Simulate:",
            options=list(user_options.keys()),
            format_func=lambda x: user_options[x],
            index=list(user_options.keys()).index(current_uid) if current_uid in user_options else 0
        )

        if selected_uid != current_uid:
            on_switch_user(selected_uid)
            st.rerun()

    with col_p2:
        st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Current User ID</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #1e293b;">{active_user['user_id']}</div>
                <div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">Email: {active_user.get('email', 'N/A')}</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("---")

    # Tabs: Browsing History, Wishlist, Preference Profiling
    tab_history, tab_wishlist, tab_fingerprint = st.tabs(["🕘 Browsing & Interaction History", "❤️ Saved Wishlist", "🧬 AI Behavioral Fingerprint"])

    # Load DB session to query user records
    from database.database import SessionLocal, Interaction, Wishlist as WishlistModel
    db = SessionLocal()
    try:
        # Tab 1: History
        with tab_history:
            interactions = (
                db.query(Interaction)
                .filter(Interaction.user_id == active_user["user_id"])
                .order_by(Interaction.timestamp.desc())
                .all()
            )

            if not interactions:
                st.info("ℹ️ No interaction history recorded yet. This is expected for cold-start new users! Browse some products to build history.")
            else:
                st.write(f"Showing **{len(interactions)}** recorded user events:")
                hist_records = []
                for inter in interactions:
                    matching_p = df_products[df_products["product_id"] == inter.product_id]
                    p_name = matching_p.iloc[0]["product_name"] if not matching_p.empty else inter.product_id
                    p_cat = matching_p.iloc[0]["category"] if not matching_p.empty else "N/A"
                    hist_records.append({
                        "Event Type": inter.interaction_type.upper(),
                        "Product Name": p_name,
                        "Category": p_cat,
                        "Timestamp": inter.timestamp.strftime("%Y-%m-%d %H:%M:%S") if inter.timestamp else "N/A"
                    })
                st.dataframe(pd.DataFrame(hist_records), use_container_width=True)

        # Tab 2: Wishlist
        with tab_wishlist:
            wishlist_items = (
                db.query(WishlistModel)
                .filter(WishlistModel.user_id == active_user["user_id"])
                .all()
            )
            if not wishlist_items:
                st.info("❤️ Your wishlist is currently empty. Click the '❤️' button on any product to save it here.")
            else:
                st.write(f"You have **{len(wishlist_items)}** saved items in your wishlist:")
                cols = st.columns(3)
                for idx, w in enumerate(wishlist_items):
                    match_p = df_products[df_products["product_id"] == w.product_id]
                    if not match_p.empty:
                        p_row = match_p.iloc[0].to_dict()
                        with cols[idx % 3]:
                            st.markdown(f"""
                                <div style="border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; margin-bottom: 10px; background: white;">
                                    <img src="{p_row['image_url']}" style="width: 100%; height: 140px; object-fit: cover; border-radius: 6px; margin-bottom: 8px;" />
                                    <div style="font-weight: 700; font-size: 0.95rem; height: 40px; overflow: hidden;">{p_row['product_name']}</div>
                                    <div style="font-weight: 800; font-size: 1.1rem; color: #0f172a; margin: 4px 0;">₹{int(p_row['price']):,}</div>
                                </div>
                            """, unsafe_allow_html=True)
                            cw1, cw2 = st.columns(2)
                            if cw1.button("View", key=f"wish_view_{p_row['product_id']}", use_container_width=True):
                                on_select_product(p_row["product_id"])
                            if cw2.button("Buy Now", key=f"wish_buy_{p_row['product_id']}", use_container_width=True):
                                on_log_interaction(p_row["product_id"], "purchase")
                                st.success("Purchased!")
                                st.rerun()

        # Tab 3: Fingerprint
        with tab_fingerprint:
            st.markdown("#### 🧬 User Preference Vectors & Cold-Start Analysis")
            user_inter_count = db.query(Interaction).filter(Interaction.user_id == active_user["user_id"]).count()

            col_f1, col_f2 = st.columns([1, 1])
            with col_f1:
                st.metric("Total User Interactions Recorded", user_inter_count)
                status = "Cold-Start (Sparse History)" if user_inter_count <= 1 else "Active Trained Profile"
                st.metric("Engine Operational Mode", status)

            with col_f2:
                # Distribution of interactions by category for active user
                query = f"""
                SELECT p.category, COUNT(i.interaction_id) as count
                FROM interactions i
                JOIN products p ON i.product_id = p.product_id
                WHERE i.user_id = '{active_user["user_id"]}'
                GROUP BY p.category
                """
                df_user_cats = pd.read_sql(query, db.bind)
                if not df_user_cats.empty:
                    import plotly.express as px
                    fig = px.pie(df_user_cats, values="count", names="category", title="User Category Affinity Weight")
                    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No interactions available yet to compute category distribution.")

    finally:
        db.close()
