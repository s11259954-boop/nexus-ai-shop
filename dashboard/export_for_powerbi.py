"""
Power BI Data Exporter.
Exports normalized relational CSV files and dimensional star-schema datasets
ready to be imported directly into Microsoft Power BI Desktop.
"""

import os
import sys
import pandas as pd
from sqlalchemy import text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from database.database import SessionLocal, Product, User, Interaction, Rating

EXPORT_DIR = os.path.join(BASE_DIR, "dashboard", "powerbi_data")
os.makedirs(EXPORT_DIR, exist_ok=True)


def export_powerbi_tables():
    """Extracts tables from DB and saves as clean, structured CSV files for Power BI import."""
    db = SessionLocal()
    try:
        # Dim_Products
        df_prod = pd.read_sql(text("SELECT * FROM products"), db.bind)
        df_prod.to_csv(os.path.join(EXPORT_DIR, "Dim_Products.csv"), index=False)

        # Dim_Users
        df_users = pd.read_sql(text("SELECT * FROM users"), db.bind)
        df_users.to_csv(os.path.join(EXPORT_DIR, "Dim_Users.csv"), index=False)

        # Fact_Interactions
        df_inter = pd.read_sql(text("SELECT * FROM interactions"), db.bind)
        df_inter.to_csv(os.path.join(EXPORT_DIR, "Fact_Interactions.csv"), index=False)

        # Fact_Ratings
        df_ratings = pd.read_sql(text("SELECT * FROM ratings"), db.bind)
        df_ratings.to_csv(os.path.join(EXPORT_DIR, "Fact_Ratings.csv"), index=False)

        # Pre-calculated aggregated view for immediate reporting
        agg_query = """
        SELECT 
            p.category,
            p.brand,
            p.product_id,
            p.product_name,
            p.price,
            p.rating,
            p.popularity_score,
            SUM(CASE WHEN i.interaction_type = 'view' THEN 1 ELSE 0 END) as views_count,
            SUM(CASE WHEN i.interaction_type = 'click' THEN 1 ELSE 0 END) as clicks_count,
            SUM(CASE WHEN i.interaction_type = 'wishlist' THEN 1 ELSE 0 END) as wishlist_count,
            SUM(CASE WHEN i.interaction_type = 'cart' THEN 1 ELSE 0 END) as cart_count,
            SUM(CASE WHEN i.interaction_type = 'purchase' THEN 1 ELSE 0 END) as purchase_count
        FROM products p
        LEFT JOIN interactions i ON p.product_id = i.product_id
        GROUP BY p.product_id
        """
        df_agg = pd.read_sql(text(agg_query), db.bind)
        df_agg["ctr_pct"] = round((df_agg["clicks_count"] / df_agg["views_count"]) * 100, 2).fillna(0)
        df_agg["conversion_rate_pct"] = round((df_agg["purchase_count"] / df_agg["clicks_count"]) * 100, 2).fillna(0)
        df_agg.to_csv(os.path.join(EXPORT_DIR, "Agg_Product_Performance.csv"), index=False)

        print(f"✅ Power BI datasets successfully exported to: {EXPORT_DIR}")
        print("   - Dim_Products.csv")
        print("   - Dim_Users.csv")
        print("   - Fact_Interactions.csv")
        print("   - Fact_Ratings.csv")
        print("   - Agg_Product_Performance.csv")
    finally:
        db.close()


if __name__ == "__main__":
    export_powerbi_tables()
