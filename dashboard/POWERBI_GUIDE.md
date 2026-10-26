# 📊 Power BI Analytics Dashboard Guide

This platform includes a complete **Star Schema** data export pipeline that integrates seamlessly with **Microsoft Power BI Desktop**.

---

## 📁 Exported Data Files

Run the export script anytime to refresh your Power BI datasets:
```bash
python dashboard/export_for_powerbi.py
```
Exported files located in `dashboard/powerbi_data/`:
1. **`Dim_Products.csv`**: Product dimension table (IDs, names, categories, subcategories, brands, prices, ratings).
2. **`Dim_Users.csv`**: User dimension table (IDs, names, emails, preferred categories).
3. **`Fact_Interactions.csv`**: Transaction and event log (Views, Clicks, Likes, Wishlists, Carts, Purchases with timestamps).
4. **`Fact_Ratings.csv`**: Explicit customer ratings and reviews.
5. **`Agg_Product_Performance.csv`**: Pre-aggregated metrics table with calculated CTR and Conversion Rates.

---

## 🛠️ Step-by-Step Power BI Setup

### Step 1: Open Power BI Desktop
1. Click **Get Data** -> **Text/CSV**.
2. Select all files from the `dashboard/powerbi_data/` folder and load them.

### Step 2: Establish Model Relationships (Star Schema)
Power BI will automatically detect or allow you to link the primary and foreign keys:
- `Dim_Products[product_id]` 1-to-Many `Fact_Interactions[product_id]`
- `Dim_Users[user_id]` 1-to-Many `Fact_Interactions[user_id]`
- `Dim_Products[product_id]` 1-to-Many `Fact_Ratings[product_id]`
- `Dim_Users[user_id]` 1-to-Many `Fact_Ratings[user_id]`

### Step 3: Useful DAX Measures
Create the following DAX Measures in Power BI:

- **Total Views**:
  ```dax
  Total Views = CALCULATE(COUNT(Fact_Interactions[interaction_id]), Fact_Interactions[interaction_type] = "view")
  ```

- **Total Clicks**:
  ```dax
  Total Clicks = CALCULATE(COUNT(Fact_Interactions[interaction_id]), Fact_Interactions[interaction_type] = "click")
  ```

- **Total Purchases**:
  ```dax
  Total Purchases = CALCULATE(COUNT(Fact_Interactions[interaction_id]), Fact_Interactions[interaction_type] = "purchase")
  ```

- **Click-Through Rate (CTR %)**:
  ```dax
  CTR % = DIVIDE([Total Clicks], [Total Views], 0) * 100
  ```

- **Conversion Rate %**:
  ```dax
  Conversion Rate % = DIVIDE([Total Purchases], [Total Clicks], 0) * 100
  ```

---

## 📈 Recommended Visualizations to Build

1. **Card Visuals (Top KPIs)**:
   - Total Users
   - Total Products
   - Total Interactions
   - CTR %
   - Conversion Rate %
2. **Funnel Chart**:
   - Stages: Views ➔ Clicks ➔ Wishlists ➔ Carts ➔ Purchases.
3. **Clustered Bar Chart**:
   - Axis: `Dim_Products[category]`
   - Values: `Total Views` vs `Total Purchases`.
4. **Donut Chart**:
   - Legend: `Dim_Products[brand]`
   - Values: `Total Clicks`.
5. **Matrix / Table Visual**:
   - Product Name, Category, Price, Rating, Views, Purchases, CTR %.
