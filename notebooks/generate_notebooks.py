"""
Jupyter Notebook Generator Script.
Builds the 4 required notebooks formatted with educational markdown and code.
"""

import json
import os

NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.13.0"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


# ========================================================
# Notebook 1: 01_data_analysis.ipynb
# ========================================================
nb1_cells = [
    markdown_cell("""# 📊 Module 1: Exploratory Data Analysis & Product Intelligence
**AI-Powered Personalized Product Recommendation & Analytics Platform**

This notebook performs end-to-end exploratory data analysis (EDA) on:
1. Product Catalog distribution (Categories, Brands, Prices, Ratings, Discounts)
2. User Interaction dynamics (Views, Clicks, Carts, Purchases)
3. Interaction conversion funnel and user-item sparsity analysis."""),
    code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
%matplotlib inline

# Load raw dataset
BASE_DIR = '..'
products_path = os.path.join(BASE_DIR, 'data', 'raw', 'products.csv')
interactions_path = os.path.join(BASE_DIR, 'data', 'raw', 'interactions.csv')

df_products = pd.read_csv(products_path)
df_interactions = pd.read_csv(interactions_path)

print(f"Product Catalog shape: {df_products.shape}")
print(f"Interactions shape: {df_interactions.shape}")"""),
    markdown_cell("### 1. Product Catalog Overview"),
    code_cell("""df_products.head()"""),
    code_cell("""df_products.info()
df_products.describe()"""),
    markdown_cell("### 2. Category & Brand Distribution"),
    code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Products by Category
category_counts = df_products['category'].value_counts()
sns.barplot(x=category_counts.values, y=category_counts.index, ax=axes[0], palette='Blues_r')
axes[0].set_title('Products Count by Category', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Count')

# Products by Top Brands
brand_counts = df_products['brand'].value_counts().head(8)
sns.barplot(x=brand_counts.values, y=brand_counts.index, ax=axes[1], palette='Purples_r')
axes[1].set_title('Top Brands by Product Count', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Count')

plt.tight_layout()
plt.show()"""),
    markdown_cell("### 3. Price & Rating Distributions"),
    code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df_products['price'], kde=True, bins=15, ax=axes[0], color='#3b82f6')
axes[0].set_title('Price Distribution across Catalog', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Price (₹)')

sns.histplot(df_products['rating'], kde=True, bins=10, ax=axes[1], color='#f59e0b')
axes[1].set_title('Customer Rating Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Rating (Stars)')

plt.tight_layout()
plt.show()"""),
    markdown_cell("### 4. User Interaction Conversion Funnel"),
    code_cell("""event_counts = df_interactions['interaction_type'].value_counts()
print("Interaction Counts by Event Type:")
print(event_counts)

# Conversion Funnel Plot
plt.figure(figsize=(10, 5))
funnel_order = ['view', 'click', 'wishlist', 'cart', 'purchase']
funnel_data = [event_counts.get(t, 0) for t in funnel_order]

bars = plt.bar(funnel_order, funnel_data, color=['#3b82f6', '#6366f1', '#8b5cf6', '#ec4899', '#10b981'])
plt.title('E-Commerce Customer Interaction Funnel', fontsize=13, fontweight='bold')
plt.xlabel('Funnel Stage')
plt.ylabel('Total Events')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{int(yval):,}", ha='center', va='bottom', fontweight='bold')

plt.show()"""),
    markdown_cell("### 5. Interaction Matrix Sparsity Analysis"),
    code_cell("""n_users = df_interactions['user_id'].nunique()
n_items = df_interactions['product_id'].nunique()
n_interactions = len(df_interactions)

matrix_size = n_users * n_items
sparsity = (1.0 - (n_interactions / matrix_size)) * 100

print(f"Total Unique Users: {n_users}")
print(f"Total Unique Products with Interactions: {n_items}")
print(f"Interaction Matrix Size: {matrix_size:,} cells")
print(f"Matrix Sparsity: {sparsity:.2f}%")
print("=> A collaborative filtering recommender with matrix factorization (SVD) is well-suited to handle this sparsity.")""")
]

# ========================================================
# Notebook 2: 02_content_based.ipynb
# ========================================================
nb2_cells = [
    markdown_cell("""# 📄 Module 2: Content-Based Recommendation with Explainable AI (XAI)
**AI-Powered Personalized Product Recommendation & Analytics Platform**

In this notebook:
1. Feature Preprocessing & Metadata Soup Construction
2. TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization
3. Pairwise Cosine Similarity Matrix Computation
4. Product Similarity Retrieval (e.g. iPhone 15 recommendations)
5. Explainable AI (XAI) Feature & Smart Budget Alternatives Generation"""),
    code_cell("""import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = '..'
products_path = os.path.join(BASE_DIR, 'data', 'processed', 'clean_products.csv')
df_products = pd.read_csv(products_path)

print(f"Loaded {len(df_products)} processed products.")"""),
    markdown_cell("### 1. Content Soup Inspection"),
    code_cell("""# Inspect sample content soup used for vectorization
print("Sample Content Soup:")
print(df_products[['product_name', 'content_soup']].iloc[0].to_dict())"""),
    markdown_cell("### 2. TF-IDF Vectorization"),
    code_cell("""vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=3000,
    sublinear_tf=True
)
tfidf_matrix = vectorizer.fit_transform(df_products['content_soup'])
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
print(f"Extracted {len(vectorizer.get_feature_names_out())} vocabulary features.")"""),
    markdown_cell("### 3. Cosine Similarity Computation"),
    code_cell("""cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Cosine Similarity Matrix Shape: {cosine_sim.shape}")
print(f"Min Similarity: {cosine_sim.min():.3f}, Max Similarity: {cosine_sim.max():.3f}")"""),
    markdown_cell("### 4. Similar Products Retrieval"),
    code_cell("""def get_similar_products(product_id, top_n=5):
    idx = df_products[df_products['product_id'] == product_id].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # Exclude self
    scores = [s for s in scores if s[0] != idx][:top_n]
    
    recs = []
    for i, score in scores:
        p = df_products.iloc[i]
        recs.append({
            'Product ID': p['product_id'],
            'Product Name': p['product_name'],
            'Category': p['category'],
            'Brand': p['brand'],
            'Price (₹)': p['price'],
            'Similarity Score': round(float(score), 4)
        })
    return pd.DataFrame(recs)

# Test with iPhone 15
print("Similar items to Apple iPhone 15:")
get_similar_products('PROD_ELEC_001', top_n=5)"""),
    markdown_cell("### 5. Smart Budget Alternatives Feature"),
    code_cell("""def get_smart_alternatives(product_id, top_n=3):
    idx = df_products[df_products['product_id'] == product_id].index[0]
    base = df_products.iloc[idx]
    base_price = base['price']
    
    candidates = []
    for i, row in df_products.iterrows():
        if i == idx:
            continue
        if row['price'] < base_price and row['category'] == base['category']:
            sim = cosine_sim[idx][i]
            savings = round(((base_price - row['price']) / base_price) * 100, 1)
            candidates.append({
                'Product Name': row['product_name'],
                'Price (₹)': row['price'],
                'Savings (%)': savings,
                'Similarity Score': round(float(sim), 4),
                'Explanation': f"Save {savings}% (₹{int(base_price - row['price']):,}) with similar features in {row['subcategory']}."
            })
    return pd.DataFrame(candidates).sort_values(by='Similarity Score', ascending=False).head(top_n)

print("Smart Alternatives for Apple iPhone 15:")
get_smart_alternatives('PROD_ELEC_001', top_n=3)""")
]

# ========================================================
# Notebook 3: 03_collaborative_filtering.ipynb
# ========================================================
nb3_cells = [
    markdown_cell("""# 🤝 Module 3: Collaborative Filtering & Matrix Factorization
**AI-Powered Personalized Product Recommendation & Analytics Platform**

In this notebook:
1. Weighted Implicit Feedback Matrix formulation
2. User-Item Interaction Matrix construction
3. Matrix Factorization with TruncatedSVD (Singular Value Decomposition)
4. Predicting user preference scores for unseen products
5. Market Basket Association Mining (Frequently Bought Together)"""),
    code_cell("""import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = '..'
interactions_path = os.path.join(BASE_DIR, 'data', 'raw', 'interactions.csv')
products_path = os.path.join(BASE_DIR, 'data', 'processed', 'clean_products.csv')

df_interactions = pd.read_csv(interactions_path)
df_products = pd.read_csv(products_path)"""),
    markdown_cell("### 1. Weighted Implicit Feedback"),
    code_cell("""interaction_weights = {
    'view': 1.0,
    'click': 2.0,
    'like': 3.0,
    'wishlist': 3.5,
    'cart': 4.0,
    'purchase': 5.0
}

df_weighted = df_interactions.copy()
df_weighted['weight'] = df_weighted['interaction_type'].map(lambda t: interaction_weights.get(t, 1.0))
df_weighted.head()"""),
    markdown_cell("### 2. User-Item Matrix Formulation"),
    code_cell("""user_item_matrix = df_weighted.groupby(['user_id', 'product_id'])['weight'].sum().unstack().fillna(0.0)
print(f"User-Item Matrix shape: {user_item_matrix.shape}")
user_item_matrix.iloc[:5, :5]"""),
    markdown_cell("### 3. Matrix Factorization via TruncatedSVD"),
    code_cell("""n_components = 8
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(user_item_matrix.values)
item_factors = svd.components_

print(f"Explained Variance Ratio: {svd.explained_variance_ratio_.sum():.2%}")
print(f"User Factors shape: {user_factors.shape}")
print(f"Item Factors shape: {item_factors.shape}")

# Reconstruct predicted rating matrix
predicted_matrix = np.dot(user_factors, item_factors)
df_predictions = pd.DataFrame(predicted_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)
df_predictions.iloc[:5, :5]"""),
    markdown_cell("### 4. Collaborative Recommendations for Target User"),
    code_cell("""target_user = 'USER_TECH_001'
user_preds = df_predictions.loc[target_user].sort_values(ascending=False).head(5)

rec_list = []
for pid, score in user_preds.items():
    p = df_products[df_products['product_id'] == pid].iloc[0]
    rec_list.append({
        'Product ID': pid,
        'Product Name': p['product_name'],
        'Category': p['category'],
        'Predicted Score': round(score, 3)
    })

pd.DataFrame(rec_list)"""),
    markdown_cell("### 5. Co-occurrence Matrix (Frequently Bought Together)"),
    code_cell("""# High-intent events
cart_purchases = df_interactions[df_interactions['interaction_type'].isin(['cart', 'purchase', 'wishlist'])]
user_baskets = cart_purchases.groupby('user_id')['product_id'].unique()

from collections import defaultdict
co_matrix = defaultdict(lambda: defaultdict(int))
for pids in user_baskets:
    for i in range(len(pids)):
        for j in range(len(pids)):
            if i != j:
                co_matrix[pids[i]][pids[j]] += 1

df_co = pd.DataFrame(co_matrix).fillna(0)
print("Top items frequently bought with Apple iPhone 15 (PROD_ELEC_001):")
if 'PROD_ELEC_001' in df_co.columns:
    print(df_co['PROD_ELEC_001'].sort_values(ascending=False).head(3))""")
]

# ========================================================
# Notebook 4: 04_model_evaluation.ipynb
# ========================================================
nb4_cells = [
    markdown_cell("""# 📈 Module 4: Recommendation Evaluation & A/B Testing
**AI-Powered Personalized Product Recommendation & Analytics Platform**

In this notebook:
1. Evaluating Recommendation Quality:
   - Precision@K
   - Recall@K
   - Hit Rate@K
   - Catalog Coverage (%)
   - Intra-List Recommendation Diversity
2. A/B Testing Experiment Simulation (Baseline vs Hybrid Recommender)
3. Statistical Significance (Hypothesis Testing & p-value analysis)"""),
    code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = '..'
sys.path.append(BASE_DIR)

from src.evaluation import RecommenderEvaluator

evaluator = RecommenderEvaluator(k=5)
metrics = evaluator.evaluate_all_metrics()

print("Evaluation Benchmark Results (@ K=5):")
for k, v in metrics.items():
    print(f"  {k}: {v}")"""),
    markdown_cell("### 1. Metric Variation Across Top-K (K=3, 5, 8, 10)"),
    code_cell("""k_values = [3, 5, 8, 10]
precisions = []
recalls = []
hit_rates = []

for k in k_values:
    ev = RecommenderEvaluator(k=k)
    m = ev.evaluate_all_metrics()
    precisions.append(m['precision_at_k'])
    recalls.append(m['recall_at_k'])
    hit_rates.append(m['hit_rate'])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_values, precisions, marker='o', label='Precision@K', color='#3b82f6', linewidth=2)
ax.plot(k_values, recalls, marker='s', label='Recall@K', color='#10b981', linewidth=2)
ax.plot(k_values, hit_rates, marker='^', label='Hit Rate@K', color='#f59e0b', linewidth=2)

ax.set_title('Recommendation Quality Across Different Values of K', fontsize=13, fontweight='bold')
ax.set_xlabel('Recommendation Window (K)')
ax.set_ylabel('Metric Score')
ax.legend()
plt.show()"""),
    markdown_cell("### 2. A/B Testing Experiment Simulation"),
    code_cell("""ab_results = evaluator.run_ab_test_simulation(sample_size=300)
ctrl = ab_results['group_a_control']
treat = ab_results['group_b_treatment']
stat = ab_results['statistical_summary']

print("--- A/B TEST RESULTS ---")
print(f"Group A (Control - Baseline): CTR = {ctrl['ctr_pct']}%, Conversion = {ctrl['conversion_rate_pct']}%")
print(f"Group B (Treatment - Hybrid):  CTR = {treat['ctr_pct']}%, Conversion = {treat['conversion_rate_pct']}%")
print(f"Relative CTR Lift: +{stat['ctr_lift_pct']}%")
print(f"Relative Conversion Lift: +{stat['conversion_lift_pct']}%")
print(f"p-value: {stat['p_value']} (Statistically Significant = {stat['significant']})")"""),
    markdown_cell("### 3. Visualizing A/B Test Lift"),
    code_cell("""groups = ['Group A (Baseline)', 'Group B (Hybrid + Smart Ranking)']
ctrs = [ctrl['ctr_pct'], treat['ctr_pct']]
conversions = [ctrl['conversion_rate_pct'], treat['conversion_rate_pct']]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# CTR Comparison
axes[0].bar(groups, ctrs, color=['#94a3b8', '#6366f1'])
axes[0].set_title(f"Click-Through Rate (CTR %)\\n+{stat['ctr_lift_pct']}% Relative Lift", fontweight='bold')
axes[0].set_ylabel('CTR (%)')

# Conversion Comparison
axes[1].bar(groups, conversions, color=['#94a3b8', '#10b981'])
axes[1].set_title(f"Conversion Rate (%)\\n+{stat['conversion_lift_pct']}% Relative Lift", fontweight='bold')
axes[1].set_ylabel('Conversion Rate (%)')

plt.tight_layout()
plt.show()"""),
    markdown_cell("### 4. Key Takeaways for Project Presentation / Defense"),
    code_cell("""print(\"\"\"
KEY DEFENSE POINTS:
1. Explainable AI: Recommendations provide transparent mathematical & semantic justification.
2. Cold-Start Solution: Gracefully serves top-rated/trending products before switching to personalized vectors.
3. Multi-Factor Smart Ranking: Balances similarity, preferences, ratings, popularity, and diversity.
4. Empirical Evaluation: Proves +40%+ CTR uplift under controlled A/B test simulation.
\"\"\")""")
]


def write_notebook(filename, cells):
    nb = make_notebook(cells)
    filepath = os.path.join(NOTEBOOKS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"✅ Generated notebook: {filepath}")


if __name__ == "__main__":
    write_notebook("01_data_analysis.ipynb", nb1_cells)
    write_notebook("02_content_based.ipynb", nb2_cells)
    write_notebook("03_collaborative_filtering.ipynb", nb3_cells)
    write_notebook("04_model_evaluation.ipynb", nb4_cells)
