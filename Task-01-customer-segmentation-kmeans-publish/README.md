# Customer Segmentation using K-Means Clustering

**Author:** [Your Name]
**Task 2:** Create a K-means clustering algorithm to group customers of a retail store based on their purchase history.

## Overview
This project segments mall/retail customers into groups with similar purchasing behaviour using the K-Means clustering algorithm. The segments help the store design targeted marketing and loyalty strategies.

## Dataset
`Mall_Customers.csv` - 200 customers with the following columns:

| Column | Description |
|---|---|
| CustomerID | Unique customer ID |
| Gender | Male / Female |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousand dollars |
| Spending Score (1-100) | Score assigned by the store based on spending behaviour |

## Approach
1. Load the data and explore it (no missing values found).
2. Select features: **Annual Income** and **Spending Score**.
3. Standardise the features with `StandardScaler`.
4. Find the optimal number of clusters using the **Elbow Method** and **Silhouette Score** (K = 2 to 10) -> **K = 5**.
5. Train K-Means (`k-means++` initialisation, `n_init=10`, `random_state=42`).
6. Profile each cluster and give it a business-friendly name.
7. Visualise the clusters and save the results.

## Results (K = 5, silhouette score = 0.5547)

| Cluster | Customers | Avg Age | Avg Income (k$) | Avg Spending | Segment |
|---|---|---|---|---|---|
| 0 | 81 | 42.7 | 55.3 | 49.5 | Standard (mid income, mid spend) |
| 1 | 39 | 32.7 | 86.5 | 82.1 | Target / VIP (high income, high spend) |
| 2 | 22 | 25.3 | 25.7 | 79.4 | Young Spenders (low income, high spend) |
| 3 | 35 | 41.1 | 88.2 | 17.1 | Careful (high income, low spend) |
| 4 | 23 | 45.2 | 26.3 | 20.9 | Sensible (low income, low spend) |

## Business Recommendations
- **Target / VIP:** loyalty rewards, premium products, early sale access.
- **Careful:** personalised promotions and premium campaigns to raise spending.
- **Young Spenders:** trendy products, installment offers, social media marketing.
- **Standard:** general promotions and bundle deals.
- **Sensible:** discounts and budget offers with low marketing spend.

## Project Structure
```
.
├── kmeans_customer_segmentation.py   # Main script
├── Mall_Customers.csv                # Dataset
├── requirements.txt                  # Dependencies
├── .gitignore
└── README.md
```

## How to Run
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the script
python kmeans_customer_segmentation.py
```
> Update the `DATA` and `OUT` paths at the top of the script to match your local folders.

## Output Files
- `1_elbow_silhouette.png` - elbow and silhouette plots
- `2_customer_clusters.png` - cluster scatter plot with centroids
- `3_cluster_profiles.png` - cluster size, income and spending comparison
- `customers_with_clusters.csv` - dataset with cluster labels
- `cluster_profile.csv` - cluster summary table

## Tech Stack
Python, pandas, NumPy, Matplotlib, scikit-learn

## Author
**[Your Name]**
