"""
Task 2: K-Means Customer Segmentation for a Retail Store (Mall Customers dataset)
Features used: Annual Income (k$) and Spending Score (1-100) -> purchase behaviour
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA = "/mnt/user-data/uploads/Mall_Customers.csv"
OUT = "/mnt/user-data/outputs/"

# 1. Load & explore -----------------------------------------------------------
df = pd.read_csv(DATA)
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe().round(2))
print("\nGender counts:\n", df["Gender"].value_counts())

# 2. Feature selection & scaling ---------------------------------------------
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].values
X_scaled = StandardScaler().fit_transform(X)

# 3. Find optimal K (Elbow + Silhouette) -------------------------------------
K_range = range(2, 11)
inertia, sil = [], []
for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42).fit(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_))
    print(f"K={k}  inertia={km.inertia_:.2f}  silhouette={sil[-1]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(K_range, inertia, "o-", color="#2a6fdb")
ax[0].axvline(5, ls="--", color="red", alpha=.6)
ax[0].set(title="Elbow Method", xlabel="Number of clusters (K)", ylabel="Inertia (WCSS)")
ax[1].plot(K_range, sil, "s-", color="#e0792a")
ax[1].axvline(5, ls="--", color="red", alpha=.6)
ax[1].set(title="Silhouette Score", xlabel="Number of clusters (K)", ylabel="Score")
plt.tight_layout(); plt.savefig(OUT + "1_elbow_silhouette.png", dpi=150); plt.close()

# 4. Final model --------------------------------------------------------------
K = 5
kmeans = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)
print(f"\nFinal K={K}  silhouette={silhouette_score(X_scaled, df['Cluster']):.4f}")

# 5. Profile & name clusters --------------------------------------------------
profile = df.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean"),
).round(1)

def name_cluster(r):
    hi_inc, hi_sp = r.Avg_Income >= 60, r.Avg_Spending >= 60
    lo_sp = r.Avg_Spending < 40
    if hi_inc and hi_sp: return "Target / VIP (High income, High spend)"
    if hi_inc and lo_sp: return "Careful (High income, Low spend)"
    if (not hi_inc) and hi_sp: return "Young Spenders (Low income, High spend)"
    if (not hi_inc) and lo_sp: return "Sensible (Low income, Low spend)"
    return "Standard (Mid income, Mid spend)"

profile["Segment"] = profile.apply(name_cluster, axis=1)
print("\nCluster profile:\n", profile.to_string())
df["Segment"] = df["Cluster"].map(profile["Segment"])

# 6. Visualisations -----------------------------------------------------------
colors = ["#2a6fdb", "#e0792a", "#2aa86b", "#c0392b", "#8e44ad"]
centers = StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_)

plt.figure(figsize=(9, 6.5))
for c in range(K):
    s = df[df.Cluster == c]
    plt.scatter(s[features[0]], s[features[1]], s=55, color=colors[c],
                alpha=.8, label=f"C{c}: {profile.loc[c,'Segment']}")
plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=260, c="black",
            edgecolor="white", label="Centroids")
plt.title("Customer Segments (K-Means, K=5)")
plt.xlabel("Annual Income (k$)"); plt.ylabel("Spending Score (1-100)")
plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout(); plt.savefig(OUT + "2_customer_clusters.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
ax[0].bar(profile.index, profile.Customers, color=colors); ax[0].set_title("Customers per Cluster")
ax[1].bar(profile.index, profile.Avg_Income, color=colors); ax[1].set_title("Avg Annual Income (k$)")
ax[2].bar(profile.index, profile.Avg_Spending, color=colors); ax[2].set_title("Avg Spending Score")
for a in ax: a.set_xlabel("Cluster")
plt.tight_layout(); plt.savefig(OUT + "3_cluster_profiles.png", dpi=150); plt.close()

# 7. Save results -------------------------------------------------------------
df.to_csv(OUT + "customers_with_clusters.csv", index=False)
profile.to_csv(OUT + "cluster_profile.csv")
print("\nSaved outputs to", OUT)
