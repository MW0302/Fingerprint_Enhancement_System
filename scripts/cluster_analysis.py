"""
Cheap validation test: does clustering the 320 images by their 7 diagnostic
degradation features (results/per_image_metrics.csv, produced by
compute_diagnostic_features.py) reveal genuinely new sub-groups, or does it
just rediscover the DB1_B..DB4_B split already used everywhere else in this
project?

- K-Means with K=4, compared against the true db column via Adjusted Rand
  Index (ARI near 1 = clusters ~= DB split; near 0 = clusters cut across DB
  boundaries).
- A cluster x db crosstab, so the raw counts behind the ARI are visible.
- A K=2..8 silhouette sweep to see what number of clusters the data itself
  suggests, independent of the K=4 comparison.
- Flags images sitting in a cluster where their own db is not that
  cluster's majority db, i.e. genuine cross-boundary cases rather than the
  1-2 stray images every clustering run produces.

Run with:
    python scripts/cluster_analysis.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))

from config import RESULTS_DIR  # noqa: E402

FEATURE_COLS = [
    "contrast_std",
    "michelson_contrast",
    "illumination_unevenness",
    "noise_sigma",
    "sharpness_laplacian_var",
    "foreground_ratio",
    "orientation_coherence",
]

RANDOM_STATE = 42


def main():
    csv_path = os.path.join(RESULTS_DIR, "per_image_metrics.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}\n")

    X = StandardScaler().fit_transform(df[FEATURE_COLS].values)

    # --- K=4 vs. true DB split ---------------------------------------------
    km4 = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
    cluster_labels = km4.fit_predict(X)
    df["cluster"] = cluster_labels

    ari = adjusted_rand_score(df["db"], cluster_labels)
    print(f"K=4 Adjusted Rand Index vs. true db label: {ari:.4f}\n")

    crosstab = pd.crosstab(df["cluster"], df["db"])
    print("Cluster x DB crosstab (K=4):")
    print(crosstab.to_string())
    print()

    # --- flag cross-boundary images -----------------------------------------
    # For each cluster, its majority db is the "expected" db for members of
    # that cluster. Any image whose own db differs from its cluster's
    # majority db is a cross-boundary case worth a visual spot-check.
    majority_db_per_cluster = crosstab.idxmax(axis=1)
    df["cluster_majority_db"] = df["cluster"].map(majority_db_per_cluster)
    crossers = df[df["db"] != df["cluster_majority_db"]].copy()

    print(f"Images assigned to a cluster whose majority db differs from their own db: "
          f"{len(crossers)} / {len(df)}\n")
    if len(crossers) > 0:
        cols = ["file", "db", "cluster", "cluster_majority_db"] + FEATURE_COLS
        print(crossers[cols].sort_values(["cluster", "db"]).to_string(index=False))
    print()

    # Per-db breakdown: does any single db split non-trivially (>1 cluster,
    # each with a non-trivial share) rather than just 1-2 strays?
    print("Per-DB cluster distribution (row = true db, values = image count per cluster):")
    db_cluster_dist = pd.crosstab(df["db"], df["cluster"])
    print(db_cluster_dist.to_string())
    print()

    # --- K=2..8 silhouette sweep --------------------------------------------
    print("Silhouette sweep, K=2..8:")
    sil_scores = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        sil_scores[k] = score
        print(f"  K={k}: silhouette={score:.4f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"\nSilhouette-suggested optimal K: {best_k} (score={sil_scores[best_k]:.4f})")

    out_path = os.path.join(RESULTS_DIR, "cluster_assignments_k4.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote per-image cluster assignments to {out_path}")


if __name__ == "__main__":
    main()
