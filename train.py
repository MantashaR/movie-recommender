"""
Train the clustering recommender and cache it to models/model.pkl.

Run:  python train.py
"""

import time

from recommender import (
    load_data, engineer_features, build_feature_matrix,
    choose_k, fit_clusters, MovieRecommender, save_recommender,
)

# Default K. Set to None to auto-pick the best silhouette score (slower).
N_CLUSTERS = 10


def main():
    t0 = time.time()

    print("[1/5] Loading TMDB 5000 dataset ...")
    df = load_data()
    print(f"      {len(df):,} movies loaded.")

    print("[2/5] Engineering features (genres, keywords, cast, director) ...")
    df = engineer_features(df)

    print("[3/5] Building feature matrix (TF-IDF + SVD + numeric) ...")
    features, pieces = build_feature_matrix(df)
    print(f"      feature matrix shape: {features.shape}")

    k = N_CLUSTERS
    if k is None:
        print("[4/5] Choosing K via silhouette score ...")
        scores = choose_k(features)
        print(scores.to_string(index=False))
        k = int(scores.sort_values("silhouette", ascending=False).iloc[0]["k"])
        print(f"      best k = {k}")
    else:
        print(f"[4/5] Clustering with k = {k} ...")

    kmeans, labels = fit_clusters(features, n_clusters=k)

    rec = MovieRecommender(df, features, labels, kmeans, pieces)
    save_recommender(rec)

    print("[5/5] Saved model -> models/model.pkl")
    print("\nCluster themes:")
    print(rec.cluster_summary().to_string(index=False))

    print(f"\nDone in {time.time() - t0:.1f}s")
    print("\nSample recommendations for 'The Dark Knight':")
    recs = rec.recommend("The Dark Knight", n=5)
    if not recs.empty:
        print(recs[["title", "genres", "release_year",
                    "vote_average", "similarity"]].to_string(index=False))


if __name__ == "__main__":
    main()
