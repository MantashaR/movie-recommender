"""
Movie Recommendation System using Clustering — Streamlit demo.

Run:  streamlit run app.py
(Train first with:  python train.py)
"""

import os

import pandas as pd
import streamlit as st

from recommender import (
    load_recommender, ARTIFACT_PATH, load_data, engineer_features,
    build_feature_matrix, fit_clusters, MovieRecommender,
)

st.set_page_config(page_title="Movie Recommender · Clustering",
                   page_icon="🎬", layout="wide")

N_CLUSTERS = 10


@st.cache_resource
def get_recommender():
    # Use the pre-trained model if present (fast local start) ...
    if os.path.exists(ARTIFACT_PATH):
        try:
            return load_recommender()
        except Exception:
            pass  # fall through and rebuild (e.g. version mismatch on cloud)
    # ... otherwise build it on first run (downloads data automatically).
    with st.spinner("First run: downloading data & training the model (~20s)…"):
        df = engineer_features(load_data())
        features, pieces = build_feature_matrix(df)
        kmeans, labels = fit_clusters(features, n_clusters=N_CLUSTERS)
        return MovieRecommender(df, features, labels, kmeans, pieces)


rec = get_recommender()

# ----------------------------- header ------------------------------------- #
st.title("🎬 Movie Recommendation System")
st.caption("Content-based **K-Means clustering** on the TMDB 5000 dataset · "
           "pick a movie, get on-theme recommendations from its cluster.")

if rec is None:
    st.error("No trained model found. Run `python train.py` first, then reload.")
    st.stop()

# ----------------------------- sidebar ------------------------------------ #
with st.sidebar:
    st.header("About")
    st.markdown(
        "**How it works**\n"
        "1. Each movie → a feature vector from *genres, keywords, cast, "
        "director, overview* (TF-IDF + SVD) plus numeric signals.\n"
        "2. **K-Means** groups the 4,803 movies into themed clusters.\n"
        "3. Recommendations = closest movies *within the same cluster* "
        "(cosine similarity).\n"
    )
    summary = rec.cluster_summary()
    st.metric("Movies", f"{len(rec.df):,}")
    st.metric("Clusters", rec.df['cluster'].nunique())
    st.markdown("**Cluster themes**")
    st.dataframe(summary[["cluster", "size", "top_genres"]],
                 hide_index=True, use_container_width=True)

# ----------------------------- main: pick a movie ------------------------- #
tab_recommend, tab_explore = st.tabs(["🔍 Recommend", "📊 Explore clusters"])

with tab_recommend:
    col1, col2 = st.columns([3, 1])
    with col1:
        # popular movies first so the demo opens on something recognisable
        popular = (rec.df.sort_values("vote_count", ascending=False)
                   ["title"].astype(str).tolist())
        movie = st.selectbox("Choose a movie", popular, index=0)
    with col2:
        n = st.slider("How many?", 5, 20, 10)

    if movie:
        idx = rec.find_index(movie)
        info = rec.df.iloc[idx]
        cl = int(info["cluster"])
        theme = summary.loc[summary["cluster"] == cl, "top_genres"].values[0]

        st.markdown(
            f"**{movie}** ({int(info['release_year']) if not pd.isna(info['release_year']) else 'N/A'}) "
            f"· ⭐ {info['vote_average']} · belongs to **Cluster {cl}** "
            f"_({theme})_"
        )

        recs = rec.recommend(movie, n=n)
        if recs.empty:
            st.warning("No recommendations found.")
        else:
            recs_display = recs.copy()
            recs_display["match"] = (recs_display["similarity"] * 100).round(1).astype(str) + "%"
            recs_display["⭐"] = recs_display["vote_average"]
            recs_display["year"] = recs_display["release_year"].apply(
                lambda y: int(y) if not pd.isna(y) else "—")
            st.dataframe(
                recs_display[["title", "genres", "year", "⭐", "match"]],
                hide_index=True, use_container_width=True,
                column_config={
                    "title": "Recommended movie",
                    "genres": "Genres",
                    "match": "Similarity",
                },
            )

with tab_explore:
    st.subheader("Cluster overview")
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.subheader("Browse a cluster")
    c = st.selectbox("Cluster", sorted(rec.df["cluster"].unique()))
    sub = (rec.df[rec.df["cluster"] == c]
           .sort_values("vote_count", ascending=False)
           .head(30))
    show = sub[["title", "release_year", "vote_average", "vote_count"]].copy()
    show["release_year"] = show["release_year"].apply(
        lambda y: int(y) if not pd.isna(y) else "—")
    st.dataframe(show, hide_index=True, use_container_width=True,
                 column_config={"title": "Movie", "release_year": "Year",
                                "vote_average": "Rating", "vote_count": "Votes"})

    st.subheader("Cluster sizes")
    st.bar_chart(summary.set_index("cluster")["size"])
