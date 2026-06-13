"""
Movie Recommendation System using Clustering
--------------------------------------------
Core ML module: loads the TMDB 5000 dataset, engineers features,
clusters movies with K-Means, and recommends similar movies.

The recommendation strategy is HYBRID:
  1. Movies are grouped into clusters (K-Means) on a combined
     content + numeric feature space.
  2. For a chosen movie we look inside its own cluster and rank
     neighbours by cosine similarity -> tight, on-theme picks.
"""

import os
import ast
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_DIR = os.path.join(HERE, "models")
ARTIFACT_PATH = os.path.join(MODEL_DIR, "model.pkl")


# --------------------------------------------------------------------------- #
# 1. Loading & parsing
# --------------------------------------------------------------------------- #
def _parse_names(text, top_n=None):
    """TMDB stores list-columns as JSON-ish strings: [{'id':28,'name':'Action'}]."""
    try:
        items = ast.literal_eval(text) if isinstance(text, str) else []
    except (ValueError, SyntaxError):
        return []
    names = [d["name"] for d in items if isinstance(d, dict) and "name" in d]
    return names[:top_n] if top_n else names


def _get_director(crew_text):
    try:
        crew = ast.literal_eval(crew_text) if isinstance(crew_text, str) else []
    except (ValueError, SyntaxError):
        return ""
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


# Public mirror of the Kaggle TMDB 5000 dataset (used if data is missing,
# e.g. on a fresh Streamlit Cloud deploy where the CSVs aren't committed).
_MIRROR = "https://raw.githubusercontent.com/procodingclass/kaggle-movie-data/main"


def ensure_data(data_dir=DATA_DIR):
    """Download the TMDB CSVs into data_dir if they aren't present."""
    import urllib.request
    os.makedirs(data_dir, exist_ok=True)
    for fname in ("tmdb_5000_movies.csv", "tmdb_5000_credits.csv"):
        dest = os.path.join(data_dir, fname)
        if not os.path.exists(dest) or os.path.getsize(dest) < 100000:
            urllib.request.urlretrieve(f"{_MIRROR}/{fname}", dest)
    return data_dir


def load_data(data_dir=DATA_DIR):
    """Load and merge the TMDB movies + credits CSVs."""
    ensure_data(data_dir)
    movies = pd.read_csv(os.path.join(data_dir, "tmdb_5000_movies.csv"))
    credits = pd.read_csv(os.path.join(data_dir, "tmdb_5000_credits.csv"))
    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")
    return df


# --------------------------------------------------------------------------- #
# 2. Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df):
    """Add parsed/derived columns used for the content 'soup' and numeric model."""
    df = df.copy()
    df["genres_list"] = df["genres"].apply(lambda x: _parse_names(x))
    df["keywords_list"] = df["keywords"].apply(lambda x: _parse_names(x))
    df["cast_list"] = df["cast"].apply(lambda x: _parse_names(x, top_n=3))
    df["director"] = df["crew"].apply(_get_director)
    df["overview"] = df["overview"].fillna("")
    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year

    def _clean(name):
        return name.lower().replace(" ", "")

    def make_soup(row):
        # genres weighted x3 so the cluster theme is genre-led
        genres = (" ".join(_clean(g) for g in row["genres_list"]) + " ") * 3
        keywords = " ".join(_clean(k) for k in row["keywords_list"])
        cast = " ".join(_clean(c) for c in row["cast_list"])
        director = (_clean(row["director"]) + " ") * 2 if row["director"] else ""
        return f"{genres}{keywords} {cast} {director} {row['overview']}".strip()

    df["soup"] = df.apply(make_soup, axis=1)
    return df


NUMERIC_COLS = ["vote_average", "vote_count", "popularity", "runtime", "release_year"]


def build_feature_matrix(df, max_features=5000, svd_components=120, random_state=42):
    """Return combined (content + numeric) feature matrix and the fitted pieces."""
    # --- content features: TF-IDF on the soup, reduced with SVD ---
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["soup"])

    n_comp = min(svd_components, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    content = svd.fit_transform(tfidf_matrix)

    # --- numeric features: log-scale skewed cols, then min-max ---
    num = df[NUMERIC_COLS].copy()
    num["vote_count"] = np.log1p(num["vote_count"].fillna(0))
    num["popularity"] = np.log1p(num["popularity"].fillna(0))
    num = num.fillna(num.median())
    scaler = MinMaxScaler()
    numeric = scaler.fit_transform(num)

    # content carries most of the signal; numeric nudges (weight 0.5)
    features = np.hstack([content, numeric * 0.5])
    pieces = {"tfidf": tfidf, "svd": svd, "scaler": scaler}
    return features, pieces


# --------------------------------------------------------------------------- #
# 3. Clustering
# --------------------------------------------------------------------------- #
def choose_k(features, k_range=range(4, 16), random_state=42):
    """Evaluate K-Means across k and return per-k inertia + silhouette scores."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(features)
        sil = silhouette_score(features, labels, sample_size=2000,
                               random_state=random_state)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(results)


def fit_clusters(features, n_clusters, random_state=42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(features)
    return km, labels


# --------------------------------------------------------------------------- #
# 4. Recommendation
# --------------------------------------------------------------------------- #
class MovieRecommender:
    """Wraps the trained artifacts and serves recommendations."""

    def __init__(self, df, features, labels, kmeans, pieces):
        self.df = df.reset_index(drop=True)
        self.features = features
        self.labels = labels
        self.kmeans = kmeans
        self.pieces = pieces
        self.df["cluster"] = labels
        self._title_to_idx = {
            t.lower(): i for i, t in enumerate(self.df["title"].astype(str))
        }

    # ---- lookups ----
    def titles(self):
        return self.df["title"].astype(str).tolist()

    def find_index(self, title):
        return self._title_to_idx.get(str(title).lower())

    def search_titles(self, query, limit=10):
        q = str(query).lower()
        hits = [t for t in self.titles() if q in t.lower()]
        return hits[:limit]

    # ---- the core recommendation call ----
    def recommend(self, title, n=10):
        idx = self.find_index(title)
        if idx is None:
            return pd.DataFrame()
        cluster_id = self.labels[idx]
        member_idx = np.where(self.labels == cluster_id)[0]

        sims = cosine_similarity(
            self.features[idx].reshape(1, -1), self.features[member_idx]
        ).flatten()

        order = member_idx[np.argsort(sims)[::-1]]
        order = [i for i in order if i != idx][:n]

        out = self.df.iloc[order][
            ["title", "genres_list", "release_year", "vote_average",
             "vote_count", "cluster"]
        ].copy()
        out["similarity"] = [
            float(sims[list(member_idx).index(i)]) for i in order
        ]
        out["genres"] = out["genres_list"].apply(lambda g: ", ".join(g))
        return out.drop(columns=["genres_list"]).reset_index(drop=True)

    def cluster_summary(self):
        """Top genres per cluster -> human-readable theme labels."""
        rows = []
        for c in sorted(self.df["cluster"].unique()):
            sub = self.df[self.df["cluster"] == c]
            genre_counts = (
                sub["genres_list"].explode().value_counts().head(3)
            )
            rows.append({
                "cluster": c,
                "size": len(sub),
                "top_genres": ", ".join(genre_counts.index.tolist()),
                "avg_rating": round(sub["vote_average"].mean(), 2),
            })
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Persistence
# --------------------------------------------------------------------------- #
def save_recommender(rec, path=ARTIFACT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "df": rec.df,
        "features": rec.features,
        "labels": rec.labels,
        "kmeans": rec.kmeans,
        "pieces": rec.pieces,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_recommender(path=ARTIFACT_PATH):
    with open(path, "rb") as f:
        p = pickle.load(f)
    return MovieRecommender(
        p["df"], p["features"], p["labels"], p["kmeans"], p["pieces"]
    )
