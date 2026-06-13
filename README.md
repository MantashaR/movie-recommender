# 🎬 Movie Recommendation System using Clustering

A content-based movie recommender built on **K-Means clustering** over the
**TMDB 5000 Movies** dataset (from Kaggle). Pick a movie and get on-theme
recommendations drawn from the same cluster, ranked by similarity.

## How it works

1. **Feature engineering** — each movie becomes a feature vector from its
   *genres, keywords, top cast, director and overview* (combined into a text
   "soup" → TF-IDF → reduced with Truncated SVD) plus scaled numeric signals
   (rating, vote count, popularity, runtime, year).
2. **Clustering** — K-Means groups all 4,803 movies into themed clusters
   (e.g. *Action/Sci-Fi*, *Romance/Drama*, *Horror/Thriller*).
3. **Recommendation** — for a chosen movie we stay inside its cluster and rank
   the other members by **cosine similarity** → tight, relevant picks.

## Project structure

```
movie-recommender/
├── data/                 # TMDB 5000 CSVs (downloaded)
├── models/               # cached trained model (model.pkl)
├── recommender.py        # core ML module (load, features, cluster, recommend)
├── train.py              # trains + caches the model, prints metrics
├── app.py                # Streamlit demo UI
├── notebook.ipynb        # step-by-step ML walkthrough (EDA → clusters → recs)
├── requirements.txt
└── README.md
```

## Setup & run

```powershell
# 1. install dependencies
pip install -r requirements.txt

# 2. train the model (creates models/model.pkl) — ~15s
python train.py

# 3. launch the demo
streamlit run app.py
```

Then open the notebook (`notebook.ipynb`) for the full ML walkthrough including
the elbow/silhouette analysis used to choose the number of clusters.

## Dataset

TMDB 5000 Movie Dataset — `tmdb_5000_movies.csv` + `tmdb_5000_credits.csv`
(~4,800 movies with genres, keywords, cast, crew, ratings).
