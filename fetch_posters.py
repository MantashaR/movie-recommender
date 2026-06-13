"""
Cache poster images for every movie in the trained model.

Usage:
    python fetch_posters.py YOUR_TMDB_API_KEY
    # or set $env:TMDB_API_KEY first, then:
    python fetch_posters.py
"""

import os
import sys

from recommender import load_recommender
from posters import fetch_all


def main():
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TMDB_API_KEY")
    if not api_key:
        print("No API key. Get a free one at "
              "https://www.themoviedb.org/settings/api")
        print("Then run:  python fetch_posters.py YOUR_KEY")
        sys.exit(1)

    rec = load_recommender()
    ids = rec.df["id"].tolist()
    fetch_all(api_key, ids)


if __name__ == "__main__":
    main()
