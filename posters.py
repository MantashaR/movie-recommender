"""
Poster fetching + caching for the movie recommender.

Posters come from the TMDB API (free key). We cache poster paths to
models/posters.json so the Streamlit app never hits the network live
(fast, no rate limits during a demo).

Get a free key at https://www.themoviedb.org/settings/api  then either:
  * set an env var:   $env:TMDB_API_KEY="your_key"
  * or pass it to:    python fetch_posters.py your_key
"""

import os
import json
import time
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "models", "posters.json")
IMG_BASE = "https://image.tmdb.org/t/p/w342"


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def poster_url(tmdb_id, cache):
    """Return a full poster image URL for a TMDB id, or None if unknown."""
    path = cache.get(str(tmdb_id))
    return f"{IMG_BASE}{path}" if path else None


def _fetch_one(tmdb_id, api_key):
    url = (f"https://api.themoviedb.org/3/movie/{tmdb_id}"
           f"?api_key={api_key}")
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("poster_path")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return None


def fetch_all(api_key, ids):
    """Populate the poster cache for the given list of TMDB ids."""
    cache = load_cache()
    todo = [i for i in ids if str(i) not in cache]
    print(f"Fetching posters for {len(todo)} movies "
          f"({len(cache)} already cached) ...")
    for n, tmdb_id in enumerate(todo, 1):
        path = _fetch_one(tmdb_id, api_key)
        cache[str(tmdb_id)] = path  # store even None to avoid refetching
        if n % 100 == 0:
            print(f"  {n}/{len(todo)}")
            save_cache(cache)
        time.sleep(0.02)  # be polite to the API
    save_cache(cache)
    found = sum(1 for v in cache.values() if v)
    print(f"Done. {found} posters cached -> {CACHE_PATH}")
    return cache
