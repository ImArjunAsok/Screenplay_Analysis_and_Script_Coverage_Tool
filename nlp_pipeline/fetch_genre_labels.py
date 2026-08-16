"""
Week 6 (prep) -- Fetch genre labels from IMSDB
----------------------------------------------------
Genre classification needs labels to train against, and your scraped
screenplay text doesn't carry genre info on its own. Good news: you
don't need an external API for this -- IMSDB already tags every script,
right on its own genre listing pages (imsdb.com/genre/Action,
imsdb.com/genre/Drama, etc). A title can appear on more than one genre
page (e.g. an "action comedy" appears on both), which is realistic --
most real films have more than one genre, so this naturally supports
multi-label classification rather than forcing one genre per film.

This is a SEPARATE, additive script -- it doesn't touch or need your
existing scraper. It builds its own title -> [genres] lookup, then joins
it onto your corpus by title.

Run:
    python nlp_pipeline/fetch_genre_labels.py
    python nlp_pipeline/fetch_genre_labels.py --join dataset/corpus_clean_characters.jsonl

First run (no --join) just builds and saves the genre lookup table --
worth doing once and inspecting before joining it onto your corpus, in
case IMSDB's title formatting doesn't exactly match your scraped
filenames (common gotcha: "Abyss, The" vs "The Abyss" -- IMSDB
consistently uses the comma-suffix style, so if your own scraper's
filenames came from IMSDB too, they should already match).

Outputs:
    dataset/genre_labels.json          -- title -> [genres]
    dataset/corpus_with_genres.jsonl   -- only written with --join; your
                                           corpus with a "genres" field
                                           added to each screenplay
                                           (empty list if no match found)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama",
    "Family", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Short", "Thriller", "War", "Western",
]

OUT_DIR = Path(__file__).parent.parent / "dataset"
LABELS_FILE = OUT_DIR / "genre_labels.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (research script for MSc dissertation)"}


def normalize_title(title: str) -> str:
    """Loose normalization for matching IMSDB titles against your own
    scraped filenames -- lowercase, strip punctuation, collapse spaces."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fetch_genre_page(genre: str) -> list[str]:
    """Returns the list of movie titles on this genre's IMSDB page."""
    url = f"https://imsdb.com/genre/{genre}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    titles = []
    for a in soup.find_all("a", href=True):
        if a["href"].startswith("/Movie Scripts/"):
            title = a.get_text(strip=True)
            if title:
                titles.append(title)
    return titles


def build_genre_labels() -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    for genre in GENRES:
        print(f"  Fetching {genre}...", end=" ")
        try:
            titles = fetch_genre_page(genre)
        except Exception as e:
            print(f"FAILED ({e})")
            continue
        print(f"{len(titles)} titles")
        for t in titles:
            labels.setdefault(t, [])
            if genre not in labels[t]:
                labels[t].append(genre)
        time.sleep(1)  # be polite to IMSDB's server
    return labels


def join_to_corpus(corpus_path: str, labels: dict[str, list[str]]):
    norm_lookup = {normalize_title(t): genres for t, genres in labels.items()}

    matched, unmatched = 0, 0
    out_path = OUT_DIR / "corpus_with_genres.jsonl"
    with open(corpus_path, encoding="utf-8") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            sp = json.loads(line)
            genres = norm_lookup.get(normalize_title(sp["title"]), [])
            if genres:
                matched += 1
            else:
                unmatched += 1
            sp["genres"] = genres
            out_f.write(json.dumps(sp) + "\n")

    print(f"\nJoined onto corpus: {matched} matched, {unmatched} unmatched (no genre found)")
    print(f"Written to {out_path}")
    if unmatched:
        print("Unmatched titles are kept with genres=[] -- check dataset/genre_labels.json")
        print("for the exact IMSDB title spelling if you want to fix specific ones by hand.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", help="Path to a corpus.jsonl to attach genre labels onto")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if LABELS_FILE.exists():
        print(f"Found existing {LABELS_FILE}, loading it instead of re-fetching.")
        print("Delete it first if you want a fresh fetch.")
        labels = json.loads(LABELS_FILE.read_text())
    else:
        print(f"Fetching genre labels for {len(GENRES)} genres from IMSDB...\n")
        labels = build_genre_labels()
        LABELS_FILE.write_text(json.dumps(labels, indent=2))
        print(f"\nSaved {len(labels)} titles' genre labels to {LABELS_FILE}")

    multi_genre = sum(1 for g in labels.values() if len(g) > 1)
    print(f"({multi_genre} of {len(labels)} titles have more than one genre)")

    if args.join:
        join_to_corpus(args.join, labels)


if __name__ == "__main__":
    main()