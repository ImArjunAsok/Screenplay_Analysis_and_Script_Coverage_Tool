import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

OUT_DIR = Path(__file__).parent.parent / "dataset"
LABELS_FILE = OUT_DIR / "viability_labels.json"
OMDB_URL = "http://www.omdbapi.com/"


def clean_number(value: str):
    if not value or value == "N/A":
        return None
    cleaned = re.sub(r"[^0-9.]", "", value)
    return float(cleaned) if cleaned else None


def lookup_title(title: str, api_key: str) -> dict:
    resp = requests.get(OMDB_URL, params={"t": title, "apikey": api_key}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("Response") != "True":
        return {"matched": False}

    return {
        "matched": True,
        "omdb_title": data.get("Title"),
        "year": data.get("Year"),
        "imdb_rating": clean_number(data.get("imdbRating")),
        "imdb_votes": clean_number(data.get("imdbVotes")),
        "box_office": clean_number(data.get("BoxOffice")),
        "metascore": clean_number(data.get("Metascore")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_path")
    ap.add_argument("--api-key", required=True, help="Your free OMDb API key")
    ap.add_argument("--join", action="store_true", help="Also join labels onto the corpus")
    ap.add_argument("--daily-limit", type=int, default=1000,
                     help="Stop after this many NEW lookups this run (free tier default: 1000)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels: dict[str, dict] = {}
    if LABELS_FILE.exists():
        labels = json.loads(LABELS_FILE.read_text())
        print(f"Resuming: {len(labels)} titles already looked up previously.")

    screenplays = [json.loads(l) for l in open(args.corpus_path, encoding="utf-8")]
    titles = [sp["title"] for sp in screenplays]
    remaining = [t for t in titles if t not in labels]
    print(f"{len(titles)} total scripts, {len(remaining)} not yet looked up.")

    todo = remaining[:args.daily_limit]
    if len(remaining) > args.daily_limit:
        print(f"Only doing {args.daily_limit} this run (free-tier daily cap). "
              f"Re-run tomorrow to continue -- progress is saved after every lookup.")

    matched_count = 0
    for i, title in enumerate(todo, 1):
        try:
            result = lookup_title(title, args.api_key)
        except Exception as e:
            print(f"  [{i}/{len(todo)}] ERROR on '{title}': {e}")
            result = {"matched": False, "error": str(e)}

        labels[title] = result
        if result.get("matched"):
            matched_count += 1
            print(f"  [{i}/{len(todo)}] {title:<40} rating={result['imdb_rating']}")
        else:
            print(f"  [{i}/{len(todo)}] {title:<40} NOT FOUND")

        # Save after every single lookup -- if this gets interrupted
        # (daily cap, network drop, anything), nothing already done is lost.
        LABELS_FILE.write_text(json.dumps(labels, indent=2))
        time.sleep(0.1)

    print(f"\nThis run: {matched_count}/{len(todo)} matched.")
    print(f"Total so far: {len(labels)}/{len(titles)} titles looked up.")

    if args.join:
        join_to_corpus(args.corpus_path, labels)


def join_to_corpus(corpus_path: str, labels: dict):
    out_path = OUT_DIR / "corpus_with_viability.jsonl"
    matched, unmatched, no_box_office = 0, 0, 0

    with open(corpus_path, encoding="utf-8") as in_f, open(out_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            sp = json.loads(line)
            label = labels.get(sp["title"], {"matched": False})
            sp["viability"] = label
            if label.get("matched") and label.get("imdb_rating") is not None:
                matched += 1
                if label.get("box_office") is None:
                    no_box_office += 1
            else:
                unmatched += 1
            out_f.write(json.dumps(sp) + "\n")

    print(f"\nJoined onto corpus: {matched} have a usable rating, {unmatched} don't.")
    print(f"Of the {matched} matched, {no_box_office} have no box office figure "
          f"(expected -- this is why rating, not box office, is the main target).")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()