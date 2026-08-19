"""
Week 7 -- FastAPI backend
------------------------------
Exposes the analysis pipeline (pipeline.py) as a web API. This is the
literal Week 7 deliverable: "connect all analysis modules through a
FastAPI service" producing "an end-to-end API response containing all
analysis outputs."

Run:
    uvicorn backend.main:app --reload

Then either:
  - open http://127.0.0.1:8000/docs for FastAPI's automatic interactive
    API tester (upload a file right in the browser, no client code needed)
  - or POST a file directly:
        curl -X POST http://127.0.0.1:8000/analyze -F "file=@data/Black_Panther.txt"

IMPORTANT: startup will take a while the FIRST time the server boots --
this is when pipeline.py's models actually get loaded (spaCy, the
sentiment model, genre model, viability model). That's expected and
only happens once, not per-request; watch the console for "All models
loaded. Pipeline ready." before sending requests.
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Importing this triggers pipeline.py's model loading -- deliberately
# done at server startup (import time), not per-request. See pipeline.py
# for why.
from backend import pipeline

app = FastAPI(
    title="Screenplay Analysis API",
    description="Upload a screenplay (.txt) and receive parsed structure, "
                 "sentiment arc, story beats, character relationships, "
                 "predicted genre, and a viability estimate in one response.",
    version="0.1.0",
)

# Allows a frontend (Week 8) running on a different port/origin during
# development to call this API from the browser. Tightened to specific
# origins once a real frontend URL exists.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Simple liveness check -- also confirms all models loaded
    successfully at startup, since this endpoint only responds once
    pipeline.py has finished importing."""
    return {
        "status": "ok",
        "genre_labels_available": len(pipeline.GENRE_LABELS),
        "viability_uses_genre": pipeline.VIABILITY_INCLUDES_GENRE,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Accepts a plain-text screenplay (.txt) and returns the full
    analysis: parsed structure, characters, sentiment arc, story beats,
    character relationships, predicted genre, and viability estimate."""
    if not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt screenplay files are supported right now. "
                   "PDF support is planned but not yet built.",
        )

    # Write the upload to a temp file -- the parser and every downstream
    # module work off a real file path, not an in-memory stream.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        original_title = Path(file.filename).stem
        result = pipeline.analyze_screenplay(tmp_path, title_override=original_title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not result.get("success"):
        raise HTTPException(status_code=422, detail=result)

    return result