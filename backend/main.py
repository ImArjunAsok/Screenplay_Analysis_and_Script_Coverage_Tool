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
        curl -X POST http://127.0.0.1:8000/analyze -F "file=@data/Black_Panther.pdf"

IMPORTANT: startup will take a while the FIRST time the server boots --
this is when pipeline.py's models actually get loaded (spaCy, the
sentiment model, genre model, viability model). That's expected and
only happens once, not per-request; watch the console for "All models
loaded. Pipeline ready." before sending requests.
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Importing this triggers pipeline.py's model loading -- deliberately
# done at server startup (import time), not per-request. See pipeline.py
# for why.
from backend import pipeline
from backend.report_generator import generate_report

app = FastAPI(
    title="Screenplay Analysis API",
    description="Upload a screenplay (.txt or .pdf) and receive parsed structure, "
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


ALLOWED_EXTENSIONS = (".txt", ".pdf")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Accepts a screenplay (.txt or .pdf) and returns the full
    analysis: parsed structure, characters, sentiment arc, story beats,
    character relationships, predicted genre, and viability estimate."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(ALLOWED_EXTENSIONS)}.",
        )

    # Write the upload to a temp file -- the parser and every downstream
    # module work off a real file path, not an in-memory stream. The
    # suffix MUST match the real upload (not hardcoded to .txt) since
    # the parser decides how to read the file based on this extension --
    # a PDF saved with a .txt suffix would be read as raw text and fail.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
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


@app.post("/report")
async def report(file: UploadFile = File(...)):
    """Same analysis as /analyze, but returns a downloadable PDF coverage
    report instead of raw JSON -- the Week 8 'automated PDF report
    generation' deliverable. Runs the full pipeline itself rather than
    accepting an already-computed analysis, so this endpoint is
    self-contained and usable on its own."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(ALLOWED_EXTENSIONS)}.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    pdf_path = None
    try:
        original_title = Path(file.filename).stem
        result = pipeline.analyze_screenplay(tmp_path, title_override=original_title)
        if not result.get("success"):
            raise HTTPException(status_code=422, detail=result)

        pdf_fd = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf_path = pdf_fd.name
        pdf_fd.close()
        generate_report(result, pdf_path)

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"{original_title}_coverage_report.pdf",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        # Note: pdf_path is deliberately NOT deleted here -- FileResponse
        # streams it back to the client after this function returns, so
        # deleting it now would race against that. It's a temp file, so
        # the OS will clean it up eventually; fine for now, worth
        # revisiting with an explicit cleanup (e.g. BackgroundTask) if
        # disk usage becomes a concern at higher traffic.


@app.post("/report-from-analysis")
async def report_from_analysis(analysis: dict = Body(...)):
    """Generates a PDF from an analysis result the client ALREADY HAS --
    e.g. the frontend, right after a successful /analyze call -- instead
    of re-running the full pipeline (parsing, sentiment scoring,
    genre/viability prediction) a second time just to build a PDF from
    data that was already computed seconds earlier. This is the endpoint
    the frontend should call for its "Download PDF Report" button; the
    file-upload /report endpoint above still exists for standalone use
    (e.g. going straight from a file to a PDF via curl or /docs, with no
    prior /analyze call), but it does the full, slower pipeline run by
    necessity, since it only has a file, not a result."""
    if not analysis.get("success"):
        raise HTTPException(status_code=422, detail="Cannot generate a report from a failed analysis.")

    pdf_fd = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf_path = pdf_fd.name
    pdf_fd.close()
    try:
        generate_report(analysis, pdf_path)
        title = analysis.get("title", "report")
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"{title}_coverage_report.pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")