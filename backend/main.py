import shutil
import tempfile
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend import pipeline
from backend.report_generator import generate_report

app = FastAPI(
    title="Screenplay Analysis API",
    description="Upload a screenplay (.txt or .pdf) and receive parsed structure, "
                 "sentiment arc, story beats, character relationships, "
                 "predicted genre, and a viability estimate in one response.",
    version="0.1.0",
)

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


@app.post("/report-from-analysis")
async def report_from_analysis(analysis: dict = Body(...)):
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