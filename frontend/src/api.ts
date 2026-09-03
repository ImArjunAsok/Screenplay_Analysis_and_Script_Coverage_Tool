import type { AnalysisResult, AnalysisFailure } from "./types";

const API_BASE = "http://127.0.0.1:8000";

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function analyzeScreenplay(
  file: File
): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new ApiError(
      "Couldn't reach the analysis server. Is it running? " +
        "(uvicorn backend.main:app --reload)"
    );
  }

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as
      | { detail?: AnalysisFailure | string }
      | null;
    const detail = body?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.error ?? `Analysis failed (status ${response.status}).`;
    throw new ApiError(message, response.status);
  }

  return (await response.json()) as AnalysisResult;
}

export async function downloadReport(analysis: AnalysisResult): Promise<void> {
  // Sends the analysis we ALREADY HAVE (from the earlier /analyze call)
  // instead of re-uploading the file, which used to make the backend
  // re-run the entire pipeline -- parsing, sentiment scoring, genre and
  // viability prediction -- a second time just to build a PDF from data
  // that was already computed. This is why "Download PDF" used to feel
  // as slow as the original analysis; it's now just formatting.
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/report-from-analysis`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(analysis),
    });
  } catch {
    throw new ApiError("Couldn't reach the analysis server to generate the report.");
  }

  if (!response.ok) {
    throw new ApiError(`Report generation failed (status ${response.status}).`);
  }

  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${analysis.title}_coverage_report.pdf`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}