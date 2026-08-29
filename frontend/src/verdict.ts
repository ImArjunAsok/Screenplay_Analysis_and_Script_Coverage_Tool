export type Verdict = "RECOMMEND" | "CONSIDER" | "PASS";

// Mirrors backend/report_generator.py's _recommendation() thresholds
// exactly -- keep these in sync so the on-screen verdict and the PDF
// verdict never disagree with each other.
export function getVerdict(rating: number): Verdict {
  if (rating >= 7.0) return "RECOMMEND";
  if (rating >= 5.5) return "CONSIDER";
  return "PASS";
}

export function verdictClassName(verdict: Verdict): string {
  return `verdict-stamp verdict-${verdict.toLowerCase()}`;
}
