export type Verdict = "RECOMMEND" | "CONSIDER" | "PASS";

export function getVerdict(rating: number): Verdict {
  if (rating >= 7.0) return "RECOMMEND";
  if (rating >= 5.5) return "CONSIDER";
  return "PASS";
}

export function verdictClassName(verdict: Verdict): string {
  return `verdict-stamp verdict-${verdict.toLowerCase()}`;
}
