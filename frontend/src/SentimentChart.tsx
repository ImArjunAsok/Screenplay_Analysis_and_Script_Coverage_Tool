import { useMemo } from "react";
import type { PredictedBeat } from "./types";

interface Props {
  scores: number[];
  smoothed: number[];
  beats: PredictedBeat[];
}

const WIDTH = 760;
const HEIGHT = 400;
const PAD_LEFT = 42;
const PAD_RIGHT = 12;
const PAD_TOP = 145; // long rotated labels (e.g. "Dark Night of the Soul", 23 chars)
                      // were getting clipped by the SVG edge at a smaller value -- verified visually
const PAD_BOTTOM = 34;
const PLOT_W = WIDTH - PAD_LEFT - PAD_RIGHT;
const PLOT_H = HEIGHT - PAD_TOP - PAD_BOTTOM;

function scaleX(i: number, n: number): number {
  if (n <= 1) return PAD_LEFT;
  return PAD_LEFT + (i / (n - 1)) * PLOT_W;
}

function scaleY(v: number): number {
  const clamped = Math.max(-1, Math.min(1, v));
  return PAD_TOP + (1 - (clamped + 1) / 2) * PLOT_H;
}

function buildPath(values: number[]): string {
  return values
    .map((v, i) => `${i === 0 ? "M" : "L"} ${scaleX(i, values.length).toFixed(1)} ${scaleY(v).toFixed(1)}`)
    .join(" ");
}

export default function SentimentChart({ scores, smoothed, beats }: Props) {
  const n = scores.length;
  const rawPath = useMemo(() => buildPath(scores), [scores]);
  const smoothedPath = useMemo(() => buildPath(smoothed), [smoothed]);

  // Same horizontal-nudge-with-connector-line approach as the PDF chart:
  // beats landing on the same (or very close) scene index get their
  // LABEL nudged sideways so the text doesn't overlap, while the
  // vertical line always marks the true scene position. Tested against
  // real data where two beats landed on the exact same scene -- a
  // simple vertical stagger wasn't enough separation for long rotated
  // labels, this was.
  const beatMarkers = useMemo(() => {
    if (n === 0) return [];
    const sorted = [...beats].sort((a, b) => a.scene_index - b.scene_index);
    const minGap = Math.max(2, n * 0.03);
    const nudge = n * 0.022;
    let lastX = -Infinity;
    let clusterStep = 0;

    return sorted
      .filter((b) => b.scene_index < n)
      .map((b) => {
        const idx = b.scene_index;
        clusterStep = idx - lastX < minGap ? clusterStep + 1 : 0;
        lastX = idx;

        const trueX = scaleX(idx, n);
        const labelIdx = Math.min(idx + clusterStep * nudge, n - 1);
        const labelX = scaleX(labelIdx, n);

        return { beat: b.beat, trueX, labelX, offset: clusterStep > 0 };
      });
  }, [beats, n]);

  if (n === 0) return null;

  const yTicks = [-1, -0.5, 0, 0.5, 1];
  const xTickCount = Math.min(6, n);
  const xTicks = Array.from({ length: xTickCount }, (_, i) =>
    Math.round((i / (xTickCount - 1)) * (n - 1))
  );

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="sentiment-chart"
      role="img"
      aria-label="Sentiment arc across the script, with predicted story beats marked"
    >
      {yTicks.map((v) => (
        <g key={v}>
          <line
            x1={PAD_LEFT}
            x2={WIDTH - PAD_RIGHT}
            y1={scaleY(v)}
            y2={scaleY(v)}
            stroke={v === 0 ? "#1c1a17" : "#ded9cd"}
            strokeDasharray={v === 0 ? "4 3" : undefined}
            strokeWidth={v === 0 ? 1 : 0.6}
            opacity={v === 0 ? 0.5 : 1}
          />
          <text x={PAD_LEFT - 8} y={scaleY(v) + 3} textAnchor="end" fontSize="9" fill="#6b665c" fontFamily="var(--font-mono)">
            {v.toFixed(1)}
          </text>
        </g>
      ))}

      {xTicks.map((i) => (
        <text
          key={i}
          x={scaleX(i, n)}
          y={HEIGHT - PAD_BOTTOM + 16}
          textAnchor="middle"
          fontSize="9"
          fill="#6b665c"
          fontFamily="var(--font-mono)"
        >
          {i}
        </text>
      ))}
      <text x={WIDTH / 2} y={HEIGHT - 4} textAnchor="middle" fontSize="9" fill="#6b665c" fontFamily="var(--font-mono)">
        Scene index
      </text>

      {beatMarkers.map((m, i) => (
        <g key={i}>
          <line x1={m.trueX} x2={m.trueX} y1={PAD_TOP} y2={HEIGHT - PAD_BOTTOM} stroke="#a33d2e" strokeWidth={1} opacity={0.3} />
          {m.offset && (
            <line x1={m.trueX} x2={m.labelX} y1={PAD_TOP} y2={PAD_TOP - 6} stroke="#a33d2e" strokeWidth={0.6} opacity={0.4} />
          )}
          <text
            x={m.labelX}
            y={PAD_TOP - 8}
            fontSize="7.5"
            fill="#4a4a4a"
            fontFamily="var(--font-mono)"
            textAnchor="start"
            transform={`rotate(-90 ${m.labelX} ${PAD_TOP - 8})`}
          >
            {m.beat}
          </text>
        </g>
      ))}

      <path d={rawPath} fill="none" stroke="#c9c9c9" strokeWidth={1} />
      <path d={smoothedPath} fill="none" stroke="#1F3864" strokeWidth={2} />

      <rect x={PAD_LEFT} y={PAD_TOP} width={PLOT_W} height={PLOT_H} fill="none" stroke="#1c1a17" strokeWidth={1} />
    </svg>
  );
}
