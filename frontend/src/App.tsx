import { useCallback, useState } from "react";
import "./App.css";
import type { AnalysisResult } from "./types";
import { analyzeScreenplay, downloadReport, ApiError } from "./api";
import { getVerdict, verdictClassName } from "./verdict";
import SentimentChart from "./SentimentChart";

type Status = "idle" | "analyzing" | "results" | "error";

function StatCard({ value, label }: { value: string | number; label: string }) {
  return (
    <div className="stat-card">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

function UploadZone({
  onFileSelected,
  disabled,
}: {
  onFileSelected: (file: File) => void;
  disabled: boolean;
}) {
  const [dragActive, setDragActive] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      if (disabled) return;
      const file = e.dataTransfer.files?.[0];
      if (file) onFileSelected(file);
    },
    [disabled, onFileSelected]
  );

  return (
    <div
      className={`upload-zone${dragActive ? " drag-active" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragActive(true);
      }}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
    >
      <p>Drop a screenplay (.txt or .pdf) here, or choose a file</p>
      <label className="btn" style={{ display: "inline-block" }}>
        Choose File
        <input
          type="file"
          accept=".txt,.pdf"
          disabled={disabled}
          style={{ display: "none" }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFileSelected(file);
          }}
        />
      </label>
    </div>
  );
}

function LayerDivider({ label }: { label: string }) {
  return <div className="layer-divider">{label}</div>;
}

function ResultsView({
  result,
  onDownloadReport,
  downloading,
}: {
  result: AnalysisResult;
  onDownloadReport: () => void;
  downloading: boolean;
}) {
  const verdict = getVerdict(result.viability.predicted_imdb_rating);

  return (
    <div>
      <div className="verdict-wrap">
        <span className={verdictClassName(verdict)}>{verdict}</span>
      </div>
      <p className="verdict-caption">
        Predicted IMDb rating: <strong>{result.viability.predicted_imdb_rating}/10</strong>
        {result.viability.confidence && ` (confidence: ${result.viability.confidence})`}
      </p>
      {result.viability.caveat && (
        <p className="caveat-note" style={{ textAlign: "center" }}>
          {result.viability.caveat}
        </p>
      )}

      <div className="section">
        <div className="section-title">Overview</div>
        <div className="stat-grid">
          <StatCard value={result.overview.scene_count} label="Scenes" />
          <StatCard value={result.overview.character_count} label="Characters" />
          <StatCard value={result.overview.dialogue_count} label="Dialogue Lines" />
        </div>
        {result.parser_notes.map((note, i) => (
          <p className="parser-note" key={i}>
            {note}
          </p>
        ))}
      </div>

      <div className="section">
        <div className="section-title">Predicted Genre</div>
        <div className="genre-tags">
          {result.predicted_genres.length > 0 ? (
            result.predicted_genres.map((g) => (
              <span className="genre-tag" key={g}>
                {g}
              </span>
            ))
          ) : (
            <span className="caveat-note">No genre predicted with confidence.</span>
          )}
        </div>
        {result.genre_confidence?.length > 0 && (
          <table className="data-table" style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th>Genre</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {result.genre_confidence.slice(0, 5).map((c) => (
                <tr key={c.genre}>
                  <td>{c.genre}</td>
                  <td className="beat-index">{c.probability.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <LayerDivider label="LAYER 1 — DESCRIPTIVE ANALYSIS: What is in the screenplay?" />

      <div className="section">
        <div className="section-title">Characters</div>
        <p className="caveat-note" style={{ marginBottom: 12 }}>
          {result.characters.likely_real_names.length} identified,{" "}
          {result.characters.likely_role_labels.length} generic role labels filtered out,{" "}
          {result.characters.uncertain.length} uncertain
        </p>
        <div className="character-chip-row">
          {result.characters.likely_real_names.map((name) => (
            <span className="character-chip" key={name}>
              {name}
            </span>
          ))}
        </div>
      </div>

      {result.character_relationships.most_central_characters.length > 0 && (
        <div className="section">
          <div className="section-title">Character Network</div>
          <p className="caveat-note" style={{ marginBottom: 12 }}>
            {result.character_relationships.character_count_in_network} characters,{" "}
            {result.character_relationships.relationship_count} relationships
          </p>
          {result.character_relationships.network_interpretation && (
            <p style={{ fontSize: 13, fontStyle: "italic", marginBottom: 12 }}>
              {result.character_relationships.network_interpretation}
            </p>
          )}
          <table className="data-table">
            <thead>
              <tr>
                <th>Character</th>
                <th>Scenes Shared</th>
                <th>Bridge Score</th>
              </tr>
            </thead>
            <tbody>
              {result.character_relationships.most_central_characters.map((c) => (
                <tr key={c.name}>
                  <td>
                    {c.name}
                    {c.name === result.character_relationships.likely_protagonist && (
                      <span className="protagonist-badge">Likely Protagonist</span>
                    )}
                  </td>
                  <td>{c.weighted_degree}</td>
                  <td>{c.betweenness_centrality.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <LayerDivider label="LAYER 2 — STRUCTURAL ANALYSIS: How is the screenplay constructed?" />

      <div className="section">
        <div className="section-title">Emotional Arc</div>
        <p style={{ fontSize: 14, lineHeight: 1.6 }}>
          Overall tone: <strong>{result.sentiment_arc.sentiment_label}</strong>{" "}
          <span style={{ color: "var(--muted)" }}>
            (raw score: {result.sentiment_arc.average_sentiment.toFixed(3)})
          </span>
          <br />
          Most positive scene: <em>{result.sentiment_arc.most_positive_scene}</em>
          <br />
          Most negative scene: <em>{result.sentiment_arc.most_negative_scene}</em>
          <br />
          {result.sentiment_arc.turning_point_count} emotional turning points detected
        </p>
        {result.sentiment_arc.scene_scores?.length > 0 && (
          <div className="chart-wrap">
            <SentimentChart
              scores={result.sentiment_arc.scene_scores}
              smoothed={result.sentiment_arc.smoothed_scores}
              beats={result.story_structure.predicted_beats}
            />
            <p className="caveat-note">
              Red vertical lines mark each predicted story beat at its scene position.
            </p>
          </div>
        )}
        <p className="caveat-note">{result.sentiment_arc.sentiment_label_caveat}</p>
        <p className="caveat-note">Model: {result.sentiment_arc.model_source}</p>
      </div>

      <div className="section">
        <div className="section-title">{result.story_structure.detection_type || "Story Structure"}</div>
        {result.story_structure.detection_note && (
          <p className="caveat-note" style={{ marginBottom: 10 }}>
            {result.story_structure.detection_note}
          </p>
        )}
        <table className="data-table">
          <thead>
            <tr>
              <th>Beat</th>
              <th>Scene</th>
              <th>Method</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {result.story_structure.predicted_beats.map((b) => (
              <tr key={b.beat}>
                <td>{b.beat}</td>
                <td className="beat-index">{b.scene_index}</td>
                <td style={{ fontSize: 12, color: "var(--muted)" }}>{b.method}</td>
                <td>
                  <span className={`confidence-badge confidence-${b.confidence?.toLowerCase()}`}>
                    {b.confidence}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <LayerDivider label="LAYER 3 — PREDICTIVE ANALYSIS: What might happen commercially?" />

      <div className="section">
        <div className="section-title">Viability Assessment</div>
        <p style={{ fontSize: 14 }}>
          Predicted IMDb rating: <strong>{result.viability.predicted_imdb_rating}/10</strong>
          {result.viability.confidence && ` (confidence: ${result.viability.confidence})`}
        </p>
        {result.viability.caveat && <p className="caveat-note">{result.viability.caveat}</p>}
      </div>

      {result.limitations?.length > 0 && (
        <div className="section limitations-box">
          <div className="section-title">Automated Analysis Limitations</div>
          <ul className="limitations-list">
            {result.limitations.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="report-cta">
        <button className="btn" onClick={onDownloadReport} disabled={downloading}>
          {downloading ? "Generating…" : "Download PDF Coverage Report"}
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState<Status>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  const handleFileSelected = (selected: File) => {
    setFile(selected);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setStatus("analyzing");
    setError(null);
    try {
      const analysis = await analyzeScreenplay(file);
      setResult(analysis);
      setStatus("results");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong analysing the script.");
      setStatus("error");
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setStatus("idle");
  };

  const handleDownloadReport = async () => {
    if (!result) return;
    setDownloading(true);
    try {
      await downloadReport(result);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Couldn't generate the report.");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <>
      <header className="masthead">
        <p className="masthead-eyebrow">Automated Script Coverage</p>
        <h1 className="masthead-title">Screenplay Analysis</h1>
      </header>

      {status !== "results" && (
        <>
          <UploadZone onFileSelected={handleFileSelected} disabled={status === "analyzing"} />
          {file && (
            <p className="upload-filename" style={{ textAlign: "center" }}>
              {file.name}
            </p>
          )}
          <div className="btn-row">
            <button className="btn" onClick={handleAnalyze} disabled={!file || status === "analyzing"}>
              {status === "analyzing" ? "Analysing…" : "Analyse Script"}
            </button>
          </div>
          {error && <p className="error-box">{error}</p>}
        </>
      )}

      {status === "results" && result && (
        <>
          <ResultsView result={result} onDownloadReport={handleDownloadReport} downloading={downloading} />
          <div className="btn-row" style={{ marginTop: 24 }}>
            <button className="btn btn-secondary" onClick={handleReset}>
              Analyse Another Script
            </button>
          </div>
        </>
      )}
    </>
  );
}
