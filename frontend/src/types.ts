// Mirrors the JSON shape returned by backend/pipeline.py's
// analyze_screenplay(). Keep these in sync if the backend response
// shape changes -- there's no runtime schema validation, just this
// compile-time contract.

export interface OverviewStats {
  scene_count: number;
  character_count: number;
  dialogue_count: number;
}

export interface CharacterBreakdown {
  all_characters: string[];
  likely_real_names: string[];
  likely_role_labels: string[];
  uncertain: string[];
}

export interface SentimentArc {
  average_sentiment: number;
  sentiment_label: string;
  sentiment_label_caveat: string;
  emotional_volatility: number;
  emotional_volatility_label: string;
  arc_interpretation: string;
  most_positive_scene: string;
  most_negative_scene: string;
  turning_point_count: number;
  model_source: string;
  scene_scores: number[];
  smoothed_scores: number[];
}

export interface CharacterArc {
  character: string;
  scene_appearances: number;
  introduction_sentiment: number;
  midpoint_sentiment: number;
  final_sentiment: number;
  arc_direction: string;
  arc_strength: number;
}

export interface DialogueShare {
  character: string;
  dialogue_lines: number;
  share_pct: number;
}

export interface PacingOutlier {
  scene_index: number;
  heading: string;
  dialogue_density: number;
  type: string;
}

export interface PacingAnalysis {
  scene_count: number;
  average_scene_length_lines: number;
  shortest_scene: { scene_index: number; heading: string; length_lines: number };
  longest_scene: { scene_index: number; heading: string; length_lines: number };
  average_dialogue_density: number;
  pacing_outliers: PacingOutlier[];
  pacing_note: string;
}

export interface PredictedBeat {
  beat: string;
  scene_index: number;
  method: string;
  confidence: string;
}

export interface StoryStructure {
  detection_type: string;
  detection_note: string;
  predicted_beats: PredictedBeat[];
}

export interface CentralCharacter {
  name: string;
  degree: number;
  weighted_degree: number;
  betweenness_centrality: number;
  closeness_centrality: number;
}

export interface BridgeCharacter {
  name: string;
  betweenness: number;
}

export interface CharacterRelationships {
  character_count_in_network: number;
  relationship_count: number;
  likely_protagonist: string | null;
  top_bridge_character: string | null;
  network_interpretation: string | null;
  most_central_characters: CentralCharacter[];
  top_bridge_characters: BridgeCharacter[];
}

export interface Viability {
  predicted_imdb_rating: number;
  confidence?: string;
  caveat?: string;
}

export interface GenreConfidence {
  genre: string;
  probability: number;
}

export interface AnalysisResult {
  success: true;
  title: string;
  overview: OverviewStats;
  parser_notes: string[];
  characters: CharacterBreakdown;
  dialogue_distribution: DialogueShare[];
  pacing: PacingAnalysis;
  sentiment_arc: SentimentArc;
  character_arcs: CharacterArc[];
  story_structure: StoryStructure;
  character_relationships: CharacterRelationships;
  predicted_genres: string[];
  genre_confidence: GenreConfidence[];
  viability: Viability;
  limitations: string[];
}

export interface AnalysisFailure {
  success: false;
  error: string;
  parser_notes?: string[];
}
