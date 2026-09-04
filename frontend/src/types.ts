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
  most_positive_scene: string;
  most_negative_scene: string;
  turning_point_count: number;
  model_source: string;
  scene_scores: number[];
  smoothed_scores: number[];
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
  sentiment_arc: SentimentArc;
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
