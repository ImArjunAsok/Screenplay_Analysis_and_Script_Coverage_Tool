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
  most_positive_scene: string;
  most_negative_scene: string;
  turning_point_count: number;
  model_source: string;
}

export interface PredictedBeat {
  beat: string;
  scene_index: number;
  method: string;
}

export interface StoryStructure {
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
  most_central_characters: CentralCharacter[];
  top_bridge_characters: BridgeCharacter[];
}

export interface Viability {
  predicted_imdb_rating: number;
  confidence?: string;
  caveat?: string;
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
  viability: Viability;
}

export interface AnalysisFailure {
  success: false;
  error: string;
  parser_notes?: string[];
}
