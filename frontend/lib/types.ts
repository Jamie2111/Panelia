export type SourceType = "mangadex_url" | "comix_to_url" | "zip" | "pdf" | "images" | "folder";
export type DuplicateHandlingMode = "auto_pick_best" | "prefer_official" | "prefer_fan" | "prefer_consistent_group";
export type StageStatus = "pending" | "ready" | "running" | "completed" | "failed" | "cancelled" | "needs_review";
export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type PipelineStage =
  | "ingestion"
  | "panel_detection"
  | "panel_review"
  | "character_review"
  | "character_portrait"
  | "panel_vision_extraction"
  | "panel_vision_quality"
  | "script_generation"
  | "narration_generation"
  | "video_rendering"
  | "youtube_bundle";

export type Orientation = "landscape" | "vertical";
export type PanelLayout = "card" | "fullscreen";
export type OutputFormat = "mp4" | "mov";

export interface PanelBox {
  id: string;
  page: number;
  panel: number;
  x: number;
  y: number;
  width: number;
  height: number;
  order: number;
  keep: boolean;
  duration_seconds?: number | null;
  narration?: string | null;
  zoom_hint?: string | null;
  merged_from: string[];
  ocr_text?: string | null;
  text_detected?: boolean | null;
  auto_skipped: boolean;
  skip_reason?: string | null;
  manual_keep: boolean;
  manual_narration: boolean;
  narration_locked: boolean;
  manual_ocr_text: boolean;
  visual_caption?: string | null;
  narration_source?:
    | "ocr"
    | "vision_caption"
    | "fallback"
    | "panel_vision_narrator"
    | "vision_failed"
    | "vision_needs_regenerate"
    | "manual"
    | "restored_backup_20260430"
    | "restored_and_generated"
    | "aligned_visual_order"
    | "aligned_to_visual_order"
    | string
    | null;
  review_flags: string[];
}

export type PanelRewriteMode = "balanced" | "closer_to_ocr" | "shorten";

export interface PanelRewriteResponse {
  panel_id: string;
  narration: string;
  mode: PanelRewriteMode;
}

export interface StageState {
  stage: PipelineStage;
  status: StageStatus;
  progress: number;
  message?: string | null;
  updated_at: string;
}

export interface VoiceConfig {
  voice: string;
  lang_code: string;
  speed: number;
}

export interface MusicTrack {
  name: string;
  file: string;
  duration_seconds?: number;
  mood?: string;
  available: boolean;
  url?: string | null;
  path?: string;
  source?: "builtin" | "uploaded" | string;
}

export interface MusicConfig {
  enabled: boolean;
  track_name?: string | null;
  volume: number;
  fade_in_seconds: number;
  fade_out_seconds: number;
}

export interface VideoConfig {
  width: number;
  height: number;
  orientation: Orientation;
  panel_layout: PanelLayout;
  intro_thumbnail_enabled: boolean;
  intro_thumbnail_seconds: number;
  output_format: OutputFormat;
  fps: number;
  background_color: string;
}

export type NarrationMode = "panel";

export interface PipelineConfig {
  auto_run_end_to_end: boolean;
  narration_mode: NarrationMode;
}

export interface ChapterMetadata {
  chapter_id?: string | null;
  source_url?: string | null;
  manga_title?: string | null;
  chapter_title?: string | null;
  chapter_number?: string | null;
  volume_number?: string | null;
  language?: string | null;
  page_count?: number | null;
  raw: Record<string, unknown>;
}

export interface JobRecord {
  id: string;
  project_id: string;
  stage: PipelineStage;
  status: JobStatus;
  progress: number;
  message?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  payload: Record<string, unknown>;
}

export interface AudioFile {
  panel_id: string;
  path: string;
  url: string;
  duration_seconds: number;
}

export interface VideoFile {
  name: string;
  path: string;
  url: string;
  width: number;
  height: number;
  output_format: OutputFormat;
  created_at: string;
  duration_seconds?: number | null;
}

export interface ProjectSummary {
  id: string;
  name: string;
  source_type: SourceType;
  source_reference?: string | null;
  created_at: string;
  updated_at: string;
  chapter_metadata: ChapterMetadata;
  stage_states: Record<PipelineStage, StageState>;
  page_count: number;
  panel_count: number;
  kept_panel_count: number;
  thumbnail_url?: string | null;
  video_thumbnail_url?: string | null;
  active_jobs: JobRecord[];
  latest_video?: VideoFile | null;
  voice_config: VoiceConfig;
  video_config: VideoConfig;
  music_config: MusicConfig;
  pipeline_config: PipelineConfig;
}

export interface ProjectDetail extends ProjectSummary {
  panels: PanelBox[];
  script_lines: string[];
  script_story?: string | null;
  story_segments: StorySegment[];
  script_display_metadata?: ScriptDisplayMetadata;
  audio_files: AudioFile[];
  videos: VideoFile[];
  available_music_tracks: MusicTrack[];
}

export interface ScriptDisplayMetadata {
  displayed_script_path?: string | null;
  displayed_script_job_id?: string | null;
  displayed_script_created_at?: string | null;
  latest_job_id?: string | null;
  latest_job_status?: string | null;
  latest_completed_script_path?: string | null;
  latest_completed_script_job_id?: string | null;
  is_displaying_stale_script?: boolean;
  stale_reason?: string | null;
}

export interface StorySegment {
  id: string;
  order: number;
  text: string;
  keep?: boolean;
  panel_ids: string[];
  panel_start?: number | null;
  panel_end?: number | null;
  scene_id?: number | null;
  title?: string | null;
  representative_panel_id?: string | null;
  visual_only?: boolean;
  suppression_reason?: string | null;
}

export type CharacterReviewStatus = "suggested" | "confirmed" | "unknown";

export interface CharacterReviewSample {
  sample_id: string;
  image_url?: string | null;
  image_path?: string | null;
  panel_id?: string | null;
  page?: number | null;
  panel?: number | null;
  bbox: number[];
}

export interface CharacterReviewIdentity {
  review_id: string;
  stable_character_ids: string[];
  source_character_ids: string[];
  suggested_name?: string | null;
  remembered_name?: string | null;
  memory_matches: string[];
  name?: string | null;
  status: CharacterReviewStatus;
  role_hint?: string | null;
  appearance_count: number;
  pages: number[];
  panel_ids: string[];
  sample_images: CharacterReviewSample[];
  notes?: string | null;
}

export interface CharacterReviewState {
  project_id: string;
  series_key: string;
  protagonist_name?: string | null;
  memory_names: string[];
  identities: CharacterReviewIdentity[];
  generated_at: string;
  updated_at: string;
}

export interface CharacterDictionaryEntry {
  key: string;
  name: string;
}

export interface CharacterDictionaryResponse {
  project_id: string;
  entries: CharacterDictionaryEntry[];
}

export interface CanonicalCharacter {
  stable_id: string;
  name: string;
  role: string;
  visual_description: string;
  portrait_panel_ids: string[];
  portrait_pages: number[];
  confidence?: number | null;
  aliases: string[];
}

export interface CharacterPortraitsResponse {
  project_id: string;
  characters: CanonicalCharacter[];
}

export interface LanguageOption {
  code: string;
  label: string;
  description: string;
  sample_text: string;
}

export interface VoiceOption {
  id: string;
  lang_code: string;
  label: string;
  description: string;
  quality_note?: string | null;
  style_tags: string[];
}

export interface CatalogOptions {
  languages: LanguageOption[];
  voices: VoiceOption[];
  music_tracks: MusicTrack[];
}

export interface DetectorTrainingStatus {
  training_status: "idle" | "running" | "completed" | "failed" | string;
  is_training: boolean;
  ready_to_train: boolean;
  progress_percent: number;
  current_epoch: number;
  total_epochs: number;
  train_loss?: number | null;
  val_loss?: number | null;
  min_new_annotations: number;
  panel_annotations_total: number;
  new_panel_annotations: number;
  ocr_annotations_total: number;
  new_ocr_annotations: number;
  remaining_annotations_until_ready: number;
  last_trained_at?: string | null;
  checkpoint_path?: string | null;
  log_path?: string | null;
  pid?: number | null;
  message?: string | null;
}
