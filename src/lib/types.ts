// IPC contract between Svelte frontend and Tauri Rust backend.
// MUST stay byte-aligned with the structs in src-tauri/src/mocks/commands.rs
// (and later src-tauri/src/commands/*.rs once Plan B replaces the mock).

export type Bbox = [number, number, number, number]; // [minLon, minLat, maxLon, maxLat], WGS84
export type Source = "esri" | "google" | "auto";
// Known stages emitted by the backend, plus open-ended strings like
// "resuming (N/M tiles already cached)" that the UI just displays verbatim.
export type Stage =
  | "downloading"
  | "stitching"
  | "writing_cog"
  | "writing_preview"
  | (string & {});

export interface ParseVectorFileOk {
  bbox: Bbox;
  geometry: GeoJSON.Geometry;
  layer_count: number;
}
export type ParseVectorFileError =
  | { kind: "unsupported_format"; message: string }
  | { kind: "no_geometry"; message: string }
  | { kind: "io_error"; message: string };

export interface EstimateOutput {
  tile_count: number;
  pixel_w: number;
  pixel_h: number;
  est_size_mb: number;
  est_seconds: number;
}

export interface StartDownloadArgs {
  bbox: Bbox;
  zoom: number;            // 8..23
  source: Source;
  output_path: string;
  max_concurrency: number; // default 50
  retry_per_tile: number;  // default 3
  write_preview_png: boolean; // default true
}
export type StartDownloadError =
  | { kind: "invalid_bbox"; message: string }
  | { kind: "output_not_writable"; message: string };

export interface ProgressEvent {
  download_id: string;
  completed: number;
  total: number;
  bytes_downloaded: number;
  current_speed_mbps: number;
  elapsed_sec: number;
  eta_sec: number;
}

export interface TileFailedEvent {
  download_id: string;
  x: number;
  y: number;
  z: number;
  attempt: number;
  error: string;
}

export interface StageEvent {
  download_id: string;
  stage: Stage;
}

export type DoneEvent =
  | {
      download_id: string;
      ok: true;
      output_path: string;
      preview_path: string | null;
      bbox: Bbox;
      zoom: number;
      source_used: Source;
      duration_sec: number;
      total_tiles: number;
      failed_tiles: number;
      output_size_mb: number;
    }
  | { download_id: string; ok: false; error: string };

export type HistoryStatus = "in_progress" | "completed" | "cancelled" | "failed";

export interface HistoryEntry {
  bbox: Bbox;
  zoom: number;
  source: Source;
  output_path: string;
  /** Legacy success flag — true iff status === "completed" and failed_tiles === 0. */
  ok: boolean;
  duration_sec: number;
  total_tiles: number;
  failed_tiles: number;
  output_size_mb: number;
  finished_at: string; // "epoch:<unix-secs>" or ISO 8601
  /** Tiles already downloaded when this row was last persisted. */
  completed_tiles: number;
  status: HistoryStatus;
}
