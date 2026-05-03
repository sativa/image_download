import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  Bbox, Source, EstimateOutput, StartDownloadArgs,
  ProgressEvent, TileFailedEvent, StageEvent, DoneEvent,
  HistoryEntry, ParseVectorFileOk,
} from "./types";

export function parseVectorFile(path: string): Promise<ParseVectorFileOk> {
  return invoke("parse_vector_file", { path });
}

export function estimateOutput(bbox: Bbox, zoom: number, source: Source): Promise<EstimateOutput> {
  return invoke("estimate_output", { bbox, zoom, source });
}

export function startDownload(args: StartDownloadArgs): Promise<{ download_id: string }> {
  return invoke("start_download", { args });
}

export function cancelDownload(downloadId: string): Promise<{ ok: true }> {
  return invoke("cancel_download", { downloadId });
}

export function retryFailed(downloadId: string): Promise<{ ok: true }> {
  return invoke("retry_failed", { downloadId });
}

export function listHistory(): Promise<HistoryEntry[]> {
  return invoke("list_history");
}

export function clearHistory(): Promise<{ ok: true }> {
  return invoke("clear_history");
}

export function onProgress(cb: (e: ProgressEvent) => void): Promise<UnlistenFn> {
  return listen<ProgressEvent>("download://progress", (e) => cb(e.payload));
}

export function onTileFailed(cb: (e: TileFailedEvent) => void): Promise<UnlistenFn> {
  return listen<TileFailedEvent>("download://tile-failed", (e) => cb(e.payload));
}

export function onStage(cb: (e: StageEvent) => void): Promise<UnlistenFn> {
  return listen<StageEvent>("download://stage", (e) => cb(e.payload));
}

export function onDone(cb: (e: DoneEvent) => void): Promise<UnlistenFn> {
  return listen<DoneEvent>("download://done", (e) => cb(e.payload));
}
