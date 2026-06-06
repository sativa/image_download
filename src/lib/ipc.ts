import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  Bbox, Source, EstimateOutput, StartDownloadArgs,
  ProgressEvent, TileFailedEvent, StageEvent, DoneEvent,
  HistoryEntry, ParseVectorFileOk,
  StartClassifyArgs, LandformStageEvent, LandformProgressEvent, LandformDoneEvent,
} from "./types";

// True only inside the actual Tauri webview. In `pnpm dev` (browser preview)
// the global is absent, so we short-circuit every IPC entrypoint to keep the
// UI usable for development without crashing on missing Tauri internals.
export const HAS_TAURI =
  typeof window !== "undefined" &&
  ("__TAURI_INTERNALS__" in window || "__TAURI_IPC__" in window);

const noTauriError = (cmd: string) =>
  new Error(`[browser preview] '${cmd}' requires Tauri runtime — run with 'pnpm tauri dev'`);

export function parseVectorFile(path: string): Promise<ParseVectorFileOk> {
  if (!HAS_TAURI) return Promise.reject(noTauriError("parse_vector_file"));
  return invoke("parse_vector_file", { path });
}

export function estimateOutput(bbox: Bbox, zoom: number, source: Source): Promise<EstimateOutput> {
  if (!HAS_TAURI) return Promise.reject(noTauriError("estimate_output"));
  return invoke("estimate_output", { bbox, zoom, source });
}

export function startDownload(args: StartDownloadArgs): Promise<{ download_id: string }> {
  if (!HAS_TAURI) return Promise.reject(noTauriError("start_download"));
  return invoke("start_download", { args });
}

export function cancelDownload(downloadId: string): Promise<{ ok: true }> {
  if (!HAS_TAURI) return Promise.reject(noTauriError("cancel_download"));
  return invoke("cancel_download", { downloadId });
}

export function retryFailed(downloadId: string): Promise<{ ok: true }> {
  if (!HAS_TAURI) return Promise.reject(noTauriError("retry_failed"));
  return invoke("retry_failed", { downloadId });
}

export function listHistory(): Promise<HistoryEntry[]> {
  if (!HAS_TAURI) return Promise.resolve([]);
  return invoke("list_history");
}

export function clearHistory(): Promise<{ ok: true }> {
  if (!HAS_TAURI) return Promise.resolve({ ok: true });
  return invoke("clear_history");
}

const noopUnlisten: UnlistenFn = () => {};

export function onProgress(cb: (e: ProgressEvent) => void): Promise<UnlistenFn> {
  void cb;
  if (!HAS_TAURI) return Promise.resolve(noopUnlisten);
  return listen<ProgressEvent>("download://progress", (e) => cb(e.payload));
}

export function onTileFailed(cb: (e: TileFailedEvent) => void): Promise<UnlistenFn> {
  void cb;
  if (!HAS_TAURI) return Promise.resolve(noopUnlisten);
  return listen<TileFailedEvent>("download://tile-failed", (e) => cb(e.payload));
}

export function onStage(cb: (e: StageEvent) => void): Promise<UnlistenFn> {
  void cb;
  if (!HAS_TAURI) return Promise.resolve(noopUnlisten);
  return listen<StageEvent>("download://stage", (e) => cb(e.payload));
}

export function onDone(cb: (e: DoneEvent) => void): Promise<UnlistenFn> {
  void cb;
  if (!HAS_TAURI) return Promise.resolve(noopUnlisten);
  return listen<DoneEvent>("download://done", (e) => cb(e.payload));
}

// ── Landform classification IPC ─────────────────────────────────────────

export function startClassify(args: StartClassifyArgs): Promise<{ classify_id: string }> {
  if (!HAS_TAURI) return Promise.reject(noTauriError("start_classify"));
  return invoke("start_classify", { args });
}

export function cancelClassify(classifyId: string): Promise<boolean> {
  if (!HAS_TAURI) return Promise.reject(noTauriError("cancel_classify"));
  return invoke("cancel_classify", { classifyId });
}

export function resolveHistoryTif(outputPath: string, zoom: number, source: string): Promise<string | null> {
  if (!HAS_TAURI) return Promise.resolve(null);
  return invoke("resolve_history_tif", { outputPath, zoom, source });
}

export function onLandformStage(cb: (e: LandformStageEvent) => void): Promise<UnlistenFn> {
  void cb;
  if (!HAS_TAURI) return Promise.resolve(noopUnlisten);
  return listen<LandformStageEvent>("landform://stage", (e) => cb(e.payload));
}

export function onLandformProgress(cb: (e: LandformProgressEvent) => void): Promise<UnlistenFn> {
  void cb;
  if (!HAS_TAURI) return Promise.resolve(noopUnlisten);
  return listen<LandformProgressEvent>("landform://progress", (e) => cb(e.payload));
}

export function onLandformDone(cb: (e: LandformDoneEvent) => void): Promise<UnlistenFn> {
  void cb;
  if (!HAS_TAURI) return Promise.resolve(noopUnlisten);
  return listen<LandformDoneEvent>("landform://done", (e) => cb(e.payload));
}
