import type { Bbox, Source, HistoryEntry, ProgressEvent, Stage } from "./types";

// Input form state — bound to InputPanel and reflected by MapPanel.
export const input = $state({
  bbox: [100, 30, 110, 40] as Bbox,
  zoom: 17,
  source: "esri" as Source,
  outputPath: "" as string,
  // 16 is a safe default for Esri/Google tile servers; higher rates can trigger 429.
  // User can crank to 64 in the UI if they're hitting their own / unthrottled source.
  maxConcurrency: 16,
  retryPerTile: 3,
  writePreviewPng: true,
});

// Monotonically-incrementing token. Bumped from InputPanel.pickVector after a
// successful import so MapPanel can fit the new bbox unconditionally —
// bypassing the anti-jitter "already in view, skip" guard that's correct for
// rectangle dragging but wrong for "I just dropped a file from another region".
export const fitRequest = $state({ token: 0 });

export function requestMapFit(): void {
  fitRequest.token += 1;
}

// Latest estimate, refreshed on bbox/zoom/source change (debounced 200ms).
export const estimate = $state<{
  loading: boolean;
  data: { tile_count: number; pixel_w: number; pixel_h: number; est_size_mb: number; est_seconds: number } | null;
  error: string | null;
}>({ loading: false, data: null, error: null });

// Active download status. Null when no download in flight.
export const download = $state<{
  id: string | null;
  stage: Stage | null;
  progress: ProgressEvent | null;
  failedTiles: number;
  finished: boolean;
  error: string | null;
}>({
  id: null,
  stage: null,
  progress: null,
  failedTiles: 0,
  finished: false,
  error: null,
});

// History — populated from list_history on mount, refreshed after each done.
export const history = $state<{ entries: HistoryEntry[] }>({ entries: [] });

// Toasts — append, auto-removed by Toast component after 4s.
export const toasts = $state<{ items: { id: string; level: "info" | "warn" | "error"; text: string }[] }>({
  items: [],
});

// Active landform classification overlay. Set by HistoryPanel when a
// classify run completes successfully; cleared by the user via the
// MapPanel "remove overlay" control. MapPanel watches this and adds /
// updates a MapLibre image source so the colored landform raster sits
// on top of the downloaded imagery.
//
// `bbox` is the source download's WGS84 bbox — we trust it instead of
// re-reading the GeoTIFF's CRS, because the COG was already written in
// EPSG:3857 from this same bbox.
export const landformOverlay = $state<{
  /** WGS84 GeoJSON path; MapPanel fetches and renders as fill layer. */
  geojsonPath: string | null;
  /** Camera fit hint — derived from the actual GPKG bbox so the map can
   *  pan even before the GeoJSON FeatureCollection loads. */
  bbox: Bbox | null;
  classifyId: string | null;
  /** Toggle for the map layer — true: drawn, false: hidden but kept in
   *  state so the user can re-show without re-running. Clear() resets
   *  the source entirely. */
  visible: boolean;
}>({
  geojsonPath: null,
  bbox: null,
  classifyId: null,
  visible: true,
});

export function setLandformOverlay(geojsonPath: string, bbox: Bbox, classifyId: string): void {
  landformOverlay.geojsonPath = geojsonPath;
  landformOverlay.bbox = bbox;
  landformOverlay.classifyId = classifyId;
  landformOverlay.visible = true;
}

export function toggleLandformOverlay(): void {
  if (!landformOverlay.geojsonPath) return;
  landformOverlay.visible = !landformOverlay.visible;
}

export function clearLandformOverlay(): void {
  landformOverlay.geojsonPath = null;
  landformOverlay.bbox = null;
  landformOverlay.classifyId = null;
  landformOverlay.visible = true;
}

export function pushToast(level: "info" | "warn" | "error", text: string): void {
  const id = crypto.randomUUID();
  toasts.items = [...toasts.items, { id, level, text }];
  setTimeout(() => {
    toasts.items = toasts.items.filter((t) => t.id !== id);
  }, 4000);
}
