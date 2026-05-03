import type { Bbox, Source, HistoryEntry, ProgressEvent, Stage } from "./types";

// Input form state — bound to InputPanel and reflected by MapPanel.
export const input = $state({
  bbox: [100, 30, 110, 40] as Bbox,
  zoom: 17,
  source: "esri" as Source,
  outputPath: "" as string,
  maxConcurrency: 50,
  retryPerTile: 3,
  writePreviewPng: true,
});

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

export function pushToast(level: "info" | "warn" | "error", text: string): void {
  const id = crypto.randomUUID();
  toasts.items = [...toasts.items, { id, level, text }];
  setTimeout(() => {
    toasts.items = toasts.items.filter((t) => t.id !== id);
  }, 4000);
}
