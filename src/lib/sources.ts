import type { Source } from "./types";

// XYZ tile URL templates. {x}/{y}/{z} substituted by MapLibre.
export const TILE_URL: Record<Exclude<Source, "auto">, string> = {
  esri: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  google: "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
};

// "auto" is decided server-side; for the preview map we pick esri.
export function previewTileUrl(s: Source): string {
  if (s === "auto") return TILE_URL.esri;
  return TILE_URL[s];
}
