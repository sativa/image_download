import type { Bbox } from "./types";

export function validateBbox(b: Bbox): string | null {
  if (b.some((x) => !Number.isFinite(x))) return "All bbox values must be finite numbers";
  const [minLon, minLat, maxLon, maxLat] = b;
  if (minLon < -180 || minLon > 180 || maxLon < -180 || maxLon > 180)
    return "Longitude must be in -180..180";
  if (minLat < -90 || minLat > 90 || maxLat < -90 || maxLat > 90)
    return "Latitude must be in -90..90";
  if (minLon === maxLon && minLat === maxLat) return "bbox has zero area";
  if (minLon >= maxLon) return "minLongitude must be less than maxLongitude";
  if (minLat >= maxLat) return "minLatitude must be less than maxLatitude";
  return null;
}

export function validateZoom(z: number): string | null {
  if (!Number.isInteger(z)) return "zoom must be an integer";
  if (z < 8 || z > 23) return "zoom must be in range 8..23";
  return null;
}
