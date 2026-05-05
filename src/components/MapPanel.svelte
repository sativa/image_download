<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import maplibregl from "maplibre-gl";
  import "maplibre-gl/dist/maplibre-gl.css";
  import { input, fitRequest } from "../lib/state.svelte";
  import { previewTileUrl } from "../lib/sources";
  import { parseVectorFile } from "../lib/ipc";
  import { pushToast } from "../lib/state.svelte";

  let dragOver = $state(false);

  async function handleDrop(e: DragEvent) {
    e.preventDefault();
    dragOver = false;
    const files = Array.from(e.dataTransfer?.files || []);
    if (!files.length) return;
    const f = files[0];
    pushToast(
      "warn",
      `Vector file dropped: ${f.name} — drag-drop parsing pending Plan A. Use the picker for now.`,
    );
    void parseVectorFile; // imported for Plan B; mark used
  }

  let container: HTMLDivElement;
  let map: maplibregl.Map | null = null;

  // Live mirrors of the map's current view, updated by MapLibre move/zoom events.
  let viewZoom = $state(4);
  let viewCenterLat = $state(35);

  let drawing = $state(false);
  let drawStart: maplibregl.LngLat | null = null;
  let drawRect: maplibregl.LngLatBounds | null = null;

  function bboxRing(b: maplibregl.LngLatBounds): GeoJSON.Feature<GeoJSON.Polygon> {
    const w = b.getWest(), s = b.getSouth(), e = b.getEast(), n = b.getNorth();
    return {
      type: "Feature",
      properties: {},
      geometry: {
        type: "Polygon",
        coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
      },
    };
  }

  function drawPreviewLayer() {
    if (!map || !drawRect) return;
    const ring = bboxRing(drawRect);
    if (map.getSource("draw-preview")) {
      (map.getSource("draw-preview") as maplibregl.GeoJSONSource).setData(ring);
    } else {
      map.addSource("draw-preview", { type: "geojson", data: ring });
      map.addLayer({
        id: "draw-preview",
        type: "line",
        source: "draw-preview",
        paint: { "line-color": "#0066cc", "line-width": 2, "line-dasharray": [2, 2] },
      });
    }
  }

  function persistBboxLayer(b: maplibregl.LngLatBounds) {
    if (!map) return;
    const ring = bboxRing(b);
    if (map.getLayer("draw-preview")) map.removeLayer("draw-preview");
    if (map.getSource("draw-preview")) map.removeSource("draw-preview");
    if (map.getSource("bbox")) {
      (map.getSource("bbox") as maplibregl.GeoJSONSource).setData(ring);
    } else {
      map.addSource("bbox", { type: "geojson", data: ring });
      map.addLayer({
        id: "bbox-fill",
        type: "fill",
        source: "bbox",
        paint: { "fill-color": "#0066cc", "fill-opacity": 0.15 },
      });
      map.addLayer({
        id: "bbox-line",
        type: "line",
        source: "bbox",
        paint: { "line-color": "#0066cc", "line-width": 2 },
      });
    }
  }

  function enableDraw() {
    if (!map) return;
    drawing = true;
    map.getCanvas().style.cursor = "crosshair";
    map.dragPan.disable();
  }

  function disableDraw() {
    drawing = false;
    if (map) {
      map.getCanvas().style.cursor = "";
      map.dragPan.enable();
    }
    drawStart = null;
    drawRect = null;
  }

  onMount(() => {
    map = new maplibregl.Map({
      container,
      style: {
        version: 8,
        sources: {
          base: {
            type: "raster",
            tiles: [previewTileUrl(input.source)],
            tileSize: 256,
            maxzoom: 22,
            attribution: input.source === "google" ? "© Google" : "© Esri, Maxar",
          },
        },
        layers: [{ id: "base", type: "raster", source: "base" }],
      },
      center: [(input.bbox[0] + input.bbox[2]) / 2, (input.bbox[1] + input.bbox[3]) / 2],
      zoom: 4,
    });

    // Track view state for the badge — updates live as user pans/zooms.
    const syncView = () => {
      if (!map) return;
      viewZoom = map.getZoom();
      viewCenterLat = map.getCenter().lat;
    };
    syncView();
    map.on("zoom", syncView);
    map.on("move", syncView);

    // Normalize two LngLat points into proper (sw, ne) regardless of drag direction.
    // MapLibre's LngLatBounds constructor doesn't auto-normalize — passing (NE, SW)
    // gives you a bbox with min > max coordinates and breaks downstream validation.
    const normalize = (a: maplibregl.LngLat, b: maplibregl.LngLat) => {
      const sw = new maplibregl.LngLat(Math.min(a.lng, b.lng), Math.min(a.lat, b.lat));
      const ne = new maplibregl.LngLat(Math.max(a.lng, b.lng), Math.max(a.lat, b.lat));
      return new maplibregl.LngLatBounds(sw, ne);
    };

    map.on("mousedown", (e) => {
      if (!drawing) return;
      drawStart = e.lngLat;
      e.preventDefault();
    });
    map.on("mousemove", (e) => {
      if (!drawing || !drawStart) return;
      drawRect = normalize(drawStart, e.lngLat);
      drawPreviewLayer();
    });
    map.on("mouseup", (e) => {
      if (!drawing || !drawStart) return;
      const bounds = normalize(drawStart, e.lngLat);
      input.bbox = [
        bounds.getWest(), bounds.getSouth(),
        bounds.getEast(), bounds.getNorth(),
      ];
      disableDraw();
      persistBboxLayer(bounds);
    });
  });

  onDestroy(() => map?.remove());

  $effect(() => {
    if (!map) return;
    const url = previewTileUrl(input.source);
    const src = map.getSource("base") as maplibregl.RasterTileSource | undefined;
    if (src) src.setTiles([url]);
  });

  $effect(() => {
    if (!map) return;
    const [w, s, e, n] = input.bbox;
    if (![w, s, e, n].every(Number.isFinite)) return;
    // Avoid jitter: only fit if currently outside view
    const cur = map.getBounds();
    if (cur.contains([w, s]) && cur.contains([e, n])) return;
    map.fitBounds([[w, s], [e, n]], { padding: 40, animate: false });
    persistBboxLayer(new maplibregl.LngLatBounds([w, s], [e, n]));
  });

  // Force-fit: triggered after a file import even if the bbox happens to fall
  // inside the current view. We bump fitRequest.token from InputPanel and
  // animate here so the user sees the camera move.
  let lastFitToken = 0;
  $effect(() => {
    const t = fitRequest.token;
    if (!map) return;
    if (t === lastFitToken) return; // skip mount-time pass and re-runs from other deps
    lastFitToken = t;
    const [w, s, e, n] = input.bbox;
    if (![w, s, e, n].every(Number.isFinite)) return;
    map.fitBounds([[w, s], [e, n]], { padding: 60, animate: true, duration: 600 });
    persistBboxLayer(new maplibregl.LngLatBounds([w, s], [e, n]));
  });

  // Web-mercator ground sample distance at integer or fractional zoom z and latitude lat.
  // Formula: (Earth_circumference_m * cos(lat)) / (2^z * tile_pixels)
  function mpp(z: number, lat: number): number {
    const cos = Math.cos((lat * Math.PI) / 180);
    return (40075016.686 * Math.abs(cos)) / (Math.pow(2, z) * 256);
  }

  // What the user sees on the map RIGHT NOW (z and m/px at view center).
  let viewMpp = $derived(mpp(viewZoom, viewCenterLat));

  // What the download will produce (slider z + bbox center latitude).
  let downloadMpp = $derived.by(() => {
    const lat = (input.bbox[1] + input.bbox[3]) / 2;
    if (!Number.isFinite(lat)) return null;
    return mpp(input.zoom, lat);
  });

  function fmtMpp(m: number): string {
    if (m < 1) return `${(m * 100).toFixed(1)} cm/px`;
    if (m < 1000) return `${m.toFixed(1)} m/px`;
    return `${(m / 1000).toFixed(2)} km/px`;
  }
</script>

<div
  class="wrap"
  ondragover={(e) => { e.preventDefault(); dragOver = true; }}
  ondragleave={() => (dragOver = false)}
  ondrop={handleDrop}
>
  <div class="map" bind:this={container}></div>
  <div class="controls">
    {#if drawing}
      <button onclick={disableDraw}>Cancel draw</button>
    {:else}
      <button onclick={enableDraw}>Draw rectangle</button>
    {/if}
  </div>
  <div class="zoom-badge">
    <div class="row">
      <span class="label">view</span>
      <span>z{viewZoom.toFixed(1)} · {fmtMpp(viewMpp)}</span>
    </div>
    {#if downloadMpp !== null}
      <div class="row dl">
        <span class="label">dl</span>
        <strong>z{input.zoom}</strong>
        <span>· {fmtMpp(downloadMpp)}</span>
      </div>
    {/if}
  </div>
  {#if dragOver}
    <div class="drop-overlay">Drop vector file…</div>
  {/if}
</div>

<style>
  .wrap { position: relative; width: 100%; height: 100%; }
  .map { width: 100%; height: 100%; }
  :global(.maplibregl-canvas) { outline: none; }
  .controls {
    position: absolute;
    top: 1rem;
    left: 1rem;
    z-index: 10;
  }
  .zoom-badge {
    position: absolute;
    top: 1rem;
    right: 1rem;
    z-index: 10;
    background: rgba(0, 0, 0, 0.65);
    color: white;
    padding: 0.4rem 0.7rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-family: ui-monospace, Menlo, monospace;
    pointer-events: none;
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    line-height: 1.2;
  }
  .zoom-badge .row { display: flex; gap: 0.4rem; align-items: baseline; }
  .zoom-badge .label {
    color: #9aa0a6;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    width: 2.2em;
  }
  .zoom-badge .dl strong { color: #4fb0ff; font-weight: 600; }
  .drop-overlay {
    position: absolute;
    inset: 0;
    background: rgba(0, 102, 204, 0.4);
    color: white;
    display: grid;
    place-items: center;
    font-size: 1.5rem;
    pointer-events: none;
    z-index: 20;
  }
</style>
