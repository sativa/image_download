<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import maplibregl from "maplibre-gl";
  import "maplibre-gl/dist/maplibre-gl.css";
  import { input } from "../lib/state.svelte";
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

    map.on("mousedown", (e) => {
      if (!drawing) return;
      drawStart = e.lngLat;
      e.preventDefault();
    });
    map.on("mousemove", (e) => {
      if (!drawing || !drawStart) return;
      drawRect = new maplibregl.LngLatBounds(drawStart, e.lngLat);
      drawPreviewLayer();
    });
    map.on("mouseup", (e) => {
      if (!drawing || !drawStart) return;
      const bounds = new maplibregl.LngLatBounds(drawStart, e.lngLat);
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
