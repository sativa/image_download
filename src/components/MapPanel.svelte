<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import maplibregl from "maplibre-gl";
  import "maplibre-gl/dist/maplibre-gl.css";
  import { input } from "../lib/state.svelte";
  import { previewTileUrl } from "../lib/sources";

  let container: HTMLDivElement;
  let map: maplibregl.Map | null = null;

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
  });

  onDestroy(() => map?.remove());
</script>

<div class="wrap" bind:this={container}></div>

<style>
  .wrap { width: 100%; height: 100%; }
  :global(.maplibregl-canvas) { outline: none; }
</style>
