<script lang="ts">
  import { input, estimate, download, pushToast, requestMapFit } from "../lib/state.svelte";
  import { validateBbox, validateZoom } from "../lib/validate";
  import type { Source } from "../lib/types";
  import { open } from "@tauri-apps/plugin-dialog";
  import { estimateOutput, startDownload, parseVectorFile } from "../lib/ipc";
  import { formatNumber, formatDuration } from "../lib/format";

  type Mode = "numeric" | "draw" | "import";
  let mode = $state<Mode>("numeric");

  let bboxErr = $derived(validateBbox(input.bbox));
  let zoomErr = $derived(validateZoom(input.zoom));

  let debounceTimer: number | null = null;
  $effect(() => {
    // Re-runs whenever bbox/zoom/source mutate.
    const b = [...input.bbox];
    const z = input.zoom;
    const s = input.source;

    if (bboxErr || zoomErr) {
      estimate.data = null;
      estimate.loading = false;
      estimate.error = null;
      return;
    }

    if (debounceTimer) clearTimeout(debounceTimer);
    estimate.loading = true;
    debounceTimer = setTimeout(async () => {
      try {
        estimate.data = await estimateOutput(b as [number, number, number, number], z, s);
        estimate.error = null;
      } catch (e) {
        estimate.error = String(e);
      } finally {
        estimate.loading = false;
      }
    }, 200) as unknown as number;
  });

  const sources: Source[] = ["esri", "google", "auto"];

  let canStart = $derived(
    !bboxErr && !zoomErr && input.outputPath.length > 0 && download.id === null,
  );

  let blockedReason = $derived.by(() => {
    if (download.id) return "Download in progress…";
    if (bboxErr) return bboxErr;
    if (zoomErr) return zoomErr;
    if (input.outputPath.length === 0) return "Pick an output folder first";
    return null;
  });

  async function start() {
    try {
      download.finished = false;
      download.error = null;
      download.progress = null;
      download.failedTiles = 0;
      download.stage = null;
      const r = await startDownload({
        bbox: input.bbox,
        zoom: input.zoom,
        source: input.source,
        output_path: input.outputPath,
        max_concurrency: input.maxConcurrency,
        retry_per_tile: input.retryPerTile,
        write_preview_png: input.writePreviewPng,
      });
      download.id = r.download_id;
    } catch (e) {
      pushToast("error", String(e));
    }
  }

  async function pickVector() {
    const p = await open({
      title: "Select vector file",
      filters: [
        { name: "Vector", extensions: ["geojson", "json", "shp", "gpkg", "parquet", "geoparquet"] },
      ],
    });
    if (!p || Array.isArray(p)) return;
    try {
      const r = await parseVectorFile(p);
      input.bbox = r.bbox;
      // Force the map to pan/zoom to the imported bbox even if the new region
      // happens to fall inside the current view (anti-jitter would otherwise skip).
      requestMapFit();
      pushToast("info", `Loaded ${r.layer_count} layer(s) from ${p}`);
    } catch (e) {
      pushToast("warn", `Failed to parse vector file: ${String(e)}`);
    }
  }

  async function pickOutput() {
    // Output is now a directory; the backend auto-names each run as
    // `imagery_z{zoom}_{source}_{ts}.tif` inside it (see make_output_stem
    // in src-tauri/src/commands/download.rs).
    const p = await open({
      title: "Choose output folder",
      directory: true,
      multiple: false,
    });
    if (typeof p === "string") input.outputPath = p;
  }
</script>

<section class="panel">
  <nav class="tabs">
    <button class:active={mode === "numeric"} onclick={() => (mode = "numeric")}>Numeric</button>
    <button class:active={mode === "draw"} onclick={() => (mode = "draw")}>Draw</button>
    <button class:active={mode === "import"} onclick={() => (mode = "import")}>Import</button>
  </nav>

  {#if mode === "numeric"}
    <div class="grid">
      <label>min Lon
        <input type="number" step="any" class:invalid={bboxErr}
               bind:value={input.bbox[0]} />
      </label>
      <label>min Lat
        <input type="number" step="any" class:invalid={bboxErr}
               bind:value={input.bbox[1]} />
      </label>
      <label>max Lon
        <input type="number" step="any" class:invalid={bboxErr}
               bind:value={input.bbox[2]} />
      </label>
      <label>max Lat
        <input type="number" step="any" class:invalid={bboxErr}
               bind:value={input.bbox[3]} />
      </label>
    </div>
    {#if bboxErr}<p class="err">{bboxErr}</p>{/if}
  {:else if mode === "draw"}
    <p class="muted">Draw a rectangle on the map. Coordinates appear here once you release.</p>
  {:else}
    <p class="muted">Pick a .geojson / .shp / .gpkg / .parquet (GeoParquet) file:</p>
    <button onclick={pickVector}>Choose file…</button>
    <p class="muted small">Parsing is implemented by Plan A; picker will wire through then.</p>
  {/if}

  <hr />

  <label>Zoom <span class="hint">{input.zoom}</span>
    <input type="range" min="8" max="23" step="1" bind:value={input.zoom} />
  </label>
  {#if zoomErr}<p class="err">{zoomErr}</p>{/if}

  <label>Source
    <select bind:value={input.source}>
      {#each sources as s}<option value={s}>{s}</option>{/each}
    </select>
  </label>

  <div class="grid">
    <label>Concurrent downloads <span class="hint">{input.maxConcurrency}</span>
      <input
        type="range"
        min="1"
        max="64"
        step="1"
        bind:value={input.maxConcurrency}
      />
      <p class="muted small">
        Parallel HTTP requests. Higher is faster but may trigger rate limits / 429.
        Recommended 8–32 for Esri/Google.
      </p>
    </label>
    <label>Retries per tile <span class="hint">{input.retryPerTile}</span>
      <input
        type="number"
        min="0"
        max="10"
        step="1"
        bind:value={input.retryPerTile}
      />
      <p class="muted small">Per-tile retry attempts before giving up.</p>
    </label>
  </div>

  <label>Output folder
    <div class="row">
      <input type="text" placeholder="… choose a folder for the imagery" readonly value={input.outputPath} />
      <button onclick={pickOutput}>Pick folder…</button>
    </div>
    <p class="muted small">Files will be named <code>imagery_z{input.zoom}_{input.source}_&lt;timestamp&gt;.tif</code> inside the folder.</p>
  </label>

  <hr />
  <div class="estimate">
    {#if estimate.loading}
      <em class="muted">computing estimate…</em>
    {:else if estimate.error}
      <em class="err">{estimate.error}</em>
    {:else if estimate.data}
      <div>{formatNumber(estimate.data.tile_count)} tiles · {estimate.data.pixel_w} × {estimate.data.pixel_h} px</div>
      <div>≈ {estimate.data.est_size_mb.toFixed(1)} MB · {formatDuration(estimate.data.est_seconds)}</div>
    {:else}
      <em class="muted">enter a valid bbox to see estimate</em>
    {/if}
  </div>

  <button class="primary" disabled={!canStart} onclick={start}>
    {download.id ? "Downloading…" : "Start download"}
  </button>
  {#if blockedReason}
    <p class="hint-blocked">{blockedReason}</p>
  {/if}
</section>

<style>
  .panel { padding: 1rem; display: flex; flex-direction: column; gap: 0.7rem; }
  .tabs { display: flex; gap: 0.3rem; }
  .tabs button { flex: 1; }
  .tabs .active { background: var(--accent); color: white; }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
  }
  label { display: flex; flex-direction: column; font-size: 0.85rem; gap: 0.2rem; }
  .hint { float: right; color: var(--fg-muted); }
  .err { color: var(--error); font-size: 0.8rem; margin: 0; }
  .muted { color: var(--fg-muted); font-size: 0.9rem; }
  hr { border: none; border-top: 1px solid var(--border); margin: 0.3rem 0; }
  .row { display: flex; gap: 0.3rem; }
  .row input { flex: 1; }
  .estimate {
    background: var(--bg-elev);
    padding: 0.5rem 0.7rem;
    border-radius: 6px;
    font-size: 0.85rem;
  }
  .hint-blocked {
    color: var(--fg-muted);
    font-size: 0.8rem;
    margin: -0.3rem 0 0;
    text-align: center;
  }
  .primary {
    background: var(--accent);
    color: white;
    border-color: var(--accent);
    padding: 0.6rem;
    font-weight: 600;
  }
  .small { font-size: 0.75rem; }
</style>
