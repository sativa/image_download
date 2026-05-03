<script lang="ts">
  import { input } from "../lib/state.svelte";
  import { validateBbox, validateZoom } from "../lib/validate";
  import type { Source } from "../lib/types";

  type Mode = "numeric" | "draw" | "import";
  let mode = $state<Mode>("numeric");

  let bboxErr = $derived(validateBbox(input.bbox));
  let zoomErr = $derived(validateZoom(input.zoom));

  const sources: Source[] = ["esri", "google", "auto"];
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
    <p class="muted">Drag a .geojson / .shp / .gpkg into the map area, or use the picker:</p>
    <button disabled>Choose file… (Plan A)</button>
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
</style>
