<script lang="ts">
  import { onMount } from "svelte";
  import { history, input, pushToast } from "../lib/state.svelte";
  import { listHistory, clearHistory } from "../lib/ipc";
  import { formatDuration, formatNumber } from "../lib/format";
  import type { HistoryEntry, Source } from "../lib/types";

  async function load() {
    try {
      history.entries = await listHistory();
    } catch (e) {
      pushToast("error", `Load history failed: ${e}`);
    }
  }
  onMount(load);

  async function clear() {
    if (!confirm("Clear all history?")) return;
    await clearHistory();
    history.entries = [];
  }

  function restore(e: HistoryEntry) {
    input.bbox = e.bbox;
    input.zoom = e.zoom;
    input.source = e.source as Source;
    input.outputPath = e.output_path;
    pushToast("info", "History entry restored");
  }
</script>

<section class="panel">
  <header>
    <h2>History</h2>
    {#if history.entries.length > 0}
      <button class="link" onclick={clear}>Clear</button>
    {/if}
  </header>

  {#if history.entries.length === 0}
    <p class="muted">No previous downloads.</p>
  {:else}
    <ul>
      {#each history.entries as e (e.finished_at + e.zoom)}
        <li>
          <button class="entry" onclick={() => restore(e)}>
            <div class="meta">z{e.zoom} · {e.source} · {e.ok ? "✓" : "✗"}</div>
            <div class="bbox">[{e.bbox.map((n) => n.toFixed(2)).join(", ")}]</div>
            <div class="row muted">
              {formatNumber(e.total_tiles)} tiles · {formatDuration(e.duration_sec)} ·
              {e.output_size_mb.toFixed(1)} MB
            </div>
            <div class="ts muted">{e.finished_at}</div>
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .panel { padding: 1rem; flex: 1; overflow: auto; }
  header { display: flex; justify-content: space-between; align-items: center; }
  h2 { margin: 0; font-size: 1rem; }
  .link { background: none; border: none; color: var(--accent); padding: 0; }
  .muted { color: var(--fg-muted); font-size: 0.85rem; margin: 0; }
  ul { list-style: none; padding: 0; margin: 0.5rem 0 0; display: flex; flex-direction: column; gap: 0.4rem; }
  .entry {
    width: 100%;
    text-align: left;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.5rem 0.7rem;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  .entry:hover { background: var(--accent); color: white; }
  .entry:hover .muted { color: rgba(255,255,255,0.9); }
  .meta { font-weight: 600; font-size: 0.9rem; }
  .bbox { font-family: ui-monospace, Menlo, monospace; font-size: 0.75rem; }
  .ts { font-size: 0.7rem; }
</style>
