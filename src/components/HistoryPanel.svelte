<script lang="ts">
  import { onMount } from "svelte";
  import { history, input, pushToast, requestMapFit } from "../lib/state.svelte";
  import { listHistory, clearHistory, onDone } from "../lib/ipc";
  import { formatDuration, formatNumber } from "../lib/format";
  import type { HistoryEntry, Source } from "../lib/types";

  async function load() {
    try {
      history.entries = await listHistory();
    } catch (e) {
      pushToast("error", `Load history failed: ${e}`);
    }
  }

  onMount(() => {
    void load();
    // Reload history whenever ANY download finishes — backend already wrote
    // the history row before emitting `done` (see commands/download.rs), so
    // listHistory() at this point reflects the new entry. Without this, new
    // runs only became visible on next app launch.
    let unlisten: (() => void) | null = null;
    onDone(() => {
      void load();
    }).then((u) => {
      unlisten = u;
    });
    return () => {
      unlisten?.();
    };
  });

  async function clear() {
    if (!confirm("Clear all history?")) return;
    await clearHistory();
    history.entries = [];
  }

  function restore(e: HistoryEntry) {
    input.bbox = e.bbox;
    input.zoom = e.zoom;
    input.source = e.source as Source;
    // history.output_path is already the user-chosen folder (the backend
    // migrates legacy `.tif` paths on Store::open before sending them
    // here, see core/history.rs::looks_like_file_path).
    input.outputPath = e.output_path;
    // Force-fit the map to this entry's bbox even if the new region happens
    // to fall inside the current view (anti-jitter would otherwise skip).
    // Same mechanism vector-import uses; see state.svelte.ts:fitRequest.
    requestMapFit();
    pushToast("info", "History entry restored");
  }

  function statusLabel(s: HistoryEntry["status"]): string {
    switch (s) {
      case "completed": return "✓ done";
      case "in_progress": return "● running";
      case "cancelled": return "✕ cancelled";
      case "failed": return "! failed";
    }
  }

  // Backend writes "epoch:<unix-secs>"; pre-existing rows might be ISO 8601.
  function formatFinishedAt(s: string): string {
    if (s.startsWith("epoch:")) {
      const secs = Number(s.slice(6));
      if (Number.isFinite(secs)) return new Date(secs * 1000).toLocaleString();
    }
    return s;
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
      {#each history.entries as e (e.finished_at + "_" + e.zoom + "_" + e.source)}
        {@const pct = e.total_tiles > 0 ? Math.round((e.completed_tiles / e.total_tiles) * 100) : 0}
        {@const isDone = e.status === "completed"}
        <li>
          <button class="entry status-{e.status}" onclick={() => restore(e)}>
            <div class="meta">
              z{e.zoom} · {e.source}
              <span class="badge status-{e.status}">{statusLabel(e.status)}</span>
              {#if !isDone && e.total_tiles > 0}
                <span class="muted small">— {pct}%</span>
              {/if}
            </div>
            <div class="bbox">[{e.bbox.map((n) => n.toFixed(2)).join(", ")}]</div>
            <div class="row muted">
              {#if isDone}
                {formatNumber(e.total_tiles)} tiles · {formatDuration(e.duration_sec)} ·
                {e.output_size_mb.toFixed(1)} MB
              {:else}
                {formatNumber(e.completed_tiles)} / {formatNumber(e.total_tiles)} tiles
                {#if e.duration_sec > 0} · {formatDuration(e.duration_sec)}{/if}
              {/if}
            </div>
            <div class="ts muted">{formatFinishedAt(e.finished_at)}</div>
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
  .meta { font-weight: 600; font-size: 0.9rem; display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; }
  .bbox { font-family: ui-monospace, Menlo, monospace; font-size: 0.75rem; }
  .ts { font-size: 0.7rem; }
  .small { font-size: 0.8rem; }
  .badge {
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.1rem 0.4rem;
    border-radius: 999px;
    background: var(--bg);
    border: 1px solid var(--border);
  }
  .badge.status-completed   { background: #16a34a22; border-color: #16a34a; color: #15803d; }
  .badge.status-in_progress { background: #2563eb22; border-color: #2563eb; color: #1d4ed8; }
  .badge.status-cancelled   { background: #ca8a0422; border-color: #ca8a04; color: #a16207; }
  .badge.status-failed      { background: #dc262622; border-color: #dc2626; color: #b91c1c; }
  /* Left-edge accent so unfinished items stand out at a glance. */
  .entry.status-in_progress { border-left: 3px solid #2563eb; }
  .entry.status-cancelled   { border-left: 3px solid #ca8a04; }
  .entry.status-failed      { border-left: 3px solid #dc2626; }
  .entry:hover .badge { color: white; background: rgba(255,255,255,0.15); border-color: rgba(255,255,255,0.4); }
</style>
