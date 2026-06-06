<script lang="ts">
  import { onMount } from "svelte";
  import {
    history, input, pushToast, requestMapFit,
    setLandformOverlay,
  } from "../lib/state.svelte";
  import {
    listHistory, clearHistory, onDone,
    startClassify, onLandformStage, onLandformProgress, onLandformDone,
    resolveHistoryTif,
  } from "../lib/ipc";
  import { formatDuration, formatNumber } from "../lib/format";
  import type { HistoryEntry, Source } from "../lib/types";

  async function load() {
    try {
      history.entries = await listHistory();
    } catch (e) {
      pushToast("error", `Load history failed: ${e}`);
    }
  }

  // Per-row classification state. Keyed by the input TIF path the user
  // clicked. classify_id is needed to correlate landform://* events back
  // to a row; the same input path can only have one classification job
  // running at a time (UI disables the button).
  type RowClassifyState =
    | { phase: "idle" }
    | { phase: "running"; classifyId: string; stage: string; done: number; total: number }
    | { phase: "succeeded"; labelGpkg: string; topClasses: { id: string; pct: number }[] }
    | { phase: "failed"; error: string };
  const classifyByInput = $state<Record<string, RowClassifyState>>({});
  // Selected classification model/backend (applies to the next Classify click).
  let classifyBackend = $state("cropland");
  const BACKEND_OPTIONS = [
    { value: "cropland", label: "耕地二分类 (快)" },
    { value: "parcel_bh", label: "逐地块·边界头+8类 (推荐)" },
    { value: "parcel", label: "逐地块 SAM3+耕地 (慢)" },
    { value: "landcover", label: "8类地物 (像素)" },
  ];
  // Reverse map so landform events (which only carry classify_id) can
  // find the input path that owns the row state.
  const inputByClassifyId = new Map<string, string>();
  // Bbox associated with each in-flight classify (not on the done event).
  const bboxByClassifyId = new Map<string, [number, number, number, number]>();
  // Which row triggered which classification — keyed by stable history
  // row identity (same key used in the {#each} block). Without this we
  // can't tell apart two rows pointing at the same folder, and the
  // running-state badge bleeds across every entry in that folder.
  const tifPathByHistoryKey = $state<Record<string, string>>({});

  function historyKey(e: HistoryEntry): string {
    return `${e.finished_at}_${e.zoom}_${e.source}`;
  }

  function getRowState(path: string): RowClassifyState {
    return classifyByInput[path] ?? { phase: "idle" };
  }

  async function copyPath(path: string) {
    try {
      await navigator.clipboard.writeText(path);
      pushToast("info", `路径已复制 / Path copied: ${path.split("/").pop()}`);
    } catch {
      pushToast("info", path);
    }
  }

  async function classifyRow(e: HistoryEntry) {
    // Resolve the .tif path at click time. Avoids eager async work for
    // every visible row and makes failures (e.g. file moved/deleted)
    // surface as a user-facing toast, not a missing button.
    const inputTif = await resolveHistoryTif(e.output_path, e.zoom, e.source);
    if (!inputTif) {
      pushToast("error", `No .tif found under ${e.output_path} for z${e.zoom}/${e.source}`);
      return;
    }
    tifPathByHistoryKey[historyKey(e)] = inputTif;
    if (getRowState(inputTif).phase === "running") return;
    classifyByInput[inputTif] = {
      phase: "running", classifyId: "", stage: "starting", done: 0, total: 0,
    };
    try {
      const { classify_id } = await startClassify({ input_path: inputTif, backend: classifyBackend });
      inputByClassifyId.set(classify_id, inputTif);
      bboxByClassifyId.set(classify_id, e.bbox);
      classifyByInput[inputTif] = {
        phase: "running", classifyId: classify_id, stage: "spawning_sidecar", done: 0, total: 0,
      };
      // Also restore the row's input parameters into the form so the
      // MapPanel pans to the right bbox immediately; the overlay can
      // then land in view by the time it's ready.
      input.bbox = e.bbox;
      input.zoom = e.zoom;
      input.source = e.source;
      requestMapFit();
      pushToast("info", `Classify started — ${inputTif.split("/").pop()}`);
    } catch (err) {
      classifyByInput[inputTif] = { phase: "failed", error: String(err) };
      pushToast("error", `Classify start failed: ${err}`);
    }
  }

  // No more eager .tif resolution — the button always appears on
  // completed rows; classifyRow() does the path lookup at click time.

  onMount(() => {
    void load();
    const unlisteners: (() => void)[] = [];
    onDone(() => { void load(); }).then((u) => unlisteners.push(u));

    onLandformStage((ev) => {
      const input = inputByClassifyId.get(ev.classify_id);
      if (!input) return;
      const cur = classifyByInput[input];
      if (cur?.phase !== "running") return;
      classifyByInput[input] = { ...cur, stage: ev.stage };
    }).then((u) => unlisteners.push(u));

    onLandformProgress((ev) => {
      const input = inputByClassifyId.get(ev.classify_id);
      if (!input) return;
      const cur = classifyByInput[input];
      if (cur?.phase !== "running") return;
      classifyByInput[input] = {
        ...cur, done: ev.done, total: ev.total,
        stage: ev.current_prompt ? `prompt: ${ev.current_prompt}` : cur.stage,
      };
    }).then((u) => unlisteners.push(u));

    onLandformDone((ev) => {
      const input = inputByClassifyId.get(ev.classify_id);
      if (!input) return;
      inputByClassifyId.delete(ev.classify_id);
      if (ev.ok) {
        const entries = Object.entries(ev.stats)
          .filter(([id, s]) => id !== "0" || s.area_pct > 5)
          .sort((a, b) => b[1].area_pct - a[1].area_pct)
          .slice(0, 3)
          .map(([id, s]) => ({ id, pct: s.area_pct }));
        classifyByInput[input] = { phase: "succeeded", labelGpkg: ev.label_gpkg, topClasses: entries };
        const fallbackBbox = bboxByClassifyId.get(ev.classify_id);
        bboxByClassifyId.delete(ev.classify_id);
        const overlayBbox = ev.overlay_bbox_wgs84 ?? fallbackBbox ?? null;
        if (ev.overlay_geojson && overlayBbox) {
          setLandformOverlay(ev.overlay_geojson, overlayBbox, ev.classify_id);
        }
        pushToast("info", `Landform written: ${ev.label_gpkg.split("/").pop()}`);
      } else {
        classifyByInput[input] = { phase: "failed", error: ev.error };
        pushToast("error", `Classify failed: ${ev.error}`);
      }
    }).then((u) => unlisteners.push(u));

    return () => unlisteners.forEach((u) => u());
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

  {#if history.entries.length > 0}
    <label class="model-select">
      分类模型 / Model:
      <select bind:value={classifyBackend}>
        {#each BACKEND_OPTIONS as opt}
          <option value={opt.value}>{opt.label}</option>
        {/each}
      </select>
    </label>
  {/if}

  {#if history.entries.length === 0}
    <p class="muted">No previous downloads.</p>
  {:else}
    <ul>
      {#each history.entries as e (e.finished_at + "_" + e.zoom + "_" + e.source)}
        {@const pct = e.total_tiles > 0 ? Math.round((e.completed_tiles / e.total_tiles) * 100) : 0}
        {@const isDone = e.status === "completed"}
        {@const _hk = historyKey(e)}
        {@const _tif = tifPathByHistoryKey[_hk]}
        {@const rowState = (_tif ? classifyByInput[_tif] : undefined) ?? ({ phase: "idle" } as RowClassifyState)}
        <li>
          <button class="entry status-{e.status}" onclick={() => restore(e)}>
            <div class="meta">
              z{e.zoom} · {e.source}
              <span class="badge status-{e.status}">{statusLabel(e.status)}</span>
              {#if !isDone && e.total_tiles > 0}
                <span class="muted small">— {pct}%</span>
              {/if}
              {#if rowState.phase === "succeeded"}
                <span class="badge landform-ok" title={rowState.labelGpkg}>
                  ✓ landform
                </span>
              {:else if rowState.phase === "failed"}
                <span class="badge landform-err" title={rowState.error}>! landform</span>
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
            {#if rowState.phase === "running"}
              <div class="row muted small">
                classifying — {rowState.stage}
                {#if rowState.total > 0} ({rowState.done}/{rowState.total}){/if}
              </div>
            {:else if rowState.phase === "succeeded"}
              {#if rowState.topClasses.length > 0}
                <div class="row muted small">
                  {rowState.topClasses.map((c) => `class ${c.id}: ${c.pct.toFixed(1)}%`).join(" · ")}
                </div>
              {/if}
              <div
                class="row small poly-path"
                role="button"
                tabindex="0"
                title="点击复制路径 / Click to copy: {rowState.labelGpkg}"
                onclick={(event) => { event.stopPropagation(); void copyPath(rowState.labelGpkg); }}
                onkeydown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.stopPropagation(); void copyPath(rowState.labelGpkg); } }}
              >
                📄 多边形 / polygons: {rowState.labelGpkg}
              </div>
            {/if}
            <div class="ts muted">{formatFinishedAt(e.finished_at)}</div>
          </button>
          {#if isDone}
            <button
              class="classify-btn"
              disabled={rowState.phase === "running"}
              onclick={(event) => { event.stopPropagation(); void classifyRow(e); }}
              title={rowState.phase === "running" ? "Classification in progress" : "Run SAM 3 land-cover classification on this download"}
            >
              {#if rowState.phase === "running"}…{:else if rowState.phase === "succeeded"}Re-classify{:else}Classify{/if}
            </button>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .model-select {
    display: flex; align-items: center; gap: 0.4rem;
    font-size: 0.8rem; color: var(--fg-muted); margin: 0.4rem 0 0;
  }
  .model-select select { flex: 1; padding: 0.2rem; font-size: 0.8rem; }
  .poly-path {
    color: var(--accent);
    font-size: 0.78rem;
    word-break: break-all;
    cursor: pointer;
    margin-top: 0.15rem;
  }
  .poly-path:hover { text-decoration: underline; }
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
  .badge.landform-ok  { background: #6366f122; border-color: #6366f1; color: #4338ca; }
  .badge.landform-err { background: #f5934222; border-color: #f59342; color: #b45309; }
  /* Position each entry's classify button at the top-right corner so it
     overlays the restore button without stealing its click area. */
  li { position: relative; }
  .classify-btn {
    position: absolute; top: 0.4rem; right: 0.4rem;
    font-size: 0.7rem; padding: 0.15rem 0.45rem;
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    color: var(--fg);
    cursor: pointer;
  }
  .classify-btn:hover:not(:disabled) { background: var(--accent); color: white; border-color: var(--accent); }
  .classify-btn:disabled { opacity: 0.5; cursor: progress; }
</style>
