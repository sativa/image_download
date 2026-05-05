<script lang="ts">
  import { onMount } from "svelte";
  import { download, pushToast } from "../lib/state.svelte";
  import { onProgress, onStage, onDone, onTileFailed, cancelDownload, retryFailed } from "../lib/ipc";
  import { formatBytes, formatDuration, formatNumber } from "../lib/format";

  // Time after successful done.ok before the panel auto-clears so the user
  // can immediately start a new download without clicking Dismiss.
  const AUTO_DISMISS_MS = 3000;
  let autoDismissTimer: ReturnType<typeof setTimeout> | null = null;

  function clearAutoDismiss() {
    if (autoDismissTimer !== null) {
      clearTimeout(autoDismissTimer);
      autoDismissTimer = null;
    }
  }

  onMount(() => {
    const offs: (() => void)[] = [];
    onProgress((p) => { if (p.download_id === download.id) download.progress = p; })
      .then((u) => offs.push(u));
    onStage((s) => { if (s.download_id === download.id) download.stage = s.stage; })
      .then((u) => offs.push(u));
    onTileFailed((t) => { if (t.download_id === download.id) download.failedTiles += 1; })
      .then((u) => offs.push(u));
    onDone((d) => {
      if (d.download_id !== download.id) return;
      download.finished = true;
      if (d.ok) {
        pushToast("info", `Done · ${formatNumber(d.total_tiles)} tiles · ${d.duration_sec.toFixed(1)}s`);
        // Successful run: auto-clear to "No active download" so the Start
        // button (in InputPanel) re-enables. Failed runs stay visible so
        // the user can read the error and decide to retry.
        clearAutoDismiss();
        autoDismissTimer = setTimeout(() => {
          // Only reset if still the same download (user might have already started another).
          if (download.finished && !download.error) reset();
        }, AUTO_DISMISS_MS);
      } else {
        download.error = d.error;
        pushToast("error", `Download failed: ${d.error}`);
      }
    }).then((u) => offs.push(u));

    return () => {
      clearAutoDismiss();
      offs.forEach((u) => u());
    };
  });

  let pct = $derived(
    download.progress
      ? Math.min(100, (download.progress.completed / Math.max(1, download.progress.total)) * 100)
      : 0,
  );

  // After all tiles are fetched (pct==100) but the pipeline is still working
  // (stitching / writing_cog / writing_preview), show an indeterminate bar
  // so the panel doesn't look frozen.
  let busyAfterFetch = $derived(
    pct >= 100 &&
      !download.finished &&
      download.stage !== null &&
      download.stage !== "downloading"
  );

  async function cancel() {
    if (!download.id) return;
    clearAutoDismiss();
    try { await cancelDownload(download.id); } catch (e) { pushToast("error", String(e)); }
  }
  async function retry() {
    if (!download.id) return;
    clearAutoDismiss();
    try { await retryFailed(download.id); } catch (e) { pushToast("error", String(e)); }
  }
  function reset() {
    clearAutoDismiss();
    download.id = null;
    download.progress = null;
    download.stage = null;
    download.failedTiles = 0;
    download.finished = false;
    download.error = null;
  }
</script>

<section class="panel">
  <h2>Progress</h2>

  {#if !download.id}
    <p class="muted">No active download.</p>
  {:else}
    <div class="meta">
      <span>{download.stage ?? "starting…"}</span>
      {#if download.progress}
        <span>· {formatNumber(download.progress.completed)} / {formatNumber(download.progress.total)}</span>
      {/if}
    </div>

    <div class="bar" class:indeterminate={busyAfterFetch}>
      <div class="fill" style="width:{pct}%"></div>
    </div>

    {#if download.progress}
      <div class="row">
        <span>{download.progress.current_speed_mbps.toFixed(1)} MB/s</span>
        <span>· {formatBytes(download.progress.bytes_downloaded)}</span>
        <span>· ETA {formatDuration(download.progress.eta_sec)}</span>
      </div>
    {/if}

    {#if download.failedTiles > 0}
      <p class="warn">{download.failedTiles} tile(s) failed</p>
    {/if}

    <div class="actions">
      {#if !download.finished}
        <button onclick={cancel}>Cancel</button>
      {:else}
        {#if download.failedTiles > 0 && !download.error}
          <button onclick={retry}>Retry failed</button>
        {/if}
        <button onclick={reset}>Dismiss</button>
      {/if}
    </div>
  {/if}
</section>

<style>
  .panel { padding: 1rem; display: flex; flex-direction: column; gap: 0.5rem; border-bottom: 1px solid var(--border); }
  h2 { margin: 0; font-size: 1rem; }
  .muted { color: var(--fg-muted); font-size: 0.9rem; margin: 0; }
  .meta, .row {
    display: flex;
    gap: 0.5rem;
    color: var(--fg-muted);
    font-size: 0.85rem;
    flex-wrap: wrap;
  }
  .bar { height: 8px; background: var(--bg-elev); border-radius: 4px; overflow: hidden; position: relative; }
  .fill { height: 100%; background: var(--accent); transition: width 200ms; }
  .bar.indeterminate .fill {
    background: linear-gradient(
      90deg,
      transparent 0%, var(--accent) 50%, transparent 100%
    );
    background-size: 200% 100%;
    animation: stripe 1.4s linear infinite;
  }
  @keyframes stripe {
    from { background-position: 200% 0; }
    to   { background-position: -200% 0; }
  }
  .warn { color: var(--warn); margin: 0; font-size: 0.85rem; }
  .actions { display: flex; gap: 0.5rem; margin-top: 0.3rem; }
</style>
