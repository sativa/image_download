<script lang="ts">
  import InputPanel from "./components/InputPanel.svelte";
  import MapPanel from "./components/MapPanel.svelte";
  import ProgressPanel from "./components/ProgressPanel.svelte";
  import HistoryPanel from "./components/HistoryPanel.svelte";
  import Toast from "./components/Toast.svelte";
</script>

<main class="layout">
  <aside class="left">
    <InputPanel />
  </aside>
  <section class="center">
    <MapPanel />
  </section>
  <aside class="right">
    <ProgressPanel />
    <HistoryPanel />
  </aside>
</main>
<Toast />

<style>
  .layout {
    display: grid;
    grid-template-columns: 320px 1fr 360px;
    height: 100vh;
    gap: 1px;
    background: var(--border);
  }
  .left, .center, .right {
    background: var(--bg);
    overflow: auto;
  }
  .right {
    display: flex;
    flex-direction: column;
  }
  @media (max-width: 1100px) {
    .layout { grid-template-columns: 280px 1fr 320px; }
  }
</style>
