<script lang="ts">
  import { toasts } from "../lib/state.svelte";
</script>

<div class="stack">
  {#each toasts.items as t (t.id)}
    <div class="toast {t.level}">{t.text}</div>
  {/each}
</div>

<style>
  .stack {
    position: fixed;
    bottom: 1rem;
    right: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    z-index: 100;
    pointer-events: none;
  }
  .toast {
    background: var(--bg-elev);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    pointer-events: auto;
    max-width: 30rem;
  }
  .toast.warn { border-color: var(--warn); }
  .toast.error { border-color: var(--error); }
</style>
