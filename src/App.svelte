<script lang="ts">
  let pong = $state<string>("...");
  async function ping() {
    pong = "Tauri scaffold OK at " + new Date().toISOString();
  }
</script>

<main>
  <h1>Imagery Downloader</h1>
  <p>Scaffold-only build. Plans A/B/C will replace this UI.</p>
  <button onclick={ping}>Ping</button>
  <p>{pong}</p>
</main>

<style>
  main { font-family: system-ui, sans-serif; padding: 2rem; }
  button { padding: 0.5rem 1rem; }
</style>
