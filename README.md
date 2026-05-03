# Imagery Downloader

Desktop application to download satellite XYZ tiles for a bounding box and
write a Cloud-Optimized GeoTIFF.

> **Status: scaffold only.** This repo is mid-rewrite from a Python CLI
> (preserved in [`legacy/`](./legacy)) to a Tauri 2.x desktop app. The UI
> currently shows a placeholder; download functionality is being added in
> tracked plans under `docs/superpowers/plans/`.

## Develop

Prerequisites:
- Rust stable (`rustup install stable`)
- Node 20+ via Corepack (`corepack enable && corepack prepare pnpm@9 --activate`)
- macOS: Xcode Command Line Tools
- Windows: WebView2 Runtime (Evergreen Bootstrapper) and Microsoft C++ Build Tools

```bash
pnpm install
pnpm tauri dev      # launch dev window
pnpm tauri build    # produce platform-native installer in src-tauri/target/release/bundle
```

## Releases

Tag a `v*` release on `main`:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

GitHub Actions runs `tauri-action` across macOS + Windows runners and uploads
the resulting `.dmg` / `.msi` / `.exe` artifacts to a draft GitHub Release.

See [`docs/RELEASING.md`](./docs/RELEASING.md) for the full release procedure
and [`docs/SIGNING.md`](./docs/SIGNING.md) for code-signing notes.

## Legacy

The original Python CLI lives in [`legacy/`](./legacy) for reference. It is
not built or tested by CI.
