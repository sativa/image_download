#!/usr/bin/env node
// Verify that `pnpm tauri build` produced the artifacts we expect.
// Exit 0 if all expected artifacts exist with non-trivial size, exit 1 otherwise.
//
// Policy: strict. Native legs MUST land bundles at src-tauri/target/release/bundle/.
// If they don't, the diagnostic walker (below) prints where the bundles *did* land,
// so we can see path drift from a CI log without rerunning.

import { existsSync, statSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { platform, arch } from "node:process";

const root = "src-tauri/target/release/bundle";
const targetDir = "src-tauri/target";

// File extensions / directory suffixes that indicate a bundle artifact.
const BUNDLE_HINTS = [".app", ".dmg", ".msi", ".exe", ".AppImage", ".deb"];

function pickExpected() {
  if (platform === "darwin") {
    const a = arch === "arm64" ? "aarch64" : "x64";
    return [
      { path: `${root}/macos/Imagery Downloader.app`, kind: "dir" },
      { path: `${root}/dmg/Imagery Downloader_0.0.1_${a}.dmg`, kind: "file" },
    ];
  }
  if (platform === "win32") {
    return [
      { path: `${root}/msi/Imagery Downloader_0.0.1_x64_en-US.msi`, kind: "file" },
      { path: `${root}/nsis/Imagery Downloader_0.0.1_x64-setup.exe`, kind: "file" },
    ];
  }
  console.error(`Unsupported platform for bundle check: ${platform}`);
  process.exit(2);
}

function dirSize(p) {
  let total = 0;
  for (const e of readdirSync(p, { withFileTypes: true })) {
    const child = join(p, e.name);
    if (e.isDirectory()) total += dirSize(child);
    else if (e.isFile()) total += statSync(child).size;
  }
  return total;
}

// TODO(you): implement findStrayBundles(root).
//
// Purpose: when an expected artifact is MISSING, walk `root` (src-tauri/target/)
// recursively and return an array of paths whose basename ends with any of
// BUNDLE_HINTS. We use this to surface bundles that landed in an unexpected
// place (e.g. target/aarch64-apple-darwin/release/bundle/...) so the next CI
// log tells us exactly where Tauri put them.
//
// Decisions you should make:
//   - Skip the `deps/`, `build/`, `incremental/` subtrees? They never hold
//     bundle output but contain thousands of files; walking them blindly will
//     drown the useful signal and slow CI.
//   - Should a `.app` directory short-circuit (don't recurse into it)? Tauri
//     puts a full bundle tree inside `.app`, and we only want the `.app`
//     itself in the report.
//   - Return absolute or repo-relative paths? Relative is friendlier in logs.
//
// Constraints:
//   - Must not throw if `root` doesn't exist (CI may fail before any build).
//   - Keep it ≤ ~15 lines; this is a diagnostic, not a search engine.
//
// Signature:
function findStrayBundles(root) {
  // TODO: your implementation here.
  return [];
}

const expected = pickExpected();
let bad = 0;
for (const { path, kind } of expected) {
  if (!existsSync(path)) {
    console.error(`MISSING: ${path}`);
    bad++;
    continue;
  }
  const size = kind === "dir" ? dirSize(path) : statSync(path).size;
  const sizeMB = size / 1e6;
  if (sizeMB < 1) {
    console.error(`TOO SMALL (${sizeMB.toFixed(2)} MB): ${path}`);
    bad++;
  } else {
    console.log(`OK ${sizeMB.toFixed(1)} MB  ${path}`);
  }
}

if (bad > 0) {
  console.error("\n--- diagnostic: scanning src-tauri/target/ for stray bundles ---");
  const strays = findStrayBundles(targetDir);
  if (strays.length === 0) {
    console.error("(no bundle-shaped artifacts found anywhere under src-tauri/target/)");
  } else {
    for (const p of strays) console.error(`FOUND: ${p}`);
  }
}

process.exit(bad === 0 ? 0 : 1);
