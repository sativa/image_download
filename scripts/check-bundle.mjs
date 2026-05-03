#!/usr/bin/env node
// Verify that `pnpm tauri build` produced the artifacts we expect.
// Exit 0 if all expected artifacts exist with non-trivial size, exit 1 otherwise.

import { existsSync, statSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { platform, arch } from "node:process";

const root = "src-tauri/target/release/bundle";

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

process.exit(bad === 0 ? 0 : 1);
