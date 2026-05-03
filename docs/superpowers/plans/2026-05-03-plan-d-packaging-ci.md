# Plan D — 打包 + CI 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `imagery-downloader` 项目从一个 Python CLI 仓库变成"打 git tag 即自动产出 macOS `.dmg`（Intel + Apple Silicon）+ Windows `.msi` / `.exe` 的 Tauri 应用骨架"，为后续 Plan A/B/C 提供可填充的结构与持续可发布的 release pipeline。

**Architecture:**
- 仓库重排：旧 Python 代码迁到 `legacy/`，新建 `src-tauri/`（Rust）+ `src/`（前端 Svelte 5 + Vite + TS）。
- 最小可运行 Tauri 2.x scaffold：一个 hello-world 窗口，仅用来证明编译/打包链可工作；Plan A/B/C 后续替换其内部模块。
- 两条 GitHub Actions 流水线：`ci.yml`（每 push / PR 跑 lint + test + build 验证）与 `release.yml`（tag `v*` 触发 `tauri-apps/tauri-action`，三平台并行构建后挂到 GitHub Release draft）。
- 签名当前不做，但 conf 与 workflow 预留 env 接口，后续仅添加 secrets 即可启用。

**Tech Stack:** Tauri 2.x · Svelte 5 + Vite + TypeScript · pnpm（Corepack）· Rust stable · GitHub Actions · `tauri-apps/tauri-action@v0` · `actionlint`（CI YAML 校验）· Node 20 LTS

---

## 文件结构（计划落地后）

```
imagery_downloader/
├── .github/
│   └── workflows/
│       ├── ci.yml              # PR / push 校验：cargo fmt/clippy/test + pnpm build
│       └── release.yml         # tag v* 触发，调用 tauri-action 矩阵打包
├── .gitignore                  # 已存在，需追加 Rust + Node 条目
├── legacy/                     # 旧 Python 脚本（保留只读）
│   ├── README.md
│   ├── download_imagery.py
│   ├── monitor_download.py
│   ├── verify_polygon_imagery.py
│   ├── visualize_imagery.py
│   └── requirements.txt
├── src/                        # 前端（Plan C 会大幅扩展）
│   ├── App.svelte              # hello world，Plan C 替换
│   ├── main.ts
│   ├── app.css
│   └── vite-env.d.ts
├── src-tauri/                  # 后端（Plan A/B 会扩展 src/ 下的模块）
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   ├── build.rs
│   ├── capabilities/
│   │   └── default.json
│   ├── icons/                  # 占位 icon set
│   └── src/
│       ├── main.rs             # 仅启动 Tauri，Plan B 会注册 commands
│       └── lib.rs
├── docs/
│   ├── superpowers/specs/2026-05-03-imagery-downloader-tauri-design.md  # 已存在
│   ├── superpowers/plans/2026-05-03-plan-d-packaging-ci.md              # 本文件
│   ├── RELEASING.md            # 本计划新增
│   └── SIGNING.md              # 本计划新增
├── package.json                # pnpm workspace root
├── pnpm-lock.yaml
├── pnpm-workspace.yaml         # 仅一个 root，但保留以便未来加 packages/
├── tsconfig.json
├── vite.config.ts
└── README.md                   # 替换为面向桌面用户的安装/使用说明
```

**核心职责切分：**
- `src-tauri/tauri.conf.json` 是打包行为的唯一真源（targets / identifier / 版本 / WebView2 引导器）。
- `.github/workflows/release.yml` 不重复 conf 里的产物决定，只声明触发条件与上传目标。
- `legacy/` 完全只读，不参与 build；CI 显式排除其路径以避免误触。
- `docs/SIGNING.md` 是"目前不签 → 未来签"切换文档，避免把决策写死在 conf 注释里。

---

## 阶段总览

| Phase | 名称 | 任务数 | 产出验证 |
|---|---|---|---|
| 0 | 仓库重排 + Tauri scaffold | 6 | `pnpm tauri dev` 弹出 hello world 窗口 |
| 1 | Bundle 配置（macOS + Windows） | 5 | 本地 `pnpm tauri build` 产出 `.dmg` 或 `.msi` 可双击安装 |
| 2 | CI workflow（push / PR 校验） | 3 | 一次 PR 跑通三平台 build + test 全绿 |
| 3 | Release workflow（tag → 产物） | 4 | tag `v0.0.1-test` 后 GitHub Release 草稿出现 6 个 artifact |
| 4 | 签名预留 + 文档 | 3 | `docs/SIGNING.md` + `README.md` 完整可读 |

合计 21 个任务。每个任务都包含失败条件 / 通过条件 / 提交命令。

**前置依赖：**
- 本机已装：`git`、`rustup` (stable)、Node 20+ (建议通过 `corepack enable` 拉 pnpm)、Xcode CLT（macOS）。
- 你有这个仓库的 GitHub Actions 写权限。
- 当前在 `main` 分支，工作区干净（已是状态）。

---

## Phase 0 — 仓库重排 + Tauri scaffold

> 目标：把旧 Python 代码搬到 `legacy/`，新建一个能跑的最小 Tauri 应用作为 Plan A/B/C 的骨架。

### Task 0.1：把旧 Python 代码迁到 `legacy/`

**Files:**
- Create: `legacy/README.md`
- Move: `download_imagery.py` → `legacy/download_imagery.py`
- Move: `monitor_download.py` → `legacy/monitor_download.py`
- Move: `verify_polygon_imagery.py` → `legacy/verify_polygon_imagery.py`
- Move: `visualize_imagery.py` → `legacy/visualize_imagery.py`
- Move: `requirements.txt` → `legacy/requirements.txt`

- [ ] **Step 1: 创建 legacy 目录与 README**

```bash
mkdir -p legacy
```

写 `legacy/README.md`：

```markdown
# Legacy Python CLI

This directory preserves the original Python CLI from before the Tauri rewrite.
**It is not built or tested by CI.** Kept for reference only.

Original entry points:
- `download_imagery.py` — main CLI
- `monitor_download.py` — progress poller
- `verify_polygon_imagery.py` — output QA
- `visualize_imagery.py` — visualization helper

To run:

    cd legacy
    pip install -r requirements.txt
    python download_imagery.py --help

The new Tauri app supersedes all of the above.
```

- [ ] **Step 2: 用 `git mv` 搬运（保留历史）**

```bash
git mv download_imagery.py legacy/download_imagery.py
git mv monitor_download.py legacy/monitor_download.py
git mv verify_polygon_imagery.py legacy/verify_polygon_imagery.py
git mv visualize_imagery.py legacy/visualize_imagery.py
git mv requirements.txt legacy/requirements.txt
```

- [ ] **Step 3: 验证仓库根目录已清空旧 .py**

Run: `ls *.py 2>/dev/null; echo "exit=$?"`
Expected output: `exit=2`（zsh 的 "no matches found" 退出码；表示根目录无 `.py` 文件）。

Run: `ls legacy/`
Expected output 包含全部 5 个文件 + `README.md`。

- [ ] **Step 4: 提交**

```bash
git add legacy/ -A
git commit -m "chore: move legacy Python CLI into legacy/ subdirectory

Preserves original Python implementation as read-only reference; the
upcoming Tauri rewrite (Plans A/B/C) will live under src-tauri/ and src/."
```

---

### Task 0.2：初始化前端（Svelte 5 + Vite + TS + pnpm）

**Files:**
- Create: `package.json`, `pnpm-lock.yaml`, `pnpm-workspace.yaml`
- Create: `vite.config.ts`, `tsconfig.json`, `svelte.config.js`
- Create: `index.html`, `src/main.ts`, `src/App.svelte`, `src/app.css`, `src/vite-env.d.ts`

- [ ] **Step 1: 启用 pnpm 并初始化项目**

```bash
corepack enable
corepack prepare pnpm@9.12.0 --activate
pnpm --version  # 应输出 9.12.0
```

- [ ] **Step 2: 写 package.json**

```json
{
  "name": "imagery-downloader",
  "version": "0.0.1",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "check": "svelte-check --tsconfig ./tsconfig.json",
    "tauri": "tauri"
  },
  "devDependencies": {
    "@sveltejs/vite-plugin-svelte": "^4.0.0",
    "@tauri-apps/cli": "^2.1.0",
    "@tsconfig/svelte": "^5.0.4",
    "svelte": "^5.1.0",
    "svelte-check": "^4.0.0",
    "tslib": "^2.8.0",
    "typescript": "^5.6.0",
    "vite": "^5.4.0"
  },
  "dependencies": {
    "@tauri-apps/api": "^2.1.0"
  },
  "packageManager": "pnpm@9.12.0"
}
```

- [ ] **Step 3: 写 pnpm-workspace.yaml（占位，便于后续加 packages/）**

```yaml
packages:
  - "."
```

- [ ] **Step 4: 写 vite.config.ts**

```ts
import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? { protocol: "ws", host, port: 1421 }
      : undefined,
    watch: { ignored: ["**/src-tauri/**"] },
  },
});
```

- [ ] **Step 5: 写 tsconfig.json**

```json
{
  "extends": "@tsconfig/svelte/tsconfig.json",
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "module": "ESNext",
    "resolveJsonModule": true,
    "allowJs": true,
    "checkJs": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "moduleResolution": "bundler",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src/**/*.ts", "src/**/*.svelte"]
}
```

- [ ] **Step 6: 写 svelte.config.js**

```js
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

export default { preprocess: vitePreprocess() };
```

- [ ] **Step 7: 写 index.html**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/png" href="/icon.png" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Imagery Downloader</title>
  </head>
  <body>
    <div id="app"></div>
    <script type="module" src="/src/main.ts"></script>
  </body>
</html>
```

- [ ] **Step 8: 写 src/main.ts**

```ts
import "./app.css";
import App from "./App.svelte";
import { mount } from "svelte";

const app = mount(App, { target: document.getElementById("app")! });

export default app;
```

- [ ] **Step 9: 写 src/App.svelte（hello world 占位）**

```svelte
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
```

- [ ] **Step 10: 写 src/app.css 与 src/vite-env.d.ts**

`src/app.css`:
```css
:root { color-scheme: light dark; }
body { margin: 0; }
```

`src/vite-env.d.ts`:
```ts
/// <reference types="svelte" />
/// <reference types="vite/client" />
```

- [ ] **Step 11: 安装依赖并验证前端能 build**

```bash
pnpm install
pnpm check
pnpm build
```

Expected:
- `pnpm install` 成功，生成 `pnpm-lock.yaml`。
- `pnpm check` 输出 `0 errors and 0 warnings`。
- `pnpm build` 在 `dist/` 下生成 `index.html` 与 `assets/`。

- [ ] **Step 12: 提交**

```bash
git add package.json pnpm-lock.yaml pnpm-workspace.yaml vite.config.ts \
        tsconfig.json svelte.config.js index.html src/
git commit -m "feat(scaffold): bootstrap Svelte 5 + Vite + TS frontend

Hello-world UI; will be replaced by Plan C. Vite is configured to ignore
src-tauri/ from HMR watch and runs on the standard Tauri dev port 1420."
```

---

### Task 0.3：初始化 Tauri Rust 后端

**Files:**
- Create: `src-tauri/Cargo.toml`, `src-tauri/build.rs`
- Create: `src-tauri/tauri.conf.json`
- Create: `src-tauri/capabilities/default.json`
- Create: `src-tauri/src/main.rs`, `src-tauri/src/lib.rs`
- Create: `src-tauri/icons/` (icon set 由 `pnpm tauri icon` 生成)

- [ ] **Step 1: 创建目录与 icon 占位**

```bash
mkdir -p src-tauri/src src-tauri/capabilities src-tauri/icons
```

生成一张 1024×1024 占位 PNG 然后用 `pnpm tauri icon` 派生全套：

```bash
python3 - <<'PY'
from struct import pack
import zlib
w=h=1024
raw=b''.join(b'\x00'+b'\x33\x66\xff\xff'*w for _ in range(h))
def chunk(t,d):
    return pack('>I',len(d))+t+d+pack('>I',zlib.crc32(t+d)&0xffffffff)
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',pack('>IIBBBBB',w,h,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
open('src-tauri/icons/app-icon.png','wb').write(png)
print('OK')
PY
pnpm tauri icon src-tauri/icons/app-icon.png
```

Expected: `src-tauri/icons/` 下出现 `32x32.png`, `128x128.png`, `128x128@2x.png`, `icon.icns`, `icon.ico`, `Square*.png`, `StoreLogo.png` 等约 10 个文件。

- [ ] **Step 2: 写 src-tauri/Cargo.toml**

```toml
[package]
name = "imagery-downloader"
version = "0.0.1"
description = "Download satellite XYZ tiles for a bounding box and write COG GeoTIFF"
authors = ["zhangfeng04@gmail.com"]
edition = "2021"
rust-version = "1.77"

[lib]
name = "imagery_downloader_lib"
crate-type = ["staticlib", "cdylib", "rlib"]

[build-dependencies]
tauri-build = { version = "2.0", features = [] }

[dependencies]
tauri = { version = "2.1", features = [] }
tauri-plugin-log = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[features]
custom-protocol = ["tauri/custom-protocol"]
```

- [ ] **Step 3: 写 src-tauri/build.rs**

```rust
fn main() {
    tauri_build::build()
}
```

- [ ] **Step 4: 写 src-tauri/src/lib.rs**

```rust
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::default().build())
        .invoke_handler(tauri::generate_handler![])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

- [ ] **Step 5: 写 src-tauri/src/main.rs**

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    imagery_downloader_lib::run()
}
```

- [ ] **Step 6: 写 capabilities/default.json**

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "Default capability for the main window",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "core:window:default",
    "core:webview:default"
  ]
}
```

- [ ] **Step 7: 写 src-tauri/tauri.conf.json（最小可用配置；bundle 配置 Phase 1 再细化）**

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "Imagery Downloader",
  "version": "0.0.1",
  "identifier": "com.zhangfeng.imagery-downloader",
  "build": {
    "beforeDevCommand": "pnpm dev",
    "beforeBuildCommand": "pnpm build",
    "devUrl": "http://localhost:1420",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "Imagery Downloader",
        "width": 1100,
        "height": 720,
        "minWidth": 900,
        "minHeight": 600,
        "resizable": true,
        "fullscreen": false
      }
    ],
    "security": {
      "csp": null
    }
  },
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/128x128@2x.png",
      "icons/icon.icns",
      "icons/icon.ico"
    ]
  }
}
```

> **DECISION**：`identifier` 我用了 `com.zhangfeng.imagery-downloader`。如果你的 GitHub 是组织账号或想用 `dev.<github>.<app>` 风格，**这里改一次**比 release 之后改容易得多——macOS 用 identifier 区分应用安装记录。

- [ ] **Step 8: 跑 cargo check 验证**

```bash
cd src-tauri
cargo check
cd ..
```

Expected: 通过。如果 fail，按报错修。

- [ ] **Step 9: 验证 dev 模式能起来**

```bash
pnpm tauri dev
```

Expected: 几十秒后弹出原生窗口，标题 "Imagery Downloader"，内容是 hello world 页面。点 Ping 按钮文本变化。Ctrl-C 关闭。

> **失败兜底**：如果首次启动报 "WebView2 未安装"（仅 Windows），到 https://developer.microsoft.com/en-us/microsoft-edge/webview2/ 装 Evergreen Bootstrapper。Phase 1 我们会把这个嵌入安装包，开发期手动装一次即可。

- [ ] **Step 10: 提交**

```bash
git add src-tauri/
git commit -m "feat(scaffold): add minimal Tauri 2.x Rust backend

Hello-world Tauri app with empty invoke_handler — Plan B will register
the full command set. tauri.conf.json carries identity but no platform-
specific bundle config yet (added in Phase 1)."
```

---

### Task 0.4：更新 .gitignore（排除 build 产物）

**Files:**
- Modify: `.gitignore`（追加 Rust + Node 条目）

- [ ] **Step 1: 读现有 .gitignore**

```bash
cat .gitignore
```

记下当前条目。

- [ ] **Step 2: 在 `.gitignore` 末尾追加（不要覆盖已有内容）**

```gitignore

# Node / Vite
node_modules/
dist/
.vite/

# Rust / Tauri
src-tauri/target/
src-tauri/gen/
src-tauri/Cargo.lock

# OS noise
.DS_Store
Thumbs.db

# IDE
.vscode/*
!.vscode/extensions.json
.idea/
```

> **注意**：上面把 `src-tauri/Cargo.lock` 排除了。**应用最终交付二进制时应该提交 lock**——但目前 scaffold 阶段依赖会频繁变动，先排除避免噪音。Phase 1 末尾会重新评估。

- [ ] **Step 3: 验证 status 干净**

```bash
git status
```

Expected：只显示 `.gitignore` 一个修改。如果还显示 `node_modules/` 或 `target/`，说明 ignore 规则没生效。

- [ ] **Step 4: 提交**

```bash
git add .gitignore
git commit -m "chore: extend .gitignore for Node and Tauri build outputs"
```

---

### Task 0.5：更新 README（替换为新项目说明）

**Files:**
- Modify: `README.md`（整体替换）

- [ ] **Step 1: 写新 README**

````markdown
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
````

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs: replace README with desktop-app overview"
```

---

### Task 0.6：Phase 0 阶段验收

- [ ] **Step 1: 运行端到端 dev**

```bash
pnpm tauri dev
```

Expected: 窗口弹出，标题 "Imagery Downloader"，UI 渲染。

- [ ] **Step 2: 运行端到端 build（当前平台）**

```bash
pnpm tauri build
```

Expected:
- macOS: `src-tauri/target/release/bundle/dmg/Imagery Downloader_0.0.1_aarch64.dmg`（或 `_x64.dmg`）。
- Windows: `src-tauri/target/release/bundle/msi/Imagery Downloader_0.0.1_x64_en-US.msi` 与 `nsis/...`。

打开 dmg / 双击 msi，安装后能从 Applications / 开始菜单启动。

- [ ] **Step 3: 检查提交历史**

```bash
git log --oneline | head -5
```

Expected: 5 条提交，依次为 README、gitignore、tauri scaffold、frontend scaffold、legacy move。

✅ Phase 0 完成。Plan A/B/C 现在有了可填充的 `src-tauri/src/` 与 `src/`，且本地能产出可安装包。

---

## Phase 1 — Bundle 配置（macOS + Windows）

> 目标：把 `tauri.conf.json` 的 bundle section 配齐，使三个目标平台（macOS arm64、macOS x86_64、Windows x64）的产物都符合发行要求。

### Task 1.1：写 bundle 期望验证脚本（fail-first）

**Files:**
- Create: `scripts/check-bundle.mjs`
- Modify: `package.json`

我们把"产物路径与命名是否符合预期"做成一个本地脚本，CI 也复用——这是 bundle 配置的"测试"。脚本只用 `fs`，不调外部进程。

- [ ] **Step 1: 创建 scripts 目录与脚本**

```bash
mkdir -p scripts
```

写 `scripts/check-bundle.mjs`：

```js
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
```

- [ ] **Step 2: 把脚本声明到 package.json**

修改 `package.json` 的 `scripts` block：

```json
"scripts": {
  "dev": "vite",
  "build": "vite build",
  "preview": "vite preview",
  "check": "svelte-check --tsconfig ./tsconfig.json",
  "check:bundle": "node scripts/check-bundle.mjs",
  "tauri": "tauri"
}
```

- [ ] **Step 3: 跑一次（应该 PASS，因为 Phase 0 已经 build 过）**

```bash
pnpm check:bundle
```

Expected: 输出 `OK <size> MB  <path>` 行，进程返回 0。

如果 fail 说明 Phase 0 build 没成功或 conf 里产物名跟脚本不一致，**先修脚本与 conf 同步**，再继续。

- [ ] **Step 4: 提交**

```bash
git add scripts/check-bundle.mjs package.json
git commit -m "test: add bundle artifact check script for local + CI"
```

---

### Task 1.2：配置 macOS bundle（`.dmg` + `.app`，最低系统版本）

**Files:**
- Modify: `src-tauri/tauri.conf.json`

- [ ] **Step 1: 在 `bundle` 下追加 `macOS` 配置**

把 `tauri.conf.json` 的 `bundle` block 改为：

```json
"bundle": {
  "active": true,
  "targets": ["app", "dmg"],
  "icon": [
    "icons/32x32.png",
    "icons/128x128.png",
    "icons/128x128@2x.png",
    "icons/icon.icns",
    "icons/icon.ico"
  ],
  "category": "DeveloperTool",
  "shortDescription": "Download satellite tiles for a bounding box",
  "longDescription": "Imagery Downloader fetches XYZ tiles from configurable sources (ESRI, Google) for a user-specified bounding box and writes a Cloud-Optimized GeoTIFF.",
  "macOS": {
    "minimumSystemVersion": "10.15",
    "frameworks": [],
    "exceptionDomain": "",
    "signingIdentity": null,
    "providerShortName": null,
    "entitlements": null
  }
}
```

> **DECISION**：`minimumSystemVersion: "10.15"` 是个判断。10.15 (Catalina, 2019) 覆盖几乎所有还在用的 Mac，且 Tauri webview crate 实际上要求 ≥ 10.13——别再低。如果你想往新（如 11.0），告诉我。

- [ ] **Step 2: 重新打包验证**

```bash
pnpm tauri build
pnpm check:bundle
```

Expected: bundle check 输出 `OK ... .app` + `OK ... .dmg` 两行。

- [ ] **Step 3: 手动安装验证（仅当前在 macOS）**

```bash
open "src-tauri/target/release/bundle/dmg/Imagery Downloader_0.0.1_aarch64.dmg"
# 拖到 Applications
open "/Applications/Imagery Downloader.app"
```

Expected: 应用启动，无 "App is damaged" 弹窗（未签名时首次右键 → 打开 一次即可）。

- [ ] **Step 4: 提交**

```bash
git add src-tauri/tauri.conf.json
git commit -m "feat(bundle): configure macOS .app + .dmg with minimum 10.15

Signing identity left null — see docs/SIGNING.md for ad-hoc signing path
when Apple Developer ID becomes available."
```

---

### Task 1.3：配置 Windows bundle（`.msi` + `.exe`，嵌入 WebView2 引导器）

**Files:**
- Modify: `src-tauri/tauri.conf.json`

- [ ] **Step 1: 把 `targets` 扩到 `["app","dmg","msi","nsis"]`，并在 `bundle` 内追加 `windows`**

```json
"windows": {
  "certificateThumbprint": null,
  "digestAlgorithm": "sha256",
  "timestampUrl": "",
  "tsp": false,
  "webviewInstallMode": {
    "type": "embedBootstrapper"
  },
  "wix": {
    "language": ["en-US"],
    "template": null,
    "fragmentPaths": [],
    "componentGroupRefs": [],
    "componentRefs": [],
    "featureGroupRefs": [],
    "featureRefs": [],
    "mergeRefs": []
  },
  "nsis": {
    "license": null,
    "headerImage": null,
    "sidebarImage": null,
    "installerIcon": "icons/icon.ico",
    "installMode": "perMachine",
    "languages": ["English"],
    "displayLanguageSelector": false
  }
}
```

> `webviewInstallMode: embedBootstrapper` 把 WebView2 在线引导器（约 200 KB）打进安装包。**非 offline-installer**——目标机器首次安装时会从微软下 WebView2，需短暂网络。如果要离线，改 `"type": "offlineInstaller"`，安装包会膨胀约 +120 MB。Phase 1 默认走 bootstrapper。

- [ ] **Step 2（仅 Windows runner 上）：重新打包**

```bash
pnpm tauri build
pnpm check:bundle
```

Expected: bundle check 输出 `OK ... .msi` 与 `OK ... -setup.exe`。

> **macOS 上跳过此步**：cross-platform Windows build 不支持，靠 CI matrix 补完。

- [ ] **Step 3: 提交**

```bash
git add src-tauri/tauri.conf.json
git commit -m "feat(bundle): configure Windows .msi + NSIS .exe with embedded WebView2 bootstrapper

NSIS installer is per-machine (admin); WebView2 fetched online at install
time. Code signing left disabled — see docs/SIGNING.md."
```

---

### Task 1.4：把版本号从 `Cargo.toml` 单源化

**Files:**
- Modify: `src-tauri/tauri.conf.json`

`tauri.conf.json` 的 `version` 字段不写时 Tauri 2.x 默认读 `Cargo.toml` 的 `version`。让 Cargo.toml 成为唯一真源。

- [ ] **Step 1: 删除 conf 里的顶层 `version`**

把 `tauri.conf.json` 顶部的：

```json
"productName": "Imagery Downloader",
"version": "0.0.1",
"identifier": "com.zhangfeng.imagery-downloader",
```

改成：

```json
"productName": "Imagery Downloader",
"identifier": "com.zhangfeng.imagery-downloader",
```

**版本现在唯一来源是 `src-tauri/Cargo.toml` 的 `version`**。Release workflow 会从 git tag 注入版本号到 Cargo.toml，避免 conf 与 tag 不一致。

- [ ] **Step 2: 验证 build 仍然产出正确版本号**

```bash
pnpm tauri build
ls "src-tauri/target/release/bundle/dmg/" 2>/dev/null \
  || ls "src-tauri/target/release/bundle/msi/"
```

Expected: 文件名仍含 `_0.0.1_`。

- [ ] **Step 3: 提交**

```bash
git add src-tauri/tauri.conf.json
git commit -m "refactor(bundle): single-source version from Cargo.toml

Removes duplication between conf and Cargo.toml. Release workflow will
write the git tag's version into Cargo.toml before building."
```

---

### Task 1.5：Phase 1 阶段验收

- [ ] **Step 1: 本地端到端**

```bash
pnpm tauri build
pnpm check:bundle
```

Expected: bundle 脚本 PASS。

- [ ] **Step 2: 检查产物体积合理（< 50 MB）**

Expected:
- macOS `.dmg`: 8-15 MB
- macOS `.app`: 12-25 MB
- Windows `.msi` / `-setup.exe`: 6-14 MB（embedBootstrapper 模式）

如果显著超过 50 MB，多半是误把 `node_modules` 或 `dist/.vite/` 打进去——检查 `frontendDist` 是否指向 `../dist`。

✅ Phase 1 完成。本地能稳定打出符合预期的安装包。

---

## Phase 2 — CI workflow（push / PR 校验）

> 目标：每次 push / PR 触发跨平台 build + test 矩阵，绿色才允许合并。**这一步不会上传任何 release artifact**——纯校验。

### Task 2.1：写 CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: 创建目录与 workflow 文件**

```bash
mkdir -p .github/workflows
```

写 `.github/workflows/ci.yml`：

```yaml
name: CI

on:
  push:
    branches: [main]
    paths-ignore:
      - "legacy/**"
      - "docs/**"
      - "README.md"
  pull_request:
    branches: [main]
    paths-ignore:
      - "legacy/**"
      - "docs/**"
      - "README.md"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  frontend:
    name: Frontend (lint + build)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with:
          version: 9.12.0
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm
      - run: pnpm install --frozen-lockfile
      - run: pnpm check
      - run: pnpm build

  rust:
    name: Rust (fmt + clippy + test)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: src-tauri
      - name: Install Tauri Linux build deps
        run: |
          sudo apt-get update
          sudo apt-get install -y libwebkit2gtk-4.1-dev \
            build-essential curl wget file libxdo-dev libssl-dev \
            libayatana-appindicator3-dev librsvg2-dev
      - run: cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
      - run: cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets -- -D warnings
      - run: cargo test --manifest-path src-tauri/Cargo.toml --all

  build-matrix:
    name: Build (${{ matrix.platform }} ${{ matrix.target }})
    needs: [frontend, rust]
    strategy:
      fail-fast: false
      matrix:
        include:
          - platform: macos-latest
            target: aarch64-apple-darwin
          - platform: macos-latest
            target: x86_64-apple-darwin
          - platform: windows-latest
            target: ""
    runs-on: ${{ matrix.platform }}
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with:
          version: 9.12.0
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}
      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: src-tauri
          key: ${{ matrix.platform }}-${{ matrix.target }}
      - run: pnpm install --frozen-lockfile
      - name: Tauri build
        shell: bash
        run: |
          if [ -n "${{ matrix.target }}" ]; then
            pnpm tauri build --target ${{ matrix.target }}
          else
            pnpm tauri build
          fi
      - name: Verify bundle artifacts (current host arch only)
        if: matrix.target == '' || matrix.target == 'aarch64-apple-darwin'
        shell: bash
        run: node scripts/check-bundle.mjs
```

> **关键决定：**
> 1. `paths-ignore` 跳过 `legacy/`、`docs/`、`README.md`——这些路径修改不应触发跨平台 build。
> 2. `concurrency` 让同一 PR 的新 push 自动取消旧的 run，省 CI 分钟。
> 3. Linux runner **不打 bundle**，只跑 fmt/clippy/test——Linux 不在产品目标里，但跑 Rust 测试比 macOS/Windows runner 便宜得多。
> 4. macOS Intel 用 `--target x86_64-apple-darwin` 在 ARM runner 交叉编译。`check-bundle.mjs` 里的路径假设 host arch == target arch，所以只对原生 leg 跑校验；Intel 交叉编译产物路径包含 `x86_64-apple-darwin/` 子目录，校验逻辑放进 release.yml 里更合适（那里我们要明确路径）。
> 5. `cargo clippy ... -D warnings` 把 clippy warning 当错误——一旦放水后续要回收非常痛苦。

- [ ] **Step 2: 本地用 actionlint 校验语法**

```bash
brew install actionlint || true
actionlint .github/workflows/ci.yml
```

Expected: 无输出（PASS）。如果报 syntax error，按提示修。

- [ ] **Step 3: 提交并推到一个 PR 分支验证**

```bash
git checkout -b ci/initial-pipeline
git add .github/workflows/ci.yml
git commit -m "ci: add cross-platform build verification workflow

Runs frontend lint + Rust fmt/clippy/test on Linux, plus full Tauri
build on macOS (arm64+x64) and Windows. paths-ignore avoids burning CI
on legacy/ or docs-only changes."
git push -u origin ci/initial-pipeline
```

打开 GitHub UI 创建 PR。

- [ ] **Step 4: 在 GitHub Actions 标签里观察**

Expected: 三个 job (frontend / rust / build-matrix(×3)) 全绿，总耗时 8-20 分钟。

**常见 fail 修法：**
- `pnpm check` 报 svelte-check 错误：通常是 Phase 0 写的 hello-world 有未使用变量；按提示删掉。
- Rust clippy 报 `dead_code`：在对应 fn 上加 `#[allow(dead_code)]`，**或**直接删掉（scaffold 阶段没人调）。
- macOS Intel 交叉编译报 linker 错误：在 `dtolnay/rust-toolchain` step 里确认 `targets:` 字段拼对；通常装 `targets: x86_64-apple-darwin` 即够。

- [ ] **Step 5: 全绿后合 PR**

UI 上 squash-merge。然后：

```bash
git checkout main
git pull
git branch -d ci/initial-pipeline
```

---

### Task 2.2：把 `actionlint` 加到本地 pre-push 防呆

**Files:**
- Create: `.githooks/pre-push`
- Modify: `package.json`

- [ ] **Step 1: 写 hook**

```bash
mkdir -p .githooks
```

`.githooks/pre-push`:
```bash
#!/usr/bin/env bash
set -euo pipefail
if command -v actionlint >/dev/null 2>&1; then
  actionlint .github/workflows/*.yml
else
  echo "warn: actionlint not installed; skipping workflow lint"
fi
```

```bash
chmod +x .githooks/pre-push
```

- [ ] **Step 2: 在 package.json 加 `prepare` script 自动配 hooks 路径**

```json
"scripts": {
  "dev": "vite",
  "build": "vite build",
  "preview": "vite preview",
  "check": "svelte-check --tsconfig ./tsconfig.json",
  "check:bundle": "node scripts/check-bundle.mjs",
  "tauri": "tauri",
  "prepare": "git config core.hooksPath .githooks || true"
}
```

- [ ] **Step 3: 触发 prepare**

```bash
pnpm install
git config --get core.hooksPath
```

Expected: 输出 `.githooks`。

- [ ] **Step 4: 提交**

```bash
git add .githooks/pre-push package.json
git commit -m "chore: add pre-push hook to lint GitHub workflows locally"
```

---

### Task 2.3：在 main 上观察 CI 持续绿

- [ ] **Step 1: 触发一次 main push**

```bash
git push origin main
```

- [ ] **Step 2: 在 Actions UI 看最新 run**

Expected: 三个 job 都绿，build-matrix 三个 leg 都绿，总时 < 25 min（首次冷缓存 30+ min 也正常，第二次起会快）。

- [ ] **Step 3: 把 main 设为 required check（GitHub UI）**

到 Settings → Branches → Branch protection → main：
- Require status checks to pass before merging
- 勾选 `Frontend (lint + build)`、`Rust (fmt + clippy + test)`、`Build (macos-latest aarch64-apple-darwin)`、`Build (windows-latest )`

> **DECISION**：是否要把 macOS x86_64 leg 也设 required？我建议**不要**——Intel 交叉编译偶发 linker bug，作 required 容易卡 PR。让它作为"warn"信号即可。如果你倾向严格，告诉我。

✅ Phase 2 完成。

---

## Phase 3 — Release workflow（tag → 产物）

> 目标：在 `main` 上打 `v*` tag，自动跨三平台构建，把所有产物挂到一个 GitHub Release 草稿。

### Task 3.1：写 release workflow

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: 写 release workflow**

```yaml
name: Release

on:
  push:
    tags:
      - "v*.*.*"
      - "v*.*.*-*"

permissions:
  contents: write

concurrency:
  group: release-${{ github.ref }}
  cancel-in-progress: false

jobs:
  release:
    strategy:
      fail-fast: false
      matrix:
        include:
          - platform: macos-latest
            target: aarch64-apple-darwin
            args: "--target aarch64-apple-darwin"
          - platform: macos-latest
            target: x86_64-apple-darwin
            args: "--target x86_64-apple-darwin"
          - platform: windows-latest
            target: ""
            args: ""
    runs-on: ${{ matrix.platform }}
    steps:
      - uses: actions/checkout@v4

      - name: Resolve version from tag
        id: ver
        shell: bash
        run: |
          tag="${GITHUB_REF##*/}"
          ver="${tag#v}"
          echo "tag=$tag"   >> "$GITHUB_OUTPUT"
          echo "version=$ver" >> "$GITHUB_OUTPUT"

      - name: Patch version into Cargo.toml
        shell: bash
        run: |
          cd src-tauri
          python3 - <<PY
          import re
          p = 'Cargo.toml'
          s = open(p).read()
          s = re.sub(r'^version\s*=.*$',
                     'version = "${{ steps.ver.outputs.version }}"',
                     s, count=1, flags=re.M)
          open(p, 'w').write(s)
          PY
          grep '^version' Cargo.toml

      - uses: pnpm/action-setup@v4
        with:
          version: 9.12.0

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm

      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}

      - uses: Swatinem/rust-cache@v2
        with:
          workspaces: src-tauri
          key: release-${{ matrix.platform }}-${{ matrix.target }}

      - run: pnpm install --frozen-lockfile

      - uses: tauri-apps/tauri-action@v0
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          APPLE_CERTIFICATE: ${{ secrets.APPLE_CERTIFICATE }}
          APPLE_CERTIFICATE_PASSWORD: ${{ secrets.APPLE_CERTIFICATE_PASSWORD }}
          APPLE_SIGNING_IDENTITY: ${{ secrets.APPLE_SIGNING_IDENTITY }}
          APPLE_ID: ${{ secrets.APPLE_ID }}
          APPLE_PASSWORD: ${{ secrets.APPLE_PASSWORD }}
          APPLE_TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}
        with:
          tagName: ${{ steps.ver.outputs.tag }}
          releaseName: "Imagery Downloader ${{ steps.ver.outputs.tag }}"
          releaseBody: |
            See attached installers below.

            - macOS Apple Silicon: `*_aarch64.dmg`
            - macOS Intel: `*_x64.dmg`
            - Windows: `*.msi` (recommended) or `*-setup.exe`
          releaseDraft: true
          prerelease: ${{ contains(steps.ver.outputs.tag, '-') }}
          args: ${{ matrix.args }}
```

> **关键决定：**
> 1. **`releaseDraft: true`**：所有 leg 完成后，release 仍是草稿——你要点 "Publish" 才公开。这是必要的安全网，避免 CI 误推。
> 2. **`prerelease`** 由 tag 是否含 `-` 自动决定：`v0.1.0` 是稳定版，`v0.1.0-rc1` 是预发版。
> 3. **三个 leg 共享同一个 release**：tauri-action 在所有 runner 上用相同 `tagName`，第一个跑完的 leg 创建 release，后续 leg 上传到同一个。
> 4. **版本号注入**：从 tag `v0.1.0` 抽出 `0.1.0` 写进 Cargo.toml。这要求每次 tag 前 Cargo.toml 的 version 字段长得像 `version = "x.y.z"`——单行格式。
> 5. **签名 env 留空当前不影响 build**——`tauri-action` 见到 secret 为空会跳过签名步骤。

- [ ] **Step 2: actionlint 校验**

```bash
actionlint .github/workflows/release.yml
```

Expected: 无输出。

- [ ] **Step 3: 提交（不 push tag）**

```bash
git checkout -b ci/release-pipeline
git add .github/workflows/release.yml
git commit -m "ci(release): add tag-triggered cross-platform release workflow

On 'v*' tag push, builds three Tauri targets in parallel and uploads to
a single GitHub Release draft. Code signing slots are wired but inactive
until the corresponding secrets are added."
git push -u origin ci/release-pipeline
```

PR review，合并到 main。

---

### Task 3.2：用一个测试 tag 跑通

- [ ] **Step 1: 在 main 上打一个测试 tag**

```bash
git checkout main
git pull
git tag v0.0.1-test
git push origin v0.0.1-test
```

- [ ] **Step 2: 在 Actions 看 release run**

Expected: 三个 leg 全绿，耗时 15-30 min。GitHub UI 的 Releases 页面出现一个 draft，名为 "Imagery Downloader v0.0.1-test"，含 6 个 artifact：
- `Imagery.Downloader_0.0.1-test_aarch64.dmg`
- `Imagery.Downloader_0.0.1-test_aarch64.app.tar.gz`（updater sig 用）
- `Imagery.Downloader_0.0.1-test_x64.dmg`
- `Imagery.Downloader_0.0.1-test_x64.app.tar.gz`
- `Imagery.Downloader_0.0.1-test_x64_en-US.msi`
- `Imagery.Downloader_0.0.1-test_x64-setup.exe`

> **常见 fail：**
> - 第一个 leg 创建 release，但其余 leg 报 `release already exists`：tauri-action 会自动追加，等所有 leg 跑完再核对清单是否完整。
> - 版本号注入失败（产物文件名仍是 `0.0.1` 而非 `0.0.1-test`）：检查 "Patch version into Cargo.toml" step 的输出是否真的把版本号写进去了。

- [ ] **Step 3: 下载 macOS .dmg 验证**

在 macOS 上从 release UI 下载 `_aarch64.dmg`，双击挂载，把 `.app` 拖到 Applications。右键 → 打开（首次绕过 Gatekeeper），窗口正常弹出。

- [ ] **Step 4: 删除测试 release 与 tag（防污染）**

```bash
gh release delete v0.0.1-test --yes
git tag -d v0.0.1-test
git push origin :refs/tags/v0.0.1-test
```

如果没装 `gh`，到 GitHub UI Releases 页面手动删除草稿，然后：

```bash
git push origin :refs/tags/v0.0.1-test
git tag -d v0.0.1-test
```

---

### Task 3.3：写 RELEASING.md 让未来发版步骤固化

**Files:**
- Create: `docs/RELEASING.md`

- [ ] **Step 1: 写文档**

````markdown
# Releasing

Imagery Downloader uses tag-triggered GitHub Actions for releases.

## Cut a release

1. Make sure `main` is green on CI.
2. Bump `src-tauri/Cargo.toml` version (or rely on tag injection — both work).
3. Tag and push:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. Watch `Release` workflow in GitHub Actions. Expect ~20 min for all
   three legs (macOS arm64, macOS x64, Windows x64).
5. Open the auto-created **draft** release on GitHub. Review artifacts
   match the matrix (6 files: dmg×2, app.tar.gz×2, msi, exe).
6. Edit release notes, then click **Publish**.

## Pre-release / RC

Tags containing `-` are automatically marked as pre-release:

```bash
git tag v0.2.0-rc1
git push origin v0.2.0-rc1
```

## Roll back

To unpublish a release:
1. UI: Releases → … → Delete release. Choose to also delete the tag.
2. Command line:
   ```bash
   gh release delete v0.2.0 --yes
   git push origin :refs/tags/v0.2.0
   git tag -d v0.2.0
   ```

The next tag pushed afterwards will produce a fresh release.

## Versioning

We follow SemVer:
- `0.x.y`: scaffold + early development. Breaking changes in minor.
- `1.x.y`: stable IPC contract. Breaking changes only in major.
````

- [ ] **Step 2: 提交**

```bash
git add docs/RELEASING.md
git commit -m "docs: add release procedure"
```

---

### Task 3.4：Phase 3 阶段验收

- [ ] **Step 1: 打第一个真实 tag `v0.0.1`（项目第一个里程碑，留着别删）**

```bash
git tag v0.0.1
git push origin v0.0.1
```

- [ ] **Step 2: workflow 全绿后**，在 release UI 写发版说明，留草稿——别 Publish。

理由：scaffold 还没 ship 实际功能，留草稿当首个里程碑产物。等 Plan A/B/C 都完成才 Publish 第一个公开版。

✅ Phase 3 完成。

---

## Phase 4 — 签名预留 + 文档

### Task 4.1：写 SIGNING.md（说明当前不签 + 未来如何启用）

**Files:**
- Create: `docs/SIGNING.md`

- [ ] **Step 1: 写文档**

````markdown
# Code Signing

This project ships **unsigned binaries by default**. Signing is wired in
`.github/workflows/release.yml` but inactive until the corresponding
secrets are populated.

## Current behavior

### macOS
- Unsigned `.app` triggers Gatekeeper. First-launch UX:
  1. User downloads `.dmg`, drags `.app` to `/Applications`.
  2. Double-click → "App is from an unidentified developer." → user must
     **right-click → Open** the first time.
  3. Subsequent launches work normally.
- Some users report the binary is "damaged" after download. Fix:
  ```bash
  xattr -cr "/Applications/Imagery Downloader.app"
  ```

### Windows
- Unsigned `.msi` and `-setup.exe` trigger SmartScreen. First-launch UX:
  1. User downloads, double-clicks installer.
  2. SmartScreen says "Windows protected your PC". User clicks
     **More info → Run anyway**.
  3. Once installed, app launches without further warnings.

## Enabling Apple Developer ID signing (future)

Required:
- Apple Developer Program membership.
- A Developer ID Application certificate exported as `.p12`.
- An app-specific password for `notarytool`.

Steps:
1. Add the following GitHub repository secrets:
   - `APPLE_CERTIFICATE` — base64 of the `.p12`
   - `APPLE_CERTIFICATE_PASSWORD` — password protecting the `.p12`
   - `APPLE_SIGNING_IDENTITY` — full identity string, e.g.
     `Developer ID Application: Your Name (TEAMID)`
   - `APPLE_ID` — Apple ID email
   - `APPLE_PASSWORD` — app-specific password
   - `APPLE_TEAM_ID` — 10-char team ID
2. Tauri-action picks up these env vars automatically — no conf change
   needed. It will sign **and** notarize.
3. Cut a new release.

## Enabling Windows code signing (future)

Required:
- A code-signing certificate (OV or EV) as `.pfx`.

Steps:
1. Add secrets:
   - `WINDOWS_CERTIFICATE` — base64 of `.pfx`
   - `WINDOWS_CERTIFICATE_PASSWORD` — password
2. Wire them into release.yml's tauri-action env block.
3. In `tauri.conf.json` under `bundle.windows`, set
   `certificateThumbprint`, or rely on tauri-action env injection.
4. Cut a new release.

## Why not now

- Apple Developer Program is $99/yr — defer until a public release is
  imminent.
- Windows OV cert is ~$200/yr; EV is ~$300+/yr. Same rationale.
- Until then, the documented Gatekeeper / SmartScreen workarounds are
  acceptable for a small user base.
````

- [ ] **Step 2: 提交**

```bash
git add docs/SIGNING.md
git commit -m "docs: document signing strategy and future activation steps"
```

---

### Task 4.2：把 Phase 3-4 的链接接到 README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 检查 README 的 Releases 段已含 RELEASING/SIGNING 链接（Task 0.5 写的）**

```bash
grep -E "RELEASING|SIGNING" README.md
```

Expected: 看到指向 `docs/RELEASING.md` 与 `docs/SIGNING.md` 的链接。如果没有（说明 Task 0.5 时还没决定文档名），把 README 的 Releases 段改为：

````markdown
## Releases

Tag a `v*` release on `main`:
```bash
git tag v0.1.0 && git push origin v0.1.0
```

GitHub Actions runs `tauri-action` across macOS + Windows runners and uploads
the resulting `.dmg` / `.msi` / `.exe` artifacts to a draft GitHub Release.

See [`docs/RELEASING.md`](./docs/RELEASING.md) for the full release procedure
and [`docs/SIGNING.md`](./docs/SIGNING.md) for code-signing notes.
````

- [ ] **Step 2: 提交（如果有改动）**

```bash
git add README.md
git commit -m "docs: cross-link RELEASING.md and SIGNING.md from README" \
  || echo "no changes"
```

---

### Task 4.3：Plan D 全局验收

- [ ] **Step 1: 仓库结构 sanity check**

```bash
ls -la
```

Expected 顶层包含：`legacy/  src-tauri/  src/  docs/  scripts/  .github/  package.json  pnpm-lock.yaml  vite.config.ts  tsconfig.json  index.html  README.md  .gitignore`，**不再有任何根目录的 `.py` 或 `requirements.txt`**。

- [ ] **Step 2: 本地全链路**

```bash
pnpm install
pnpm check
pnpm tauri build
pnpm check:bundle
```

Expected: 三步全绿。

- [ ] **Step 3: 远端全链路**

确认 `main` 上：
- 最近一次 push 触发的 `CI` workflow 是绿的。
- `v0.0.1` tag 触发的 `Release` workflow 是绿的，draft release 含 6 个 artifact。

- [ ] **Step 4: 写一份 Plan D 完成简报追加到 spec 文档**

```bash
cat >> docs/superpowers/specs/2026-05-03-imagery-downloader-tauri-design.md <<'EOF'

---

## Plan D Implementation Status (2026-05-03)

✅ Repo restructured: `legacy/` + `src-tauri/` + `src/`.
✅ Minimal Tauri 2.x scaffold compiles and runs (`pnpm tauri dev`).
✅ Local `pnpm tauri build` produces `.dmg` / `.msi` / `.exe` (signing pending).
✅ CI workflow gates `main`: Linux fmt/clippy/test + macOS×2 + Windows build matrix.
✅ Release workflow on tag `v*` produces 6 artifacts to a GitHub Release draft.
✅ Signing slots wired (inactive). See `docs/SIGNING.md`.

Plans A/B/C now have a stable, releasable container to fill in.
EOF

git add docs/superpowers/specs/2026-05-03-imagery-downloader-tauri-design.md
git commit -m "docs(spec): record Plan D completion status"
```

✅ Plan D 全部完成。

---

## Self-Review

按 skill 要求的三项 checklist 自审：

### 1. Spec 覆盖

| Spec 段落 | 覆盖 task | 备注 |
|---|---|---|
| §7 macOS `.dmg` | 1.2, 3.1, 3.2 | ✅ |
| §7 Windows `.msi` + `.exe` | 1.3, 3.1, 3.2 | ✅ |
| §7 WebView2 引导器嵌入 | 1.3 (`webviewInstallMode: embedBootstrapper`) | ✅ |
| §7 零 GDAL | 0.3 (Cargo.toml 不引入 GDAL crate) + Plan A 的 cog 模块负责保持 | ✅（Plan D 范围内） |
| §7 GitHub Actions matrix `[macos-14, macos-13, windows-latest]` × tauri-action | 3.1（用 `macos-latest` + 交叉编译代替 macos-13，等价但更省 CI） | ✅ 设计偏离 |
| §7 tag push 自动发 release | 3.1, 3.2 | ✅ |
| §9 Apple Developer ID 签名 待定 | 4.1（SIGNING.md 全文档化未来启用步骤） | ✅ |
| §9 Windows 签名暂不做 | 4.1 | ✅ |
| §10 旧 Python 移到 legacy/ | 0.1 | ✅ |
| §10 新 README | 0.5, 4.2 | ✅ |

**一处偏离**：spec 写 `[macos-14, macos-13, windows-latest]`，计划改为 `macos-latest + 交叉编译`。理由：
- macos-14 与 macos-13 在 GitHub runner 上分别是 ARM 与 Intel，用两个 runner 各跑一次 native build。
- macos-latest（当前 = ARM）+ `--target x86_64-apple-darwin` 在同一 runner 跑两轮，节省一个 runner pool 等待时间。
- 交叉编译产物可在原生 Intel 上正常运行（Tauri 与 webview crate 都支持）。

如果你倾向严格按 spec（双 runner 各 native），告诉我，把 release.yml 第二条 leg 换成 `macos-13` 即可，工程量 1 行。

### 2. Placeholder 扫描

无 "TBD" / "TODO" / "implement later" / "appropriate error handling" / "similar to Task N" 等模糊措辞。每个 task 都包含具体命令与预期输出。两处 `DECISION:` 标记是给用户的明确决策点，**不是** placeholder。

### 3. 类型 / 命名一致性

- `pnpm@9.12.0` 在所有 task 里统一。
- `tauri-action@v0`、`actions/checkout@v4`、`actions/setup-node@v4`、`Swatinem/rust-cache@v2`、`dtolnay/rust-toolchain@stable` 在 ci.yml 与 release.yml 版本一致。
- `productName` = "Imagery Downloader"，`identifier` = `com.zhangfeng.imagery-downloader`，`Cargo.toml` 包名 `imagery-downloader`，`lib.rs` `pub fn run()` 在 main.rs 调 `imagery_downloader_lib::run()`——`-` 与 `_` 切换正确（Cargo 自动转）。
- `frontendDist` = `../dist`，`vite build` 默认输出 `dist/`，对齐。
- `check-bundle.mjs` 里的产物路径与 §1.2/1.3 配置出来的产物一致。

无不一致。

---

## Execution Handoff

Plan 已保存到 `docs/superpowers/plans/2026-05-03-plan-d-packaging-ci.md`。

**两种执行模式：**

1. **Subagent-Driven（推荐）** — 每个 task 我派一个 fresh subagent 执行，每完成一个我做 review，再派下一个。优点：单 task 上下文干净；缺点：每 task 启动有冷启延迟。
2. **Inline Execution** — 我在当前 session 里按 `executing-plans` 流程批量执行，每个 Phase 末尾跟你 checkpoint。优点：快、连贯；缺点：长 session 我自己上下文可能塌。

哪一种？
