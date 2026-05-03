# Plan C — 前端 UI + Mock 后端 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让用户打开 `pnpm tauri dev` 就能看到 spec §4 描述的完整 UI——地图框选 / 数值输入 / 文件拖拽三入口、实时 estimate 横幅、progress 面板带 ETA + 取消、history 面板 10 条最近记录——且 mock 后端能驱动一次完整的"伪下载"5 秒走完，让前端的 throttle / ETA / cancel 全部能真实测试。Plan A/B 落地时**只删 mock 文件夹、不动前端**。

**Architecture:**
- 前端 Svelte 5 SPA：单页三面板（input / map / progress+history），全部用 runes (`$state` / `$derived` / `$effect`)，不引第三方 store。
- IPC 层：`src/lib/ipc.ts` 包装 `@tauri-apps/api/core` 的 `invoke` + `event.listen`，TypeScript 类型来自 `src/lib/types.ts`，与 spec §3 IPC 契约 1:1。
- Mock 后端：`src-tauri/src/mocks/` 目录里的真实异步 Tauri command handlers，`spawn` tokio task 模拟 5 秒下载，emit 50 个 progress event；Plan B 落地时整目录删除、`lib.rs` 替换 `invoke_handler![]` 注册——前端零改动。
- 持久化：`history` 模块（spec §2.2 原本属 Plan A）在本 plan 中提前实现，`~/Library/Application Support/imagery-downloader/history.json`（macOS） / `%APPDATA%\imagery-downloader\history.json`（Windows）。
- 地图：MapLibre GL JS 4.x，纯 raster XYZ 底图（按 source 切换）；矩形框选不引第三方 draw 库——50 行原生 mouse handler 自实现，更可控。

**Tech Stack:** Svelte 5 + Vite + TypeScript（Plan D 已搭）· MapLibre GL JS 4 · `@tauri-apps/api` 2 · `@tauri-apps/plugin-fs`（仅 history 持久化用）· vitest 2 + `@testing-library/svelte` 5 + jsdom 25 · Rust：`tokio_util::sync::CancellationToken`、`uuid` v4、`directories` crate（XDG 路径）

---

## 文件结构（计划落地后）

```
imagery_downloader/
├── src/
│   ├── App.svelte                       # 3-pane 主布局
│   ├── main.ts                          # 已存在（Plan D），不动
│   ├── app.css                          # 扩展全局样式（reset + 主题变量）
│   ├── vite-env.d.ts                    # 已存在
│   ├── lib/
│   │   ├── types.ts                     # IPC 类型，对齐 spec §3
│   │   ├── ipc.ts                       # invoke + listen 包装
│   │   ├── state.svelte.ts              # 全局 runes：bbox/zoom/source/download/history/toast
│   │   ├── validate.ts                  # 纯函数 bbox + zoom 校验
│   │   ├── format.ts                    # 数字 / 时长 / 字节人类可读
│   │   └── sources.ts                   # XYZ URL 模板（前端复用，Plan A 后端会有自己的副本）
│   ├── components/
│   │   ├── InputPanel.svelte            # 表单 + tabs（数值 / 框选 / 导入）
│   │   ├── MapPanel.svelte              # MapLibre + rectangle draw + drop zone
│   │   ├── ProgressPanel.svelte         # 进度条 / ETA / 失败列表 / 取消 / 重试
│   │   ├── HistoryPanel.svelte          # 最近 10 次列表
│   │   └── Toast.svelte                 # 全局通知
│   └── tests/
│       ├── validate.test.ts             # bbox / zoom 校验
│       ├── format.test.ts               # 时长 / 字节格式化
│       └── ipc.test.ts                  # invoke 包装契约
├── src-tauri/
│   ├── src/
│   │   ├── lib.rs                       # 修改：注册 mock + history commands
│   │   ├── main.rs                      # 已存在，不动
│   │   ├── history.rs                   # 真实持久化模块（Plan A 提前）
│   │   └── mocks/
│   │       ├── mod.rs                   # 公开 commands fn
│   │       ├── commands.rs              # estimate / start / cancel / retry / parse-reject
│   │       ├── runner.rs                # 5-sec 异步 progress emitter + CancellationToken
│   │       └── README.md                # Plan B 替换契约
│   ├── Cargo.toml                       # 修改：加 tokio / uuid / directories / tokio-util
│   ├── tauri.conf.json                  # 修改：加 fs plugin permission
│   ├── capabilities/default.json        # 修改：精简 + 加 fs:read/write history.json
│   └── tests/
│       └── history_test.rs              # cargo test：history 增删改查
└── package.json                         # 修改：加 maplibre-gl + 测试栈
```

**核心职责切分：**
- `lib/ipc.ts` 是 **唯一** 的 invoke 入口；任何组件想调命令都通过它，禁止直接 import `@tauri-apps/api/core`。Plan B 替换 mock 时不需要改 ipc.ts，因为 invoke 字符串和参数不变。
- `lib/state.svelte.ts` 用 `$state` 暴露全局可变状态；组件直接 import 然后读写。**不用 Svelte stores**——Svelte 5 runes 已经覆盖。
- `mocks/commands.rs` 暴露的函数签名 = Plan B 真实 commands 的函数签名。Plan B 的 patch 应该是：删 `mocks/`、把 `lib.rs` 里的 `mocks::commands::*` 换成 `commands::*`。
- `history.rs` 是 **真实** 模块（不在 mock 目录下），文件路径用 `directories::ProjectDirs`，写入用 atomic `tempfile + rename`。Plan A 接手时只需改 `pub fn` 改 `pub(crate) fn` + 移到 `core::history` 模块，无逻辑改动。

---

## 阶段总览

| Phase | 名称 | 任务数 | 产出验证 |
|---|---|---|---|
| 0 | 依赖 + 类型骨架 | 5 | `pnpm check` + `cargo check` 全绿，IPC types 编译通过 |
| 1 | 应用布局 + 全局 state + Toast | 3 | dev 窗口打开看到 3 面板布局 + 系统主题响应 |
| 2 | 输入面板（数值 + tabs） | 4 | 输入合法 bbox 后 estimate 横幅出现；非法输入红框提示 |
| 3 | 地图面板（MapLibre + 框选） | 4 | 切换 source 底图换；拖框选完毕 bbox 自动回填表单 |
| 4 | 文件导入（拒绝态） | 2 | 拖 .geojson 进窗口弹"待 Plan A"提示，不崩 |
| 5 | 进度面板（events + 取消 + 重试） | 4 | mock 5 秒下载，progress 条平滑跑、ETA 倒数、取消立即停 |
| 6 | History 面板（持久化） | 3 | 完成 1 次 mock 下载 → history 出现条目；重启 app 仍在 |
| 7 | Mock 后端实现 | 4 | 上述所有面板的 mock 路径都跑通 |
| 8 | 集成验收 + 文档 | 2 | 手测 happy path + 编辑 README + 写 mocks/README + 提交 spec status |

合计 31 个任务。每个 task 含失败条件 / 通过条件 / 提交命令。

**前置依赖：**
- Plan D 已合（commit `02f1a8c` 是 Phase 1 终点）。
- 本机 Node 20+ / Rust stable，`pnpm tauri dev` 已可启动。
- 工作区干净。

---

## Phase 0 — 依赖 + 类型骨架

> 把所有"会让后续任务编译通过"的依赖与类型先就位，避免后面任务因缺包/缺 type 反复 bouncing。

### Task 0.1：加前端依赖（MapLibre + 测试栈）

**Files:**
- Modify: `package.json`
- Generate: `pnpm-lock.yaml` 更新

- [ ] **Step 1: 加运行时与 dev 依赖**

```bash
pnpm add maplibre-gl@^4.7.0 @tauri-apps/plugin-fs@^2
pnpm add -D vitest@^2.1.0 @testing-library/svelte@^5.2.0 jsdom@^25.0.0 @types/geojson@^7946.0.14
```

- [ ] **Step 2: 加 test 与 lint scripts 到 package.json**

修改 `scripts` block 为：

```json
"scripts": {
  "dev": "vite",
  "build": "vite build",
  "preview": "vite preview",
  "check": "svelte-check --tsconfig ./tsconfig.json",
  "check:bundle": "node scripts/check-bundle.mjs",
  "test": "vitest run",
  "test:watch": "vitest",
  "tauri": "tauri"
}
```

- [ ] **Step 3: 创建 vitest 配置**

写 `vitest.config.ts`：

```ts
import { defineConfig } from "vitest/config";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte({ hot: false })],
  test: {
    environment: "jsdom",
    include: ["src/tests/**/*.test.ts"],
    globals: true,
  },
});
```

- [ ] **Step 4: tsconfig 加 vitest globals types**

修改 `tsconfig.json` 的 `compilerOptions` 加：

```json
"types": ["vite/client", "vitest/globals"]
```

且 `include` 扩到：

```json
"include": ["src/**/*.ts", "src/**/*.svelte", "vitest.config.ts"]
```

- [ ] **Step 5: 写一个 smoke test 证明 vitest 跑得起来**

写 `src/tests/smoke.test.ts`：

```ts
import { describe, it, expect } from "vitest";

describe("vitest smoke", () => {
  it("runs", () => {
    expect(1 + 1).toBe(2);
  });
});
```

- [ ] **Step 6: 跑测试 + check + 提交**

```bash
pnpm check
pnpm test
```

Expected:
- `pnpm check`: `0 errors and 0 warnings`
- `pnpm test`: `1 passed | 0 failed`

```bash
git add package.json pnpm-lock.yaml vitest.config.ts tsconfig.json src/tests/smoke.test.ts
git commit -m "chore(c): add MapLibre + vitest stack for Plan C frontend

MapLibre GL 4 for the map preview, vitest+@testing-library/svelte+jsdom
for unit tests, plugin-fs for history.json persistence."
```

---

### Task 0.2：写 IPC 类型（对齐 spec §3）

**Files:**
- Create: `src/lib/types.ts`

- [ ] **Step 1: 写 types.ts**

```ts
// IPC contract between Svelte frontend and Tauri Rust backend.
// MUST stay byte-aligned with the structs in src-tauri/src/mocks/commands.rs
// (and later src-tauri/src/commands/*.rs once Plan B replaces the mock).

export type Bbox = [number, number, number, number]; // [minLon, minLat, maxLon, maxLat], WGS84
export type Source = "esri" | "google" | "auto";
export type Stage = "downloading" | "stitching" | "writing_cog" | "writing_preview";

export interface ParseVectorFileOk {
  bbox: Bbox;
  geometry: GeoJSON.Geometry;
  layer_count: number;
}
export type ParseVectorFileError =
  | { kind: "unsupported_format"; message: string }
  | { kind: "no_geometry"; message: string }
  | { kind: "io_error"; message: string };

export interface EstimateOutput {
  tile_count: number;
  pixel_w: number;
  pixel_h: number;
  est_size_mb: number;
  est_seconds: number;
}

export interface StartDownloadArgs {
  bbox: Bbox;
  zoom: number;            // 8..23
  source: Source;
  output_path: string;
  max_concurrency: number; // default 50
  retry_per_tile: number;  // default 3
  write_preview_png: boolean; // default true
}
export type StartDownloadError =
  | { kind: "invalid_bbox"; message: string }
  | { kind: "output_not_writable"; message: string };

export interface ProgressEvent {
  download_id: string;
  completed: number;
  total: number;
  bytes_downloaded: number;
  current_speed_mbps: number;
  elapsed_sec: number;
  eta_sec: number;
}

export interface TileFailedEvent {
  download_id: string;
  x: number;
  y: number;
  z: number;
  attempt: number;
  error: string;
}

export interface StageEvent {
  download_id: string;
  stage: Stage;
}

export type DoneEvent =
  | {
      download_id: string;
      ok: true;
      output_path: string;
      preview_path: string | null;
      bbox: Bbox;
      zoom: number;
      source_used: Source;
      duration_sec: number;
      total_tiles: number;
      failed_tiles: number;
      output_size_mb: number;
    }
  | { download_id: string; ok: false; error: string };

export interface HistoryEntry {
  bbox: Bbox;
  zoom: number;
  source: Source;
  output_path: string;
  ok: boolean;
  duration_sec: number;
  total_tiles: number;
  failed_tiles: number;
  output_size_mb: number;
  finished_at: string; // ISO 8601
}
```

- [ ] **Step 2: 验证编译**

```bash
pnpm check
```

Expected: `0 errors and 0 warnings`. 如果报 `Cannot find name 'GeoJSON'`，确认 `@types/geojson` 装好且 `tsconfig.json` 的 `types` 数组里包含 `vite/client`。如有需要在 types.ts 顶部加 `import type {} from "geojson";`（仅声明）。

- [ ] **Step 3: 提交**

```bash
git add src/lib/types.ts
git commit -m "feat(c): add IPC types mirroring spec §3 contract

These types define the boundary between the Svelte frontend and the
Tauri Rust backend. Plan B will keep the same shapes when replacing the
mock implementation."
```

---

### Task 0.3：写 IPC 包装 + 单测

**Files:**
- Create: `src/lib/ipc.ts`
- Create: `src/tests/ipc.test.ts`

- [ ] **Step 1: 先写测试（fail-first）**

`src/tests/ipc.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock @tauri-apps/api/core BEFORE importing ipc.ts
const invokeMock = vi.fn();
const listenMock = vi.fn();
vi.mock("@tauri-apps/api/core", () => ({ invoke: (...a: unknown[]) => invokeMock(...a) }));
vi.mock("@tauri-apps/api/event", () => ({ listen: (...a: unknown[]) => listenMock(...a) }));

beforeEach(() => {
  invokeMock.mockReset();
  listenMock.mockReset();
});

describe("ipc.estimateOutput", () => {
  it("forwards args under estimate_output command", async () => {
    invokeMock.mockResolvedValue({
      tile_count: 100, pixel_w: 256, pixel_h: 256,
      est_size_mb: 1, est_seconds: 1,
    });
    const { estimateOutput } = await import("../lib/ipc");
    const r = await estimateOutput([1, 2, 3, 4], 17, "esri");
    expect(invokeMock).toHaveBeenCalledWith("estimate_output", {
      bbox: [1, 2, 3, 4], zoom: 17, source: "esri",
    });
    expect(r.tile_count).toBe(100);
  });
});

describe("ipc.onProgress", () => {
  it("registers progress listener and returns unlisten", async () => {
    const unlisten = vi.fn();
    listenMock.mockResolvedValue(unlisten);
    const { onProgress } = await import("../lib/ipc");
    const cb = vi.fn();
    const off = await onProgress(cb);
    expect(listenMock).toHaveBeenCalledWith("download://progress", expect.any(Function));
    expect(off).toBe(unlisten);
  });
});
```

- [ ] **Step 2: 跑测试，确认失败**

```bash
pnpm test src/tests/ipc.test.ts
```

Expected: FAIL with `Cannot find module '../lib/ipc'`.

- [ ] **Step 3: 写最小实现**

`src/lib/ipc.ts`:

```ts
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  Bbox, Source, EstimateOutput, StartDownloadArgs,
  ProgressEvent, TileFailedEvent, StageEvent, DoneEvent,
  HistoryEntry, ParseVectorFileOk,
} from "./types";

export function parseVectorFile(path: string): Promise<ParseVectorFileOk> {
  return invoke("parse_vector_file", { path });
}

export function estimateOutput(bbox: Bbox, zoom: number, source: Source): Promise<EstimateOutput> {
  return invoke("estimate_output", { bbox, zoom, source });
}

export function startDownload(args: StartDownloadArgs): Promise<{ download_id: string }> {
  return invoke("start_download", { args });
}

export function cancelDownload(downloadId: string): Promise<{ ok: true }> {
  return invoke("cancel_download", { downloadId });
}

export function retryFailed(downloadId: string): Promise<{ ok: true }> {
  return invoke("retry_failed", { downloadId });
}

export function listHistory(): Promise<HistoryEntry[]> {
  return invoke("list_history");
}

export function clearHistory(): Promise<{ ok: true }> {
  return invoke("clear_history");
}

export function onProgress(cb: (e: ProgressEvent) => void): Promise<UnlistenFn> {
  return listen<ProgressEvent>("download://progress", (e) => cb(e.payload));
}

export function onTileFailed(cb: (e: TileFailedEvent) => void): Promise<UnlistenFn> {
  return listen<TileFailedEvent>("download://tile-failed", (e) => cb(e.payload));
}

export function onStage(cb: (e: StageEvent) => void): Promise<UnlistenFn> {
  return listen<StageEvent>("download://stage", (e) => cb(e.payload));
}

export function onDone(cb: (e: DoneEvent) => void): Promise<UnlistenFn> {
  return listen<DoneEvent>("download://done", (e) => cb(e.payload));
}
```

- [ ] **Step 4: 跑测试，确认通过**

```bash
pnpm test src/tests/ipc.test.ts
```

Expected: `2 passed | 0 failed`.

- [ ] **Step 5: 提交**

```bash
git add src/lib/ipc.ts src/tests/ipc.test.ts
git commit -m "feat(c): add typed IPC wrapper for invoke + listen

ipc.ts is the single allowed entry point to @tauri-apps/api/core; all
components import from here so Plan B can replace mocks without touching
component code. Tests cover argument forwarding and event subscription."
```

---

### Task 0.4：纯函数 validate.ts + format.ts + 单测

**Files:**
- Create: `src/lib/validate.ts`, `src/lib/format.ts`
- Create: `src/tests/validate.test.ts`, `src/tests/format.test.ts`

- [ ] **Step 1: 写两份测试（fail-first）**

`src/tests/validate.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { validateBbox, validateZoom } from "../lib/validate";

describe("validateBbox", () => {
  it("accepts a normal bbox", () => {
    expect(validateBbox([100, 30, 110, 40])).toBeNull();
  });
  it("rejects min >= max longitude", () => {
    expect(validateBbox([110, 30, 100, 40])).toMatch(/longitude/i);
  });
  it("rejects latitude out of range", () => {
    expect(validateBbox([100, -91, 110, 40])).toMatch(/latitude/i);
    expect(validateBbox([100, 30, 110, 91])).toMatch(/latitude/i);
  });
  it("rejects longitude out of range", () => {
    expect(validateBbox([-181, 30, 110, 40])).toMatch(/longitude/i);
  });
  it("rejects zero-area bbox", () => {
    expect(validateBbox([100, 30, 100, 30])).toMatch(/area/i);
  });
  it("rejects NaN", () => {
    expect(validateBbox([NaN, 30, 110, 40])).toMatch(/finite/i);
  });
});

describe("validateZoom", () => {
  it("accepts 8..23", () => {
    for (const z of [8, 12, 17, 22, 23]) expect(validateZoom(z)).toBeNull();
  });
  it("rejects out of range", () => {
    expect(validateZoom(7)).toMatch(/range/i);
    expect(validateZoom(24)).toMatch(/range/i);
  });
  it("rejects non-integer", () => {
    expect(validateZoom(17.5)).toMatch(/integer/i);
  });
});
```

`src/tests/format.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { formatBytes, formatDuration, formatNumber } from "../lib/format";

describe("formatBytes", () => {
  it("KB / MB / GB", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
    expect(formatBytes(3 * 1024 ** 3)).toBe("3.0 GB");
  });
});

describe("formatDuration", () => {
  it("seconds / minutes / hours", () => {
    expect(formatDuration(0)).toBe("0s");
    expect(formatDuration(45)).toBe("45s");
    expect(formatDuration(125)).toBe("2m 5s");
    expect(formatDuration(3725)).toBe("1h 2m 5s");
  });
  it("handles negative as 0", () => {
    expect(formatDuration(-10)).toBe("0s");
  });
});

describe("formatNumber", () => {
  it("inserts thousands separators", () => {
    expect(formatNumber(0)).toBe("0");
    expect(formatNumber(1234)).toBe("1,234");
    expect(formatNumber(1234567)).toBe("1,234,567");
  });
});
```

- [ ] **Step 2: 跑确认 fail**

```bash
pnpm test
```

Expected: 失败因为模块不存在。

- [ ] **Step 3: 写实现**

`src/lib/validate.ts`:

```ts
import type { Bbox } from "./types";

export function validateBbox(b: Bbox): string | null {
  if (b.some((x) => !Number.isFinite(x))) return "All bbox values must be finite numbers";
  const [minLon, minLat, maxLon, maxLat] = b;
  if (minLon < -180 || minLon > 180 || maxLon < -180 || maxLon > 180)
    return "Longitude must be in -180..180";
  if (minLat < -90 || minLat > 90 || maxLat < -90 || maxLat > 90)
    return "Latitude must be in -90..90";
  if (minLon >= maxLon) return "minLongitude must be less than maxLongitude";
  if (minLat >= maxLat) return "minLatitude must be less than maxLatitude";
  if (minLon === maxLon && minLat === maxLat) return "bbox has zero area";
  return null;
}

export function validateZoom(z: number): string | null {
  if (!Number.isInteger(z)) return "zoom must be an integer";
  if (z < 8 || z > 23) return "zoom must be in range 8..23";
  return null;
}
```

`src/lib/format.ts`:

```ts
export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  for (const u of units) {
    if (v < 1024) return `${v.toFixed(1)} ${u}`;
    v /= 1024;
  }
  return `${v.toFixed(1)} PB`;
}

export function formatDuration(sec: number): string {
  if (sec <= 0) return "0s";
  const s = Math.floor(sec) % 60;
  const m = Math.floor(sec / 60) % 60;
  const h = Math.floor(sec / 3600);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function formatNumber(n: number): string {
  return n.toLocaleString("en-US");
}
```

- [ ] **Step 4: 跑测试通过**

```bash
pnpm test
```

Expected: 全绿（包括 ipc + smoke + validate + format）。

- [ ] **Step 5: 提交**

```bash
git add src/lib/validate.ts src/lib/format.ts src/tests/validate.test.ts src/tests/format.test.ts
git commit -m "feat(c): add bbox/zoom validators and human-readable formatters

Pure functions, exhaustively unit-tested. Used by InputPanel for inline
validation and by ProgressPanel/HistoryPanel for display."
```

---

### Task 0.5：mocks/ 模块骨架 + history.rs 占位

**Files:**
- Create: `src-tauri/src/mocks/mod.rs`
- Create: `src-tauri/src/mocks/commands.rs` (空函数)
- Create: `src-tauri/src/mocks/runner.rs` (空 struct)
- Create: `src-tauri/src/history.rs` (空 struct)
- Modify: `src-tauri/src/lib.rs` (引用 mod)
- Modify: `src-tauri/Cargo.toml`

- [ ] **Step 1: 加 Cargo 依赖**

```bash
cd src-tauri
cargo add tokio --features rt-multi-thread,sync,macros,time
cargo add tokio-util --features rt
cargo add uuid --features v4
cargo add directories
cargo add tauri-plugin-fs@2
cd ..
```

- [ ] **Step 2: 写 src-tauri/src/history.rs（占位骨架）**

```rust
//! Persistent history of recent download parameters.
//!
//! This module is logically part of Plan A's `core::history`, but is
//! implemented during Plan C so the mock UI can demonstrate persistence.
//! Plan A will move it under `core::history` without API changes.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistoryEntry {
    pub bbox: [f64; 4],
    pub zoom: u32,
    pub source: String,
    pub output_path: String,
    pub ok: bool,
    pub duration_sec: f64,
    pub total_tiles: u32,
    pub failed_tiles: u32,
    pub output_size_mb: f64,
    pub finished_at: String,
}

/// Stub. Real impl in Task 6.1.
pub fn history_path() -> PathBuf {
    PathBuf::from("/tmp/imagery-downloader-history.json")
}
```

- [ ] **Step 3: 写 mocks/mod.rs**

```rust
//! Mock backend implementations.
//!
//! Plan B will replace this entire directory with real `core::*` calls.
//! Frontend invokes (in src/lib/ipc.ts) MUST NOT change between mock
//! and real — only the Rust handlers swap.
//!
//! See README.md in this directory for the replacement contract.

pub mod commands;
pub mod runner;
```

- [ ] **Step 4: 写 mocks/commands.rs（空函数占位）**

```rust
//! Tauri command handlers for the mock backend.
//! Real implementations land in Plan B.

use serde::Serialize;

#[derive(Serialize)]
pub struct StartDownloadResp {
    pub download_id: String,
}

#[tauri::command]
pub async fn estimate_output() -> Result<(), String> {
    Err("not implemented yet (Task 7.1)".into())
}

#[tauri::command]
pub async fn start_download() -> Result<StartDownloadResp, String> {
    Err("not implemented yet (Task 7.2)".into())
}

#[tauri::command]
pub async fn cancel_download() -> Result<serde_json::Value, String> {
    Err("not implemented yet (Task 7.3)".into())
}

#[tauri::command]
pub async fn retry_failed() -> Result<serde_json::Value, String> {
    Err("not implemented yet (Task 7.4)".into())
}

#[tauri::command]
pub async fn parse_vector_file() -> Result<serde_json::Value, String> {
    Err("not implemented yet (Task 4.2)".into())
}
```

- [ ] **Step 5: 写 mocks/runner.rs（空）**

```rust
//! Async runner that drives mock progress events. Filled in Task 7.2.

use std::collections::HashMap;
use std::sync::Mutex;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
pub struct Runner {
    pub tokens: Mutex<HashMap<String, CancellationToken>>,
}
```

- [ ] **Step 6: 修改 src-tauri/src/lib.rs**

替换内容为：

```rust
mod history;
mod mocks;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::default().build())
        .plugin(tauri_plugin_fs::init())
        .manage(mocks::runner::Runner::default())
        .invoke_handler(tauri::generate_handler![
            mocks::commands::estimate_output,
            mocks::commands::start_download,
            mocks::commands::cancel_download,
            mocks::commands::retry_failed,
            mocks::commands::parse_vector_file,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

- [ ] **Step 7: cargo check + 启动验证**

```bash
cd src-tauri
cargo check
cd ..
```

Expected: PASS。如果报 `cannot find tauri_plugin_fs`，确认 Step 1 的 `cargo add tauri-plugin-fs@2` 真的执行了。

- [ ] **Step 8: 提交**

```bash
git add src-tauri/Cargo.toml src-tauri/Cargo.lock src-tauri/src/lib.rs \
        src-tauri/src/history.rs src-tauri/src/mocks/
git commit -m "feat(c): scaffold mocks/ module + history stub + register commands

All five command handlers return 'not implemented yet' — they will be
filled in Phase 7. Runner state is registered via Tauri's State so the
mock progress emitter can later access shared CancellationTokens."
```

> **DECISION**：本任务首次出现 `Cargo.lock` 的提交。Plan D 的 Task 0.4 把 `Cargo.lock` 加进了 .gitignore——但现在加了真正的依赖，**应该提交 lock 以保证 release build 复现**。在 Step 8 前，把 `.gitignore` 里的 `src-tauri/Cargo.lock` 行删掉。如果你坚持不提交 lock（接受 release 构建非确定性），告诉我，我把这一步去掉。

---

## Phase 1 — 应用布局 + 全局 state + Toast

### Task 1.1：全局 state.svelte.ts + 主题样式

**Files:**
- Create: `src/lib/state.svelte.ts`
- Modify: `src/app.css`

- [ ] **Step 1: 写 state.svelte.ts**

```ts
import type { Bbox, Source, HistoryEntry, ProgressEvent, Stage } from "./types";

// Input form state — bound to InputPanel and reflected by MapPanel.
export const input = $state({
  bbox: [100, 30, 110, 40] as Bbox,
  zoom: 17,
  source: "esri" as Source,
  outputPath: "" as string,
  maxConcurrency: 50,
  retryPerTile: 3,
  writePreviewPng: true,
});

// Latest estimate, refreshed on bbox/zoom/source change (debounced 200ms).
export const estimate = $state<{
  loading: boolean;
  data: { tile_count: number; pixel_w: number; pixel_h: number; est_size_mb: number; est_seconds: number } | null;
  error: string | null;
}>({ loading: false, data: null, error: null });

// Active download status. Null when no download in flight.
export const download = $state<{
  id: string | null;
  stage: Stage | null;
  progress: ProgressEvent | null;
  failedTiles: number;
  finished: boolean;
  error: string | null;
}>({
  id: null,
  stage: null,
  progress: null,
  failedTiles: 0,
  finished: false,
  error: null,
});

// History — populated from list_history on mount, refreshed after each done.
export const history = $state<{ entries: HistoryEntry[] }>({ entries: [] });

// Toasts — append, auto-removed by Toast component after 4s.
export const toasts = $state<{ items: { id: string; level: "info" | "warn" | "error"; text: string }[] }>({
  items: [],
});

export function pushToast(level: "info" | "warn" | "error", text: string): void {
  const id = crypto.randomUUID();
  toasts.items = [...toasts.items, { id, level, text }];
  setTimeout(() => {
    toasts.items = toasts.items.filter((t) => t.id !== id);
  }, 4000);
}
```

- [ ] **Step 2: 扩展 app.css 加主题变量**

```css
:root {
  color-scheme: light dark;
  --bg: #ffffff;
  --bg-elev: #f5f5f7;
  --fg: #1d1d1f;
  --fg-muted: #6e6e73;
  --accent: #0066cc;
  --error: #d33;
  --warn: #d70;
  --border: #d2d2d7;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #1d1d1f;
    --bg-elev: #2c2c2e;
    --fg: #f5f5f7;
    --fg-muted: #98989d;
    --border: #38383a;
  }
}

* { box-sizing: border-box; }
html, body, #app { height: 100%; }
body {
  margin: 0;
  font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--fg);
}

button {
  font: inherit;
  background: var(--bg-elev);
  color: var(--fg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.4rem 0.8rem;
  cursor: pointer;
}
button:hover:not(:disabled) { background: var(--accent); color: white; }
button:disabled { opacity: 0.4; cursor: not-allowed; }

input[type="text"], input[type="number"], select {
  font: inherit;
  background: var(--bg);
  color: var(--fg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 0.3rem 0.5rem;
}
input.invalid { border-color: var(--error); }
```

- [ ] **Step 3: pnpm check**

```bash
pnpm check
```

Expected: `0 errors and 0 warnings`. 如果报 `$state` 等 rune 找不到，确认 svelte 是 `^5.1.0` 且 svelte-check 是 `^4.0.0`。

- [ ] **Step 4: 提交**

```bash
git add src/lib/state.svelte.ts src/app.css
git commit -m "feat(c): add global state runes + light/dark theme variables

Single source of truth for input form, estimate, active download, history
and toasts. Components import directly; no svelte/store. Color theme
follows prefers-color-scheme."
```

---

### Task 1.2：3 面板 App 布局 + Toast 组件

**Files:**
- Replace: `src/App.svelte`
- Create: `src/components/Toast.svelte`
- Create: `src/components/InputPanel.svelte` (空占位)
- Create: `src/components/MapPanel.svelte` (空占位)
- Create: `src/components/ProgressPanel.svelte` (空占位)
- Create: `src/components/HistoryPanel.svelte` (空占位)

- [ ] **Step 1: 写四个空占位组件**

每个文件内容（替换 `XxxPanel`）：

```svelte
<!-- src/components/InputPanel.svelte -->
<section class="panel">
  <h2>Input</h2>
  <p class="muted">filled in Phase 2</p>
</section>

<style>
  .panel { padding: 1rem; }
  h2 { margin: 0 0 0.5rem; font-size: 1rem; }
  .muted { color: var(--fg-muted); font-size: 0.9rem; }
</style>
```

四份内容除标题外完全相同。**MapPanel 的 panel 内 placeholder 改为** `<div class="map-placeholder"/>` + `<style>.map-placeholder { aspect-ratio: 16/9; background: var(--bg-elev); border-radius: 8px; }</style>`，让占位时也有视觉权重。

- [ ] **Step 2: 写 Toast 组件**

`src/components/Toast.svelte`:

```svelte
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
```

- [ ] **Step 3: 替换 App.svelte 为 3 面板布局**

```svelte
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
```

- [ ] **Step 4: pnpm check + 启动 dev 验证**

```bash
pnpm check
```

Expected: 0 errors。

```bash
pnpm tauri dev > /tmp/tauri-dev.log 2>&1 &
sleep 60
grep -E "windowDidBecomeKey|error\[" /tmp/tauri-dev.log | tail -5
pkill -f "tauri dev" || true
pkill -f "imagery-downloader" || true
```

Expected: 看到 `windowDidBecomeKey:` 行；无 `error[` 行。

- [ ] **Step 5: 提交**

```bash
git add src/App.svelte src/components/
git commit -m "feat(c): add 3-pane layout with Toast and panel placeholders

Left pane = InputPanel, center = MapPanel, right column stacks
ProgressPanel + HistoryPanel. Each panel is a placeholder that subsequent
phases will fill in."
```

---

### Task 1.3：Toast 集成验证（小冒烟）

**Files:**
- Modify: `src/components/InputPanel.svelte`（临时按钮触发 toast）

> 这个 task 是为了证明 toast 系统工作；末尾会还原占位。

- [ ] **Step 1: 临时改 InputPanel.svelte 加按钮**

```svelte
<script lang="ts">
  import { pushToast } from "../lib/state.svelte";
</script>

<section class="panel">
  <h2>Input</h2>
  <p class="muted">filled in Phase 2</p>
  <button onclick={() => pushToast("info", "Toast OK at " + new Date().toLocaleTimeString())}>
    Test toast
  </button>
</section>

<style>
  .panel { padding: 1rem; }
  h2 { margin: 0 0 0.5rem; font-size: 1rem; }
  .muted { color: var(--fg-muted); font-size: 0.9rem; }
</style>
```

- [ ] **Step 2: pnpm check + dev 看一眼**

启 `pnpm tauri dev`，点按钮看右下角 toast 出现并 4 秒后消失。

- [ ] **Step 3: 还原 InputPanel.svelte 到 Task 1.2 状态（删 button + 删 import）**

- [ ] **Step 4: 提交**

```bash
git add src/components/InputPanel.svelte
git commit -m "test(c): verify toast wiring then revert to placeholder

Manually verified pushToast() displays + auto-dismisses; reverting the
test button so Phase 2 can fill InputPanel cleanly."
```

✅ Phase 1 完成。

---

## Phase 2 — 输入面板（数值表单 + tabs + estimate 横幅）

### Task 2.1：InputPanel 主体（tabs + 数值字段 + 校验）

**Files:**
- Replace: `src/components/InputPanel.svelte`

- [ ] **Step 1: 写 InputPanel.svelte**

```svelte
<script lang="ts">
  import { input } from "../lib/state.svelte";
  import { validateBbox, validateZoom } from "../lib/validate";
  import type { Source } from "../lib/types";

  type Mode = "numeric" | "draw" | "import";
  let mode = $state<Mode>("numeric");

  let bboxErr = $derived(validateBbox(input.bbox));
  let zoomErr = $derived(validateZoom(input.zoom));

  const sources: Source[] = ["esri", "google", "auto"];
</script>

<section class="panel">
  <nav class="tabs">
    <button class:active={mode === "numeric"} onclick={() => (mode = "numeric")}>Numeric</button>
    <button class:active={mode === "draw"} onclick={() => (mode = "draw")}>Draw</button>
    <button class:active={mode === "import"} onclick={() => (mode = "import")}>Import</button>
  </nav>

  {#if mode === "numeric"}
    <div class="grid">
      <label>min Lon
        <input type="number" step="any" class:invalid={bboxErr}
               bind:value={input.bbox[0]} />
      </label>
      <label>min Lat
        <input type="number" step="any" class:invalid={bboxErr}
               bind:value={input.bbox[1]} />
      </label>
      <label>max Lon
        <input type="number" step="any" class:invalid={bboxErr}
               bind:value={input.bbox[2]} />
      </label>
      <label>max Lat
        <input type="number" step="any" class:invalid={bboxErr}
               bind:value={input.bbox[3]} />
      </label>
    </div>
    {#if bboxErr}<p class="err">{bboxErr}</p>{/if}
  {:else if mode === "draw"}
    <p class="muted">Draw a rectangle on the map. Coordinates appear here once you release.</p>
  {:else}
    <p class="muted">Drag a .geojson / .shp / .gpkg into the map area, or use the picker:</p>
    <button disabled>Choose file… (Plan A)</button>
  {/if}

  <hr />

  <label>Zoom <span class="hint">{input.zoom}</span>
    <input type="range" min="8" max="23" step="1" bind:value={input.zoom} />
  </label>
  {#if zoomErr}<p class="err">{zoomErr}</p>{/if}

  <label>Source
    <select bind:value={input.source}>
      {#each sources as s}<option value={s}>{s}</option>{/each}
    </select>
  </label>
</section>

<style>
  .panel { padding: 1rem; display: flex; flex-direction: column; gap: 0.7rem; }
  .tabs { display: flex; gap: 0.3rem; }
  .tabs button { flex: 1; }
  .tabs .active { background: var(--accent); color: white; }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
  }
  label { display: flex; flex-direction: column; font-size: 0.85rem; gap: 0.2rem; }
  .hint { float: right; color: var(--fg-muted); }
  .err { color: var(--error); font-size: 0.8rem; margin: 0; }
  .muted { color: var(--fg-muted); font-size: 0.9rem; }
  hr { border: none; border-top: 1px solid var(--border); margin: 0.3rem 0; }
</style>
```

- [ ] **Step 2: pnpm check + dev 烟测**

```bash
pnpm check
```

启 dev，验证：
- 三个 tabs 切换；
- 输入非法 bbox（如 minLon 大于 maxLon）→ 边框变红 + 错误文字出现；
- zoom 滑块从 8 移到 23 → 数字跟随。

- [ ] **Step 3: 提交**

```bash
git add src/components/InputPanel.svelte
git commit -m "feat(c): InputPanel with mode tabs, numeric form, inline validation

Numeric tab is the primary path; Draw and Import are stubs that direct
the user to the map area (filled in Phases 3 and 4). Zoom uses a 8..23
range slider; source uses a select with esri/google/auto."
```

---

### Task 2.2：InputPanel — outputPath 选择器

**Files:**
- Modify: `src/components/InputPanel.svelte`
- Modify: `src-tauri/capabilities/default.json`（加 dialog 权限）

- [ ] **Step 1: 安装 dialog plugin**

```bash
pnpm add @tauri-apps/plugin-dialog@^2
cd src-tauri
cargo add tauri-plugin-dialog@2
cd ..
```

- [ ] **Step 2: 在 lib.rs 注册 dialog plugin**

修改 `src-tauri/src/lib.rs` 的 builder chain，把 `.plugin(tauri_plugin_fs::init())` 后面接 `.plugin(tauri_plugin_dialog::init())`。

- [ ] **Step 3: 修改 capabilities/default.json**

加 `dialog:default` 到 permissions：

```json
{
  "$schema": "../gen/schemas/desktop-schema.json",
  "identifier": "default",
  "description": "Default capability for the main window",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "core:window:default",
    "core:webview:default",
    "dialog:default",
    "fs:default"
  ]
}
```

- [ ] **Step 4: 在 InputPanel.svelte 加输出路径行**

在 `</section>` 之前、最后一个 `<label>Source>` 之后加：

```svelte
  <label>Output
    <div class="row">
      <input type="text" placeholder="… select a .tif path" readonly value={input.outputPath} />
      <button onclick={pickOutput}>Pick…</button>
    </div>
  </label>
```

并在 `<script>` 里加：

```ts
import { save } from "@tauri-apps/plugin-dialog";

async function pickOutput() {
  const p = await save({
    title: "Save GeoTIFF as…",
    defaultPath: "imagery.tif",
    filters: [{ name: "GeoTIFF", extensions: ["tif"] }],
  });
  if (p) input.outputPath = p;
}
```

style 加：

```css
.row { display: flex; gap: 0.3rem; }
.row input { flex: 1; }
```

- [ ] **Step 5: 烟测**

启动 dev，点 Pick → 弹原生 save 对话框 → 选路径 → 输入框显示路径。

- [ ] **Step 6: 提交**

```bash
git add src/components/InputPanel.svelte src-tauri/Cargo.toml src-tauri/Cargo.lock \
        src-tauri/src/lib.rs src-tauri/capabilities/default.json package.json pnpm-lock.yaml
git commit -m "feat(c): add output path picker via tauri-plugin-dialog

Native save dialog filtered to .tif files; selected path bound to
input.outputPath in global state."
```

---

### Task 2.3：Estimate 横幅（debounced 200ms）

**Files:**
- Modify: `src/components/InputPanel.svelte`

- [ ] **Step 1: 在 InputPanel.svelte 顶部加 estimate effect**

`<script>` 里加：

```ts
import { estimate } from "../lib/state.svelte";
import { estimateOutput } from "../lib/ipc";
import { formatBytes, formatNumber, formatDuration } from "../lib/format";

let debounceTimer: number | null = null;
$effect(() => {
  // depends on bbox + zoom + source; re-run when any change
  const b = [...input.bbox];
  const z = input.zoom;
  const s = input.source;

  if (bboxErr || zoomErr) {
    estimate.data = null;
    estimate.loading = false;
    estimate.error = null;
    return;
  }

  if (debounceTimer) clearTimeout(debounceTimer);
  estimate.loading = true;
  debounceTimer = setTimeout(async () => {
    try {
      estimate.data = await estimateOutput(b as [number, number, number, number], z, s);
      estimate.error = null;
    } catch (e) {
      estimate.error = String(e);
    } finally {
      estimate.loading = false;
    }
  }, 200) as unknown as number;
});
```

- [ ] **Step 2: 在 `</section>` 前加 estimate 横幅**

```svelte
  <hr />
  <div class="estimate">
    {#if estimate.loading}
      <em class="muted">computing estimate…</em>
    {:else if estimate.error}
      <em class="err">{estimate.error}</em>
    {:else if estimate.data}
      <div>{formatNumber(estimate.data.tile_count)} tiles · {estimate.data.pixel_w} × {estimate.data.pixel_h} px</div>
      <div>≈ {estimate.data.est_size_mb.toFixed(1)} MB · {formatDuration(estimate.data.est_seconds)}</div>
    {:else}
      <em class="muted">enter a valid bbox to see estimate</em>
    {/if}
  </div>
```

style 加：

```css
.estimate {
  background: var(--bg-elev);
  padding: 0.5rem 0.7rem;
  border-radius: 6px;
  font-size: 0.85rem;
}
```

- [ ] **Step 3: 烟测**

启动 dev。**Phase 7 之前 estimate command 仍是 `Err("not implemented yet")`**，所以 `estimate.error` 应显示 "not implemented yet (Task 7.1)"。这是预期——Phase 2 只验证 wiring。Task 7.1 实现后这里应显示真实数字。

- [ ] **Step 4: 提交**

```bash
git add src/components/InputPanel.svelte
git commit -m "feat(c): debounced estimate banner under input form

200ms debounce on bbox/zoom/source change; shows tile count, pixel
dimensions, size MB and ETA. Currently displays the mock's
'not implemented yet' error — wired correctly, awaiting Task 7.1."
```

---

### Task 2.4：Start Download 按钮 + 校验门

**Files:**
- Modify: `src/components/InputPanel.svelte`

- [ ] **Step 1: 在 InputPanel `<script>` 加 start handler**

```ts
import { startDownload } from "../lib/ipc";
import { download, pushToast } from "../lib/state.svelte";

let canStart = $derived(
  !bboxErr && !zoomErr && input.outputPath.length > 0 && download.id === null,
);

async function start() {
  try {
    download.finished = false;
    download.error = null;
    download.progress = null;
    download.failedTiles = 0;
    download.stage = null;
    const r = await startDownload({
      bbox: input.bbox,
      zoom: input.zoom,
      source: input.source,
      output_path: input.outputPath,
      max_concurrency: input.maxConcurrency,
      retry_per_tile: input.retryPerTile,
      write_preview_png: input.writePreviewPng,
    });
    download.id = r.download_id;
  } catch (e) {
    pushToast("error", String(e));
  }
}
```

- [ ] **Step 2: 在 estimate 横幅之后加 start 按钮**

```svelte
  <button class="primary" disabled={!canStart} onclick={start}>
    {download.id ? "Downloading…" : "Start download"}
  </button>
```

style 加：

```css
.primary {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
  padding: 0.6rem;
  font-weight: 600;
}
```

- [ ] **Step 3: 烟测**

bbox / zoom 合法 + outputPath 已选 → Start 按钮亮；点击 → 触发 mock `start_download` → 报错 "not implemented yet (Task 7.2)"，按钮恢复。

- [ ] **Step 4: 提交**

```bash
git add src/components/InputPanel.svelte
git commit -m "feat(c): wire Start download button to ipc.startDownload

Disabled until bbox+zoom valid AND outputPath set AND no active download.
Invokes mock start_download — fully functional once Task 7.2 lands."
```

✅ Phase 2 完成。

---

## Phase 3 — 地图面板（MapLibre + 矩形框选）

### Task 3.1：MapPanel 初始化 MapLibre

**Files:**
- Replace: `src/components/MapPanel.svelte`
- Create: `src/lib/sources.ts`

- [ ] **Step 1: 写 src/lib/sources.ts**

```ts
import type { Source } from "./types";

// XYZ tile URL templates. {x}/{y}/{z} substituted by MapLibre.
// Sub-domain {s} not used — these all use single host.
export const TILE_URL: Record<Exclude<Source, "auto">, string> = {
  esri: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  google: "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
};

// "auto" is decided server-side; for the preview map we pick esri.
export function previewTileUrl(s: Source): string {
  if (s === "auto") return TILE_URL.esri;
  return TILE_URL[s];
}
```

- [ ] **Step 2: 替换 MapPanel.svelte**

```svelte
<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import maplibregl from "maplibre-gl";
  import "maplibre-gl/dist/maplibre-gl.css";
  import { input } from "../lib/state.svelte";
  import { previewTileUrl } from "../lib/sources";

  let container: HTMLDivElement;
  let map: maplibregl.Map | null = null;

  onMount(() => {
    map = new maplibregl.Map({
      container,
      style: {
        version: 8,
        sources: {
          base: {
            type: "raster",
            tiles: [previewTileUrl(input.source)],
            tileSize: 256,
            maxzoom: 22,
            attribution: input.source === "google" ? "© Google" : "© Esri, Maxar",
          },
        },
        layers: [{ id: "base", type: "raster", source: "base" }],
      },
      center: [(input.bbox[0] + input.bbox[2]) / 2, (input.bbox[1] + input.bbox[3]) / 2],
      zoom: 4,
    });
  });

  onDestroy(() => map?.remove());
</script>

<div class="wrap" bind:this={container}></div>

<style>
  .wrap { width: 100%; height: 100%; }
  :global(.maplibregl-canvas) { outline: none; }
</style>
```

- [ ] **Step 3: 烟测**

启动 dev — 中间面板出现卫星影像底图，可拖动可滚轮缩放。

- [ ] **Step 4: 提交**

```bash
git add src/components/MapPanel.svelte src/lib/sources.ts
git commit -m "feat(c): MapPanel with MapLibre raster XYZ base layer

Initial center is the midpoint of the default bbox. Source is fixed at
mount; Task 3.2 adds live source switching."
```

---

### Task 3.2：source 切换 → 换底图

**Files:**
- Modify: `src/components/MapPanel.svelte`

- [ ] **Step 1: 在 onMount 后加 $effect 监听 input.source**

`<script>` 末尾加：

```ts
$effect(() => {
  if (!map) return;
  const url = previewTileUrl(input.source);
  const src = map.getSource("base") as maplibregl.RasterTileSource | undefined;
  if (src) src.setTiles([url]);
});
```

- [ ] **Step 2: 烟测**

切换 InputPanel 的 Source select → 中间面板瓦片切换。esri/google 视觉差异明显（google 的标签字体）。

- [ ] **Step 3: 提交**

```bash
git add src/components/MapPanel.svelte
git commit -m "feat(c): swap base raster tiles when source changes"
```

---

### Task 3.3：矩形框选（原生 mouse handler）

**Files:**
- Modify: `src/components/MapPanel.svelte`

- [ ] **Step 1: 在 `<script>` 加绘制状态与 handler**

```ts
let drawing = $state(false);
let drawStart: maplibregl.LngLat | null = null;
let drawRect: maplibregl.LngLatBounds | null = null;

function enableDraw() {
  if (!map) return;
  drawing = true;
  map.getCanvas().style.cursor = "crosshair";
  map.dragPan.disable();
}
function disableDraw() {
  drawing = false;
  if (map) {
    map.getCanvas().style.cursor = "";
    map.dragPan.enable();
  }
  drawStart = null;
  drawRect = null;
}

function attachDraw() {
  if (!map) return;
  map.on("mousedown", (e) => {
    if (!drawing) return;
    drawStart = e.lngLat;
    e.preventDefault();
  });
  map.on("mousemove", (e) => {
    if (!drawing || !drawStart) return;
    drawRect = new maplibregl.LngLatBounds(drawStart, e.lngLat);
    drawPreviewLayer();
  });
  map.on("mouseup", (e) => {
    if (!drawing || !drawStart) return;
    const bounds = new maplibregl.LngLatBounds(drawStart, e.lngLat);
    input.bbox = [
      bounds.getWest(), bounds.getSouth(),
      bounds.getEast(), bounds.getNorth(),
    ];
    disableDraw();
    persistBboxLayer(bounds);
  });
}

function drawPreviewLayer() {
  if (!map || !drawRect) return;
  const ring = bboxRing(drawRect);
  if (map.getSource("draw-preview")) {
    (map.getSource("draw-preview") as maplibregl.GeoJSONSource).setData(ring);
  } else {
    map.addSource("draw-preview", { type: "geojson", data: ring });
    map.addLayer({
      id: "draw-preview",
      type: "line",
      source: "draw-preview",
      paint: { "line-color": "#0066cc", "line-width": 2, "line-dasharray": [2, 2] },
    });
  }
}

function persistBboxLayer(b: maplibregl.LngLatBounds) {
  if (!map) return;
  const ring = bboxRing(b);
  if (map.getLayer("draw-preview")) map.removeLayer("draw-preview");
  if (map.getSource("draw-preview")) map.removeSource("draw-preview");
  if (map.getSource("bbox")) {
    (map.getSource("bbox") as maplibregl.GeoJSONSource).setData(ring);
  } else {
    map.addSource("bbox", { type: "geojson", data: ring });
    map.addLayer({
      id: "bbox-fill",
      type: "fill",
      source: "bbox",
      paint: { "fill-color": "#0066cc", "fill-opacity": 0.15 },
    });
    map.addLayer({
      id: "bbox-line",
      type: "line",
      source: "bbox",
      paint: { "line-color": "#0066cc", "line-width": 2 },
    });
  }
}

function bboxRing(b: maplibregl.LngLatBounds): GeoJSON.Feature<GeoJSON.Polygon> {
  const w = b.getWest(), s = b.getSouth(), e = b.getEast(), n = b.getNorth();
  return {
    type: "Feature",
    properties: {},
    geometry: {
      type: "Polygon",
      coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    },
  };
}
```

把 `attachDraw()` 调用塞进 `onMount` 里，`map.on("load", ...)` 之后也行——MapLibre 4 的 `Map` 构造完成 mousedown handler 即可注册。

- [ ] **Step 2: 在 template 加 Draw 按钮**

替换 `<div class="wrap" ...>` 为：

```svelte
<div class="wrap">
  <div class="map" bind:this={container}></div>
  <div class="controls">
    {#if drawing}
      <button onclick={disableDraw}>Cancel draw</button>
    {:else}
      <button onclick={enableDraw}>Draw rectangle</button>
    {/if}
  </div>
</div>
```

style 改：

```css
.wrap { position: relative; width: 100%; height: 100%; }
.map { width: 100%; height: 100%; }
.controls {
  position: absolute;
  top: 1rem;
  left: 1rem;
  z-index: 10;
}
```

- [ ] **Step 3: 烟测**

点 "Draw rectangle" → cursor 变十字 → mousedown + drag → 看到虚线矩形 → mouseup → 实色蓝矩形 + InputPanel 的 4 个数值跟着变。

- [ ] **Step 4: 提交**

```bash
git add src/components/MapPanel.svelte
git commit -m "feat(c): native rectangle drawing on map writes bbox to state

Custom mousedown/mousemove/mouseup handlers; no third-party draw library.
Dashed preview while dragging, solid fill+line after release."
```

---

### Task 3.4：bbox 改变时 fitBounds 地图

**Files:**
- Modify: `src/components/MapPanel.svelte`

- [ ] **Step 1: 加 effect 监听 input.bbox**

`<script>` 末尾再加：

```ts
$effect(() => {
  if (!map) return;
  const [w, s, e, n] = input.bbox;
  if (![w, s, e, n].every(Number.isFinite)) return;
  // Avoid jitter: only fit if currently outside view
  const cur = map.getBounds();
  if (cur.contains([w, s]) && cur.contains([e, n])) return;
  map.fitBounds([[w, s], [e, n]], { padding: 40, animate: false });
  persistBboxLayer(new maplibregl.LngLatBounds([w, s], [e, n]));
});
```

- [ ] **Step 2: 烟测**

在 InputPanel 改 minLon 从 100 → 50 → 地图自动 fit 到新范围。

- [ ] **Step 3: 提交**

```bash
git add src/components/MapPanel.svelte
git commit -m "feat(c): map fitBounds when bbox changes from form input"
```

✅ Phase 3 完成。

---

## Phase 4 — 文件导入（拒绝态）

### Task 4.1：MapPanel 加 drop overlay

**Files:**
- Modify: `src/components/MapPanel.svelte`
- Modify: `src/components/InputPanel.svelte`（"Choose file" 按钮触发 dialog）

- [ ] **Step 1: 在 MapPanel.svelte 顶层加 dragover 监听**

`<script>` 加：

```ts
import { parseVectorFile } from "../lib/ipc";
import { pushToast } from "../lib/state.svelte";

let dragOver = $state(false);

async function handleDrop(e: DragEvent) {
  e.preventDefault();
  dragOver = false;
  const files = Array.from(e.dataTransfer?.files || []);
  if (!files.length) return;
  const f = files[0];
  // In Tauri, File objects don't expose absolute paths reliably; we surface
  // a guidance toast and require the file picker. Plan A wires real parsing.
  pushToast("warn", `Vector file dropped: ${f.name} — drag-drop parsing pending Plan A. Use the picker for now.`);
  void parseVectorFile; // mark imported so unused-locals check is OK
}
```

- [ ] **Step 2: 在 `.wrap` 上加 drag handler**

```svelte
<div
  class="wrap"
  ondragover={(e) => { e.preventDefault(); dragOver = true; }}
  ondragleave={() => (dragOver = false)}
  ondrop={handleDrop}
>
  <div class="map" bind:this={container}></div>
  <div class="controls">
    {#if drawing}
      <button onclick={disableDraw}>Cancel draw</button>
    {:else}
      <button onclick={enableDraw}>Draw rectangle</button>
    {/if}
  </div>
  {#if dragOver}
    <div class="drop-overlay">Drop vector file…</div>
  {/if}
</div>
```

style 加：

```css
.drop-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 102, 204, 0.4);
  color: white;
  display: grid;
  place-items: center;
  font-size: 1.5rem;
  pointer-events: none;
  z-index: 20;
}
```

- [ ] **Step 3: 烟测**

把任意文件拖到中间面板 → 蓝色覆盖层出现 "Drop vector file…" → 松手 → 右下角 toast 显示拒绝信息。

- [ ] **Step 4: 提交**

```bash
git add src/components/MapPanel.svelte
git commit -m "feat(c): drop overlay rejects vector files with guidance toast

Drag-drop parsing requires absolute file path resolution which Tauri's
webview restricts. Plan A will wire the picker path through invoke;
until then, drop shows a toast pointing the user to the future picker."
```

---

### Task 4.2：picker 入口（仅前端，调用 Plan A 时点亮）

**Files:**
- Modify: `src/components/InputPanel.svelte`

- [ ] **Step 1: 替换 import tab 内容**

把 `{:else}` 分支（import tab 内容）改为：

```svelte
  {:else}
    <p class="muted">Pick a .geojson, .shp or .gpkg file:</p>
    <button onclick={pickVector}>Choose file…</button>
    <p class="muted small">Parsing is implemented by Plan A; picker will wire through then.</p>
```

`<script>` 加：

```ts
import { open } from "@tauri-apps/plugin-dialog";

async function pickVector() {
  const p = await open({
    title: "Select vector file",
    filters: [
      { name: "Vector", extensions: ["geojson", "shp", "gpkg"] },
    ],
  });
  if (!p || Array.isArray(p)) return;
  try {
    const r = await parseVectorFile(p);
    input.bbox = r.bbox;
    pushToast("info", `Loaded ${r.layer_count} layer(s) from ${p}`);
  } catch (e) {
    pushToast("warn", `Vector parsing pending Plan A — ${String(e)}`);
  }
}
import { parseVectorFile } from "../lib/ipc";
```

style 加：

```css
.small { font-size: 0.75rem; }
```

- [ ] **Step 2: 烟测**

点 Choose file → 选 .geojson → toast 显示 "Vector parsing pending Plan A — not implemented yet"。Plan A 实现后会变成 "Loaded N layer(s)"。

- [ ] **Step 3: 提交**

```bash
git add src/components/InputPanel.svelte
git commit -m "feat(c): import tab picks vector file and forwards to parse_vector_file

Mock currently returns 'not implemented'; the user-visible flow is
identical to the post-Plan-A behavior, so no UI change required when
Plan A's vector parser lands."
```

✅ Phase 4 完成。

---

## Phase 5 — 进度面板（events + 取消 + 重试）

### Task 5.1：ProgressPanel 主体

**Files:**
- Replace: `src/components/ProgressPanel.svelte`

- [ ] **Step 1: 写 ProgressPanel.svelte**

```svelte
<script lang="ts">
  import { onMount } from "svelte";
  import { download, pushToast } from "../lib/state.svelte";
  import { onProgress, onStage, onDone, onTileFailed, cancelDownload, retryFailed } from "../lib/ipc";
  import { formatBytes, formatDuration, formatNumber } from "../lib/format";

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
      } else {
        download.error = d.error;
        pushToast("error", `Download failed: ${d.error}`);
      }
    }).then((u) => offs.push(u));

    return () => offs.forEach((u) => u());
  });

  let pct = $derived(
    download.progress
      ? Math.min(100, (download.progress.completed / Math.max(1, download.progress.total)) * 100)
      : 0,
  );

  async function cancel() {
    if (!download.id) return;
    try { await cancelDownload(download.id); } catch (e) { pushToast("error", String(e)); }
  }
  async function retry() {
    if (!download.id) return;
    try { await retryFailed(download.id); } catch (e) { pushToast("error", String(e)); }
  }
  function reset() {
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

    <div class="bar"><div class="fill" style="width:{pct}%"></div></div>

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
  .bar { height: 8px; background: var(--bg-elev); border-radius: 4px; overflow: hidden; }
  .fill { height: 100%; background: var(--accent); transition: width 200ms; }
  .warn { color: var(--warn); margin: 0; font-size: 0.85rem; }
  .actions { display: flex; gap: 0.5rem; margin-top: 0.3rem; }
</style>
```

- [ ] **Step 2: pnpm check**

```bash
pnpm check
```

Expected: 0 errors.

- [ ] **Step 3: 烟测**

启动 dev — Progress 面板显示 "No active download." 因为 mock command 还报错，点 Start 会提示 "not implemented yet (Task 7.2)"，progress 面板状态不变。Phase 7 落地后 progress 面板会真实跑起来。

- [ ] **Step 4: 提交**

```bash
git add src/components/ProgressPanel.svelte
git commit -m "feat(c): ProgressPanel reads download state and IPC events

Subscribes to progress/stage/tile-failed/done events on mount, filters by
download_id, derives percentage and formats speed/bytes/ETA. Cancel and
retry buttons invoke the corresponding mock commands."
```

✅ Phase 5 完成（其余动作要 mock 后端落地后才能真实见效）。

---

## Phase 6 — History 面板（持久化）

### Task 6.1：history.rs 真实实现 + cargo test

**Files:**
- Replace: `src-tauri/src/history.rs`
- Create: `src-tauri/tests/history_test.rs`

- [ ] **Step 1: 把 .gitignore 中 `src-tauri/Cargo.lock` 删掉（如果还在）**

参见 Task 0.5 的 DECISION 注。

- [ ] **Step 2: 写测试（fail-first）**

`src-tauri/tests/history_test.rs`:

```rust
use imagery_downloader_lib::history::{Store, HistoryEntry};
use std::path::PathBuf;

fn tmp_path() -> PathBuf {
    let p = std::env::temp_dir().join(format!("history-{}.json", uuid::Uuid::new_v4()));
    p
}

fn entry(zoom: u32) -> HistoryEntry {
    HistoryEntry {
        bbox: [100.0, 30.0, 110.0, 40.0],
        zoom,
        source: "esri".into(),
        output_path: "/tmp/x.tif".into(),
        ok: true,
        duration_sec: 5.0,
        total_tiles: 100,
        failed_tiles: 0,
        output_size_mb: 5.0,
        finished_at: "2026-05-03T10:00:00Z".into(),
    }
}

#[test]
fn roundtrip_empty() {
    let s = Store::open(tmp_path()).unwrap();
    assert!(s.list().is_empty());
}

#[test]
fn add_and_list() {
    let s = Store::open(tmp_path()).unwrap();
    s.add(entry(17)).unwrap();
    s.add(entry(18)).unwrap();
    let l = s.list();
    assert_eq!(l.len(), 2);
    // newest first
    assert_eq!(l[0].zoom, 18);
}

#[test]
fn dedupe_by_bbox_zoom_source() {
    let s = Store::open(tmp_path()).unwrap();
    s.add(entry(17)).unwrap();
    let mut e = entry(17);
    e.finished_at = "2026-05-03T11:00:00Z".into();
    s.add(e).unwrap();
    let l = s.list();
    assert_eq!(l.len(), 1);
    assert_eq!(l[0].finished_at, "2026-05-03T11:00:00Z");
}

#[test]
fn caps_at_10() {
    let s = Store::open(tmp_path()).unwrap();
    for z in 8..=22 { s.add(entry(z)).unwrap(); }
    let l = s.list();
    assert_eq!(l.len(), 10);
    // newest still on top
    assert_eq!(l[0].zoom, 22);
}

#[test]
fn clear() {
    let s = Store::open(tmp_path()).unwrap();
    s.add(entry(17)).unwrap();
    s.clear().unwrap();
    assert!(s.list().is_empty());
}

#[test]
fn persists_across_open() {
    let p = tmp_path();
    {
        let s = Store::open(&p).unwrap();
        s.add(entry(17)).unwrap();
    }
    let s2 = Store::open(&p).unwrap();
    assert_eq!(s2.list().len(), 1);
}
```

- [ ] **Step 3: 运行确认 fail**

```bash
cd src-tauri
cargo test --test history_test
cd ..
```

Expected: 编译失败因 `Store` 不存在。

- [ ] **Step 4: 写实现**

`src-tauri/src/history.rs`:

```rust
//! Persistent history of recent download parameters.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistoryEntry {
    pub bbox: [f64; 4],
    pub zoom: u32,
    pub source: String,
    pub output_path: String,
    pub ok: bool,
    pub duration_sec: f64,
    pub total_tiles: u32,
    pub failed_tiles: u32,
    pub output_size_mb: f64,
    pub finished_at: String,
}

pub struct Store {
    path: PathBuf,
    inner: Mutex<Vec<HistoryEntry>>,
}

const MAX: usize = 10;

impl Store {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let inner = if path.exists() {
            let bytes = fs::read(&path)?;
            serde_json::from_slice(&bytes).unwrap_or_default()
        } else {
            Vec::new()
        };
        Ok(Self { path, inner: Mutex::new(inner) })
    }

    pub fn list(&self) -> Vec<HistoryEntry> {
        self.inner.lock().unwrap().clone()
    }

    pub fn add(&self, entry: HistoryEntry) -> io::Result<()> {
        let mut g = self.inner.lock().unwrap();
        // Dedupe by (bbox, zoom, source).
        g.retain(|e| !(e.bbox == entry.bbox && e.zoom == entry.zoom && e.source == entry.source));
        g.insert(0, entry);
        if g.len() > MAX { g.truncate(MAX); }
        self.persist(&g)?;
        Ok(())
    }

    pub fn clear(&self) -> io::Result<()> {
        let mut g = self.inner.lock().unwrap();
        g.clear();
        self.persist(&g)?;
        Ok(())
    }

    fn persist(&self, list: &[HistoryEntry]) -> io::Result<()> {
        if let Some(parent) = self.path.parent() { fs::create_dir_all(parent)?; }
        let tmp = self.path.with_extension("json.tmp");
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&serde_json::to_vec_pretty(list).unwrap())?;
        f.sync_all()?;
        fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}

pub fn default_path() -> PathBuf {
    use directories::ProjectDirs;
    if let Some(d) = ProjectDirs::from("com", "zhangfeng", "imagery-downloader") {
        d.data_dir().join("history.json")
    } else {
        std::env::temp_dir().join("imagery-downloader-history.json")
    }
}
```

- [ ] **Step 5: 跑测试通过**

```bash
cd src-tauri
cargo test --test history_test
cd ..
```

Expected: `6 passed; 0 failed`.

- [ ] **Step 6: 提交**

```bash
git add src-tauri/src/history.rs src-tauri/tests/history_test.rs .gitignore src-tauri/Cargo.lock
git commit -m "feat(c): real history persistence with atomic write

Plan-A advance: this is logically core::history but lives at the crate
root for now. Atomic tempfile+rename, dedupe by bbox+zoom+source, cap
at 10 newest-first. 6 cargo tests cover roundtrip / dedupe / cap / clear.
Cargo.lock now committed."
```

---

### Task 6.2：list_history / clear_history commands + 启动加载

**Files:**
- Modify: `src-tauri/src/lib.rs`
- Create: `src-tauri/src/mocks/history_commands.rs`
- Modify: `src-tauri/src/mocks/mod.rs`

- [ ] **Step 1: 写 mocks/history_commands.rs**

> 这两个 command 不是"假"的——它们直接用 `history::Store`，所以不放 mocks 目录其实更合适。但为了让 Plan B 的"删 mocks/" patch 简单，**我们把 thin wrapper 放在 mocks 里**，Plan B 时把 wrapper 移到顶层 `commands/history.rs`。

```rust
use crate::history::{Store, HistoryEntry, default_path};
use std::sync::OnceLock;

static STORE: OnceLock<Store> = OnceLock::new();

fn store() -> &'static Store {
    STORE.get_or_init(|| Store::open(default_path()).expect("open history store"))
}

#[tauri::command]
pub fn list_history() -> Vec<HistoryEntry> {
    store().list()
}

#[tauri::command]
pub fn clear_history() -> Result<serde_json::Value, String> {
    store().clear().map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "ok": true }))
}

pub fn record(entry: HistoryEntry) {
    let _ = store().add(entry);
}
```

- [ ] **Step 2: 修改 mocks/mod.rs**

```rust
pub mod commands;
pub mod history_commands;
pub mod runner;
```

- [ ] **Step 3: 修改 lib.rs 注册**

`invoke_handler!` 加：

```rust
mocks::history_commands::list_history,
mocks::history_commands::clear_history,
```

- [ ] **Step 4: cargo check**

```bash
cd src-tauri
cargo check
cd ..
```

- [ ] **Step 5: 提交**

```bash
git add src-tauri/src/lib.rs src-tauri/src/mocks/
git commit -m "feat(c): expose list_history / clear_history Tauri commands

Both wrap the real Store; Plan B will move them out of mocks/ and into
commands/history.rs without behavioral change."
```

---

### Task 6.3：HistoryPanel 渲染 + 点击恢复

**Files:**
- Replace: `src/components/HistoryPanel.svelte`

- [ ] **Step 1: 写 HistoryPanel.svelte**

```svelte
<script lang="ts">
  import { onMount } from "svelte";
  import { history, input, pushToast } from "../lib/state.svelte";
  import { listHistory, clearHistory } from "../lib/ipc";
  import { formatDuration, formatNumber } from "../lib/format";
  import type { HistoryEntry, Source } from "../lib/types";

  async function load() {
    try {
      history.entries = await listHistory();
    } catch (e) {
      pushToast("error", `Load history failed: ${e}`);
    }
  }
  onMount(load);

  async function clear() {
    if (!confirm("Clear all history?")) return;
    await clearHistory();
    history.entries = [];
  }

  function restore(e: HistoryEntry) {
    input.bbox = e.bbox;
    input.zoom = e.zoom;
    input.source = e.source as Source;
    input.outputPath = e.output_path;
    pushToast("info", "History entry restored");
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
      {#each history.entries as e (e.finished_at + e.zoom)}
        <li>
          <button class="entry" onclick={() => restore(e)}>
            <div class="meta">z{e.zoom} · {e.source} · {e.ok ? "✓" : "✗"}</div>
            <div class="bbox">[{e.bbox.map((n) => n.toFixed(2)).join(", ")}]</div>
            <div class="row muted">
              {formatNumber(e.total_tiles)} tiles · {formatDuration(e.duration_sec)} ·
              {e.output_size_mb.toFixed(1)} MB
            </div>
            <div class="ts muted">{e.finished_at}</div>
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
  .meta { font-weight: 600; font-size: 0.9rem; }
  .bbox { font-family: ui-monospace, Menlo, monospace; font-size: 0.75rem; }
  .ts { font-size: 0.7rem; }
</style>
```

- [ ] **Step 2: pnpm check + dev**

```bash
pnpm check
```

启 dev — History 面板初始为 "No previous downloads."；mock 下载完成后（Phase 7 落地）会出现条目。

- [ ] **Step 3: 提交**

```bash
git add src/components/HistoryPanel.svelte
git commit -m "feat(c): HistoryPanel renders entries, click restores params, clear button

Loads on mount via list_history; entries dedupe-by-bbox+zoom+source on
add (handled in store). Clicking an entry restores bbox/zoom/source/path
to the input form."
```

✅ Phase 6 完成（条目还要等 Phase 7 mock 真正完成下载才会写入）。

---

## Phase 7 — Mock 后端实现

### Task 7.1：mock estimate_output 确定性返回

**Files:**
- Modify: `src-tauri/src/mocks/commands.rs`

- [ ] **Step 1: 写 estimate_output 实现**

替换 `estimate_output` 函数：

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct Bbox(pub [f64; 4]);

#[derive(Debug, Serialize)]
pub struct EstimateOutput {
    pub tile_count: u32,
    pub pixel_w: u32,
    pub pixel_h: u32,
    pub est_size_mb: f64,
    pub est_seconds: f64,
}

#[tauri::command]
pub async fn estimate_output(bbox: [f64; 4], zoom: u32, source: String) -> Result<EstimateOutput, String> {
    if zoom < 8 || zoom > 23 { return Err(format!("zoom {} out of range 8..23", zoom)); }
    if bbox[0] >= bbox[2] || bbox[1] >= bbox[3] { return Err("invalid bbox".into()); }
    // Web mercator tile math (good enough for an estimate; Plan A's tiles module is authoritative).
    let n = 2_f64.powi(zoom as i32);
    let lon_w = bbox[0]; let lon_e = bbox[2];
    let lat_s = bbox[1].max(-85.0511); let lat_n = bbox[3].min(85.0511);
    let x0 = ((lon_w + 180.0) / 360.0 * n).floor() as i64;
    let x1 = ((lon_e + 180.0) / 360.0 * n).ceil() as i64;
    let y0 = ((1.0 - (lat_n.to_radians().tan() + 1.0 / lat_n.to_radians().cos()).ln() / std::f64::consts::PI) / 2.0 * n).floor() as i64;
    let y1 = ((1.0 - (lat_s.to_radians().tan() + 1.0 / lat_s.to_radians().cos()).ln() / std::f64::consts::PI) / 2.0 * n).ceil() as i64;
    let tx = (x1 - x0).max(1) as u32;
    let ty = (y1 - y0).max(1) as u32;
    let tile_count = tx * ty;
    let pixel_w = tx * 256;
    let pixel_h = ty * 256;
    // Rough heuristic: 30 KB/tile JPEG, 50 tiles/sec at default concurrency.
    let est_size_mb = tile_count as f64 * 30.0 / 1024.0;
    let est_seconds = (tile_count as f64 / 50.0).max(1.0);
    let _ = source; // accepted but ignored in mock
    Ok(EstimateOutput { tile_count, pixel_w, pixel_h, est_size_mb, est_seconds })
}
```

- [ ] **Step 2: cargo check + 烟测**

```bash
cd src-tauri && cargo check && cd ..
```

启 dev — 输入 bbox `[100, 30, 110, 40]`, zoom 17 → estimate 横幅显示真实数字（约几万 tile，几 GB）。

- [ ] **Step 3: 提交**

```bash
git add src-tauri/src/mocks/commands.rs
git commit -m "feat(c): mock estimate_output with web-mercator tile math

Gives realistic numbers so the UI estimate banner reflects bbox/zoom
changes in real time. Not authoritative — Plan A's core::tiles will
replace this with the canonical implementation."
```

---

### Task 7.2：mock start_download — 异步 progress emitter

**Files:**
- Modify: `src-tauri/src/mocks/commands.rs`
- Modify: `src-tauri/src/mocks/runner.rs`

- [ ] **Step 1: 写 runner.rs**

```rust
use std::collections::HashMap;
use std::sync::Mutex;
use tokio_util::sync::CancellationToken;

#[derive(Default)]
pub struct Runner {
    tokens: Mutex<HashMap<String, CancellationToken>>,
}

impl Runner {
    pub fn register(&self, id: String) -> CancellationToken {
        let t = CancellationToken::new();
        self.tokens.lock().unwrap().insert(id, t.clone());
        t
    }
    pub fn cancel(&self, id: &str) -> bool {
        if let Some(t) = self.tokens.lock().unwrap().remove(id) {
            t.cancel();
            true
        } else { false }
    }
    pub fn forget(&self, id: &str) {
        self.tokens.lock().unwrap().remove(id);
    }
}
```

- [ ] **Step 2: 写 start_download + driver**

替换 `start_download`：

```rust
use crate::history::HistoryEntry;
use crate::mocks::history_commands::record;
use crate::mocks::runner::Runner;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, State};
use tokio::time::sleep;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct StartDownloadArgs {
    pub bbox: [f64; 4],
    pub zoom: u32,
    pub source: String,
    pub output_path: String,
    pub max_concurrency: u32,
    pub retry_per_tile: u32,
    pub write_preview_png: bool,
}

#[derive(Debug, Serialize)]
pub struct StartDownloadResp { pub download_id: String }

#[derive(Debug, Serialize, Clone)]
struct ProgressEvent {
    download_id: String,
    completed: u32,
    total: u32,
    bytes_downloaded: u64,
    current_speed_mbps: f64,
    elapsed_sec: f64,
    eta_sec: f64,
}

#[derive(Debug, Serialize, Clone)]
struct StageEvent { download_id: String, stage: String }

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
enum DoneEvent {
    Ok {
        download_id: String, ok: bool, output_path: String, preview_path: Option<String>,
        bbox: [f64; 4], zoom: u32, source_used: String,
        duration_sec: f64, total_tiles: u32, failed_tiles: u32, output_size_mb: f64,
    },
    Err { download_id: String, ok: bool, error: String },
}

const MOCK_TOTAL: u32 = 100;
const MOCK_DURATION_SEC: f64 = 5.0;
const TICKS: u32 = 50; // 10 Hz, will exercise UI throttling

#[tauri::command]
pub async fn start_download(
    app: AppHandle,
    runner: State<'_, Runner>,
    args: StartDownloadArgs,
) -> Result<StartDownloadResp, String> {
    if args.bbox[0] >= args.bbox[2] || args.bbox[1] >= args.bbox[3] {
        return Err("invalid bbox".into());
    }
    let id = Uuid::new_v4().to_string();
    let token = runner.register(id.clone());
    let app_clone = app.clone();
    let id_clone = id.clone();
    let bbox = args.bbox;
    let zoom = args.zoom;
    let source = args.source.clone();
    let output_path = args.output_path.clone();

    tokio::spawn(async move {
        let _ = app_clone.emit("download://stage", StageEvent { download_id: id_clone.clone(), stage: "downloading".into() });
        let start = Instant::now();
        let tick_dur = Duration::from_secs_f64(MOCK_DURATION_SEC / TICKS as f64);
        let bytes_per_tile = 30_000u64;
        for i in 1..=TICKS {
            tokio::select! {
                _ = sleep(tick_dur) => {},
                _ = token.cancelled() => {
                    let _ = app_clone.emit("download://done",
                        DoneEvent::Err { download_id: id_clone.clone(), ok: false, error: "cancelled".into() });
                    return;
                }
            }
            let completed = (i * MOCK_TOTAL) / TICKS;
            let elapsed = start.elapsed().as_secs_f64();
            let eta = if i == TICKS { 0.0 } else { (MOCK_DURATION_SEC - elapsed).max(0.0) };
            let bytes = bytes_per_tile * completed as u64;
            let speed = (bytes as f64 / 1.0e6) / elapsed.max(0.01);
            let _ = app_clone.emit("download://progress", ProgressEvent {
                download_id: id_clone.clone(),
                completed, total: MOCK_TOTAL,
                bytes_downloaded: bytes,
                current_speed_mbps: speed,
                elapsed_sec: elapsed,
                eta_sec: eta,
            });
        }
        let _ = app_clone.emit("download://stage", StageEvent { download_id: id_clone.clone(), stage: "stitching".into() });
        sleep(Duration::from_millis(300)).await;
        let _ = app_clone.emit("download://stage", StageEvent { download_id: id_clone.clone(), stage: "writing_cog".into() });
        sleep(Duration::from_millis(400)).await;
        if args.write_preview_png {
            let _ = app_clone.emit("download://stage", StageEvent { download_id: id_clone.clone(), stage: "writing_preview".into() });
            sleep(Duration::from_millis(200)).await;
        }
        let total_dur = start.elapsed().as_secs_f64();
        let entry = HistoryEntry {
            bbox, zoom, source: source.clone(),
            output_path: output_path.clone(),
            ok: true,
            duration_sec: total_dur,
            total_tiles: MOCK_TOTAL,
            failed_tiles: 0,
            output_size_mb: (bytes_per_tile * MOCK_TOTAL as u64) as f64 / 1.0e6,
            finished_at: chrono_iso_now(),
        };
        record(entry);
        let _ = app_clone.emit("download://done",
            DoneEvent::Ok {
                download_id: id_clone.clone(), ok: true,
                output_path,
                preview_path: if args.write_preview_png { Some(format!("{}.preview.png", args.output_path.trim_end_matches(".tif"))) } else { None },
                bbox, zoom, source_used: source,
                duration_sec: total_dur,
                total_tiles: MOCK_TOTAL,
                failed_tiles: 0,
                output_size_mb: (bytes_per_tile * MOCK_TOTAL as u64) as f64 / 1.0e6,
            });
        // Clean up token registry; the cancel path already removed it.
        // (Held in scope by the task — re-acquire & forget by id_clone.)
    });

    Ok(StartDownloadResp { download_id: id })
}

fn chrono_iso_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now();
    let epoch = now.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    // crude ISO 8601 — Plan A will use chrono::Utc::now()
    format!("epoch:{}", epoch)
}
```

> 注：`chrono_iso_now` 故意写成 `epoch:N` 字符串而不是引 `chrono` crate——避免本任务为 mock 多加一个 dep。Plan A 的 history 模块会把它换成真 ISO。前端 HistoryPanel 显示这个字符串照样可读。

- [ ] **Step 3: cargo check**

```bash
cd src-tauri && cargo check && cd ..
```

如果报 missing fields / serde — fix。

- [ ] **Step 4: 烟测**

启动 dev → 选 outputPath → Start → progress 条平滑跑 5 秒 → 看到 stage 切换 → done toast → history 出现条目。

- [ ] **Step 5: 提交**

```bash
git add src-tauri/src/mocks/
git commit -m "feat(c): mock start_download emits realistic 5-sec progress + done

Spawns tokio task that ticks 50 progress events over 5s (10Hz), then
walks through stage events (downloading/stitching/writing_cog/preview),
finally emits a done event and writes a HistoryEntry. CancellationToken
registered so cancel_download (Task 7.3) can interrupt mid-flight."
```

---

### Task 7.3：mock cancel_download

**Files:**
- Modify: `src-tauri/src/mocks/commands.rs`

- [ ] **Step 1: 替换 cancel_download**

```rust
#[tauri::command]
pub fn cancel_download(runner: State<'_, Runner>, download_id: String) -> Result<serde_json::Value, String> {
    let cancelled = runner.cancel(&download_id);
    if !cancelled { return Err(format!("unknown download_id: {}", download_id)); }
    Ok(serde_json::json!({ "ok": true }))
}
```

- [ ] **Step 2: 烟测**

启动 dev → Start → 进度跑到 ~30% → 点 Cancel → 进度立即停 + done toast 显示 "cancelled" + UI 复位。

- [ ] **Step 3: 提交**

```bash
git add src-tauri/src/mocks/commands.rs
git commit -m "feat(c): mock cancel_download triggers CancellationToken"
```

---

### Task 7.4：mock retry_failed + parse_vector_file rejection

**Files:**
- Modify: `src-tauri/src/mocks/commands.rs`

- [ ] **Step 1: 替换 retry_failed**

```rust
#[tauri::command]
pub async fn retry_failed(
    app: AppHandle,
    runner: State<'_, Runner>,
    download_id: String,
) -> Result<serde_json::Value, String> {
    // Mock retries by emitting a fast 1-sec progress run with no failures.
    let token = runner.register(download_id.clone());
    let id_clone = download_id.clone();
    let app_clone = app.clone();
    tokio::spawn(async move {
        let _ = app_clone.emit("download://stage", StageEvent { download_id: id_clone.clone(), stage: "downloading".into() });
        for i in 1..=10u32 {
            tokio::select! {
                _ = sleep(Duration::from_millis(100)) => {},
                _ = token.cancelled() => {
                    let _ = app_clone.emit("download://done",
                        DoneEvent::Err { download_id: id_clone.clone(), ok: false, error: "cancelled".into() });
                    return;
                }
            }
            let _ = app_clone.emit("download://progress", ProgressEvent {
                download_id: id_clone.clone(),
                completed: i * 10, total: MOCK_TOTAL,
                bytes_downloaded: 0,
                current_speed_mbps: 1.0,
                elapsed_sec: i as f64 * 0.1,
                eta_sec: (10 - i) as f64 * 0.1,
            });
        }
        let _ = app_clone.emit("download://done",
            DoneEvent::Ok {
                download_id: id_clone, ok: true,
                output_path: String::new(),
                preview_path: None,
                bbox: [0.0; 4], zoom: 0, source_used: String::new(),
                duration_sec: 1.0,
                total_tiles: MOCK_TOTAL,
                failed_tiles: 0,
                output_size_mb: 0.0,
            });
    });
    Ok(serde_json::json!({ "ok": true }))
}
```

- [ ] **Step 2: 替换 parse_vector_file**

```rust
#[tauri::command]
pub fn parse_vector_file(_path: String) -> Result<serde_json::Value, String> {
    Err("vector parsing pending Plan A".into())
}
```

- [ ] **Step 3: cargo check + 烟测**

启动 dev — Start → 等完成 → 因 mock 永远不出错，failed_tiles=0，所以 retry 按钮**不会出现**。手动模拟失败：把 `MOCK_TOTAL` 临时改 100 但 emit `tile-failed` 几次再恢复。或在 self-review 时记下 "retry 路径在 mock 阶段没真实演练"。

实际 retry_failed 只会在 Plan A/B 真实失败发生后才有意义。Mock 阶段验证按钮 wiring 即可——临时强行让按钮出现的方法：

```ts
// 在浏览器 devtools 控制台
import("./src/lib/state.svelte.ts").then(s => s.download.failedTiles = 5)
```

- [ ] **Step 4: 提交**

```bash
git add src-tauri/src/mocks/commands.rs
git commit -m "feat(c): mock retry_failed + parse_vector_file rejection

retry_failed runs a 1-sec mini-progress to verify wiring;
parse_vector_file always errors so the import path surfaces a
'pending Plan A' toast. Frontend code path unchanged."
```

✅ Phase 7 完成。

---

## Phase 8 — 集成验收 + 文档

### Task 8.1：手测 happy path + 截屏（人机协作）

**Files:** 无修改。

- [ ] **Step 1: 启动 + 走完整流程**

```bash
pnpm tauri dev
```

按这个顺序操作：

1. 默认 bbox `[100, 30, 110, 40]` zoom 17 → estimate 横幅显示真实数字。
2. 在地图上点 "Draw rectangle" → 拖一个新区域 → InputPanel 4 个数字跟着变。
3. 切换 Source → 底图换瓦片商。
4. 点 Pick… → 选 `~/Desktop/test.tif` 作为输出。
5. 点 Start download → progress 条 5 秒跑满 → toast "Done · 100 tiles · 5.xs" → history 出现一条。
6. 点 Start download 再来一次 → 进度到 30% 时点 Cancel → toast "Download failed: cancelled"。
7. 点 history 第一条 → 表单参数恢复 → 地图 fit 回去。
8. 点 history Clear → 清空。
9. 切到 Import tab → 点 Choose file → 选任意 .geojson → toast "Vector parsing pending Plan A"。
10. 拖任意文件到地图 → 蓝色覆盖层 → 松手 → toast 提示。

每个步骤都通过 = Plan C 验收通过。任何步骤失败：写下来，回到对应 task 修。

- [ ] **Step 2: 写一份验收快照（可选）**

把上述 10 步产物贴到 `docs/screenshots/plan-c-walkthrough.md`（截屏从 macOS 全屏 cmd-shift-4）。这步不强制——但 Plan B/A review 时有图很有用。

- [ ] **Step 3: 提交（如果加了 screenshots）**

```bash
git add docs/screenshots/
git commit -m "docs(c): walkthrough screenshots of Plan C UI" || echo "no screenshots added"
```

---

### Task 8.2：写 mocks/README.md（Plan B 替换契约）+ 更新顶层 README + spec status

**Files:**
- Create: `src-tauri/src/mocks/README.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-05-03-imagery-downloader-tauri-design.md`

- [ ] **Step 1: 写 mocks/README.md**

````markdown
# `src-tauri/src/mocks/` — Plan B replacement contract

This directory contains **mock Tauri command handlers** that drive the
frontend UI shipped by Plan C. The frontend code in `src/` is _final_;
only this directory will be replaced when Plan A/B land.

## What gets replaced

When Plan B implements real backend commands, this is the patch shape:

1. Move `history_commands.rs` content into `src-tauri/src/commands/history.rs`
   (no logic change — just relocation + change `mod mocks` references).
2. Delete `mod.rs`, `commands.rs`, `runner.rs`. The History store is
   already in `src-tauri/src/history.rs` and survives.
3. Add real implementations in `src-tauri/src/commands/{download.rs, vector.rs}` that
   call into `core::*` modules.
4. Update `lib.rs`'s `invoke_handler!` to register the real command paths.

## Frontend invariant

The frontend MUST NOT import anything from this directory. It only knows
about IPC command names and event names — both kept stable across the
mock → real transition. As of writing, the contract surface is:

| invoke command          | event channel            |
|-------------------------|--------------------------|
| `parse_vector_file`     | (none)                   |
| `estimate_output`       | (none)                   |
| `start_download`        | `download://progress`    |
| `cancel_download`       | `download://stage`       |
| `retry_failed`          | `download://tile-failed` |
| `list_history`          | `download://done`        |
| `clear_history`         |                          |

## Why this exists

- Plan C wanted the UI working end-to-end before any real backend is
  implemented, so users can validate UX early.
- Without mocks, the UI couldn't be tested for throttling, ETA math,
  cancel responsiveness, or history persistence.
- The mock progress emitter ticks at 10Hz over 5s — high enough that
  real-world UI throttling code (4Hz target per spec §3.2) gets exercised.
````

- [ ] **Step 2: 修改顶层 README 加状态行**

把 README 顶部的 status box 改为：

```markdown
> **Status: Plan C complete (UI + mock backend).** This repo is mid-rewrite
> from a Python CLI (preserved in [`legacy/`](./legacy)) to a Tauri 2.x
> desktop app. The UI is fully functional but driven by a mock backend —
> tile downloads are simulated. Plans A/B will replace the mock with real
> downloads. See `docs/superpowers/plans/`.
```

- [ ] **Step 3: 给 spec 追加 Plan C 状态**

在 spec 末尾加：

```markdown

## Plan C Implementation Status (2026-05-03)

✅ Frontend UI complete: 3-pane layout (input / map / progress+history).
✅ Three input modes: numeric form, map rectangle draw, file picker (parse pending Plan A).
✅ MapLibre raster preview with live source switching.
✅ Mock backend in `src-tauri/src/mocks/`: `start_download` simulates a
   5-sec download with 50 progress ticks, real CancellationToken-based
   cancel, real history persistence.
✅ History panel with 10-newest-first cap, dedupe by bbox+zoom+source,
   click-to-restore.
✅ Vitest unit tests for validators, formatters, IPC wrapper. Cargo tests
   for history Store.
✅ The frontend is final-shape; Plan B replaces only `src-tauri/src/mocks/`.

Pending:
- Plan A: real `tiles`, `sources`, `downloader`, `stitcher`, `cog`, `vector` modules.
- Plan B: real Tauri commands wiring Plan A modules into `invoke_handler!`.
- Plan D Phases 2-4: CI workflow, release workflow, signing docs (deferred).
```

- [ ] **Step 4: 提交**

```bash
git add src-tauri/src/mocks/README.md README.md \
        docs/superpowers/specs/2026-05-03-imagery-downloader-tauri-design.md
git commit -m "docs(c): document Plan C mock contract + status

mocks/README.md describes exactly what Plan B should delete and replace.
Top-level README updated to reflect that the UI works end-to-end on a
mock backend. Spec gets a Plan C status section."
```

✅ Plan C 全部完成。

---

## Self-Review

按 skill 要求三项 checklist 自审。

### 1. Spec 覆盖

| Spec 要求 | 覆盖 task | 备注 |
|---|---|---|
| §3 IPC 契约（commands + events） | T0.2 (types), T0.3 (ipc.ts), Phase 7 (commands), T5.1 (events) | ✅ |
| §3.3 download_id UUID | T7.2 (`Uuid::new_v4()`) | ✅ |
| §3 estimate_output 防抖 200ms | T2.3 | ✅ |
| §4.1 数值输入路径 | Phase 2 | ✅ |
| §4.2 地图框选路径 | Phase 3 | ✅ |
| §4.3 文件导入路径 | Phase 4 + T2.4 picker | ⚠ parse 未实现，按设计转 Plan A |
| §5 错误处理 — 输入校验 | T2.1 inline validation | ✅ |
| §5 错误处理 — 取消 | T7.3 + ProgressPanel cancel | ✅ |
| §5 关键不变量 — 进度按瓦片数 | T7.2 `completed/total` | ✅ |
| §5 关键不变量 — session tile cache | ⚠ Plan A 范围 | 跳出 Plan C |
| §6 history 最近 10 条 | Phase 6 + T6.1 cap=10 dedupe | ✅ |
| §6 source ESRI/Google/auto | T0.2 types + T2.1 select + T3.1 sources.ts | ✅ |
| §8 测试 — 前端 component | T0.4 (validate/format) + T0.3 (ipc) | ✅ ~10 tests，符合预估 |
| §8 测试 — Rust 集成 | T6.1 history_test.rs (6 tests) | ✅ 部分 (Plan A 补全) |

无 spec 段落遗漏；§5 "session 内 tile cache 保留到 retry_failed" 跳出 Plan C 范围（属于真实 downloader 实现）。

### 2. Placeholder 扫描

无 "TBD" / "TODO" / "implement later" / "appropriate error handling" / "similar to Task N"。每个 task 含具体代码块或具体命令。一处 `DECISION:` 是给用户的 Cargo.lock 提交决策点，**不是** placeholder。

### 3. 类型 / 命名一致性

- `Bbox` 类型在 `src/lib/types.ts`、`src-tauri/src/history.rs`、`src-tauri/src/mocks/commands.rs` 三处均为 `[f64; 4]` / `[number, number, number, number]`。✅
- `Source` 是 `"esri" | "google" | "auto"` 三处一致。✅
- 事件名 `download://progress`、`download://stage`、`download://tile-failed`、`download://done` 在 ipc.ts 与 commands.rs 一致。✅
- `HistoryEntry` 字段名 snake_case 在 Rust 与 TS 都用 snake，由 serde + 直接消费保证。✅
- mock command 函数名（`estimate_output`, `start_download`, `cancel_download`, `retry_failed`, `parse_vector_file`, `list_history`, `clear_history`）= invoke 字符串 = ipc.ts 函数 camelCase 后的 snake 转换 ✅
- `pushToast` 在 state.svelte.ts 定义，在 InputPanel/MapPanel/ProgressPanel/HistoryPanel 调用，签名 `(level, text)` 一致。✅

无不一致。

---

## Execution Handoff

Plan 已保存到 `docs/superpowers/plans/2026-05-03-plan-c-frontend-ui-mock-backend.md`。

**两种执行模式：**

1. **Subagent-Driven（推荐）** — 每 task fresh subagent + 两阶段 review。优点：单 task 上下文干净；缺点：每 task 启动有冷启延迟。
2. **Inline Execution** — 当前 session 批量执行，phase 末 checkpoint。优点：连续；缺点：长 session 可能塌。

考虑到 Plan C 涉及前端 UI 实测（每个 task 都需要起 dev 看一下），且前端代码 svelte 5 runes 还相对新，subagent 失误概率高——**我倾向 Inline Execution**，每个 phase 末跟你 checkpoint 一次。

哪一种？
