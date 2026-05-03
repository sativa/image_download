# `src-tauri/src/mocks/` — Plan B replacement contract

This directory contains **mock Tauri command handlers** that drive the
frontend UI shipped by Plan C. The frontend code in `src/` is _final_;
only this directory will be replaced when Plan A/B land.

## What gets replaced

When Plan B implements real backend commands, this is the patch shape:

1. Move `history_commands.rs` content into `src-tauri/src/commands/history.rs`
   (no logic change — just relocation + swap `mod mocks` references).
2. Delete `mod.rs`, `commands.rs`, `runner.rs`. The History store is
   already in `src-tauri/src/history.rs` and survives.
3. Add real implementations in `src-tauri/src/commands/{download.rs, vector.rs}` that
   call into `core::*` modules (`tiles`, `sources`, `downloader`, `stitcher`, `cog`, `vector`).
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

Argument shapes are defined in `src/lib/types.ts` and serialize to the
matching Rust structs via serde. Any Plan B change to the wire format MUST
be accompanied by a TypeScript types update in the same commit.

## Why this exists

- Plan C wanted the UI working end-to-end before any real backend is
  implemented, so users can validate UX early.
- Without mocks, the UI couldn't be tested for throttling, ETA math,
  cancel responsiveness, or history persistence.
- The mock progress emitter ticks at 10 Hz over 5 s — high enough that
  real-world UI throttling code (4 Hz target per spec §3.2) gets exercised.

## What is NOT mocked (Plan A territory)

- `parse_vector_file`: always returns "vector parsing pending Plan A".
  Plan A's `vector` module will parse `.geojson` / `.shp` / `.gpkg`.
- Real tile fetching: `start_download` simulates progress via timers but
  does not call any HTTP API.
- COG GeoTIFF writing: `start_download` claims success after a fake
  `writing_cog` stage; no file is written. Plan A's `cog` module handles
  the real write.
- `auto` source selection: `estimate_output` accepts but ignores the
  source string. Plan A's `sources` module probes ESRI vs Google by
  latency.
