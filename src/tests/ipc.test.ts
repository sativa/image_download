import { describe, it, expect, vi, beforeEach } from "vitest";

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
