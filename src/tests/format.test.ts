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
