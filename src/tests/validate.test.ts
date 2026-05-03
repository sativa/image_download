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
