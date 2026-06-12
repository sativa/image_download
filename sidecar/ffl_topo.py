"""Topology-aware FFL (shared-edge frame-field regularization).

Diagnosis (prev run): per-instance FFL overlaps 6-14% because each parcel's shared border is
contoured+regularized independently -> the two simplified copies cross. Fix: move the frame-field
regularization onto the SHARED-EDGE topology so every border is regularized exactly ONCE.

Pipeline per cell (1m full-res, same ridge-watershed idmap):
  idmap -> rasterio.features.shapes (coverage, 0 overlap but staircase) -> gdf
  topojson.Topology(gdf, prequantize=False, shared_coords=True) -> output['arcs'] (each shared edge once)
  for each arc: CRS->px, fix endpoints, DP-simplify interior, snap segments to frame field, clamp, px->CRS
  mutate output['arcs'] in place -> to_gdf() rebuilds regularized coverage (0 overlap guaranteed).
Compare vs per-instance FFL (polygonize_ff) and shapes+coverage_simplify.
"""
import sys, time, math, json
from pathlib import Path
import numpy as np
HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))
import torch
from transformers import AutoModel
from train_dino_1m_v3 import DinoV3FreqUNetBDDF, DINOV3_SAT
from dino_parcel_eval import infer_heads
from dino_parcel_export import build_idmap, load_tif_pair
from ff_polygonize import _tiled_ff, polygonize_ff, ff_main_angle

import rasterio.features
from rasterio.transform import from_bounds
from shapely.geometry import shape
from shapely import coverage_simplify, unary_union
import geopandas as gpd
import topojson

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for fp in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(fp).exists():
        fm.fontManager.addfont(fp); plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

CKPT = "/mnt/sda/zf/landform/results/dino_v3_bddf_enh/best.pt"
TIF = "/mnt/sda/zf/landform/data/yz_full_tif"
OUT = Path("/mnt/sda/zf/landform/results"); DEV = "cuda:0"
CELLS = ["yzf_251", "yzf_400", "yzf_700"]
# zoom windows for the 4-panel close-up (row0,col0,size)
ZWIN = {"yzf_251": (760, 820, 460), "yzf_400": (900, 650, 520), "yzf_700": (700, 700, 480)}


class P:
    min_dist = 20; peak_thr = 0.4; min_area_px = 200; ridge = True; downscale = 1


# ---------- arc-level frame-field regularization (open polyline, endpoints fixed) ----------
import cv2


def _line_intersect(l1, l2):
    x1, y1, a1 = l1; x2, y2, a2 = l2
    d1 = (math.cos(a1), math.sin(a1)); d2 = (math.cos(a2), math.sin(a2))
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    t = ((x2 - x1) * d2[1] - (y2 - y1) * d2[0]) / den
    return (x1 + t * d1[0], y1 + t * d1[1])


def regularize_arc_px(coords, ffc0, ffc2, snap_deg=35.0, clamp=16.0):
    """coords: list of (col,row) px, OPEN polyline (arc). Snap each segment to local frame-field dir.
    Endpoints (coords[0], coords[-1]) are topology nodes -> kept EXACTLY (preserve shared-edge match).
    Interior vertices rebuilt as consecutive snapped-line intersections, displacement clamped."""
    n = len(coords) - 1                                            # n segments
    if n < 2:
        return coords                                             # single segment: nothing interior to move
    H, W = ffc0.shape
    lines = []
    for i in range(n):                                            # one snapped line per segment
        (x0, y0), (x1, y1) = coords[i], coords[i + 1]
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        cy = min(max(int(my), 0), H - 1); cx = min(max(int(mx), 0), W - 1)
        th = ff_main_angle(ffc0[cy, cx], ffc2[cy, cx])
        edge_ang = math.atan2(y1 - y0, x1 - x0)
        cands = [th, th + math.pi / 2]
        best = min(cands, key=lambda c: abs(((edge_ang - c + math.pi / 2) % math.pi) - math.pi / 2))
        d = abs(((edge_ang - best + math.pi / 2) % math.pi) - math.pi / 2)
        lines.append((mx, my, best if math.degrees(d) < snap_deg else edge_ang))
    out = [tuple(coords[0])]                                      # endpoint fixed
    for i in range(1, n):                                         # interior vertex i = segment(i-1) ^ segment(i)
        p = _line_intersect(lines[i - 1], lines[i])
        orig = coords[i]
        if p is None or (clamp is not None and abs(p[0] - orig[0]) + abs(p[1] - orig[1]) > clamp):
            p = orig
        out.append((float(p[0]), float(p[1])))
    out.append(tuple(coords[-1]))                                 # endpoint fixed
    return out


def topo_ffl(gdf, ffc0, ffc2, transform, simp_px=2.0, snap_deg=35.0, clamp=16.0):
    """Shared-edge frame-field regularization. Returns regularized GeoDataFrame (0 overlap)."""
    topo = topojson.Topology(gdf, prequantize=False, shared_coords=True)
    inv = ~transform                                              # CRS (lon,lat) -> px (col,row)
    arcs = topo.output["arcs"]
    new_arcs = []
    for arc in arcs:
        # arc coords are in CRS (lon,lat); map to px for frame-field lookup
        px = [inv * (x, y) for (x, y) in arc]                     # (col,row)
        # DP-simplify the open polyline interior (keep endpoints) then snap to frame field
        a = np.array(px, np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(a, simp_px, False)[:, 0, :]     # open=False keeps endpoints
        if len(approx) < 2:
            approx = np.array(px, np.float32)
        # ensure true endpoints preserved exactly
        approx = approx.tolist()
        approx[0] = list(px[0]); approx[-1] = list(px[-1])
        reg_px = regularize_arc_px([tuple(p) for p in approx], ffc0, ffc2, snap_deg, clamp)
        # snap endpoints back to original CRS node coords exactly (avoid px->CRS round-off breaking topology)
        reg_crs = [list(transform * (c, r)) for (c, r) in reg_px]
        reg_crs[0] = list(arc[0]); reg_crs[-1] = list(arc[-1])
        new_arcs.append(reg_crs)
    topo.output["arcs"] = new_arcs
    out = topo.to_gdf()
    out = out.set_crs(gdf.crs, allow_override=True)
    # repair any invalid from regularization
    out["geometry"] = [g if g.is_valid else g.buffer(0) for g in out.geometry]
    out = out[~out.geometry.is_empty].reset_index(drop=True)
    return out


# ---------- shared metrics ----------
def vert_stats(geoms):
    nv = []
    for g in geoms:
        polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
        tot = 0
        for p in polys:
            tot += len(p.exterior.coords) - 1
            for r in p.interiors:
                tot += len(r.coords) - 1
        nv.append(tot)
    return np.array(nv, float)


def to_m2(gdf, bbox):
    g = gdf.to_crs(3857)
    lat = math.radians((float(bbox[1]) + float(bbox[3])) / 2); k = math.cos(lat) ** 2
    areas = g.geometry.area * k
    uni = unary_union(g.geometry.values) if len(g) else None
    uni_a = (uni.area * k) if uni is not None and not uni.is_empty else 0.0
    return float(areas.sum()), float(uni_a)


def shapes_rows(idmap, cls_of, transform, simplify=False):
    rows = []
    for geom, val in rasterio.features.shapes(idmap, mask=idmap > 0, connectivity=8, transform=transform):
        pid = int(val); c = cls_of.get(pid)
        if not c:
            continue
        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        if g.is_empty:
            continue
        rows.append({"parcel_id": pid, "class_id": c, "geometry": g})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    if simplify and len(gdf):
        g3 = gdf.to_crs(3857)
        simp = coverage_simplify(g3.geometry.values, tolerance=3.0, simplify_boundary=True)
        g3 = g3.set_geometry(gpd.GeoSeries(simp, crs=3857))
        g3["geometry"] = [x if x.is_valid else x.buffer(0) for x in g3.geometry]
        gdf = g3.to_crs(4326)
    return gdf


def metrics(name, gdf, bbox, foot_m2):
    if len(gdf) == 0:
        return {"n": 0}
    nv = vert_stats(list(gdf.geometry))
    s, u = to_m2(gdf, bbox)
    return {"n": int(len(gdf)), "vert_mean": round(float(nv.mean()), 1),
            "vert_median": float(np.median(nv)), "vert_max": int(nv.max()),
            "overlap_pct": round(100 * (s - u) / (u + 1e-9), 2),
            "cover_pct": round(100 * u / (foot_m2 + 1e-9), 1)}


def main():
    t0 = time.time()
    d3 = AutoModel.from_pretrained(DINOV3_SAT, local_files_only=True)
    m = DinoV3FreqUNetBDDF(d3, num_classes=9, in_channels=11, unfreeze_last_n=4).to(DEV)
    sd = torch.load(CKPT, map_location=DEV, weights_only=True); msd = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    m.eval()
    print(f"[load] {CKPT} ({time.time()-t0:.0f}s)", flush=True)

    table = []
    for cell in CELLS:
        tc = time.time()
        x6, bbox = load_tif_pair(TIF, cell); _, H, W = x6.shape
        clsprob, dist, bnd = infer_heads(m, x6, DEV, cs=448, enhance=True)
        ffc0, ffc2 = _tiled_ff(m, x6, DEV, cs=448)
        idmap, cls_of = build_idmap(clsprob, dist, bnd, P())
        tr = from_bounds(*[float(b) for b in bbox], W, H)
        pix_m2 = ((float(bbox[2]) - float(bbox[0])) * 111320 * math.cos(math.radians((float(bbox[1]) + float(bbox[3])) / 2)) / W) * \
                 ((float(bbox[3]) - float(bbox[1])) * 110540 / H)
        foot_m2 = int((idmap > 0).sum()) * pix_m2

        g_sh = shapes_rows(idmap, cls_of, tr, False)
        g_cs = shapes_rows(idmap, cls_of, tr, True)
        g_ff = gpd.GeoDataFrame(polygonize_ff(idmap, cls_of, ffc0, ffc2, tr, simp_px=2.0, snap_deg=35.0),
                                geometry="geometry", crs="EPSG:4326")
        # topology-aware FFL — try clamp 8/16/unlimited
        topo_variants = {}
        for clamp in [8.0, 16.0, None]:
            tt = time.time()
            g_tf = topo_ffl(g_sh, ffc0, ffc2, tr, simp_px=2.0, snap_deg=35.0, clamp=clamp)
            topo_variants[clamp] = (g_tf, metrics(f"topoFFL_c{clamp}", g_tf, bbox, foot_m2), time.time() - tt)

        res = {"cell": cell, "foot_ha": round(foot_m2 / 1e4, 1),
               "shapes": metrics("shapes", g_sh, bbox, foot_m2),
               "cov_simplify": metrics("cov_simplify", g_cs, bbox, foot_m2),
               "FFL_perinstance": metrics("FFL", g_ff, bbox, foot_m2)}
        for clamp, (g_tf, mt, dt) in topo_variants.items():
            res[f"topoFFL_clamp{clamp}"] = {**mt, "build_s": round(dt, 1)}
        table.append(res)

        print(f"\n[{cell}] foot {foot_m2/1e4:.1f} ha ({time.time()-tc:.0f}s)", flush=True)
        for nm in ["shapes", "cov_simplify", "FFL_perinstance", "topoFFL_clamp8.0", "topoFFL_clamp16.0", "topoFFL_clampNone"]:
            r = res.get(nm, {})
            if r.get("n"):
                print(f"   {nm:20s} n={r['n']:4d} vmean={r['vert_mean']:6.1f} vmed={r['vert_median']:5.1f} "
                      f"overlap={r['overlap_pct']:6.2f}% cover={r['cover_pct']:5.1f}%"
                      + (f" build={r.get('build_s')}s" if 'build_s' in r else ""), flush=True)

        # ---- zoom 4-panel: RGB | FFL-perinstance | cov_simplify | topoFFL(clamp16) ----
        g_topo16 = topo_variants[16.0][0]
        r0, c0, sz = ZWIN[cell]; r1, c1 = min(r0 + sz, H), min(c0 + sz, W)
        def px2lon(c): return float(bbox[0]) + (float(bbox[2]) - float(bbox[0])) * c / W
        def px2lat(r): return float(bbox[3]) - (float(bbox[3]) - float(bbox[1])) * r / H
        ext = [px2lon(c0), px2lon(c1), px2lat(r1), px2lat(r0)]
        rgb = np.ascontiguousarray(x6[:3, r0:r1, c0:c1].transpose(1, 2, 0)).astype(np.uint8)
        fig, axes = plt.subplots(1, 4, figsize=(24, 6.4))
        titles = [f"{cell} crop RGB(1m)",
                  f"FFL逐地块 overlap={res['FFL_perinstance']['overlap_pct']}%",
                  f"shapes+cov_simplify overlap={res['cov_simplify']['overlap_pct']}%",
                  f"拓扑FFL(shared-edge) overlap={res['topoFFL_clamp16.0']['overlap_pct']}% vμ={res['topoFFL_clamp16.0']['vert_mean']}"]
        for ax, t in zip(axes, titles):
            ax.imshow(rgb, extent=ext, origin="upper"); ax.set_title(t, fontsize=13)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([])
        for ax, gdf in zip(axes[1:], [g_ff, g_cs, g_topo16]):
            if len(gdf):
                gdf.boundary.plot(ax=ax, color="yellow", linewidth=1.4)
        plt.tight_layout(); pp = OUT / f"ffl_topo_zoom_{cell}.png"
        fig.savefig(pp, dpi=170, bbox_inches="tight"); plt.close(fig)
        print(f"   zoom -> {pp}", flush=True)

        # full-cell topoFFL viz too
        fig2, ax2 = plt.subplots(1, 1, figsize=(11, 11))
        rgbf = np.ascontiguousarray(x6[:3].transpose(1, 2, 0)).astype(np.uint8)
        extf = [float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3])]
        ax2.imshow(rgbf, extent=extf, origin="upper")
        g_topo16.boundary.plot(ax=ax2, color="yellow", linewidth=0.5)
        ax2.set_title(f"{cell} 拓扑FFL n={len(g_topo16)} overlap={res['topoFFL_clamp16.0']['overlap_pct']}%", fontsize=13)
        ax2.set_xticks([]); ax2.set_yticks([])
        plt.tight_layout(); pp2 = OUT / f"ffl_topo_full_{cell}.png"
        fig2.savefig(pp2, dpi=160, bbox_inches="tight"); plt.close(fig2)
        print(f"   full -> {pp2}", flush=True)

    (OUT / "ffl_topo_metrics.json").write_text(json.dumps(table, indent=2, ensure_ascii=False))
    print(f"\n[done] -> {OUT/'ffl_topo_metrics.json'} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
