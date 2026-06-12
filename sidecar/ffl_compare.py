"""FFL vs shapes vs shapes+coverage_simplify — 田块级 1m 全分辨率边界质量对比.

Same idmap (ridge watershed, downscale=1) on each cell, vectorized three ways:
  A0  shapes()                          (当前法, 阶梯/波浪折线)
  A1  shapes() + coverage_simplify      (拓扑保持简化, 去阶梯, 无重叠)
  B   polygonize_ff (FFL 帧场)          (帧场 snap, 8px clamp)
Per method per cell: parcel count, vertices/parcel mean+median, area sum vs union
(overlap ratio -> FFL overflow check), coverage of idmap footprint.
4-panel viz (RGB | shapes黄边 | cov_simplify黄边 | FFL黄边) at dpi 170.
"""
import sys, time, math, json
from pathlib import Path
import numpy as np

HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))
import torch
from transformers import AutoModel
from train_dino_1m_v3 import DinoV3FreqUNetBDDF, DINOV3_SAT
from dino_parcel_eval import infer_heads
from dino_parcel_export import build_idmap, load_tif_pair, NAME_ZH, RGB
from ff_polygonize import _tiled_ff, polygonize_ff

import rasterio.features
from rasterio.transform import from_bounds
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely import coverage_simplify, unary_union
import geopandas as gpd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# CJK font
import matplotlib.font_manager as fm
for fp in ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

CKPT = "/mnt/sda/zf/landform/results/dino_v3_bddf_enh/best.pt"
TIF_DIR = "/mnt/sda/zf/landform/data/yz_full_tif"
OUT = Path("/mnt/sda/zf/landform/results")
DEV = "cuda:0"
CELLS = ["yzf_251", "yzf_400", "yzf_700"]


class P:  # build_idmap params — 田块级 ridge watershed, 全分辨率
    min_dist = 20; peak_thr = 0.4; min_area_px = 200; ridge = True; downscale = 1


def vert_stats(geoms):
    """vertices per ring (exterior + interiors) per geometry."""
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


def area_m2_factor(bbox, W, H):
    lat = (float(bbox[1]) + float(bbox[3])) / 2
    mx = (float(bbox[2]) - float(bbox[0])) * 111320 * math.cos(math.radians(lat)) / W
    my = (float(bbox[3]) - float(bbox[1])) * 110540 / H
    return mx * my  # m^2 per (deg-area unit)? -> we convert via projected; simpler: compute area in m using equal-area below


def to_m2(gdf, bbox):
    """sum area & union area in m^2 via local UTM-ish equal area (CRS 4326 -> 3857 approx ok at cell scale)."""
    g = gdf.to_crs(3857)
    # 3857 area is distorted by sec(lat)^2; correct
    lat = math.radians((float(bbox[1]) + float(bbox[3])) / 2)
    k = math.cos(lat) ** 2
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
        # coverage_simplify: topology-preserving, no gaps/overlaps. tolerance in CRS units.
        # work in 3857 metres so tolerance is physical; ~4.8m grid -> use 3.0 m
        g3 = gdf.to_crs(3857)
        simp = coverage_simplify(g3.geometry.values, tolerance=3.0, simplify_boundary=True)
        g3 = g3.set_geometry(gpd.GeoSeries(simp, crs=3857))
        # fix any invalid
        g3["geometry"] = [x if x.is_valid else x.buffer(0) for x in g3.geometry]
        gdf = g3.to_crs(4326)
    return gdf


def main():
    t0 = time.time()
    d3 = AutoModel.from_pretrained(DINOV3_SAT, local_files_only=True)
    m = DinoV3FreqUNetBDDF(d3, num_classes=9, in_channels=11, unfreeze_last_n=4).to(DEV)
    sd = torch.load(CKPT, map_location=DEV, weights_only=True); msd = m.state_dict()
    loaded = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    m.load_state_dict(loaded, strict=False)
    m.eval()
    has_ff = any(k.startswith("frame_field_head") for k in loaded)
    print(f"[load] {CKPT} | matched {len(loaded)}/{len(msd)} | frame_field_head present in ckpt: {has_ff} ({time.time()-t0:.0f}s)", flush=True)

    table = []
    for cell in CELLS:
        tc = time.time()
        x6, bbox = load_tif_pair(TIF_DIR, cell)
        _, H, W = x6.shape
        clsprob, dist, bnd = infer_heads(m, x6, DEV, cs=448, enhance=True)
        ffc0, ffc2 = _tiled_ff(m, x6, DEV, cs=448)
        idmap, cls_of = build_idmap(clsprob, dist, bnd, P())
        tr = from_bounds(*[float(b) for b in bbox], W, H)
        foot_px = int((idmap > 0).sum())  # idmap footprint
        pix_m2 = ((float(bbox[2]) - float(bbox[0])) * 111320 * math.cos(math.radians((float(bbox[1]) + float(bbox[3])) / 2)) / W) * \
                 ((float(bbox[3]) - float(bbox[1])) * 110540 / H)
        foot_m2 = foot_px * pix_m2

        # --- A0 shapes ---
        g_sh = shapes_rows(idmap, cls_of, tr, simplify=False)
        # --- A1 shapes + coverage_simplify ---
        g_cs = shapes_rows(idmap, cls_of, tr, simplify=True)
        # --- B FFL ---
        ff_rows = polygonize_ff(idmap, cls_of, ffc0, ffc2, tr, simp_px=2.0, snap_deg=35.0)
        g_ff = gpd.GeoDataFrame(ff_rows, geometry="geometry", crs="EPSG:4326")

        res = {"cell": cell, "H": H, "W": W, "idmap_inst": int(len(np.unique(idmap)) - 1), "foot_m2": foot_m2}
        for name, gdf in [("shapes", g_sh), ("cov_simplify", g_cs), ("FFL", g_ff)]:
            if len(gdf) == 0:
                res[name] = {"n": 0}; continue
            nv = vert_stats(list(gdf.geometry))
            s_m2, u_m2 = to_m2(gdf, bbox)
            res[name] = {
                "n": int(len(gdf)),
                "vert_mean": round(float(nv.mean()), 1),
                "vert_median": float(np.median(nv)),
                "vert_max": int(nv.max()),
                "area_sum_m2": round(s_m2, 0),
                "area_union_m2": round(u_m2, 0),
                "overlap_pct": round(100 * (s_m2 - u_m2) / (u_m2 + 1e-9), 2),  # sum/union excess = overlap
                "area_vs_foot_pct": round(100 * u_m2 / (foot_m2 + 1e-9), 1),    # coverage of idmap footprint
            }
        table.append(res)
        print(f"[{cell}] {res['idmap_inst']} idmap inst, foot {foot_m2/1e4:.1f} ha ({time.time()-tc:.0f}s)", flush=True)
        for nm in ["shapes", "cov_simplify", "FFL"]:
            r = res[nm]
            if r.get("n"):
                print(f"   {nm:13s} n={r['n']:4d} vert mean={r['vert_mean']:6.1f} med={r['vert_median']:5.1f} max={r['vert_max']:5d} | "
                      f"sum={r['area_sum_m2']/1e4:7.1f}ha union={r['area_union_m2']/1e4:7.1f}ha overlap={r['overlap_pct']:6.2f}% cover={r['area_vs_foot_pct']:5.1f}%", flush=True)

        # ---- viz: 4 panel RGB | shapes | cov_simplify | FFL (yellow edges on RGB) ----
        rgb = np.ascontiguousarray(x6[:3].transpose(1, 2, 0)).astype(np.uint8)
        fig, axes = plt.subplots(1, 4, figsize=(26, 7.0))
        ext = [float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3])]
        titles = [f"{cell} RGB(1m)",
                  f"shapes()  n={res['shapes'].get('n',0)} vert μ={res['shapes'].get('vert_mean','-')}",
                  f"shapes+cov_simplify  n={res['cov_simplify'].get('n',0)} vert μ={res['cov_simplify'].get('vert_mean','-')}",
                  f"FFL frame-field  n={res['FFL'].get('n',0)} vert μ={res['FFL'].get('vert_mean','-')} overlap={res['FFL'].get('overlap_pct','-')}%"]
        for ax, t in zip(axes, titles):
            ax.imshow(rgb, extent=ext, origin="upper")
            ax.set_title(t, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
        for ax, gdf in zip(axes[1:], [g_sh, g_cs, g_ff]):
            if len(gdf):
                gdf.boundary.plot(ax=ax, color="yellow", linewidth=0.6)
        plt.tight_layout()
        pp = OUT / f"ffl_cmp_{cell}.png"
        fig.savefig(pp, dpi=170, bbox_inches="tight"); plt.close(fig)
        print(f"   -> {pp}", flush=True)

    (OUT / "ffl_compare_metrics.json").write_text(json.dumps(table, indent=2, ensure_ascii=False))
    print(f"\n[done] metrics -> {OUT/'ffl_compare_metrics.json'} ({time.time()-t0:.0f}s)", flush=True)
    # compact summary table
    print("\n=== SUMMARY (vert mean / overlap% / coverage%) ===", flush=True)
    print(f"{'cell':10s} {'method':14s} {'n':>5s} {'vmean':>7s} {'vmed':>6s} {'overlap%':>9s} {'cover%':>7s}", flush=True)
    for r in table:
        for nm in ["shapes", "cov_simplify", "FFL"]:
            x = r[nm]
            if x.get("n"):
                print(f"{r['cell']:10s} {nm:14s} {x['n']:>5d} {x['vert_mean']:>7.1f} {x['vert_median']:>6.1f} {x['overlap_pct']:>9.2f} {x['area_vs_foot_pct']:>7.1f}", flush=True)


if __name__ == "__main__":
    main()
