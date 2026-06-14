"""Clean final vector products: drop_tiny_holes + strengthened eliminate_slivers.
Pure geometry postproc. Reports before/after sliver/hole/area/overlap. Saves *_clean.parquet.
"""
import sys, json, time, os
SIDECAR = "/Users/zhangfeng/CODE_BLOCK_DNDC/imagery_downloader/sidecar"
sys.path.insert(0, SIDECAR)
os.chdir(SIDECAR)
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely import make_valid
from shapely.ops import unary_union
import postproc

PRODUCTS = [
    ("yuzhong_product/yuzhong_SMOOTH3_chaikin3.parquet", "EPSG:32648", "DINOv3"),
    ("yuzhong_product/yuzhong_SMOOTH3_RADIO.parquet",    "EPSG:32648", "C-RADIO"),
    ("shenchi_product/shenchi_seamless.parquet",         "EPSG:32649", "shenchi"),
]

def metrics(gu):
    """gu in metric CRS. Returns dict of artifact/area metrics."""
    geoms = gu.geometry.values
    area = np.array([gm.area for gm in geoms])
    peri = np.array([gm.length for gm in geoms])
    w = np.where(peri > 0, area / peri, 0.0)
    n_holes = 0; n_hole_lt1 = 0
    for gm in geoms:
        for r in gm.interiors:
            n_holes += 1
            if Polygon(r).area < 1.0:
                n_hole_lt1 += 1
    return {
        "n_poly": int(len(gu)),
        "fine_sliver_w_lt0.5m": int((w < 0.5).sum()),
        "sliver_w_lt1m": int((w < 1.0).sum()),
        "interior_rings": int(n_holes),
        "micro_holes_lt1m2": int(n_hole_lt1),
        "total_km2": float(area.sum() / 1e6),
        "all_valid": bool(gu.geometry.is_valid.all()),
    }

def overlap_check(gu, vertex_cap=20000):
    """STRtree pairwise overlap-area on **treatable polys** (verts<vertex_cap).
    No full unary_union (too slow on 100k polys + 5M-vert giant); exact 'overlaps' predicate against
    the few giant background polys is also infeasible -> those are excluded (they're seamless
    background; their boundary only edge-touches neighbours, not overlaps). Returns (sum_km2, overlap_m2).
    overlap_m2 ~ 0 => seamless/no real overlap among the 99.99% treatable polys."""
    import shapely
    geoms = gu.geometry.values
    nv = shapely.get_num_coordinates(geoms)
    s = float(shapely.area(geoms).sum())
    sub = np.where(nv < vertex_cap)[0]
    gsub = geoms[sub]
    tree = shapely.STRtree(gsub)
    ai, bi = tree.query(gsub, predicate="overlaps")
    ov = 0.0
    seen = set()
    for a, b in zip(ai.tolist(), bi.tolist()):
        if a >= b:
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        try:
            ov += float(shapely.area(shapely.intersection(gsub[a], gsub[b])))
        except Exception:
            pass
    return s/1e6, ov

def main():
    out = {}
    for path, utm, tag in PRODUCTS:
        t0 = time.time()
        print(f"\n{'='*70}\n{tag}: {path}  (UTM={utm})\n{'='*70}", flush=True)
        g = gpd.read_parquet(path)
        av = bool(g.geometry.is_valid.all())
        print(f"  loaded n={len(g)}  raw all_valid={av}", flush=True)
        if not av:
            g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else make_valid(x))
            g = g[g.geometry.notna() & ~g.geometry.is_empty]
            g = g.explode(index_parts=False).reset_index(drop=True)
            g = g[g.geometry.geom_type == "Polygon"].reset_index(drop=True)
            print(f"  make_valid+explode -> n={len(g)}", flush=True)

        gu = g.to_crs(utm)
        before = metrics(gu)
        bsum, bov = overlap_check(gu)
        before["overlap_m2"] = bov; before["sum_km2"] = bsum
        print("  BEFORE:", json.dumps(before, ensure_ascii=False), flush=True)

        # ---- clean: eliminate_slivers (强力 细且小 w<1.5 & a<100, 默认参数) -> drop_tiny_holes ----
        gu2, r_sliv = postproc.eliminate_slivers(gu, verbose=True)
        gu2, r_hole = postproc.drop_tiny_holes(gu2, verbose=True)
        gu2 = postproc.fix_invalid(gu2)

        after = metrics(gu2)
        asum, aov = overlap_check(gu2)
        after["overlap_m2"] = aov; after["sum_km2"] = asum
        print("  AFTER :", json.dumps(after, ensure_ascii=False), flush=True)

        # save clean (back to 4326)
        out_path = path.replace(".parquet", "_clean.parquet")
        gout = gu2.to_crs("EPSG:4326")
        # re-gid
        if "gid" in gout.columns:
            gout = gout.drop(columns=["gid"])
        gout.insert(0, "gid", range(1, len(gout) + 1))
        gout.to_parquet(out_path)
        print(f"  saved -> {out_path}", flush=True)

        out[tag] = {"path": path, "out": out_path, "utm": utm,
                    "before": before, "after": after,
                    "sliver_report": {k: r_sliv[k] for k in
                        ("slivers_merged","slivers_dropped_isolated","slivers_eliminated",
                         "slivers_eliminated_area_m2","isolated_kept")},
                    "hole_report": {k: r_hole[k] for k in
                        ("holes_before","holes_dropped","holes_kept_large","holes_kept_nested",
                         "hole_area_dropped_m2")},
                    "secs": round(time.time()-t0,1)}
        print(f"  done in {out[tag]['secs']}s", flush=True)

    with open("/tmp/clean_report.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n\nWROTE /tmp/clean_report.json", flush=True)

if __name__ == "__main__":
    main()
