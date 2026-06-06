"""Object-level completeness: of the DLTB cropland (耕地+园地) polygons over the 120
cross-county test cells, what fraction does the current route-c ensemble MISS?
Per cropland polygon (clipped to its cell, ≥400 m2): recall = predicted-cropland pixels
inside it / its pixels. 'Missed' at recall<{0.1,0.3,0.5}. Reports count-% and area-weighted-%.
Full-cell inference (no crop) so every polygon in the cell is scored. Run pinned to a free GPU."""
import sys, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import geopandas as gpd
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import box as shp_box

HOME = Path("/home/ps/landform")
LF = Path("/mnt/sda/zf/landform")
sys.path.insert(0, str(HOME / "sidecar"))
import train_c_stage2 as T  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402

DEV = "cuda:0"  # physical GPU set via CUDA_VISIBLE_DEVICES
DLBM_TO_CLASS = {"01": 1, "02": 2, "03": 3, "04": 4, "05": 5, "06": 5,
                 "07": 5, "08": 5, "09": 5, "10": 5, "11": 5, "12": 5}
ARCH = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus, "unetplusplus": smp.UnetPlusPlus}
THRS_MISS = [0.1, 0.3, 0.5]


def load_model(d):
    name = Path(d).name
    arch = "deeplabv3plus" if "_dl" in name else ("unetplusplus" if "_pp" in name else "unet")
    m = ARCH[arch](encoder_name="efficientnet-b5", encoder_weights=None, in_channels=11, classes=3).to(DEV)
    m.load_state_dict(torch.load(Path(d) / "best.pt", map_location=DEV, weights_only=True))
    m.eval()
    return m


@torch.no_grad()
def cell_prob(models, cell):
    rgbnir = cell["rgbnir"].astype(np.float32).copy()
    for b in range(4):
        rgbnir[b] = (rgbnir[b] - T.S2_MEAN[b]) / T.S2_STD[b]
    ndvi_s2 = (cell["ndvi_s2"].astype(np.float32) - T.NDVI_MEAN) / T.NDVI_STD
    ndvi_yr = (cell["ndvi_years"].astype(np.float32) - T.NDVI_MEAN) / T.NDVI_STD
    x = np.concatenate([rgbnir, ndvi_s2[None], ndvi_yr, cell["feat2"].astype(np.float32)], 0)
    _, H, W = x.shape
    xb = torch.from_numpy(x).unsqueeze(0).to(DEV)
    ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
    xb = F.pad(xb, (0, pw, 0, ph), mode="reflect")
    acc = None
    for m in models:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pr = torch.softmax(m(xb).float(), 1)[:, 1]
        acc = pr if acc is None else acc + pr
    return (acc / len(models))[0, :H, :W].cpu().numpy()


_DCACHE = {}


def county_cropland(county):
    if county not in _DCACHE:
        g = gpd.read_parquet(HOME / "data/v11_dltb" / f"{county}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs(4326)
        try:
            g["geometry"] = g.geometry.make_valid()
        except Exception:
            g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        _DCACHE[county] = g[g["cid"].isin([1, 2])].copy()  # cropland = 耕地+园地
    return _DCACHE[county]


def analyze(models, cells, tag):
    tot = 0; area_tot = 0
    missed = {t: 0 for t in THRS_MISS}; area_missed = {t: 0 for t in THRS_MISS}
    for cell in cells:
        name = cell["name"]; county = name.split("_")[0]
        pred = cell_prob(models, cell) >= 0.5
        Hs, Ws = cell["label"].shape
        s2 = np.load(HOME / "data/v19_s2_raw" / f"{name}.npz")
        bbox = tuple(float(v) for v in s2["bbox"])
        tr = Affine(*np.asarray(s2["transform"]).flatten()[:6])
        g = county_cropland(county)
        for gi in g.iloc[list(g.sindex.intersection(bbox))].geometry:
            piece = gi.intersection(shp_box(*bbox))
            if piece.is_empty:
                continue
            pm = rasterize([(piece, 1)], out_shape=(Hs, Ws), transform=tr, fill=0, dtype="uint8").astype(bool)
            a = int(pm.sum())
            if a < 4:
                continue
            rec = int((pred & pm).sum()) / a
            tot += 1; area_tot += a
            for t in THRS_MISS:
                if rec < t:
                    missed[t] += 1; area_missed[t] += a
    print(f"\n=== 耕地多边形漏检率 [{tag}] ({len(models)}-model, argmax, {len(cells)} cells, {tot} polys≥400m2) ===", flush=True)
    for t in THRS_MISS:
        print(f"  recall<{t}: 个数 {100*missed[t]/max(tot,1):.1f}%  面积 {100*area_missed[t]/max(area_tot,1):.1f}%", flush=True)
    return {"n_polys": tot,
            "miss_count_pct": {str(t): 100 * missed[t] / max(tot, 1) for t in THRS_MISS},
            "miss_area_pct": {str(t): 100 * area_missed[t] / max(area_tot, 1) for t in THRS_MISS}}


def main():
    import copy
    R = json.loads((HOME / "data/v40_5k.json").read_text())
    te, _ = T.parallel_loadsplit_multitemp(R["test"], HOME / "data/v11_dltb", HOME / "data/v19_s2_raw",
                                           HOME / "data/v33_ndvi_multitemporal", T.EXTRA_YEARS, max_workers=8)
    T.attach_feat(te, str(LF / "data/c_stage1_feat"), True)
    te_zero = copy.deepcopy(te)
    for c in te_zero:
        c["feat2"] = np.zeros_like(c["feat2"])

    def present(dirs):
        return [str(LF / "results" / d) for d in dirs if (LF / "results" / d / "best.pt").exists()]

    rc = [load_model(d) for d in present(["c_stage2_1m", "ens_rc_s1", "ens_rc_s2", "ens_rc_dl", "ens_rc_pp"])]
    bl = [load_model(d) for d in present(["c_stage2_base", "ens_bl_s1", "ens_bl_s2", "ens_bl_dl", "ens_bl_pp"])]
    rrc = analyze(rc, te, "route-c 10m+1m")
    rbl = analyze(bl, te_zero, "baseline 10m-only")
    print("\n>>> 1m 对小地块的救回 (漏检率下降，正=变好):", flush=True)
    for t in THRS_MISS:
        dc = rbl["miss_count_pct"][str(t)] - rrc["miss_count_pct"][str(t)]
        da = rbl["miss_area_pct"][str(t)] - rrc["miss_area_pct"][str(t)]
        print(f"  recall<{t}: 个数漏检 {rbl['miss_count_pct'][str(t)]:.1f}%→{rrc['miss_count_pct'][str(t)]:.1f}% (Δ{dc:+.1f})  "
              f"面积 {rbl['miss_area_pct'][str(t)]:.1f}%→{rrc['miss_area_pct'][str(t)]:.1f}% (Δ{da:+.1f})", flush=True)
    (LF / "results/polygon_miss.json").write_text(json.dumps({"route_c": rrc, "baseline": rbl}, indent=2))


if __name__ == "__main__":
    main()
