"""Frame-field-guided polygonization (Girard et al. CVPR'21, simplified edge-regularization variant).

Instead of marching-squares + Chaikin (which leaves staircase / waviness), this snaps each polygon edge
to the LOCAL frame-field direction learned from DLTB edges: per edge, read the frame field at its
midpoint, get the two orthogonal main directions {θ, θ+90°}, rotate the edge to the nearest one (only if
within a threshold), then rebuild vertices as consecutive edge-line intersections -> regular, wave-free,
right-angle-respecting parcels. Test entry: load DinoV3FreqUNetBDDF, run a cell, compare vs plain.
"""
import argparse, json, sys, time, math
from pathlib import Path

import numpy as np
import cv2

HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))


def ff_main_angle(c0, c2):
    """Frame-field polynomial f(z)=z^4 + c2 z^2 + c0; return main edge direction θ (rad, mod π)."""
    c0 = np.asarray(c0, np.complex128); c2 = np.asarray(c2, np.complex128)
    disc = np.sqrt(c2 * c2 - 4 * c0)
    z2 = (-c2 + disc) / 2.0                                        # one root of the quadratic in z²
    return np.angle(z2) / 2.0                                      # û direction (the other is +90°)


def _line_intersect(l1, l2):
    x1, y1, a1 = l1; x2, y2, a2 = l2
    d1 = (math.cos(a1), math.sin(a1)); d2 = (math.cos(a2), math.sin(a2))
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None                                               # parallel -> keep original vertex
    t = ((x2 - x1) * d2[1] - (y2 - y1) * d2[0]) / den
    return (x1 + t * d1[0], y1 + t * d1[1])


def regularize_ring(coords, ffc0, ffc2, snap_deg=35.0):
    """coords: list of (col,row) px (closed). Snap each edge to the local frame-field direction."""
    n = len(coords) - 1
    if n < 3:
        return coords
    H, W = ffc0.shape
    lines = []
    for i in range(n):
        (x0, y0), (x1, y1) = coords[i], coords[i + 1]
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        cy = min(max(int(my), 0), H - 1); cx = min(max(int(mx), 0), W - 1)
        th = ff_main_angle(ffc0[cy, cx], ffc2[cy, cx])
        edge_ang = math.atan2(y1 - y0, x1 - x0)
        cands = [th, th + math.pi / 2]
        best = min(cands, key=lambda c: abs(((edge_ang - c + math.pi / 2) % math.pi) - math.pi / 2))
        d = abs(((edge_ang - best + math.pi / 2) % math.pi) - math.pi / 2)
        lines.append((mx, my, best if math.degrees(d) < snap_deg else edge_ang))   # snap only if close
    pts = []
    for i in range(n):
        p = _line_intersect(lines[i], lines[(i + 1) % n])
        pts.append(p if p is not None else coords[i + 1])
    pts.append(pts[0])
    return pts


def polygonize_ff(idmap, cls_of, ffc0, ffc2, transform, simp_px=2.0, snap_deg=35.0):
    """Per-instance: mask -> contour -> Douglas-Peucker -> frame-field edge regularization -> CRS polygon."""
    import rasterio.features  # noqa
    from shapely.geometry import Polygon, shape  # noqa
    rows = []
    ids = np.unique(idmap); ids = ids[ids > 0]
    for pid in ids:
        c = cls_of.get(int(pid))
        if not c:
            continue
        m = (idmap == pid).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 16:
            continue
        approx = cv2.approxPolyDP(cnt, simp_px, True)[:, 0, :]     # (K,2) col,row — drop staircase
        if len(approx) < 3:
            continue
        ring = [tuple(p) for p in approx] + [tuple(approx[0])]
        reg = regularize_ring(ring, ffc0, ffc2, snap_deg)          # snap edges to frame field
        world = [transform * (x, y) for x, y in reg]               # px -> CRS
        try:
            g = Polygon(world)
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty or g.geom_type != "Polygon":
                continue
        except Exception:
            continue
        rows.append({"parcel_id": int(pid), "class_id": c, "geometry": g})
    return rows


def main():
    import torch
    from transformers import AutoModel
    from train_dino_1m_v3 import DinoV3FreqUNetBDDF, DINOV3_SAT
    from train_dino_1m import norm6
    from dino_parcel_eval import infer_heads
    from dino_parcel_export import build_idmap, NAME_ZH, HEX
    import geopandas as gpd
    from rasterio.transform import from_bounds

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_v3_ff/last.pt")
    ap.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m_lc")
    ap.add_argument("--cell", default="620724_399")
    ap.add_argument("--snap-deg", type=float, default=35.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/mnt/sda/zf/landform/results/ff_poly")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d3 = AutoModel.from_pretrained(DINOV3_SAT, local_files_only=True)
    m = DinoV3FreqUNetBDDF(d3, num_classes=9, in_channels=11, unfreeze_last_n=4).to(a.device)
    sd = torch.load(a.ckpt, map_location=a.device, weights_only=True); msd = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    m.eval()
    z = np.load(Path(a.data_dir) / f"{a.cell}.npz"); x6 = z["x6"]; bbox = z["bbox"]; _, H, W = x6.shape
    # infer cls/dist (for idmap) + frame field
    clsprob, dist, _ = infer_heads(m, x6, a.device)               # (cls, dist, bnd)
    ff = _tiled_ff(m, x6, a.device)                               # frame-field raster (c0, c2)
    class _P:
        min_dist = 20; peak_thr = 0.4; min_area_px = 200; ridge = False; downscale = 4 if max(H, W) > 5000 else 1
    idmap, cls_of = build_idmap(clsprob, dist, np.zeros((H, W), np.float32), _P())
    tr = from_bounds(*[float(b) for b in bbox], W, H)
    rows = polygonize_ff(idmap, cls_of, ff[0], ff[1], tr, snap_deg=a.snap_deg)
    for r in rows:
        r["label"] = NAME_ZH[r["class_id"]]; r["rgb_hex"] = HEX[r["class_id"]]
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    gdf.to_parquet(out / f"{a.cell}_ff.parquet")
    print(f"[ff-poly] {a.cell}: {len(rows)} parcels (frame-field regularized) -> {a.cell}_ff.parquet", flush=True)
    nv = gdf.geometry.apply(lambda x: len(x.exterior.coords))
    print(f"  vertices/parcel mean {nv.mean():.1f} max {int(nv.max())} (低=规则无波浪)", flush=True)


def _tiled_ff(model, x6, dev, cs=448):
    """Tiled frame-field raster (c0_re,c0_im,c2_re,c2_im) -> returns (c0 complex, c2 complex) HxW."""
    import torch, torch.nn.functional as F
    from train_dino_1m import norm6
    _, H, W = x6.shape; ndvi = np.zeros((5, H, W), np.float32)
    acc = np.zeros((4, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    st = cs // 2
    ys = list(range(0, max(1, H - cs + 1), st)); xs = list(range(0, max(1, W - cs + 1), st))
    if ys[-1] != H - cs: ys.append(max(0, H - cs))
    if xs[-1] != W - cs: xs.append(max(0, W - cs))
    hw = np.hanning(cs); win = np.maximum(np.outer(hw, hw), 1e-3).astype(np.float32)
    for t in ys:
        for l in xs:
            xc = np.concatenate([norm6(x6[:, t:t + cs, l:l + cs]), ndvi[:, t:t + cs, l:l + cs]], 0)
            xb = torch.from_numpy(xc).unsqueeze(0).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ff = model(xb)[3]
                if ff.shape[-2:] != (cs, cs):
                    ff = F.interpolate(ff, size=(cs, cs), mode="bilinear", align_corners=False)
                ff = ff.float()[0].cpu().numpy()
            acc[:, t:t + cs, l:l + cs] += ff * win; cnt[t:t + cs, l:l + cs] += win
    cnt = np.maximum(cnt, 1e-6); acc /= cnt
    c0 = acc[0] + 1j * acc[1]; c2 = acc[2] + 1j * acc[3]
    return c0, c2


if __name__ == "__main__":
    main()
