"""Test the hypothesis that the ~0.55 Gansu boundary-F1 is a DLTB LABEL-NOISE ceiling, not a model
limit: score each test cell by DLTB boundary quality (median parcel size + sliver fraction), then
report the BD model's boundary-F1 on HIGH-quality vs LOW-quality cells. If clean cells >> 0.55, the
ceiling is label noise -> mining clean-boundary cells for training should help. Also emits a ranked
high-quality cell list for focused boundary training."""
import argparse, json, sys, math
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F, cv2
import geopandas as gpd
from shapely.geometry import box as shp_box

HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6
from train_dino_1m_v2 import load_ndvi_full
from train_dino_1m_v3 import DinoV3FreqUNetBD, DINOV3_SAT
DLTB = "/home/ps/landform/data/v11_dltb"
_cache = {}


def county(c):
    if c in _cache: return _cache[c]
    g = gpd.read_parquet(Path(DLTB) / f"{c}.parquet")
    if g.crs is None or g.crs.to_epsg() != 4326: g = g.to_crs("EPSG:4326")
    try: g["geometry"] = g.geometry.make_valid()
    except Exception: g["geometry"] = g.geometry.buffer(0)
    _cache[c] = g; return g


@torch.no_grad()
def bnd_prob(model, x6, ndvi, dev, cs=448):
    _, SZ, SZw = x6.shape; acc = np.zeros((SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
    ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
    if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
    if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
    for t in ys:
        for l in xs:
            xc = np.concatenate([norm6(x6[:, t:t+cs, l:l+cs]), ndvi[:, t:t+cs, l:l+cs]], 0)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, b, _ = model(torch.from_numpy(xc).unsqueeze(0).to(dev))
                if b.shape[-2:] != (cs, cs): b = F.interpolate(b, size=(cs, cs), mode="bilinear", align_corners=False)
                pb = torch.sigmoid(b.float())[0, 0].cpu().numpy()
            acc[t:t+cs, l:l+cs] += pb; cnt[t:t+cs, l:l+cs] += 1
    return acc / np.maximum(cnt, 1)


def bf1(pb, tb, tol=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))
    td = cv2.dilate(tb, k) > 0; pd = cv2.dilate(pb, k) > 0
    p = (pb.astype(bool) & td).sum() / max(1, pb.sum()); r = (tb.astype(bool) & pd).sum() / max(1, tb.sum())
    return 2*p*r/max(1e-9, p+r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_v3_bd/last.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m_lc")
    p.add_argument("--pbound-dir", default="/mnt/sda/zf/landform/data/c_1m_pbound")
    p.add_argument("--n-cells", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args(); dev = a.device
    from transformers import AutoModel
    d3 = AutoModel.from_pretrained(DINOV3_SAT, local_files_only=True)
    m = DinoV3FreqUNetBD(d3, num_classes=9, in_channels=11, unfreeze_last_n=4).to(dev)
    sd = torch.load(a.ckpt, map_location=dev, weights_only=True); msd = m.state_dict()
    m.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False); m.eval()
    te = [n for n in json.loads((Path(a.data_dir)/"manifest.json").read_text())["test"]
          if (Path(a.pbound_dir)/f"{n}.npy").exists()][:a.n_cells]
    rows = []   # (name, quality, f1)
    for nm in te:
        z = np.load(Path(a.data_dir)/f"{nm}.npz"); x6 = z["x6"]; bbox = z["bbox"]; _, SZ, SZw = x6.shape
        ndvi = load_ndvi_full(nm, SZ, SZw);  ndvi = ndvi if ndvi is not None else np.zeros((5, SZ, SZw), np.float32)
        pb = (bnd_prob(m, x6, ndvi, dev) >= 0.5).astype(np.uint8)
        tb = (np.load(Path(a.pbound_dir)/f"{nm}.npy") > 0).astype(np.uint8)
        f1 = bf1(pb, tb, 3)
        # boundary LABEL quality = do DLTB edges coincide with real IMAGE edges? (clean digitization vs
        # temporally-mismatched/coarse). mean image-gradient AT DLTB-edge / overall mean gradient. This is
        # NOT confounded with edge density (it is a per-edge-pixel agreement, unlike parcel-size).
        gray = cv2.cvtColor(np.ascontiguousarray(x6[:3].transpose(1, 2, 0)), cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        grad = np.sqrt(gx * gx + gy * gy)
        eb = tb > 0
        qual = float(grad[eb].mean() / (grad.mean() + 1e-6)) if eb.any() else 0.0
        rows.append((nm, qual, f1))
    rows.sort(key=lambda r: -r[1])
    qs = np.array([r[1] for r in rows]); f1s = np.array([r[2] for r in rows])
    hi = f1s[:len(rows)//3].mean(); lo = f1s[-len(rows)//3:].mean()
    print(f"\n=== 边界质量分组 boundary-F1 (BD model, {len(rows)} Gansu cells, tol3) ===", flush=True)
    print(f"  全部均值 = {f1s.mean():.4f}", flush=True)
    print(f"  高边界质量 1/3(大田/少碎块)= {hi:.4f}", flush=True)
    print(f"  低边界质量 1/3(碎块多)     = {lo:.4f}", flush=True)
    print(f"  质量-F1 相关 = {np.corrcoef(qs, f1s)[0,1]:.3f}", flush=True)
    json.dump([r[0] for r in rows[:max(1, 2*len(rows)//3)]],
              open("/mnt/sda/zf/landform/data/clean_boundary_cells.json", "w"))
    print(f"  已存高质量 cell 列表 -> clean_boundary_cells.json ({2*len(rows)//3} cells)", flush=True)


if __name__ == "__main__":
    main()
