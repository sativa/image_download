"""v19b: v19 + Hu et al. (Nature 2026) PV-style training recipe.

Differences from v19:
  - Backbone: EfficientNet-B5 (40M params, fits A6000)
  - Augmentation: PhotoMetricDistortion-style (brightness/contrast/saturation/hue)
  - cat_max_ratio=0.75 sampling — discard tiles with >75% single class
  - SGD + PolyLR (per Hu et al. config), lr=0.01
  - Longer training (10k iters via repeated epochs)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HOME = Path("/home/ps/landform")
S2_DIR = HOME / "data/v19_s2_raw"
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DLBM_TO_CLASS

S2_MEAN = np.array([400, 460, 320, 1800], dtype=np.float32)
S2_STD = np.array([200, 200, 200, 700], dtype=np.float32)
NDVI_MEAN = 0.5
NDVI_STD = 0.3


def rasterize_label_at_grid(gdf_wgs84, bbox, transform_arr, H, W):
    from rasterio.features import rasterize
    from shapely.geometry import box as shp_box
    from affine import Affine
    transform = Affine(*transform_arr.flatten()[:6])
    idx = list(gdf_wgs84.sindex.intersection(tuple(bbox)))
    if not idx: return np.zeros((H, W), dtype=np.uint8)
    sub = gdf_wgs84.iloc[idx].copy()
    sub["geometry"] = sub.geometry.intersection(shp_box(*bbox))
    sub = sub[~sub.geometry.is_empty]
    if len(sub) == 0: return np.zeros((H, W), dtype=np.uint8)
    sub["bin"] = np.where((sub["cid"] == 1) | (sub["cid"] == 2), 1, 2)
    shapes = [(g, int(c)) for g, c in zip(sub.geometry, sub["bin"])]
    return rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                     fill=0, dtype="uint8")


def photometric_distortion(rgbnir):
    """mmseg-style aug: brightness/contrast/saturation/hue (on RGBNIR before normalization).
    Operates in normalized space, applied to 4 bands."""
    # rgbnir is (4, H, W) already normalized
    # brightness
    if np.random.random() < 0.5:
        rgbnir = rgbnir + np.random.uniform(-0.3, 0.3)
    # contrast
    if np.random.random() < 0.5:
        rgbnir = rgbnir * np.random.uniform(0.7, 1.3)
    # per-channel jitter (analog of saturation/hue for 4 bands)
    if np.random.random() < 0.5:
        for b in range(4):
            rgbnir[b] = rgbnir[b] * np.random.uniform(0.85, 1.15)
    return rgbnir


class S2CellDataset(torch.utils.data.Dataset):
    def __init__(self, cells, target_size=224, training=True, cat_max_ratio=0.75,
                 max_retries=5):
        self.cells = cells; self.size = target_size; self.training = training
        self.cat_max_ratio = cat_max_ratio; self.max_retries = max_retries

    def __len__(self): return len(self.cells)

    def _crop(self, x, lbl):
        H, W = x.shape[1], x.shape[2]
        sz = self.size
        if H < sz or W < sz:
            ph = max(0, sz - H); pw = max(0, sz - W)
            x = np.pad(x, ((0,0),(0,ph),(0,pw)), mode="edge")
            lbl = np.pad(lbl, ((0,ph),(0,pw)), mode="constant")
            H, W = x.shape[1], x.shape[2]
        if self.training:
            top = np.random.randint(0, max(1, H - sz + 1))
            left = np.random.randint(0, max(1, W - sz + 1))
        else:
            top = (H - sz) // 2; left = (W - sz) // 2
        return x[:, top:top+sz, left:left+sz], lbl[top:top+sz, left:left+sz]

    def __getitem__(self, idx):
        c = self.cells[idx]
        rgbnir = c["rgbnir"].astype(np.float32)
        ndvi = c["ndvi"].astype(np.float32)
        lbl_full = c["label"].astype(np.int64)
        for b in range(4):
            rgbnir[b] = (rgbnir[b] - S2_MEAN[b]) / S2_STD[b]
        ndvi_n = (ndvi - NDVI_MEAN) / NDVI_STD
        x_full = np.concatenate([rgbnir, ndvi_n[None, ...]], axis=0).astype(np.float32)

        # cat_max_ratio sampling
        for _ in range(self.max_retries):
            x, lbl = self._crop(x_full.copy(), lbl_full.copy())
            if not self.training:
                break
            v = lbl > 0
            if v.sum() == 0:
                continue
            counts = np.bincount(lbl[v].ravel(), minlength=3)
            largest_ratio = counts[1:].max() / max(counts[1:].sum(), 1)
            if largest_ratio < self.cat_max_ratio:
                break

        if self.training:
            if np.random.random() < 0.5:
                x = x[:, :, ::-1].copy(); lbl = lbl[:, ::-1].copy()
            if np.random.random() < 0.5:
                x = x[:, ::-1, :].copy(); lbl = lbl[::-1, :].copy()
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                lbl = np.rot90(lbl, k=k, axes=(0, 1)).copy()
            x[:4] = photometric_distortion(x[:4])
        return torch.from_numpy(x), torch.from_numpy(lbl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v17_regions.json")
    p.add_argument("--s2-dir", type=Path, default=S2_DIR)
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v19b")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--backbone", default="efficientnet-b5")
    p.add_argument("--lr", type=float, default=0.01)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    regions_meta = json.loads(args.regions_json.read_text())
    print(f"[1] {len(regions_meta['train'])} train + {len(regions_meta['test'])} test", flush=True)

    import geopandas as gpd
    gdf_per_county = {}
    for r in regions_meta["train"] + regions_meta["test"]:
        code = r["county"]
        if code in gdf_per_county: continue
        g = gpd.read_parquet(args.dltb_cache / f"{code}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf_per_county[code] = g
    print(f"  {len(gdf_per_county)} counties", flush=True)

    def load_split(region_list, name):
        cells = []
        for r in region_list:
            path = args.s2_dir / f"{r['county']}_{r['idx']}.npz"
            if not path.exists(): continue
            data = np.load(path)
            rgbnir = data["rgbnir"]; ndvi = data["ndvi"]
            H, W = rgbnir.shape[1], rgbnir.shape[2]
            label = rasterize_label_at_grid(gdf_per_county[r["county"]], data["bbox"],
                                             data["transform"], H, W)
            if (label > 0).sum() < 100: continue
            cells.append({"rgbnir": rgbnir, "ndvi": ndvi, "label": label})
        print(f"  {name}: {len(cells)} cells", flush=True); return cells

    train_cells = load_split(regions_meta["train"], "train")
    test_cells = load_split(regions_meta["test"], "test")
    train_ds = S2CellDataset(train_cells, args.target_size, training=True, cat_max_ratio=0.75)
    test_ds = S2CellDataset(test_cells, args.target_size, training=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                               shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n[2] Model: UNet + {args.backbone}, 5-ch input", flush=True)
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=args.backbone, encoder_weights=None,
                     in_channels=5, classes=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e6:.1f}M", flush=True)

    bin_counts = np.zeros(3, dtype=np.float64)
    for c in train_cells: bin_counts += np.bincount(c["label"].ravel(), minlength=3)
    cw = np.where(bin_counts > 0, 1.0 / np.sqrt(bin_counts), 0.0).astype(np.float32)
    cw[0] = 0.0; cw = cw / cw.sum() * 2
    print(f"  class weights: {cw.round(3).tolist()}", flush=True)
    cw_t = torch.from_numpy(cw).to(device)

    # SGD + PolyLR (Hu et al. recipe)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                            weight_decay=5e-4)
    total_iters = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.PolynomialLR(optim, total_iters=total_iters, power=0.9)
    scaler = torch.amp.GradScaler("cuda")

    best_f1 = -1; no_improve = 0; PATIENCE = 15
    iter_count = 0
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0; n_b = 0; correct = 0; total = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                loss = F.cross_entropy(logits.float(), yb, weight=cw_t, ignore_index=0)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update(); sched.step()
            ep_loss += loss.item(); n_b += 1; iter_count += 1
            with torch.no_grad():
                p = logits.argmax(dim=1); v = yb > 0
                if v.any():
                    correct += int((p[v] == yb[v]).sum().item())
                    total += int(v.sum().item())
        train_acc = correct / max(total, 1)

        model = getattr(model, "eval")()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb); p = logits.argmax(dim=1); v = yb > 0
                if v.any():
                    pi = (p == 1) & v; ti = (yb == 1) & v
                    tp += int((pi & ti).sum().item())
                    fp += int((pi & ~ti & v).sum().item())
                    fn += int((~pi & ti).sum().item())
                    tn += int((~pi & ~ti & v).sum().item())
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        iou = tp/(tp+fp+fn) if tp+fp+fn else 0
        acc = (tp+tn)/max(tp+fp+fn+tn, 1)
        print(f"  ep{ep+1}/{args.epochs}: loss={ep_loss/n_b:.4f} train_acc={train_acc:.3f} "
              f"| acc={acc:.3f} iou={iou:.3f} prec={prec:.3f} rec={rec:.3f} F1={f1:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if f1 > best_f1:
            best_f1 = f1; no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best F1={best_f1:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[early stop]", flush=True); break
    print(f"\n[done] best F1={best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
