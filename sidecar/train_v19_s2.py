"""v19: 5-channel Sentinel-2 RGBNIR + NDVI training for binary broad-cropland.

Architecture: segmentation_models_pytorch UNet + EfficientNet-B3 backbone
  (matches FTW PRUE config). 5-input-channel modification via first-conv replacement.
Input: (R, G, B, NIR, NDVI) at 10m / ~240×240 per cell.
Loss: plain CrossEntropy (per Hu et al. Nature 2026 PV recipe).
Output: binary cropland (1 = 耕地+园地, 2 = other, 0 = nodata).
Goal: F1 ≥ 0.85 with NIR/NDVI signal.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HOME = Path("/home/ps/landform")
S2_DIR = HOME / "data/v19_s2_raw"
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DLBM_TO_CLASS

# S2 bands order in npz: B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)
# Surface reflectance in uint16, typical range 0-3000 for visible, 0-6000 for NIR.
S2_MEAN = np.array([400, 460, 320, 1800], dtype=np.float32)  # rough median
S2_STD = np.array([200, 200, 200, 700], dtype=np.float32)
NDVI_MEAN = 0.5
NDVI_STD = 0.3


def rasterize_label_at_grid(gdf_wgs84, bbox, transform_arr, H, W):
    """Rasterise broad-cropland binary label at the S2 grid."""
    from rasterio.features import rasterize
    from shapely.geometry import box as shp_box
    from affine import Affine

    transform = Affine(*transform_arr.flatten()[:6])
    idx = list(gdf_wgs84.sindex.intersection(tuple(bbox)))
    if not idx:
        return np.zeros((H, W), dtype=np.uint8)
    sub = gdf_wgs84.iloc[idx].copy()
    sub["geometry"] = sub.geometry.intersection(shp_box(*bbox))
    sub = sub[~sub.geometry.is_empty]
    if len(sub) == 0:
        return np.zeros((H, W), dtype=np.uint8)
    sub["bin"] = np.where((sub["cid"] == 1) | (sub["cid"] == 2), 1, 2)
    shapes = [(g, int(c)) for g, c in zip(sub.geometry, sub["bin"])]
    return rasterize(shapes=shapes, out_shape=(H, W), transform=transform,
                     fill=0, dtype="uint8")


class S2CellDataset(torch.utils.data.Dataset):
    """One sample = one full cell (no tiling needed, cells are ~240×240).
    Augmentations: h/v flip, 90° rotation, photometric jitter on RGBNIR only."""

    def __init__(self, cells, target_size=224, training=True):
        self.cells = cells
        self.size = target_size
        self.training = training

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        c = self.cells[idx]
        rgbnir = c["rgbnir"].astype(np.float32)
        ndvi = c["ndvi"].astype(np.float32)
        lbl = c["label"].astype(np.int64)

        # Normalize: (band - mean) / std
        for b in range(4):
            rgbnir[b] = (rgbnir[b] - S2_MEAN[b]) / S2_STD[b]
        ndvi_n = (ndvi - NDVI_MEAN) / NDVI_STD

        # Stack into (5, H, W)
        x = np.concatenate([rgbnir, ndvi_n[None, ...]], axis=0).astype(np.float32)
        H, W = x.shape[1], x.shape[2]

        # Random crop / pad to target_size
        sz = self.size
        if H < sz or W < sz:
            pad_h = max(0, sz - H); pad_w = max(0, sz - W)
            x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            lbl = np.pad(lbl, ((0, pad_h), (0, pad_w)), mode="constant")
            H, W = x.shape[1], x.shape[2]
        if self.training:
            top = np.random.randint(0, max(1, H - sz + 1))
            left = np.random.randint(0, max(1, W - sz + 1))
        else:
            top = (H - sz) // 2; left = (W - sz) // 2
        x = x[:, top:top+sz, left:left+sz]
        lbl = lbl[top:top+sz, left:left+sz]

        if self.training:
            if np.random.random() < 0.5:
                x = x[:, :, ::-1].copy(); lbl = lbl[:, ::-1].copy()
            if np.random.random() < 0.5:
                x = x[:, ::-1, :].copy(); lbl = lbl[::-1, :].copy()
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                lbl = np.rot90(lbl, k=k, axes=(0, 1)).copy()
            # Photometric jitter on RGBNIR only (not NDVI)
            jit = 1.0 + (np.random.random(4) - 0.5) * 0.2  # ±10%
            for b in range(4):
                x[b] *= jit[b]

        return torch.from_numpy(x), torch.from_numpy(lbl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v17_regions.json")
    p.add_argument("--s2-dir", type=Path, default=S2_DIR)
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v19")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--backbone", default="efficientnet-b3")
    p.add_argument("--lr", type=float, default=1e-3)
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
            npz_path = args.s2_dir / f"{r['county']}_{r['idx']}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            rgbnir = data["rgbnir"]; ndvi = data["ndvi"]
            transform_arr = data["transform"]; bbox = data["bbox"]
            H, W = rgbnir.shape[1], rgbnir.shape[2]
            label = rasterize_label_at_grid(gdf_per_county[r["county"]], bbox,
                                             transform_arr, H, W)
            if (label > 0).sum() < 100:
                continue
            cells.append({"rgbnir": rgbnir, "ndvi": ndvi, "label": label, "name": f"{r['county']}_{r['idx']}"})
        print(f"  {name}: {len(cells)} cells", flush=True)
        return cells

    print(f"\n[2] Loading", flush=True)
    train_cells = load_split(regions_meta["train"], "train")
    test_cells = load_split(regions_meta["test"], "test")

    train_ds = S2CellDataset(train_cells, target_size=args.target_size, training=True)
    test_ds = S2CellDataset(test_cells, target_size=args.target_size, training=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                               shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n[3] Model: UNet + {args.backbone}, 5 input channels, 3 classes", flush=True)
    import segmentation_models_pytorch as smp
    model = smp.Unet(
        encoder_name=args.backbone,
        encoder_weights="imagenet",
        in_channels=5,
        classes=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params/1e6:.1f}M", flush=True)

    # Compute class weights from training labels (ignore 0)
    bin_counts = np.zeros(3, dtype=np.float64)
    for c in train_cells:
        bin_counts += np.bincount(c["label"].ravel(), minlength=3)
    cw = np.where(bin_counts > 0, 1.0 / np.sqrt(bin_counts), 0.0).astype(np.float32)
    cw[0] = 0.0
    cw = cw / cw.sum() * 2
    print(f"  class pixel share: nodata={bin_counts[0]/bin_counts.sum()*100:.1f}% "
          f"crop={bin_counts[1]/(bin_counts[1]+bin_counts[2])*100:.1f}% in labelled", flush=True)
    print(f"  class weights: {cw.round(3).tolist()}", flush=True)
    cw_t = torch.from_numpy(cw).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_f1 = -1; no_improve = 0; PATIENCE = 10
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
            scaler.step(optim); scaler.update()
            ep_loss += loss.item(); n_b += 1
            with torch.no_grad():
                p = logits.argmax(dim=1)
                v = yb > 0
                if v.any():
                    correct += int((p[v] == yb[v]).sum().item())
                    total += int(v.sum().item())
        sched.step()
        train_acc = correct / max(total, 1)

        # Eval
        model = getattr(model, "eval")()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                p = logits.argmax(dim=1)
                v = yb > 0
                if v.any():
                    pi = (p == 1) & v; ti = (yb == 1) & v
                    tp += int((pi & ti).sum().item())
                    fp += int((pi & ~ti & v).sum().item())
                    fn += int((~pi & ti).sum().item())
                    tn += int((~pi & ~ti & v).sum().item())
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)

        print(f"  ep{ep+1}/{args.epochs}: loss={ep_loss/n_b:.4f} train_acc={train_acc:.3f} "
              f"| test acc={acc:.3f} iou={iou:.3f} prec={prec:.3f} rec={rec:.3f} F1={f1:.3f} "
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
