"""v23: z17 RGB only baseline at full 197-cell scale.

Same UNet+B3 architecture as v19/v19+ but input is just 3-channel z17 RGB
(no Sentinel-2). This is the critical RGB-only ablation for paper benchmark
showing the lift from spectral data.

Comparison setup at 197 cells / 90 counties:
  v19+   (S2 only,        5ch @ 10m, B3)
  v23    (z17 RGB only,   3ch @ 1m,  B3)
  v22    (z17+S2 fusion,  two-stream, B3+B0)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DLBM_TO_CLASS, IMAGENET_MEAN, IMAGENET_STD
from train_v16_binary import rasterise_dltb_binary


class RGBTilesDataset(torch.utils.data.Dataset):
    TILE = 448

    def __init__(self, cells, stride=384, training=True):
        self.training = training
        self.items = []
        for cell in cells:
            H, W = cell["rgb"].shape[:2]
            for top in range(0, max(1, H - self.TILE) + 1, stride):
                top = min(top, H - self.TILE)
                for left in range(0, max(1, W - self.TILE) + 1, stride):
                    left = min(left, W - self.TILE)
                    self.items.append((cell, top, left))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        cell, top, left = self.items[idx]
        rgb = cell["rgb"][top:top+self.TILE, left:left+self.TILE].copy()
        lbl = cell["label"][top:top+self.TILE, left:left+self.TILE].copy()
        if self.training:
            if np.random.random() < 0.5: rgb = rgb[:, ::-1, :].copy(); lbl = lbl[:, ::-1].copy()
            if np.random.random() < 0.5: rgb = rgb[::-1, :, :].copy(); lbl = lbl[::-1, :].copy()
            k = np.random.randint(4)
            if k:
                rgb = np.rot90(rgb, k=k, axes=(0,1)).copy()
                lbl = np.rot90(lbl, k=k, axes=(0,1)).copy()
            rgb_f = rgb.astype(np.float32) / 255.0
            jit = 1.0 + (np.random.random() - 0.5) * 0.3
            rgb_f = np.clip(rgb_f * jit, 0, 1)
        else:
            rgb_f = rgb.astype(np.float32) / 255.0
        x = ((rgb_f - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(lbl.astype(np.int64))


def load_cell(r, args, gdf):
    import rasterio
    bb = tuple(r["bbox"])
    for src in ["esri", "google"]:
        path = args.z17_dir / f"{r['county']}_{r['idx']}_{src}.tif"
        if not path.exists(): continue
        with rasterio.open(path) as rs:
            bands = rs.read(out_dtype="uint8")
            rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
            transform = rs.transform; H, W = rs.height, rs.width
        label = rasterise_dltb_binary(gdf, bb, transform, H, W)
        if (label > 0).sum() < 1000: return None
        return {"rgb": rgb, "label": label}
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v20_random_regions.json")
    p.add_argument("--z17-dir", type=Path, default=HOME / "data/v11_imagery")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v23")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--backbone", default="efficientnet-b3")
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    regions = json.loads(args.regions_json.read_text())
    print(f"[1] {len(regions['train'])} train + {len(regions['test'])} test", flush=True)

    import geopandas as gpd
    gdf = {}
    for r in regions["train"] + regions["test"]:
        c = r["county"]
        if c in gdf: continue
        g = gpd.read_parquet(args.dltb_cache / f"{c}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf[c] = g
    print(f"  {len(gdf)} counties", flush=True)

    def loadsplit(rs, name):
        cells = []
        for r in rs:
            c = load_cell(r, args, gdf[r["county"]])
            if c: cells.append(c)
        print(f"  {name}: {len(cells)} cells", flush=True); return cells

    t0 = time.time()
    train_cells = loadsplit(regions["train"], "train")
    test_cells = loadsplit(regions["test"], "test")
    print(f"  load time {time.time()-t0:.0f}s", flush=True)

    train_ds = RGBTilesDataset(train_cells, stride=384, training=True)
    print(f"  train tiles: {len(train_ds)}", flush=True)

    print(f"\n[2] UNet + {args.backbone}, 3-ch RGB", flush=True)
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=args.backbone, encoder_weights="imagenet",
                     in_channels=3, classes=3).to(device)

    bin_counts = np.zeros(3, dtype=np.float64)
    for c in train_cells: bin_counts += np.bincount(c["label"].ravel(), minlength=3)
    cw = np.where(bin_counts > 0, 1.0/np.sqrt(bin_counts), 0.0).astype(np.float32)
    cw[0] = 0.0; cw = cw / cw.sum() * 2
    cw_t = torch.from_numpy(cw).to(device)

    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                          shuffle=True, num_workers=4, pin_memory=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    test_ds = RGBTilesDataset(test_cells, stride=448, training=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                               shuffle=False, num_workers=2)

    best_f1 = -1; no_improve = 0
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0; n_b = 0; correct = 0; total = 0
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                loss = F.cross_entropy(logits.float(), yb, weight=cw_t, ignore_index=0)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            ep_loss += loss.item(); n_b += 1
            with torch.no_grad():
                p = logits.argmax(dim=1); v = yb > 0
                if v.any():
                    correct += int((p[v]==yb[v]).sum().item())
                    total += int(v.sum().item())
        sched.step()
        train_acc = correct / max(total, 1)

        # eval
        model = getattr(model, "eval")()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb); p = logits.argmax(dim=1); v = yb > 0
                if v.any():
                    pi = (p==1) & v; ti = (yb==1) & v
                    tp += int((pi & ti).sum().item()); fp += int((pi & ~ti & v).sum().item())
                    fn += int((~pi & ti).sum().item()); tn += int((~pi & ~ti & v).sum().item())
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        iou = tp/(tp+fp+fn) if tp+fp+fn else 0
        acc = (tp+tn)/max(tp+fp+fn+tn,1)
        print(f"  ep{ep+1}/{args.epochs}: loss={ep_loss/n_b:.4f} train_acc={train_acc:.3f} "
              f"| acc={acc:.3f} iou={iou:.3f} prec={prec:.3f} rec={rec:.3f} F1={f1:.3f} ({time.time()-t0:.0f}s)",
              flush=True)
        if f1 > best_f1:
            best_f1 = f1; no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best F1={best_f1:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 12:
                print("[early stop]", flush=True); break
    print(f"\n[done] best F1={best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
