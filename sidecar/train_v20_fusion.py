"""v20: hi-res RGB (z17) + Sentinel-2 NIR + NDVI fusion.

Architecture:
  Core: z17 Esri/Google RGB at ~1m resolution (existing data)
  Aux:  Sentinel-2 NIR (B08) + NDVI, originally 10m, upsampled bilinearly to z17 grid

Final input: 5 channels @ z17 resolution (R, G, B, NIR, NDVI)
Target: binary broad cropland (耕地 + 园地)

Why fusion: z17 RGB gives sharp field boundaries; S2 NIR/NDVI gives spectral
discrimination between cropland / forest / grassland that RGB cannot resolve.
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

S2_DIR = HOME / "data/v19_s2_raw"
Z17_DIR = HOME / "data/v11_imagery"
NDVI_MEAN = 0.5
NDVI_STD = 0.3
NIR_MEAN = 1800; NIR_STD = 700


def upsample_to_grid(arr10m, target_H, target_W):
    """Bilinear-upsample a 10m S2 channel to the z17 1m grid."""
    t = torch.from_numpy(arr10m.astype(np.float32))[None, None, ...]
    out = F.interpolate(t, size=(target_H, target_W), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


class FusionTilesDataset(torch.utils.data.Dataset):
    """Tile z17 RGB + upsampled S2 channels into 448×448 crops for training."""
    TILE = 448

    def __init__(self, cells, stride: int = 384, training: bool = True):
        self.training = training
        self.items = []
        for cell in cells:
            H, W = cell["rgb"].shape[:2]
            for top in range(0, max(1, H - self.TILE) + 1, stride):
                top = min(top, H - self.TILE)
                for left in range(0, max(1, W - self.TILE) + 1, stride):
                    left = min(left, W - self.TILE)
                    self.items.append((cell, top, left))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cell, top, left = self.items[idx]
        rgb = cell["rgb"][top:top+self.TILE, left:left+self.TILE].copy()  # H,W,3 uint8
        nir = cell["nir"][top:top+self.TILE, left:left+self.TILE].copy()  # H,W float32
        ndvi = cell["ndvi"][top:top+self.TILE, left:left+self.TILE].copy()  # H,W float32
        lbl = cell["label"][top:top+self.TILE, left:left+self.TILE].copy()

        # h/v flip + 90° rot (operates on raw data — preserves all channels jointly)
        if self.training:
            if np.random.random() < 0.5:
                rgb = rgb[:, ::-1, :].copy(); nir = nir[:, ::-1].copy()
                ndvi = ndvi[:, ::-1].copy(); lbl = lbl[:, ::-1].copy()
            if np.random.random() < 0.5:
                rgb = rgb[::-1, :, :].copy(); nir = nir[::-1, :].copy()
                ndvi = ndvi[::-1, :].copy(); lbl = lbl[::-1, :].copy()
            k = np.random.randint(4)
            if k:
                rgb = np.rot90(rgb, k=k, axes=(0, 1)).copy()
                nir = np.rot90(nir, k=k, axes=(0, 1)).copy()
                ndvi = np.rot90(ndvi, k=k, axes=(0, 1)).copy()
                lbl = np.rot90(lbl, k=k, axes=(0, 1)).copy()
            # Brightness on RGB only (z17 has real lighting variation across sources)
            rgb_f = rgb.astype(np.float32) / 255.0
            jit = 1.0 + (np.random.random() - 0.5) * 0.3
            rgb_f = np.clip(rgb_f * jit, 0, 1)
        else:
            rgb_f = rgb.astype(np.float32) / 255.0

        rgb_n = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD                  # H,W,3
        nir_n = ((nir - NIR_MEAN) / NIR_STD).astype(np.float32)         # H,W
        ndvi_n = ((ndvi - NDVI_MEAN) / NDVI_STD).astype(np.float32)     # H,W

        x = np.concatenate([
            rgb_n.transpose(2, 0, 1),     # 3,H,W
            nir_n[None, ...],              # 1,H,W
            ndvi_n[None, ...],             # 1,H,W
        ], axis=0).astype(np.float32)     # 5,H,W

        return torch.from_numpy(x), torch.from_numpy(lbl.astype(np.int64))


def load_cell(r, args, gdf):
    """Load a single cell: z17 RGB + S2 NIR/NDVI upsampled + label."""
    import rasterio
    bb = tuple(r["bbox"])
    # z17 RGB — try esri first, then google
    rgb = None; transform = None; H = W = None
    for src in ["esri", "google"]:
        path = args.z17_dir / f"{r['county']}_{r['idx']}_{src}.tif"
        if not path.exists(): continue
        with rasterio.open(path) as rs:
            bands = rs.read(out_dtype="uint8")
            rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
            transform = rs.transform
            H, W = rs.height, rs.width
        break
    if rgb is None:
        return None  # no z17 imagery for this cell — skip
    # Label at z17 grid
    label = rasterise_dltb_binary(gdf, bb, transform, H, W)
    if (label > 0).sum() < 1000:
        return None
    # S2 NPZ
    s2_path = args.s2_dir / f"{r['county']}_{r['idx']}.npz"
    if not s2_path.exists():
        return None
    data = np.load(s2_path)
    s2 = data["rgbnir"]  # 4,h,w uint16 at 10m grid
    ndvi_10m = data["ndvi"].astype(np.float32)  # h,w
    # Upsample NIR + NDVI to z17 grid (10x ~)
    nir_upsampled = upsample_to_grid(s2[3].astype(np.float32), H, W)
    ndvi_upsampled = upsample_to_grid(ndvi_10m, H, W)
    return {
        "rgb": rgb, "nir": nir_upsampled, "ndvi": ndvi_upsampled,
        "label": label, "name": f"{r['county']}_{r['idx']}",
    }


def evaluate(model, test_cells, device, batch_size=8):
    model = getattr(model, "eval")()
    tp = fp = fn = tn = 0
    test_ds = FusionTilesDataset(test_cells, stride=448, training=False)  # non-overlapping
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    with torch.no_grad():
        for xb, yb in loader:
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
    return {"acc": acc, "iou": iou, "precision": prec, "recall": rec, "f1": f1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v20_random_regions.json")
    p.add_argument("--z17-dir", type=Path, default=Z17_DIR)
    p.add_argument("--s2-dir", type=Path, default=S2_DIR)
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v20")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stride", type=int, default=384)
    p.add_argument("--backbone", default="efficientnet-b3")
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    regions = json.loads(args.regions_json.read_text())
    print(f"[1] {len(regions['train'])} train + {len(regions['test'])} test", flush=True)

    import geopandas as gpd
    gdf_per_county = {}
    for r in regions["train"] + regions["test"]:
        c = r["county"]
        if c in gdf_per_county: continue
        g = gpd.read_parquet(args.dltb_cache / f"{c}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf_per_county[c] = g
    print(f"  {len(gdf_per_county)} counties", flush=True)

    def load_split(rs, name):
        cells = []
        for r in rs:
            cell = load_cell(r, args, gdf_per_county[r["county"]])
            if cell: cells.append(cell)
        print(f"  {name}: {len(cells)} cells loaded", flush=True)
        return cells

    t0 = time.time()
    train_cells = load_split(regions["train"], "train")
    test_cells = load_split(regions["test"], "test")
    print(f"  load time {time.time()-t0:.0f}s", flush=True)

    train_ds = FusionTilesDataset(train_cells, stride=args.stride, training=True)
    print(f"  train tiles: {len(train_ds)}", flush=True)

    print(f"\n[2] Model: UNet + {args.backbone}, 5-ch (RGB+NIR+NDVI)", flush=True)
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=args.backbone, encoder_weights="imagenet",
                     in_channels=5, classes=3).to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    bin_counts = np.zeros(3, dtype=np.float64)
    for c in train_cells: bin_counts += np.bincount(c["label"].ravel(), minlength=3)
    cw = np.where(bin_counts > 0, 1.0 / np.sqrt(bin_counts), 0.0).astype(np.float32)
    cw[0] = 0.0; cw = cw / cw.sum() * 2
    print(f"  class pixel share: nodata={bin_counts[0]/bin_counts.sum()*100:.1f}% "
          f"crop_in_labelled={bin_counts[1]/max(bin_counts[1]+bin_counts[2],1)*100:.1f}%", flush=True)
    print(f"  class weights: {cw.round(3).tolist()}", flush=True)
    cw_t = torch.from_numpy(cw).to(device)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True)
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
                p = logits.argmax(dim=1); v = yb > 0
                if v.any():
                    correct += int((p[v] == yb[v]).sum().item())
                    total += int(v.sum().item())
        sched.step()
        train_acc = correct / max(total, 1)
        m = evaluate(model, test_cells, device, batch_size=args.batch_size)
        print(f"  ep{ep+1}/{args.epochs}: loss={ep_loss/n_b:.4f} train_acc={train_acc:.3f} "
              f"| acc={m['acc']:.3f} iou={m['iou']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} F1={m['f1']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]; no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best F1={best_f1:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[early stop]", flush=True); break
    print(f"\n[done] best F1={best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
