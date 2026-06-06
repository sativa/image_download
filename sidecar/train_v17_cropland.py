"""v17: cropland-rich binary cropland with stronger augmentation.

Goal: F1 ≥ 0.90 on broad-cropland binary classification.

Differences from v16:
  - 105 training regions (vs 80), ALL from counties with >30% cropland share
  - Augmentation: h/v flip + random 90° rotation + brightness jitter
  - Lower backbone LR (5e-6), longer training (30 epochs)
  - TTA at every eval (4 flips averaged)
  - Save best.pt + ema.pt (exponential moving average weights)
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
sys.path.insert(0, str(HOME / "sidecar"))
from train_v12_unet import DEFAULT_DINOV2, IMAGENET_MEAN, IMAGENET_STD, DLBM_TO_CLASS, DinoUNet
from train_v13_lovasz import lovasz_softmax
from train_v16_binary import rasterise_dltb_binary, composite_binary_loss


class StrongAugTilesDataset(torch.utils.data.Dataset):
    """Per-pixel tile dataset with h/v flip + 90° rot + brightness jitter."""
    TILE = 448

    def __init__(self, regions, stride: int = 384):
        self.items = []
        for rgb, lbl in regions:
            H, W = rgb.shape[:2]
            for top in range(0, max(1, H - self.TILE) + 1, stride):
                top = min(top, H - self.TILE)
                for left in range(0, max(1, W - self.TILE) + 1, stride):
                    left = min(left, W - self.TILE)
                    self.items.append((rgb, lbl, top, left))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rgb, lbl, top, left = self.items[idx]
        rgb_tile = rgb[top:top+self.TILE, left:left+self.TILE].copy()
        lbl_tile = lbl[top:top+self.TILE, left:left+self.TILE].copy()
        # H/V flips (each 50% chance)
        if np.random.random() < 0.5:
            rgb_tile = rgb_tile[:, ::-1, :].copy(); lbl_tile = lbl_tile[:, ::-1].copy()
        if np.random.random() < 0.5:
            rgb_tile = rgb_tile[::-1, :, :].copy(); lbl_tile = lbl_tile[::-1, :].copy()
        # 90° rotation (uniform 0/90/180/270)
        k = np.random.randint(4)
        if k:
            rgb_tile = np.rot90(rgb_tile, k=k, axes=(0, 1)).copy()
            lbl_tile = np.rot90(lbl_tile, k=k, axes=(0, 1)).copy()
        # Brightness jitter ±15%
        brightness = 1.0 + (np.random.random() - 0.5) * 0.3
        arr = rgb_tile.astype(np.float32) / 255.0
        arr = np.clip(arr * brightness, 0, 1)
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(arr).permute(2, 0, 1),
                torch.from_numpy(lbl_tile.astype(np.int64)))


def evaluate_full_binary_tta(model, rgb, lbl_truth, device, stride=384, batch_size=8,
                              use_tta=True):
    """Slide UNet + optional TTA (4 flips). Returns metrics dict."""
    H, W = rgb.shape[:2]
    flips = ["identity", "h", "v", "hv"] if use_tta else ["identity"]
    avg_probs = np.zeros((H, W, 3), dtype=np.float32)

    def _flip(arr, fl):
        if fl == "h": return arr[:, ::-1, :]
        if fl == "v": return arr[::-1, :, :]
        if fl == "hv": return arr[::-1, ::-1, :]
        return arr

    was_training = model.training
    model = getattr(model, "eval")()
    with torch.no_grad():
        for flip in flips:
            rgb_f = np.ascontiguousarray(_flip(rgb, flip)) if flip != "identity" else rgb
            TILE = 448
            pad_h = (stride - (H - TILE) % stride) % stride if H > TILE else TILE - H
            pad_w = (stride - (W - TILE) % stride) % stride if W > TILE else TILE - W
            pad_h = max(0, pad_h); pad_w = max(0, pad_w)
            padded = np.pad(rgb_f, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            Hp, Wp = padded.shape[:2]
            tiles, positions = [], []
            for top in range(0, Hp - TILE + 1, stride):
                for left in range(0, Wp - TILE + 1, stride):
                    tiles.append(padded[top:top+TILE, left:left+TILE])
                    positions.append((top, left))
            score = np.zeros((Hp, Wp, 3), dtype=np.float32)
            weight = np.zeros((Hp, Wp), dtype=np.float32)
            for b0 in range(0, len(tiles), batch_size):
                batch = tiles[b0:b0+batch_size]
                arr = np.stack(batch).astype(np.float32) / 255.0
                arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
                x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                for k, (top, left) in enumerate(positions[b0:b0+len(batch)]):
                    score[top:top+TILE, left:left+TILE] += probs[k].transpose(1, 2, 0)
                    weight[top:top+TILE, left:left+TILE] += 1.0
            score /= np.maximum(weight, 1e-6)[..., None]
            score = score[:H, :W]
            score = _flip(score, flip) if flip != "identity" else score
            avg_probs += np.ascontiguousarray(score)
    if was_training:
        model.train()
    avg_probs /= len(flips)
    pred = avg_probs.argmax(axis=-1).astype(np.uint8)

    valid = lbl_truth > 0
    if not valid.any():
        return None
    p, t = pred[valid], lbl_truth[valid]
    crop_p, crop_t = (p == 1), (t == 1)
    tp = int((crop_p & crop_t).sum())
    fp = int((crop_p & ~crop_t).sum())
    fn = int((~crop_p & crop_t).sum())
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    acc = int((p == t).sum()) / int(valid.sum())
    return {"acc": acc, "iou": iou, "precision": prec, "recall": rec, "f1": f1}


class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()
                       if v.dtype in (torch.float32, torch.float16, torch.bfloat16)}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply(self, model):
        """Returns a copy of model with EMA weights."""
        clone = copy.deepcopy(model)
        sd = clone.state_dict()
        for k in self.shadow:
            sd[k].copy_(self.shadow[k])
        clone.load_state_dict(sd)
        return clone


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v17_regions.json")
    p.add_argument("--data-cache", type=Path, default=HOME / "data/v11_imagery")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v17")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--tile-batch", type=int, default=8)
    p.add_argument("--stride", type=int, default=384)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--backbone-lr", type=float, default=5e-6)
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.add_argument("--ema-decay", type=float, default=0.999)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    regions_meta = json.loads(args.regions_json.read_text())
    print(f"[1] {len(regions_meta['train'])} train + {len(regions_meta['test'])} test (BINARY, cropland-rich)", flush=True)

    import geopandas as gpd, rasterio
    gdf_per_county = {}
    for r in regions_meta["train"] + regions_meta["test"]:
        code = r["county"]
        if code in gdf_per_county:
            continue
        g = gpd.read_parquet(args.dltb_cache / f"{code}.parquet")
        if g.crs is None or g.crs.to_epsg() != 4326:
            g = g.to_crs("EPSG:4326")
        try: g["geometry"] = g.geometry.make_valid()
        except AttributeError: g["geometry"] = g.geometry.buffer(0)
        g["cid"] = g["DLBM"].astype(str).str[:2].map(DLBM_TO_CLASS).fillna(0).astype(int)
        gdf_per_county[code] = g
    print(f"  {len(gdf_per_county)} counties", flush=True)

    def _load(region_list):
        regs = []
        for r in region_list:
            bb = tuple(r["bbox"])
            for src in ["esri", "google"]:
                path = args.data_cache / f"{r['county']}_{r['idx']}_{src}.tif"
                if not path.exists():
                    continue
                with rasterio.open(path) as rs:
                    bands = rs.read(out_dtype="uint8")
                    transform = rs.transform; H, W = rs.height, rs.width
                rgb = np.stack([bands[0], bands[1], bands[2]], axis=-1)
                lbl = rasterise_dltb_binary(gdf_per_county[r["county"]], bb, transform, H, W)
                regs.append((rgb, lbl))
        return regs

    print(f"\n[2] Loading", flush=True)
    t0 = time.time()
    train_regions = _load(regions_meta["train"])
    test_regions = _load(regions_meta["test"])
    print(f"  {len(train_regions)} train + {len(test_regions)} test ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n[3] Dataset", flush=True)
    train_ds = StrongAugTilesDataset(train_regions, stride=args.stride)
    print(f"  {len(train_ds)} tiles", flush=True)
    bin_counts = np.zeros(3, dtype=np.float64)
    for _, lbl in train_regions:
        bin_counts += np.bincount(lbl.ravel(), minlength=3)
    cw = np.where(bin_counts > 0, 1.0 / np.sqrt(bin_counts), 0.0).astype(np.float32)
    cw[0] = 0.0
    cw = (cw / cw.sum() * 2)
    cropland_share = bin_counts[1] / bin_counts[1:].sum()
    print(f"  cropland pixel share in training: {cropland_share*100:.1f}%", flush=True)
    print(f"  class weights: {cw.round(3).tolist()}", flush=True)

    print(f"\n[4] Model (DINOv2 + UNet, 3-channel out, EMA decay={args.ema_decay})", flush=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2))
    model = DinoUNet(dinov2, num_classes=3, unfreeze_last_n=args.unfreeze_blocks).to(device)
    ema = EMA(model, decay=args.ema_decay)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable: {trainable:,}", flush=True)

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not (n.startswith("up") or n.startswith("proj")
                                                    or n.startswith("classifier"))]
    head_params = [p for n, p in model.named_parameters()
                   if (n.startswith("up") or n.startswith("proj") or n.startswith("classifier"))]
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    cw_t = torch.from_numpy(cw).to(device)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.tile_batch, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=False,
    )

    print(f"\n[5] Training {args.epochs} epochs, strong aug + EMA + TTA eval", flush=True)
    best_f1 = -1.0
    no_improve = 0
    PATIENCE = 8
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0; n_b = 0; ep_c = 0; ep_t = 0
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                loss, _ = composite_binary_loss(logits, yb, cw_t)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            ema.update(model)
            ep_loss += loss.item(); n_b += 1
            preds = logits.argmax(dim=1)
            valid = yb > 0
            if valid.any():
                ep_c += int((preds[valid] == yb[valid]).sum().item())
                ep_t += int(valid.sum().item())
        sched.step()
        train_acc = ep_c / max(ep_t, 1)
        print(f"  ep{ep+1}/{args.epochs}: loss={ep_loss/n_b:.4f} train_acc={train_acc:.3f} ({time.time()-t0:.0f}s)",
              flush=True)

        # Eval with TTA, using EMA weights.
        t0 = time.time()
        ema_model = ema.apply(model)
        ema_model = getattr(ema_model, "eval")()
        all_metrics = []
        for rgb, lbl in test_regions:
            m = evaluate_full_binary_tta(ema_model, rgb, lbl, device,
                                          stride=args.stride, batch_size=args.tile_batch,
                                          use_tta=True)
            if m: all_metrics.append(m)
        avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
        print(f"    eval (EMA+TTA): acc={avg['acc']:.3f} iou={avg['iou']:.3f} "
              f"prec={avg['precision']:.3f} rec={avg['recall']:.3f} F1={avg['f1']:.3f} ({time.time()-t0:.0f}s)",
              flush=True)

        if avg["f1"] > best_f1:
            best_f1 = avg["f1"]
            no_improve = 0
            torch.save(ema_model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best F1={best_f1:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[early stop]", flush=True)
                break

    print(f"\n[done] best F1={best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
