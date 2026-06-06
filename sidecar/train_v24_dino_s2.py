"""v24: DINOv2-large (300M params) + Sentinel-2 RGBNIR + NDVI + 197 cells.

Tests whether a much bigger SSL-pretrained backbone can break the F1 0.817 ceiling
hit by v19+ (UNet + EfNet-B3, 13M params).

Key adaptation: DINOv2 patch_embed is Conv2d(3 → 1024). We replace it with
Conv2d(5 → 1024), initializing the first 3 channels from imagenet weights and
the extra 2 (NIR, NDVI) from the mean of the RGB channels.

Otherwise identical to v19+: same data, same train/test split, same loss, same eval.
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
from train_v12_unet import DLBM_TO_CLASS, DEFAULT_DINOV2

S2_MEAN = np.array([400, 460, 320, 1800], dtype=np.float32)
S2_STD = np.array([200, 200, 200, 700], dtype=np.float32)
NDVI_MEAN = 0.5
NDVI_STD = 0.3
S2_DIR_DEFAULT = HOME / "data/v19_s2_raw"


class DinoUNet5ch(nn.Module):
    """DINOv2-large + UNet decoder, adapted for 5-channel input.

    DINOv2-large has hidden_dim=1024, patch_size=14. For 224x224 input that gives
    16x16=256 patches. We use 224x224 tiles matching v19's setup.
    """

    def __init__(self, dinov2, num_classes=3, in_channels=5, unfreeze_last_n=4):
        super().__init__()
        self.backbone = dinov2

        # Replace patch_embeddings.projection (Conv2d 3 → embed_dim) with 5-channel version
        orig_proj = self.backbone.embeddings.patch_embeddings.projection
        embed_dim = orig_proj.out_channels    # 1024 for large
        ks = orig_proj.kernel_size; stride = orig_proj.stride
        new_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=ks, stride=stride,
                             bias=(orig_proj.bias is not None))
        with torch.no_grad():
            new_proj.weight[:, :3] = orig_proj.weight              # keep RGB pretrained
            mean_rgb = orig_proj.weight.mean(dim=1, keepdim=True)   # (E, 1, k, k)
            for c in range(3, in_channels):
                new_proj.weight[:, c:c+1] = mean_rgb / (in_channels / 3)  # scale to preserve sum
            if orig_proj.bias is not None:
                new_proj.bias.copy_(orig_proj.bias)
        self.backbone.embeddings.patch_embeddings.projection = new_proj
        # Update internal num_channels check (DINOv2 has a hardcoded assertion)
        self.backbone.embeddings.patch_embeddings.num_channels = in_channels
        self.backbone.config.num_channels = in_channels

        # Freeze first transformer blocks; only fine-tune last N
        for p in self.backbone.parameters():
            p.requires_grad = False
        n_blocks = len(self.backbone.encoder.layer)
        for i, blk in enumerate(self.backbone.encoder.layer):
            if i >= n_blocks - unfreeze_last_n:
                for p in blk.parameters(): p.requires_grad = True
        # Always train the new patch_embed (extra channels need learning)
        for p in self.backbone.embeddings.patch_embeddings.projection.parameters():
            p.requires_grad = True

        # UNet-style decoder: from patch feature grid back to input resolution.
        # 224 / 14 = 16x16 patches → upsample 4 stages to 256x256, resize to 224
        self.proj = nn.Conv2d(embed_dim, 256, 1)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                                  nn.BatchNorm2d(128), nn.ReLU(inplace=True))   # 16→32
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))    # 32→64
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                                  nn.BatchNorm2d(32), nn.ReLU(inplace=True))    # 64→128
        self.up4 = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                                  nn.BatchNorm2d(16), nn.ReLU(inplace=True))    # 128→256
        self.classifier = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        # x: (B, 5, H, W). DINOv2 needs interpolate_pos_encoding when H != 224.
        out = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        tokens = out.last_hidden_state[:, 1:, :]   # drop CLS
        B, N, D = tokens.shape
        Ph = Pw = int(round(np.sqrt(N)))
        feat = tokens.permute(0, 2, 1).reshape(B, D, Ph, Pw)
        x = self.proj(feat)
        x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x)
        # final size depends on input — interpolate to input resolution
        return self.classifier(x)


class S2CellDataset(torch.utils.data.Dataset):
    """Identical to v19 S2CellDataset."""

    def __init__(self, cells, target_size=224, training=True):
        self.cells = cells; self.size = target_size; self.training = training

    def __len__(self): return len(self.cells)

    def __getitem__(self, idx):
        c = self.cells[idx]
        rgbnir = c["rgbnir"].astype(np.float32)
        ndvi = c["ndvi"].astype(np.float32)
        lbl = c["label"].astype(np.int64)
        for b in range(4): rgbnir[b] = (rgbnir[b] - S2_MEAN[b]) / S2_STD[b]
        ndvi_n = (ndvi - NDVI_MEAN) / NDVI_STD
        x = np.concatenate([rgbnir, ndvi_n[None, ...]], axis=0).astype(np.float32)
        H, W = x.shape[1], x.shape[2]; sz = self.size
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
        x = x[:, top:top+sz, left:left+sz]; lbl = lbl[top:top+sz, left:left+sz]
        if self.training:
            if np.random.random() < 0.5: x = x[:, :, ::-1].copy(); lbl = lbl[:, ::-1].copy()
            if np.random.random() < 0.5: x = x[:, ::-1, :].copy(); lbl = lbl[::-1, :].copy()
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                lbl = np.rot90(lbl, k=k, axes=(0, 1)).copy()
            jit = 1.0 + (np.random.random(4) - 0.5) * 0.2
            for b in range(4): x[b] *= jit[b]
        return torch.from_numpy(x), torch.from_numpy(lbl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v20_random_regions.json")
    p.add_argument("--s2-dir", type=Path, default=S2_DIR_DEFAULT)
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v24")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    regions = json.loads(args.regions_json.read_text())
    print(f"[1] {len(regions['train'])} train + {len(regions['test'])} test", flush=True)

    # Load DLTB
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

    # Load S2 cells + rasterize labels at S2 grid
    from rasterio.features import rasterize
    from shapely.geometry import box as shp_box
    from affine import Affine

    def rast_label(g, bbox, transform_arr, H, W):
        transform = Affine(*transform_arr.flatten()[:6])
        idx = list(g.sindex.intersection(tuple(bbox)))
        if not idx: return np.zeros((H, W), dtype=np.uint8)
        sub = g.iloc[idx].copy()
        sub["geometry"] = sub.geometry.intersection(shp_box(*bbox))
        sub = sub[~sub.geometry.is_empty]
        if len(sub) == 0: return np.zeros((H, W), dtype=np.uint8)
        sub["bin"] = np.where((sub["cid"]==1) | (sub["cid"]==2), 1, 2)
        shapes = [(geom, int(c)) for geom, c in zip(sub.geometry, sub["bin"])]
        return rasterize(shapes=shapes, out_shape=(H, W), transform=transform, fill=0, dtype="uint8")

    def loadsplit(rs, name):
        cells = []
        for r in rs:
            path = args.s2_dir / f"{r['county']}_{r['idx']}.npz"
            if not path.exists(): continue
            data = np.load(path)
            rgbnir = data["rgbnir"]; ndvi = data["ndvi"]
            H, W = rgbnir.shape[1], rgbnir.shape[2]
            label = rast_label(gdf[r["county"]], data["bbox"], data["transform"], H, W)
            if (label > 0).sum() < 100: continue
            cells.append({"rgbnir": rgbnir, "ndvi": ndvi, "label": label})
        print(f"  {name}: {len(cells)} cells", flush=True); return cells

    train_cells = loadsplit(regions["train"], "train")
    test_cells = loadsplit(regions["test"], "test")
    train_ds = S2CellDataset(train_cells, args.target_size, training=True)
    test_ds = S2CellDataset(test_cells, args.target_size, training=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size,
                                                shuffle=False, num_workers=2)

    print(f"\n[2] DINOv2-large + UNet decoder, 5-ch input adapter", flush=True)
    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    model = DinoUNet5ch(dinov2, num_classes=3, in_channels=5,
                        unfreeze_last_n=args.unfreeze_blocks).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params: {n_total/1e6:.1f}M  trainable: {n_train/1e6:.1f}M", flush=True)

    bin_counts = np.zeros(3, dtype=np.float64)
    for c in train_cells: bin_counts += np.bincount(c["label"].ravel(), minlength=3)
    cw = np.where(bin_counts > 0, 1.0/np.sqrt(bin_counts), 0.0).astype(np.float32)
    cw[0] = 0.0; cw = cw / cw.sum() * 2
    print(f"  class weights: {cw.round(3).tolist()}", flush=True)
    cw_t = torch.from_numpy(cw).to(device)

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and n.startswith("backbone")]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("backbone")]
    optim = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_f1 = -1; no_improve = 0
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0; n_b = 0; correct = 0; total = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                # Resize logits to label size if needed
                if logits.shape[-2:] != yb.shape[-2:]:
                    logits = F.interpolate(logits, size=yb.shape[-2:], mode="bilinear",
                                            align_corners=False)
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

        model = getattr(model, "eval")()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                if logits.shape[-2:] != yb.shape[-2:]:
                    logits = F.interpolate(logits, size=yb.shape[-2:], mode="bilinear",
                                            align_corners=False)
                p = logits.argmax(dim=1); v = yb > 0
                if v.any():
                    pi = (p==1) & v; ti = (yb==1) & v
                    tp += int((pi & ti).sum().item()); fp += int((pi & ~ti & v).sum().item())
                    fn += int((~pi & ti).sum().item()); tn += int((~pi & ~ti & v).sum().item())
        prec = tp/(tp+fp) if tp+fp else 0
        rec = tp/(tp+fn) if tp+fn else 0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
        iou = tp/(tp+fp+fn) if tp+fp+fn else 0
        acc = (tp+tn)/max(tp+fp+fn+tn,1)
        train_acc = correct / max(total, 1)
        print(f"  ep{ep+1}/{args.epochs}: loss={ep_loss/n_b:.4f} train_acc={train_acc:.3f} "
              f"| acc={acc:.3f} iou={iou:.3f} prec={prec:.3f} rec={rec:.3f} F1={f1:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if f1 > best_f1:
            best_f1 = f1; no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best F1={best_f1:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 10:
                print("[early stop]", flush=True); break
    print(f"\n[done] best F1={best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
