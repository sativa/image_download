"""v21: late-fusion two-stream model for z17 RGB + S2 RGBNIR+NDVI.

Why late fusion (not early concat at 1m grid as v20 did):
  - z17 RGB has sharp 1m boundaries; S2 has smooth 10m spectral signal.
  - Concatenating at 1m blurs S2 (bilinear upsample destroys spectral edges).
  - Concatenating at 10m blurs z17 (downsample destroys spatial detail).
  - SOLUTION: two encoders process each modality at its native scale,
    then fuse encoder bottleneck features.

Architecture:
  Stream RGB (z17, 448×448, 3-ch):
    smp encoder (efficientnet-b3) → 5 feature levels
  Stream S2  (S2, 56×56, 5-ch):  [native S2 cropped to match z17 footprint]
    smp encoder (efficientnet-b0) → 5 feature levels
  Fusion:
    At each decoder level, upsample S2 feature to z17 feature scale, concat,
    then 1x1 conv to merge channels.
  Decoder:
    standard UNet decoder over the fused feature pyramid
  Output:
    3-class segmentation at 448×448 (z17 1m grid)

Note: S2 footprint is set so that the same geographic bbox is covered. At z17,
1m × 448px = 448m; at 10m S2 that's ~45px. We extract 56×56 to give margin.
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
S2_MEAN = np.array([400, 460, 320, 1800], dtype=np.float32)
S2_STD = np.array([200, 200, 200, 700], dtype=np.float32)
NDVI_MEAN = 0.5
NDVI_STD = 0.3


class TwoStreamUNet(nn.Module):
    """z17 RGB encoder + S2 multi-spectral encoder fused at decoder bottleneck."""

    def __init__(self, rgb_backbone="efficientnet-b3", s2_backbone="efficientnet-b0",
                 num_classes=3):
        super().__init__()
        import segmentation_models_pytorch as smp
        from segmentation_models_pytorch.encoders import get_encoder

        self.rgb_encoder = get_encoder(
            rgb_backbone, in_channels=3, depth=5, weights="imagenet"
        )
        self.s2_encoder = get_encoder(
            s2_backbone, in_channels=5, depth=5, weights="imagenet"
        )

        rgb_chs = self.rgb_encoder.out_channels  # tuple of 6 (input + 5 stages)
        s2_chs = self.s2_encoder.out_channels    # tuple of 6

        # Per-stage 1x1 fusion: project (rgb_ch + s2_ch) → rgb_ch
        # so decoder can plug in as a normal UNet decoder.
        self.fusion = nn.ModuleList([
            nn.Conv2d(rgb_chs[i] + s2_chs[i], rgb_chs[i], kernel_size=1)
            for i in range(len(rgb_chs))
        ])

        # Reuse smp's UNet decoder definition with rgb feature channels.
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=rgb_chs,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_norm="batchnorm",
            add_center_block=False,
            attention_type=None,
        )
        self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, rgb, s2):
        # rgb: B, 3, Hr, Wr (e.g., 448×448 z17)
        # s2:  B, 5, Hs, Ws (e.g., 56×56 native S2)
        rgb_feats = self.rgb_encoder(rgb)   # list of 6 tensors
        s2_feats = self.s2_encoder(s2)

        # At each level, upsample s2 to rgb's spatial size, concat, fuse
        fused = []
        for i in range(len(rgb_feats)):
            r = rgb_feats[i]
            s = s2_feats[i]
            if s.shape[-2:] != r.shape[-2:]:
                s = F.interpolate(s, size=r.shape[-2:], mode="bilinear",
                                  align_corners=False)
            cat = torch.cat([r, s], dim=1)
            fused.append(self.fusion[i](cat))
        # Use smp decoder
        decoder_out = self.decoder(fused)
        return self.segmentation_head(decoder_out)


class TwoStreamTilesDataset(torch.utils.data.Dataset):
    TILE_RGB = 448

    def __init__(self, cells, stride=384, training=True):
        self.training = training
        self.items = []
        for cell in cells:
            H, W = cell["rgb"].shape[:2]
            for top in range(0, max(1, H - self.TILE_RGB) + 1, stride):
                top = min(top, H - self.TILE_RGB)
                for left in range(0, max(1, W - self.TILE_RGB) + 1, stride):
                    left = min(left, W - self.TILE_RGB)
                    self.items.append((cell, top, left))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cell, top, left = self.items[idx]
        # z17 RGB tile
        rgb = cell["rgb"][top:top+self.TILE_RGB, left:left+self.TILE_RGB].copy()
        lbl = cell["label"][top:top+self.TILE_RGB, left:left+self.TILE_RGB].copy()

        # Convert z17 tile bbox → S2 pixel coords
        # cell["s2_shape"] is the full-cell S2 size (e.g., 240×240)
        # cell["rgb_shape"] is the full-cell z17 size (e.g., 2560×2304)
        H, W = cell["rgb"].shape[:2]
        h2, w2 = cell["s2_rgbnir"].shape[1], cell["s2_rgbnir"].shape[2]
        # crop S2 to the corresponding region
        top_s = int(top * h2 / H); left_s = int(left * w2 / W)
        sz_s_h = int(self.TILE_RGB * h2 / H)  # ≈ 45
        sz_s_w = int(self.TILE_RGB * w2 / W)
        # pad to ensure a uniform tile size for batching (use 56)
        TARGET_S = 56
        s2_rgbnir = cell["s2_rgbnir"][:, top_s:top_s+sz_s_h, left_s:left_s+sz_s_w]
        s2_ndvi = cell["s2_ndvi"][top_s:top_s+sz_s_h, left_s:left_s+sz_s_w]
        # Pad/resize to TARGET_S × TARGET_S
        ph = TARGET_S - s2_rgbnir.shape[1]; pw = TARGET_S - s2_rgbnir.shape[2]
        if ph > 0 or pw > 0:
            s2_rgbnir = np.pad(s2_rgbnir, ((0,0),(0,max(0,ph)),(0,max(0,pw))), mode="edge")
            s2_ndvi = np.pad(s2_ndvi, ((0,max(0,ph)),(0,max(0,pw))), mode="edge")
        s2_rgbnir = s2_rgbnir[:, :TARGET_S, :TARGET_S]
        s2_ndvi = s2_ndvi[:TARGET_S, :TARGET_S]

        # Training augmentation: joint flip/rotate on all streams
        if self.training:
            if np.random.random() < 0.5:
                rgb = rgb[:, ::-1, :].copy(); lbl = lbl[:, ::-1].copy()
                s2_rgbnir = s2_rgbnir[:, :, ::-1].copy(); s2_ndvi = s2_ndvi[:, ::-1].copy()
            if np.random.random() < 0.5:
                rgb = rgb[::-1, :, :].copy(); lbl = lbl[::-1, :].copy()
                s2_rgbnir = s2_rgbnir[:, ::-1, :].copy(); s2_ndvi = s2_ndvi[::-1, :].copy()
            k = np.random.randint(4)
            if k:
                rgb = np.rot90(rgb, k=k, axes=(0,1)).copy()
                lbl = np.rot90(lbl, k=k, axes=(0,1)).copy()
                s2_rgbnir = np.rot90(s2_rgbnir, k=k, axes=(1,2)).copy()
                s2_ndvi = np.rot90(s2_ndvi, k=k, axes=(0,1)).copy()
            # RGB brightness jitter only
            rgb_f = rgb.astype(np.float32) / 255.0
            jit = 1.0 + (np.random.random() - 0.5) * 0.3
            rgb_f = np.clip(rgb_f * jit, 0, 1)
        else:
            rgb_f = rgb.astype(np.float32) / 255.0

        # Normalize
        rgb_n = (rgb_f - IMAGENET_MEAN) / IMAGENET_STD  # H,W,3
        rgb_t = torch.from_numpy(rgb_n.transpose(2, 0, 1)).float()  # 3,H,W

        s2_n = s2_rgbnir.astype(np.float32).copy()
        for b in range(4):
            s2_n[b] = (s2_n[b] - S2_MEAN[b]) / S2_STD[b]
        ndvi_n = (s2_ndvi.astype(np.float32) - NDVI_MEAN) / NDVI_STD
        s2_5 = np.concatenate([s2_n, ndvi_n[None, ...]], axis=0).astype(np.float32)
        s2_t = torch.from_numpy(s2_5)

        return rgb_t, s2_t, torch.from_numpy(lbl.astype(np.int64))


def load_cell(r, args, gdf):
    import rasterio
    bb = tuple(r["bbox"])
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
    if rgb is None: return None
    label = rasterise_dltb_binary(gdf, bb, transform, H, W)
    if (label > 0).sum() < 1000: return None
    s2_path = args.s2_dir / f"{r['county']}_{r['idx']}.npz"
    if not s2_path.exists(): return None
    data = np.load(s2_path)
    return {
        "rgb": rgb, "label": label,
        "s2_rgbnir": data["rgbnir"].astype(np.float32),
        "s2_ndvi": data["ndvi"].astype(np.float32),
        "name": f"{r['county']}_{r['idx']}",
    }


def evaluate(model, test_cells, device, batch_size=4):
    model = getattr(model, "eval")()
    tp = fp = fn = tn = 0
    test_ds = TwoStreamTilesDataset(test_cells, stride=448, training=False)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
    with torch.no_grad():
        for rgb, s2, lbl in loader:
            rgb = rgb.to(device); s2 = s2.to(device); lbl = lbl.to(device)
            logits = model(rgb, s2)
            p = logits.argmax(dim=1); v = lbl > 0
            if v.any():
                pi = (p == 1) & v; ti = (lbl == 1) & v
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
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v17_regions.json")
    p.add_argument("--z17-dir", type=Path, default=Z17_DIR)
    p.add_argument("--s2-dir", type=Path, default=S2_DIR)
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v21")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--stride", type=int, default=384)
    p.add_argument("--rgb-backbone", default="efficientnet-b3")
    p.add_argument("--s2-backbone", default="efficientnet-b0")
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

    train_ds = TwoStreamTilesDataset(train_cells, stride=args.stride, training=True)
    print(f"  train tiles: {len(train_ds)}", flush=True)

    print(f"\n[2] TwoStreamUNet: RGB={args.rgb_backbone}, S2={args.s2_backbone}", flush=True)
    model = TwoStreamUNet(rgb_backbone=args.rgb_backbone, s2_backbone=args.s2_backbone,
                          num_classes=3).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"  params: {n/1e6:.1f}M", flush=True)

    bin_counts = np.zeros(3, dtype=np.float64)
    for c in train_cells: bin_counts += np.bincount(c["label"].ravel(), minlength=3)
    cw = np.where(bin_counts > 0, 1.0 / np.sqrt(bin_counts), 0.0).astype(np.float32)
    cw[0] = 0.0; cw = cw / cw.sum() * 2
    print(f"  class weights: {cw.round(3).tolist()}", flush=True)
    cw_t = torch.from_numpy(cw).to(device)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    # NaN-resilient training:
    #   - LR warmup over first 300 iter
    #   - gradient clipping at norm 1.0
    #   - skip step if loss is NaN/Inf (don't propagate corruption)
    WARMUP_ITERS = 300
    base_lrs = [g["lr"] for g in optim.param_groups]
    iter_count = 0

    best_f1 = -1; no_improve = 0; PATIENCE = 12
    n_nan_skipped = 0
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0; n_b = 0; correct = 0; total = 0
        t0 = time.time()
        for rgb, s2, lbl in train_loader:
            rgb = rgb.to(device, non_blocking=True)
            s2 = s2.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            # LR warmup ramp
            if iter_count < WARMUP_ITERS:
                ratio = (iter_count + 1) / WARMUP_ITERS
                for g, base in zip(optim.param_groups, base_lrs):
                    g["lr"] = base * ratio
            iter_count += 1
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(rgb, s2)
                loss = F.cross_entropy(logits.float(), lbl, weight=cw_t, ignore_index=0)
            if not torch.isfinite(loss):
                n_nan_skipped += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim); scaler.update()
            ep_loss += loss.item(); n_b += 1
            with torch.no_grad():
                p = logits.argmax(dim=1); v = lbl > 0
                if v.any():
                    correct += int((p[v] == lbl[v]).sum().item())
                    total += int(v.sum().item())
        sched.step()
        train_acc = correct / max(total, 1)
        m = evaluate(model, test_cells, device, batch_size=args.batch_size)
        print(f"  ep{ep+1}/{args.epochs}: loss={ep_loss/max(n_b,1):.4f} train_acc={train_acc:.3f} "
              f"| acc={m['acc']:.3f} iou={m['iou']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} F1={m['f1']:.3f} skip_nan={n_nan_skipped} ({time.time()-t0:.0f}s)", flush=True)
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
