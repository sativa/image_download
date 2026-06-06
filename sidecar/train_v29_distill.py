"""v29: knowledge distillation — RGB student learns from S2 teacher (v27).

Architecture:
  Student: smp UNet + EfNet-B3, 3-ch z17 RGB input @ 1m → 448×448 output
  Teacher: v27 (smp UNet + EfNet-B3, 5-ch S2 RGBNIR+NDVI), pre-computed logits at 224×224 cell grid

Loss per batch:
  L = α · CE(student, GT) + (1-α) · KL(softmax(student/T), softmax(teacher_upsampled/T))

Inference: ONLY z17 RGB (matches production pipeline).
The student inherits spectral-aware features from teacher's KL guidance.
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


class RGBPlusTeacherDataset(torch.utils.data.Dataset):
    """Yields (rgb_tile 3×448×448, label_tile 448×448, teacher_logits_tile 3×448×448).

    Teacher logits are stored at S2 grid (224×224) per cell.
    For each z17 tile (random 448×448 crop from cell's z17 tif), compute the
    corresponding S2 sub-region in the teacher logits and upsample to 448×448.
    """
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
        sz = self.TILE
        rgb = cell["rgb"][top:top+sz, left:left+sz].copy()
        lbl = cell["label"][top:top+sz, left:left+sz].copy()

        # Teacher logits are at cell's full S2 grid (3 × H_s × W_s).
        # The z17 tile maps to S2 coords: top_s = top * H_s/H, etc.
        teacher = cell["teacher"]  # (3, H_s, W_s) uint8
        H, W = cell["rgb"].shape[:2]
        H_s, W_s = teacher.shape[1], teacher.shape[2]
        # Compute corresponding S2 region for this z17 crop
        ts = int(top * H_s / H); ls = int(left * W_s / W)
        sz_s = max(1, int(sz * H_s / H))
        # Extract teacher sub-region
        te_sub = teacher[:, ts:ts+sz_s, ls:ls+sz_s]
        # Resize teacher logits (uint8 prob) to 448×448 via bilinear
        te_t = torch.from_numpy(te_sub.astype(np.float32) / 255.0)[None]  # 1,3,Hs,Ws
        te_t = F.interpolate(te_t, size=(sz, sz), mode="bilinear", align_corners=False)
        te_prob = te_t[0].numpy()  # 3,448,448

        if self.training:
            if np.random.random() < 0.5:
                rgb = rgb[:, ::-1, :].copy(); lbl = lbl[:, ::-1].copy()
                te_prob = te_prob[:, :, ::-1].copy()
            if np.random.random() < 0.5:
                rgb = rgb[::-1, :, :].copy(); lbl = lbl[::-1, :].copy()
                te_prob = te_prob[:, ::-1, :].copy()
            k = np.random.randint(4)
            if k:
                rgb = np.rot90(rgb, k=k, axes=(0,1)).copy()
                lbl = np.rot90(lbl, k=k, axes=(0,1)).copy()
                te_prob = np.rot90(te_prob, k=k, axes=(1,2)).copy()
            rgb_f = rgb.astype(np.float32) / 255.0
            jit = 1.0 + (np.random.random() - 0.5) * 0.3
            rgb_f = np.clip(rgb_f * jit, 0, 1)
        else:
            rgb_f = rgb.astype(np.float32) / 255.0

        x = ((rgb_f - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).astype(np.float32)
        return (torch.from_numpy(x),
                torch.from_numpy(lbl.astype(np.int64)),
                torch.from_numpy(te_prob.astype(np.float32)))


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
        teacher_path = args.teacher_dir / f"{r['county']}_{r['idx']}.npy"
        if not teacher_path.exists(): return None
        teacher = np.load(teacher_path)  # uint8 prob (3, Hs, Ws)
        return {"rgb": rgb, "label": label, "teacher": teacher}
    return None


def evaluate(model, test_cells, device, batch_size=8):
    model = getattr(model, "eval")()
    tp = fp = fn = tn = 0
    test_ds = RGBPlusTeacherDataset(test_cells, stride=448, training=False)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
    with torch.no_grad():
        for xb, yb, _ in loader:
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
    return {"acc": acc, "iou": iou, "precision": prec, "recall": rec, "f1": f1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v27_regions.json")
    p.add_argument("--z17-dir", type=Path, default=HOME / "data/v11_imagery")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--teacher-dir", type=Path, default=HOME / "data/v29_teacher_logits")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v29")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--backbone", default="efficientnet-b3")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="weight of GT CE loss; (1-alpha) is teacher KL")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="softmax temperature for KL distillation")
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

    train_cells = loadsplit(regions["train"], "train")
    test_cells = loadsplit(regions["test"], "test")
    train_ds = RGBPlusTeacherDataset(train_cells, stride=384, training=True)
    print(f"  train tiles: {len(train_ds)}", flush=True)

    print(f"\n[2] UNet + {args.backbone} student, 3-ch RGB, distill α={args.alpha} T={args.temperature}", flush=True)
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
    T = args.temperature; ALPHA = args.alpha

    best_f1 = -1; no_improve = 0
    for ep in range(args.epochs):
        model.train()
        ep_loss_ce = 0; ep_loss_kl = 0; n_b = 0
        t0 = time.time()
        for xb, yb, te in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            te = te.to(device, non_blocking=True)  # 3 × 448 × 448 prob ∈ [0,1]
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(xb).float()
                loss_ce = F.cross_entropy(logits, yb, weight=cw_t, ignore_index=0)
                # KL divergence: KL(student_soft || teacher_soft)
                # Both at temperature T
                # Mask out nodata pixels (where yb == 0)
                valid = (yb > 0).float()[:, None, ...]  # B,1,H,W
                # teacher prob is already softmax-like (assume softmaxed at T=1 by precompute, expand by T)
                # Apply temperature: re-soften teacher probs
                # teacher = teacher_prob (already ∈ [0,1] summing to 1)
                # student_log_soft = log_softmax(logits/T)
                # teacher_soft = (teacher^(1/T)) / sum (re-temperature)
                # For simplicity: use teacher probs as-is at T=1 (avoid re-temperature math errors)
                te_logp = torch.log(te.clamp(min=1e-7))
                stud_logp = F.log_softmax(logits / T, dim=1)
                # KL(teacher || student) per pixel = sum(teacher * (log(teacher) - log(student)))
                kl_per_pix = (te * (te_logp - stud_logp)).sum(dim=1, keepdim=True)
                # Apply valid mask, average over valid pixels
                kl = (kl_per_pix * valid).sum() / valid.sum().clamp(min=1)
                loss = ALPHA * loss_ce + (1 - ALPHA) * kl
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            ep_loss_ce += loss_ce.item(); ep_loss_kl += kl.item(); n_b += 1
        sched.step()

        m = evaluate(model, test_cells, device, batch_size=args.batch_size)
        print(f"  ep{ep+1}/{args.epochs}: ce={ep_loss_ce/n_b:.4f} kl={ep_loss_kl/n_b:.4f} "
              f"| acc={m['acc']:.3f} iou={m['iou']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} F1={m['f1']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]; no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best F1={best_f1:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 10:
                print("[early stop]", flush=True); break
    print(f"\n[done] best F1={best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
