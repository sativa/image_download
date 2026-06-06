"""v31: Mask2Former + 5ch Sentinel-2 + 22K cells.

Strategy:
  - Backbone: HF Mask2Former pretrained on ADE20K semantic (Swin-L)
  - Adapt patch_embed (Swin) from 3 → 5 channels (RGB mean-init trick)
  - Replace classifier: 150 ADE classes → 3 classes (nodata/crop/other)
  - Loss: model's internal mask + class losses on our semantic GT
  - Eval: convert mask queries → per-pixel semantic logits
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
from train_v12_unet import DLBM_TO_CLASS

S2_MEAN = np.array([400, 460, 320, 1800], dtype=np.float32)
S2_STD = np.array([200, 200, 200, 700], dtype=np.float32)
NDVI_MEAN = 0.5; NDVI_STD = 0.3


class S2CellDataset(torch.utils.data.Dataset):
    """1 cell = 1 training sample. Pad/crop to target_size."""
    def __init__(self, cells, target_size=224, training=True):
        self.cells = cells; self.size = target_size; self.training = training

    def __len__(self): return len(self.cells)

    def __getitem__(self, idx):
        c = self.cells[idx]
        rgbnir = c["rgbnir"].astype(np.float32).copy()
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
                x = np.rot90(x, k=k, axes=(1,2)).copy()
                lbl = np.rot90(lbl, k=k, axes=(0,1)).copy()
        return torch.from_numpy(x), torch.from_numpy(lbl)


def patch_swin_5ch(model, in_channels=5):
    """Replace Swin patch_embed (3-ch → embed_dim) with 5-ch version."""
    # M2F: model.model.pixel_level_module.encoder.swin.embeddings.patch_embeddings.projection
    swin = model.model.pixel_level_module.encoder.swin
    pe = swin.embeddings.patch_embeddings.projection
    embed_dim = pe.out_channels
    ks, st = pe.kernel_size, pe.stride
    new_pe = nn.Conv2d(in_channels, embed_dim, kernel_size=ks, stride=st,
                        padding=pe.padding, bias=pe.bias is not None)
    with torch.no_grad():
        new_pe.weight[:, :3] = pe.weight
        mean_rgb = pe.weight.mean(dim=1, keepdim=True)
        for c in range(3, in_channels):
            new_pe.weight[:, c:c+1] = mean_rgb / (in_channels / 3)
        if pe.bias is not None:
            new_pe.bias.copy_(pe.bias)
    swin.embeddings.patch_embeddings.projection = new_pe
    # Update internal num_channels metadata
    swin.embeddings.patch_embeddings.num_channels = in_channels
    if hasattr(model.config, "backbone_config") and model.config.backbone_config is not None:
        model.config.backbone_config.num_channels = in_channels


def make_targets(label, num_classes, ignore_index=0):
    """Convert pixel semantic label → Mask2Former target format.

    Returns: (mask_labels: List[Tensor[N, H, W]], class_labels: List[Tensor[N]])
    For semantic seg, N = number of classes present (excluding ignore).
    """
    mask_labels = []
    class_labels = []
    for cls in range(1, num_classes):  # skip ignore_index=0
        if (label == cls).any():
            mask_labels.append((label == cls).float())
            class_labels.append(cls)
    if not mask_labels:  # no labels present, return empty
        return torch.zeros(0, *label.shape, device=label.device), \
               torch.zeros(0, dtype=torch.long, device=label.device)
    return torch.stack(mask_labels), torch.tensor(class_labels, dtype=torch.long,
                                                    device=label.device)


def m2f_to_semantic(class_queries_logits, masks_queries_logits, target_size):
    """Convert M2F mask queries → per-pixel semantic logits at target_size."""
    # class_queries_logits: (B, Q, num_classes + 1)  (last is "no object")
    # masks_queries_logits: (B, Q, H', W')
    # Per-pixel class prob = sum_q softmax(class)_c * sigmoid(mask)_hw
    cls_probs = class_queries_logits.softmax(dim=-1)[..., :-1]  # B, Q, C  (drop no-object)
    mask_probs = masks_queries_logits.sigmoid()                  # B, Q, h', w'
    # Resize masks to target_size
    if mask_probs.shape[-2:] != target_size:
        mask_probs = F.interpolate(mask_probs, size=target_size,
                                    mode="bilinear", align_corners=False)
    # Combine: B, C, H, W
    semantic = torch.einsum("bqc,bqhw->bchw", cls_probs, mask_probs)
    return semantic  # already softmax-like (sums close to 1 across classes when not no-object)


def evaluate(model, test_cells, device, target_size, batch_size=4, num_classes=3):
    model = getattr(model, "eval")()
    tp = fp = fn = tn = 0
    test_ds = S2CellDataset(test_cells, target_size=target_size, training=False)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
                                          shuffle=False, num_workers=2)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            outputs = model(pixel_values=xb)
            sem = m2f_to_semantic(outputs.class_queries_logits,
                                   outputs.masks_queries_logits,
                                   xb.shape[-2:])
            p = sem.argmax(dim=1); v = yb > 0
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
    acc = (tp+tn)/max(tp+fp+fn+tn,1)
    return {"acc": acc, "iou": iou, "precision": prec, "recall": rec, "f1": f1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v27_regions.json")
    p.add_argument("--s2-dir", type=Path, default=HOME / "data/v19_s2_raw")
    p.add_argument("--dltb-cache", type=Path, default=HOME / "data/v11_dltb")
    p.add_argument("--out-dir", type=Path, default=HOME / "results/v31")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--m2f-model", default="facebook/mask2former-swin-base-ade-semantic")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"[1] load Mask2Former {args.m2f_model}", flush=True)
    from transformers import Mask2FormerForUniversalSegmentation
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.m2f_model,
        num_labels=3,
        id2label={0: "nodata", 1: "crop", 2: "other"},
        label2id={"nodata": 0, "crop": 1, "other": 2},
        ignore_mismatched_sizes=True,
    )
    patch_swin_5ch(model, in_channels=5)
    model = model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_total/1e6:.1f}M", flush=True)

    print(f"[2] parallel load cells", flush=True)
    from fast_load_s2 import parallel_loadsplit
    regions = json.loads(args.regions_json.read_text())
    train_cells, sk_t = parallel_loadsplit(regions["train"], args.dltb_cache,
                                             args.s2_dir, max_workers=16)
    test_cells, sk_v = parallel_loadsplit(regions["test"], args.dltb_cache,
                                            args.s2_dir, max_workers=8)
    print(f"  train: {len(train_cells)} ({sk_t} skipped) | test: {len(test_cells)} ({sk_v} skipped)",
          flush=True)

    train_ds = S2CellDataset(train_cells, args.target_size, training=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_f1 = -1; no_improve = 0
    for ep in range(args.epochs):
        model.train()
        ep_loss = 0; n_b = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            # Build M2F targets
            mask_labels = []; class_labels = []
            for b in range(yb.shape[0]):
                ml, cl = make_targets(yb[b], num_classes=3, ignore_index=0)
                mask_labels.append(ml); class_labels.append(cl)
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(pixel_values=xb,
                                 mask_labels=mask_labels,
                                 class_labels=class_labels)
                loss = outputs.loss
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim); scaler.update()
            ep_loss += loss.item(); n_b += 1
        sched.step()

        m = evaluate(model, test_cells, device, target_size=args.target_size,
                     batch_size=args.batch_size)
        print(f"  ep{ep+1}/{args.epochs}: loss={ep_loss/max(n_b,1):.4f} "
              f"| acc={m['acc']:.3f} iou={m['iou']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f} F1={m['f1']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]; no_improve = 0
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            print(f"    ↑ new best F1={best_f1:.3f}", flush=True)
        else:
            no_improve += 1
            if no_improve >= 8: print("[early stop]", flush=True); break
    print(f"\n[done] best F1={best_f1:.3f}", flush=True)


if __name__ == "__main__":
    main()
