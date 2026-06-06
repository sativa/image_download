"""DINOv2-1m v2 — push the in-domain cropland ceiling past the 0.86 plateau.

Three additions over train_dino_1m.py (each toggleable; see EXPERIMENTS.md 2026-06 plan):
  (A1) --ignore-band B : at TRAIN time, ignore a B-px band around every cropland/non-cropland
       edge in the main CE loss. DLTB polygon rasterization at 1m has multi-pixel boundary
       error -> those edge pixels are label NOISE. Removing them lifts the label-noise ceiling.
  (A2) --boundary-head : a parallel Conv2d(16->1) head off the SAME decoder features predicts the
       thin parcel boundary (BCE). Multi-task regularization + crisper vectors (FTW 3-class / IDBT
       "semantic+boundary" both show this helps). The classifier learns clean interiors; the
       boundary head owns the edges.
  (A3) --multitemporal : append 5 yearly NDVI channels (v33, 10m, SAME bbox as c_1m -> just
       bilinear-resize the 74x74 window) -> in_channels 6->11. Adds phenology, which is genuinely
       NEW information (not more of the same) and targets the hard 耕地 vs 园地/草地 confusion.
       1m RGB still leads (6 of 11 ch); NDVI is auxiliary -> stays "1m-primary".

Eval is IDENTICAL to train_dino_1m.full_eval (all valid px, classifier branch only) so the
reported 1m-F1 is directly comparable to the 0.86 baseline. Warm-starts from dino_1m/best.pt.
"""
import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6, IMG_MEAN, IMG_STD
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2

NDVI_DIR = Path("/home/ps/landform/data/v33_ndvi_multitemporal")
NDVI_SCALE = 10000.0  # v33 ndvi_years is int16 NDVI*10000


# ----- multitemporal NDVI helper (v33 same-name, same-bbox as c_1m) -----
def load_ndvi_full(name, SZ, SZw):
    """Return (5,SZ,SZw) float32 NDVI resized to the 1m grid, normalized ~[-1,1]. None if missing."""
    f = NDVI_DIR / f"{name}.npz"
    if not f.exists():
        return None
    nd = np.clip(np.load(f)["ndvi_years"].astype(np.float32) / NDVI_SCALE, 0.0, 1.0)  # clamp nodata/extremes
    nd = np.nan_to_num((nd - 0.5) / 0.5, nan=0.0, posinf=1.0, neginf=-1.0)  # (5,74,74), ~[-1,1]
    t = torch.from_numpy(nd)[None]  # 1,5,74,74
    r = F.interpolate(t, size=(SZ, SZw), mode="bilinear", align_corners=False)[0]
    return r.numpy()


class C1mDSv2(torch.utils.data.Dataset):
    def __init__(self, names, dd, crop, training, multitemporal=False):
        self.n = names; self.d = Path(dd); self.c = crop; self.tr = training; self.mt = multitemporal

    def __len__(self):
        return len(self.n)

    def __getitem__(self, i):
        nm = self.n[i]
        z = np.load(self.d / f"{nm}.npz")
        x6 = z["x6"]; lbl = z["label"].astype(np.int64); cs = self.c
        _, SZ, SZw = x6.shape
        if SZ < cs or SZw < cs:
            ph = max(0, cs - SZ); pw = max(0, cs - SZw)
            x6 = np.pad(x6, ((0, 0), (0, ph), (0, pw)), mode="edge")
            lbl = np.pad(lbl, ((0, ph), (0, pw))); SZ, SZw = x6.shape[1:]
        if self.tr:
            t = random.randint(0, SZ - cs); l = random.randint(0, SZw - cs)
        else:
            t = (SZ - cs) // 2; l = (SZw - cs) // 2
        x = norm6(x6[:, t:t + cs, l:l + cs])                    # (6,cs,cs)
        if self.mt:
            ndvi = self._ndvi_crop(nm, t, l, cs, SZ, SZw)        # (5,cs,cs)
            x = np.concatenate([x, ndvi], 0)
        lc = lbl[t:t + cs, l:l + cs]
        if self.tr:
            if random.random() < 0.5: x = x[:, :, ::-1].copy(); lc = lc[:, ::-1].copy()
            if random.random() < 0.5: x = x[:, ::-1, :].copy(); lc = lc[::-1, :].copy()
            k = random.randint(0, 3)
            if k: x = np.rot90(x, k, (1, 2)).copy(); lc = np.rot90(lc, k).copy()
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(lc))

    def _ndvi_crop(self, nm, t, l, cs, SZ, SZw):
        f = NDVI_DIR / f"{nm}.npz"
        if not f.exists():
            return np.zeros((5, cs, cs), np.float32)
        nd = np.clip(np.load(f)["ndvi_years"].astype(np.float32) / NDVI_SCALE, 0.0, 1.0)
        nd = np.nan_to_num((nd - 0.5) / 0.5, nan=0.0, posinf=1.0, neginf=-1.0)   # (5,74,74), clamped
        H = nd.shape[1]
        H = nd.shape[1]
        # same bbox -> proportional sub-window, then resize to cs
        y0 = int(t / SZ * H); y1 = max(y0 + 1, int(np.ceil((t + cs) / SZ * H)))
        x0 = int(l / SZw * H); x1 = max(x0 + 1, int(np.ceil((l + cs) / SZw * H)))
        sub = torch.from_numpy(nd[:, y0:y1, x0:x1])[None]
        return F.interpolate(sub, size=(cs, cs), mode="bilinear", align_corners=False)[0].numpy()


class DinoUNetBoundary(nn.Module):
    """Wrap a DinoUNet5ch: reuse its decoder, add a parallel boundary head off the 16-ch features."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.boundary_head = nn.Conv2d(16, 1, 1)

    def _feat16(self, x):
        b = self.base
        out = b.backbone(pixel_values=x, interpolate_pos_encoding=True)
        tok = out.last_hidden_state[:, 1:, :]
        B, N, D = tok.shape
        P = int(round(N ** 0.5))
        feat = tok.permute(0, 2, 1).reshape(B, D, P, P)
        h = b.proj(feat); h = b.up1(h); h = b.up2(h); h = b.up3(h); h = b.up4(h)
        return h

    def forward(self, x):
        h = self._feat16(x)
        return self.base.classifier(h), self.boundary_head(h)  # (cls_logits, boundary_logit)


def edge_band(mask, k):
    """morphological gradient of a {0,1} mask via maxpool; returns bool band of half-width ~ (k-1)/2."""
    m = mask.float()[:, None]
    dil = F.max_pool2d(m, k, 1, k // 2)
    ero = -F.max_pool2d(-m, k, 1, k // 2)
    return ((dil - ero) > 0)[:, 0]


def small_feature_w(y, ncls, k, boost):
    """Per-pixel loss weight: boost pixels in SMALL/thin parcels (regions removed by a k-opening).
    Forces the model to spend capacity on small parcels (the unfiltered count-F1 bottleneck)."""
    w = torch.ones(y.shape, dtype=torch.float32, device=y.device)
    for c in range(1, ncls):
        m = (y == c).float()[:, None]
        ero = -F.max_pool2d(-m, k, 1, k // 2)            # erosion
        opened = F.max_pool2d(ero, k, 1, k // 2)          # dilation -> opening (drops features < ~k)
        w = w + boost * (m - opened).clamp(0)[:, 0]       # boost where parcel survived but opening didn't
    return w


@torch.no_grad()
def full_eval(model, names, dd, dev, cs=448, multitemporal=False):
    model.eval(); tp = fp = fn = 0
    for name in names:
        z = np.load(Path(dd) / f"{name}.npz"); x6 = z["x6"]; lbl = z["label"]
        _, SZ, SZw = x6.shape
        ndvi = load_ndvi_full(name, SZ, SZw) if multitemporal else None
        acc = np.zeros((3, SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
        ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
        if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
        if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
        for t in ys:
            for l in xs:
                xc = norm6(x6[:, t:t + cs, l:l + cs])
                if multitemporal:
                    nd = ndvi[:, t:t + cs, l:l + cs] if ndvi is not None else np.zeros((5, cs, cs), np.float32)
                    xc = np.concatenate([xc, nd], 0)
                xb = torch.from_numpy(xc).unsqueeze(0).to(dev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(xb)[0]                                  # classifier branch only
                    if lg.shape[-2:] != (cs, cs):
                        lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                    pr = torch.softmax(lg.float(), 1)[0].cpu().numpy()
                acc[:, t:t + cs, l:l + cs] += pr; cnt[t:t + cs, l:l + cs] += 1
        pred = (acc / np.maximum(cnt, 1)).argmax(0)
        v = lbl > 0; ti = (lbl == 1) & v; pi = (pred == 1) & v
        tp += int((pi & ti).sum()); fp += int((pi & ~ti & v).sum()); fn += int((~pi & ti).sum())
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9), pr, rc


def warm_start(base, ckpt, dev):
    """Load matching-shape keys from the 0.86 baseline (skips patch_embed if channel count differs)."""
    sd = torch.load(ckpt, map_location=dev, weights_only=True)
    cur = base.state_dict(); ok = 0
    for k, v in sd.items():
        if k in cur and cur[k].shape == v.shape:
            cur[k] = v; ok += 1
    base.load_state_dict(cur)
    print(f"  warm-start: loaded {ok}/{len(cur)} tensors from {Path(ckpt).name}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/dino_1m_v2")
    p.add_argument("--warm-start", default="/mnt/sda/zf/landform/results/dino_1m/best.pt")
    p.add_argument("--multitemporal", action="store_true")
    p.add_argument("--boundary-head", action="store_true")
    p.add_argument("--boundary-weight", type=float, default=0.3)
    p.add_argument("--ignore-band", type=int, default=0, help="px half-width of edge band ignored in main CE (0=off)")
    p.add_argument("--small-weight", type=float, default=0.0, help="loss boost for small-parcel pixels (0=off)")
    p.add_argument("--small-k", type=int, default=31, help="opening kernel: parcels narrower than ~k get boosted")
    p.add_argument("--crop", type=int, default=448)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--backbone-lr", type=float, default=5e-6)   # low: warm-start, stay near 0.86 basin
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--unfreeze", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    in_ch = 11 if a.multitemporal else 6

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    tr_names = man["train"]; te_names = man["test"]
    if a.max_train: tr_names = tr_names[:a.max_train]
    print(f"[dino-1m-v2] in_ch={in_ch} mt={a.multitemporal} bnd={a.boundary_head} "
          f"ignore_band={a.ignore_band} train={len(tr_names)} test={len(te_names)}", flush=True)

    trl = torch.utils.data.DataLoader(
        C1mDSv2(tr_names, a.data_dir, a.crop, True, multitemporal=a.multitemporal),
        batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)

    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    base = DinoUNet5ch(dinov2, num_classes=3, in_channels=in_ch, unfreeze_last_n=a.unfreeze)
    if a.warm_start and Path(a.warm_start).exists():
        warm_start(base, a.warm_start, "cpu")
    model = DinoUNetBoundary(base).to(a.device)
    nt = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"  trainable={nt/1e6:.1f}M", flush=True)

    bc = np.zeros(3)
    for n in tr_names[:300]:
        bc += np.bincount(np.load(Path(a.data_dir) / f"{n}.npz")["label"].ravel(), minlength=3)
    cw = np.where(bc > 0, 1 / np.sqrt(bc), 0).astype(np.float32); cw[0] = 0; cw = cw / cw.sum() * 2
    cwt = torch.from_numpy(cw).to(a.device)

    bb = [q for nm, q in model.named_parameters() if q.requires_grad and "backbone" in nm]
    hd = [q for nm, q in model.named_parameters() if q.requires_grad and "backbone" not in nm]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.backbone_lr},
                             {"params": hd, "lr": a.head_lr}], weight_decay=1e-4)
    total_steps = a.epochs * len(trl); warmup = max(50, total_steps // 20)
    def lr_lambda(s):                                                 # warmup -> cosine, PER STEP
        if s < warmup:
            return (s + 1) / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    # bf16 (not fp16): wide dynamic range -> no activation overflow -> no BN-stat NaN. No GradScaler.

    best = -1.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); el = eb = 0.0; nb = 0
        for x, y in trl:
            x = x.to(a.device); y = y.to(a.device); opt.zero_grad()
            ym = y
            cropmask = (y == 1)
            if a.ignore_band > 0:
                band = edge_band(cropmask | (y == 2), 2 * a.ignore_band + 1) & (y > 0)
                ym = y.clone(); ym[band] = 0                          # ignore noisy edge px in main CE
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cls_lg, bnd_lg = model(x)
                if cls_lg.shape[-2:] != y.shape[-2:]:
                    cls_lg = F.interpolate(cls_lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
                    bnd_lg = F.interpolate(bnd_lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
                if (ym > 0).any():                                   # guard: empty target -> CE=nan
                    if a.small_weight > 0:
                        ce = F.cross_entropy(cls_lg.float(), ym, weight=cwt, ignore_index=0, reduction="none")
                        pw = small_feature_w(ym, 3, a.small_k, a.small_weight)   # 0 at ignored px (ce=0 there)
                        loss = (ce * pw).sum() / ((pw * (ym > 0).float()).sum() + 1e-6)
                    else:
                        loss = F.cross_entropy(cls_lg.float(), ym, weight=cwt, ignore_index=0)
                else:
                    loss = cls_lg.float().sum() * 0.0
                bl = torch.zeros((), device=a.device)
                if a.boundary_head:
                    bnd_t = edge_band(cropmask, 3).float()           # thin crisp boundary target
                    vmask = (y > 0).float()
                    bce = F.binary_cross_entropy_with_logits(bnd_lg[:, 0].float(), bnd_t, reduction="none")
                    bl = (bce * vmask).sum() / (vmask.sum() + 1e-6)
                    loss = loss + a.boundary_weight * bl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            el += loss.item(); eb += float(bl.detach()); nb += 1
        f1, pr, rc = full_eval(model, te_names, a.data_dir, a.device, a.crop, a.multitemporal)
        if f1 > best:
            best = f1; torch.save(model.state_dict(), out / "best.pt")
        print(f"  ep{ep+1}/{a.epochs} loss={el/nb:.4f} bnd={eb/nb:.4f} 1m-F1={f1:.4f} "
              f"(P{pr:.3f}/R{rc:.3f}) best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n[FINAL dino-1m-v2] best 1m-F1={best:.4f}  (baseline 0.860)", flush=True)
    json.dump({"best_1m_f1": best, "in_ch": in_ch, "multitemporal": a.multitemporal,
               "boundary_head": a.boundary_head, "ignore_band": a.ignore_band},
              open(out / "final.json", "w"))


if __name__ == "__main__":
    main()
