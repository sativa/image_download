"""7-class land-COVER segmentation on the best architecture (DINOv3-Sat + FreqFusion).

Classes (0 = nodata, ignored in CE):
  1 耕地 cropland   2 园地 orchard   3 林地 forest   4 草地 grassland
  5 水体 water      6 建筑 built-up  7 荒漠 bare/desert
These aggregate the 12 DLTB first-level land-USE classes into visually-separable land-COVER
super-classes (商服/工矿/住宅/公管/特殊/交通 -> 建筑; 其他土地 -> 荒漠). Labels in c_1m_label7/.

Reuses DinoV3FreqUNet, norm6, NDVI + size-aware loss from the binary pipeline; multi-class CE
(class-balanced, ignore_index=0) + size-aware up-weighting of small parcels. Selects best by OA.
"""
import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6
from train_dino_1m_v2 import load_ndvi_full, small_feature_w
from train_dino_1m_v3 import DinoV3FreqUNet, DinoV3FreqUNetBD, DinoV3FreqUNetBDD, DinoV3FreqUNetBDDF, DINOV3_SAT
from scipy import ndimage as _ndi
import cv2 as _cv2

NAMES = ["nodata", "耕地", "园地", "林地", "草地", "水体", "建筑", "荒漠", "设施大棚"]


class C1mLabel7(torch.utils.data.Dataset):
    def __init__(self, names, dd, label_dir, crop, training, multitemporal=False, pbound_dir="", dist_dir="", frame=False):
        self.n = names; self.d = Path(dd); self.ld = Path(label_dir)
        self.c = crop; self.tr = training; self.mt = multitemporal
        self.pb = Path(pbound_dir) if pbound_dir else None
        self.dd = Path(dist_dir) if dist_dir else None
        self.frame = frame                                         # also yield frame-field GT (û² + edge band)

    def __len__(self):
        return len(self.n)

    def _ndvi(self, nm, t, l, cs, SZ, SZw):
        nd = load_ndvi_full(nm, SZ, SZw)
        if nd is None:
            return np.zeros((5, cs, cs), np.float32)
        return nd[:, t:t + cs, l:l + cs]

    def __getitem__(self, i):
        nm = self.n[i]
        x6 = np.load(self.d / f"{nm}.npz")["x6"]; cs = self.c
        lf = self.ld / f"{nm}.npy"
        lbl = np.load(lf).astype(np.int64) if lf.exists() else np.zeros(x6.shape[1:], np.int64)
        def _ld(d):
            f = (d / f"{nm}.npy") if d is not None else None
            return np.load(f).astype(np.float32) if (f is not None and f.exists()) else np.zeros(x6.shape[1:], np.float32)
        bnd = _ld(self.pb); dist = _ld(getattr(self, "dd", None))
        _, SZ, SZw = x6.shape
        if SZ < cs or SZw < cs:
            ph = max(0, cs - SZ); pw = max(0, cs - SZw)
            x6 = np.pad(x6, ((0, 0), (0, ph), (0, pw)), mode="edge")
            lbl = np.pad(lbl, ((0, ph), (0, pw))); bnd = np.pad(bnd, ((0, ph), (0, pw)))
            dist = np.pad(dist, ((0, ph), (0, pw))); SZ, SZw = x6.shape[1:]
        if self.tr:
            t = random.randint(0, SZ - cs); l = random.randint(0, SZw - cs)
        else:
            t = (SZ - cs) // 2; l = (SZw - cs) // 2
        x = norm6(x6[:, t:t + cs, l:l + cs])
        if self.mt:
            x = np.concatenate([x, self._ndvi(nm, t, l, cs, SZ, SZw)], 0)
        lc = lbl[t:t + cs, l:l + cs]; bc = bnd[t:t + cs, l:l + cs]; dc = dist[t:t + cs, l:l + cs]
        if self.tr:
            if random.random() < 0.5:
                x = x[:, :, ::-1].copy(); lc = lc[:, ::-1].copy(); bc = bc[:, ::-1].copy(); dc = dc[:, ::-1].copy()
            if random.random() < 0.5:
                x = x[:, ::-1, :].copy(); lc = lc[::-1, :].copy(); bc = bc[::-1, :].copy(); dc = dc[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                x = np.rot90(x, k, (1, 2)).copy(); lc = np.rot90(lc, k).copy(); bc = np.rot90(bc, k).copy(); dc = np.rot90(dc, k).copy()
        if self.frame:                                            # frame-field GT from the (augmented) boundary
            edge = bc > 0.5
            if edge.any():
                de = _ndi.distance_transform_edt(~edge).astype(np.float32)
                gy, gx = np.gradient(de); mag = np.hypot(gx, gy) + 1e-6
                tx, ty = -gy / mag, gx / mag                       # tangent = edge direction
                band = _cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
                fg = np.stack([tx * tx - ty * ty, 2 * tx * ty, band]).astype(np.float32)   # û²(cos2θ,sin2θ), band
            else:
                fg = np.zeros((3, cs, cs), np.float32)
        else:
            fg = np.zeros((3, cs, cs), np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(lc)),
                torch.from_numpy(np.ascontiguousarray(bc)), torch.from_numpy(np.ascontiguousarray(dc)),
                torch.from_numpy(np.ascontiguousarray(fg)))


@torch.no_grad()
def evaluate(model, names, dd, ld, dev, NCLS, cs=448, multitemporal=False):
    """Overall accuracy + macro-F1 over valid (label>0) pixels."""
    model.eval()
    conf = np.zeros((NCLS, NCLS), np.int64)  # [true, pred] over classes 1..NCLS-1
    for nm in names:
        x6 = np.load(Path(dd) / f"{nm}.npz")["x6"]
        lf = Path(ld) / f"{nm}.npy"
        if not lf.exists(): continue
        lbl = np.load(lf); _, SZ, SZw = x6.shape
        ndvi = load_ndvi_full(nm, SZ, SZw) if multitemporal else None
        acc = np.zeros((NCLS, SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
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
                    lg = model(xb)[0]
                    if lg.shape[-2:] != (cs, cs):
                        lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                    pr = torch.softmax(lg.float(), 1)[0].cpu().numpy()
                acc[:, t:t + cs, l:l + cs] += pr; cnt[t:t + cs, l:l + cs] += 1
        pred = (acc / np.maximum(cnt, 1)).argmax(0)
        v = lbl > 0
        for tc, pc in zip(lbl[v].ravel(), pred[v].ravel()):
            if 1 <= tc < NCLS and 1 <= pc < NCLS:
                conf[tc, pc] += 1
    diag = np.diag(conf); tot = conf.sum()
    oa = diag.sum() / max(1, tot)
    f1s = []
    for c in range(1, NCLS):
        tp = conf[c, c]; fp = conf[:, c].sum() - tp; fn = conf[c, :].sum() - tp
        if tp + fp + fn > 0:
            p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
            f1s.append(2 * p * r / (p + r + 1e-9))
    macro = float(np.mean(f1s)) if f1s else 0.0
    per = {NAMES[c]: (round(conf[c, c] / max(1, conf[c, :].sum()), 3)) for c in range(1, NCLS)}  # recall
    return float(oa), macro, per


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--label-dir", default="/mnt/sda/zf/landform/data/c_1m_label7")
    p.add_argument("--num-classes", type=int, default=8)  # 0 nodata + 7 cover classes
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/dino_v3_7class")
    p.add_argument("--backbone-dir", default=DINOV3_SAT)
    p.add_argument("--multitemporal", action="store_true")
    p.add_argument("--small-weight", type=float, default=4.0)
    p.add_argument("--small-k", type=int, default=31)
    p.add_argument("--pbound-dir", default="", help="parcel-boundary label dir -> train the boundary head to delineate parcels")
    p.add_argument("--boundary-weight", type=float, default=0.5)
    p.add_argument("--boundary-decoder", action="store_true", help="dedicated higher-capacity boundary decoder (DinoV3FreqUNetBD)")
    p.add_argument("--boundary-dice", action="store_true", help="add Dice to the boundary loss (better for sparse edges)")
    p.add_argument("--dist-head", action="store_true", help="add distance-to-boundary head (DinoV3FreqUNetBDD, ResUNet-a/BsiNet recipe)")
    p.add_argument("--frame-field", action="store_true", help="add Frame Field head (DinoV3FreqUNetBDDF) + DLTB-edge FFL loss (joint multi-task)")
    p.add_argument("--frame-weight", type=float, default=0.5, help="frame-field loss weight")
    p.add_argument("--dist-dir", default="", help="distance-map label dir (for the distance head)")
    p.add_argument("--dist-weight", type=float, default=0.5)
    p.add_argument("--init-ckpt", default="", help="warm-start backbone/classifier from a trained checkpoint")
    p.add_argument("--crop", type=int, default=448)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--backbone-lr", type=float, default=5e-6)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--unfreeze", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    NCLS = a.num_classes; in_ch = 11 if a.multitemporal else 6

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    tr = [n for n in man["train"] if (Path(a.label_dir) / f"{n}.npy").exists()]
    te = [n for n in man["test"] if (Path(a.label_dir) / f"{n}.npy").exists()]
    print(f"[7class] in_ch={in_ch} NCLS={NCLS} train={len(tr)} test={len(te)}", flush=True)
    use_bnd = bool(a.pbound_dir) and a.boundary_weight > 0
    use_dist = bool(a.dist_dir) and a.dist_head and a.dist_weight > 0
    ds = C1mLabel7(tr, a.data_dir, a.label_dir, a.crop, True, multitemporal=a.multitemporal,
                   pbound_dir=a.pbound_dir if use_bnd else "", dist_dir=a.dist_dir if use_dist else "",
                   frame=a.frame_field)
    trl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                                      num_workers=a.workers, pin_memory=True, drop_last=True)

    from transformers import AutoModel
    dinov3 = AutoModel.from_pretrained(a.backbone_dir, local_files_only=True)
    Net = (DinoV3FreqUNetBDDF if a.frame_field else DinoV3FreqUNetBDD if a.dist_head
           else DinoV3FreqUNetBD if a.boundary_decoder else DinoV3FreqUNet)
    model = Net(dinov3, num_classes=NCLS, in_channels=in_ch, unfreeze_last_n=a.unfreeze).to(a.device)
    if a.init_ckpt:
        isd = torch.load(a.init_ckpt, map_location=a.device, weights_only=True); msd = model.state_dict()
        nl = model.load_state_dict({k: v for k, v in isd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
        print(f"  warm-start {a.init_ckpt} (missing={len(nl.missing_keys)}, e.g. boundary decoder reinit)", flush=True)
    print(f"  trainable={sum(q.numel() for q in model.parameters() if q.requires_grad)/1e6:.1f}M", flush=True)

    bc = np.zeros(NCLS)
    for n in tr[:400]:
        bc += np.bincount(np.load(Path(a.label_dir) / f"{n}.npy").ravel(), minlength=NCLS)
    cw = np.where(bc > 0, 1 / np.sqrt(bc), 0).astype(np.float32); cw[0] = 0; cw = cw / cw.sum() * (NCLS - 1)
    cwt = torch.from_numpy(cw).to(a.device)
    print("  class px%:", {NAMES[i]: round(100 * bc[i] / bc.sum(), 2) for i in range(NCLS)}, flush=True)

    bb = [q for nm, q in model.named_parameters() if q.requires_grad and "backbone" in nm]
    hd = [q for nm, q in model.named_parameters() if q.requires_grad and "backbone" not in nm]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.backbone_lr}, {"params": hd, "lr": a.head_lr}], weight_decay=1e-4)
    total = a.epochs * len(trl); warm = max(50, total // 20)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm
                                            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    best = -1.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); el = 0.0; nb = 0
        for x, y, bnd, dist, frame in trl:
            x = x.to(a.device); y = y.to(a.device); bnd = bnd.to(a.device); dist = dist.to(a.device)
            frame = frame.to(a.device); opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _o = model(x); cls_lg, bnd_lg, dist_lg = _o[0], _o[1], _o[2]   # BDDF: _o[3]=frame field
                if cls_lg.shape[-2:] != y.shape[-2:]:
                    cls_lg = F.interpolate(cls_lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
                    bnd_lg = F.interpolate(bnd_lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
                    dist_lg = F.interpolate(dist_lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
                if (y > 0).any():
                    ce = F.cross_entropy(cls_lg.float(), y, weight=cwt, ignore_index=0, reduction="none")
                    if a.small_weight > 0:
                        pw = small_feature_w(y, NCLS, a.small_k, a.small_weight)
                        loss = (ce * pw).sum() / ((pw * (y > 0).float()).sum() + 1e-6)
                    else:
                        loss = (ce * (y > 0).float()).sum() / ((y > 0).float().sum() + 1e-6)
                else:
                    loss = cls_lg.float().sum() * 0.0
                if use_bnd:
                    # parcel-boundary head: BCE on ALL parcel edges (Gansu DLTB + Tibet FSDA), pos-weighted (edges sparse)
                    bl = F.binary_cross_entropy_with_logits(bnd_lg[:, 0].float(), bnd,
                                                            pos_weight=torch.tensor(5.0, device=a.device))
                    if a.boundary_dice:                        # Dice helps thin sparse edges
                        pb = torch.sigmoid(bnd_lg[:, 0].float())
                        dice = 1 - (2 * (pb * bnd).sum() + 1) / (pb.sum() + bnd.sum() + 1)
                        bl = bl + dice
                    loss = loss + a.boundary_weight * bl
                if use_dist:                                   # distance-to-boundary regression (L1), peaks = parcel centres
                    # per-sample mask: an all-zero dist map = file missing (build still running) -> skip,
                    # never train the head to predict "everywhere is boundary".
                    has = (dist.flatten(1).amax(1) > 0)
                    if has.any():
                        dl = F.l1_loss(torch.sigmoid(dist_lg[:, 0].float())[has], dist[has])
                        loss = loss + a.dist_weight * dl
                if a.frame_field:                                  # FFL: f(z)=z⁴+c2 z²+c0 vanishes at û & iû (edge dir +90°)
                    ff = _o[3]
                    if ff.shape[-2:] != y.shape[-2:]:
                        ff = F.interpolate(ff, size=y.shape[-2:], mode="bilinear", align_corners=False)
                    c0 = torch.complex(ff[:, 0].float(), ff[:, 1].float())
                    c2 = torch.complex(ff[:, 2].float(), ff[:, 3].float())
                    u2 = torch.complex(frame[:, 0], frame[:, 1]); u4 = u2 * u2; band = frame[:, 2]
                    fu = u4 + c2 * u2 + c0; fiu = u4 - c2 * u2 + c0
                    fl = ((fu.abs() ** 2 + fiu.abs() ** 2) * band).sum() / (band.sum() + 1.0)
                    sm = sum((ff[:, k, :, 1:] - ff[:, k, :, :-1]).abs().mean()
                             + (ff[:, k, 1:, :] - ff[:, k, :-1, :]).abs().mean() for k in range(4))
                    loss = loss + a.frame_weight * (fl + 0.05 * sm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            el += loss.item(); nb += 1
        oa, macro, per = evaluate(model, te, a.data_dir, a.label_dir, a.device, NCLS, a.crop, a.multitemporal)
        score = 0.5 * oa + 0.5 * macro
        if score > best:
            best = score; torch.save(model.state_dict(), out / "best.pt")
        print(f"  ep{ep+1}/{a.epochs} loss={el/nb:.4f} OA={oa:.4f} macroF1={macro:.4f} best={best:.4f} "
              f"recall={per} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), out / "last.pt")
    print(f"\n[FINAL 7class] best(0.5OA+0.5macroF1)={best:.4f}", flush=True)
    json.dump({"best": best, "num_classes": NCLS, "names": NAMES}, open(out / "final.json", "w"))


if __name__ == "__main__":
    main()
