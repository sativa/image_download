"""DINOv3-Sat backbone (vs DINOv2-large) for 1m cropland — does an RS-domain foundation model win?

DINOv3 ViT-L pretrained on SAT-493M (Maxar 0.6 m satellite) — domain- and resolution-matched to our
1m imagery, unlike DINOv2-large (ImageNet). Same UNet decoder + boundary head + size-aware loss as
train_dino_1m_v2, ONLY the backbone changes -> isolates the backbone. Compare parcel-level to the
z17 best (dino_1m_v2_smallw, area 0.929). bf16. Run parcel_eval afterwards (ha-based).

API deltas vs DINOv2 (verified from HF model card; re-verify against the loaded model):
  - patch_embeddings at vision_model.embeddings.patch_embeddings.projection (Conv2d 3->1024, k16 s16)
  - drop CLS + 4 register tokens before reshaping to the patch grid: last_hidden_state[:, 5:, :]
  - RoPE position encoding -> variable resolution native, no interpolate_pos_encoding
  - patch_size 16 (crop 448 -> 28x28 grid -> decoder 16x upsample -> 448, exact)
"""
import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6
from train_dino_1m_v2 import C1mDSv2, edge_band, small_feature_w, load_ndvi_full

DINOV3_SAT = "/home/ps/landform/dinov3/dinov3-vitl16-sat493m"   # local dir after Mac download+transfer


def enhance6(x6):
    """High-resolution RS image enhancement on (6,H,W) uint8: per-RGB-triplet CLAHE (adaptive contrast
    in LAB-L) + unsharp masking (edge sharpening). Targets 1 m imagery / terrace-edge detail. Returns
    (6,H,W) uint8. Applied before norm6, consistently at train and eval time."""
    import cv2
    out = np.empty_like(x6)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for t in (0, 3):
        img = np.ascontiguousarray(np.transpose(x6[t:t + 3], (1, 2, 0)))   # HWC uint8 RGB
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        blur = cv2.GaussianBlur(img, (0, 0), 1.0)
        img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)                     # unsharp, amount 0.5
        out[t:t + 3] = np.transpose(img, (2, 0, 1))
    return out


class DinoV3UNet(nn.Module):
    def __init__(self, dinov3, num_classes=3, in_channels=6, unfreeze_last_n=4, n_register=4):
        super().__init__()
        self.backbone = dinov3
        self.nreg = n_register
        vm = getattr(self.backbone, "vision_model", self.backbone)
        self.vm = vm
        emb = vm.embeddings.patch_embeddings                              # Conv2d(3, D, 16, 16) — DINOv3 flat
        D = emb.out_channels
        new = nn.Conv2d(in_channels, D, kernel_size=emb.kernel_size, stride=emb.stride,
                        padding=emb.padding, bias=emb.bias is not None)
        with torch.no_grad():
            new.weight[:, :3] = emb.weight
            mean_rgb = emb.weight.mean(1, keepdim=True)
            for c in range(3, in_channels):
                new.weight[:, c:c + 1] = mean_rgb / (in_channels / 3)
            if emb.bias is not None:
                new.bias.copy_(emb.bias)
        vm.embeddings.patch_embeddings = new
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        # generically find the transformer block ModuleList, unfreeze last N
        blocks = None
        for _, mod in vm.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 12:
                blocks = mod; break
        if blocks is not None and unfreeze_last_n > 0:
            for blk in blocks[-unfreeze_last_n:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
        for p in new.parameters():
            p.requires_grad_(True)
        self.proj = nn.Conv2d(D, 256, 1)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True))
        self.classifier = nn.Conv2d(16, num_classes, 1)
        self.boundary_head = nn.Conv2d(16, 1, 1)
        self.gdlx_head = nn.Conv2d(16, 3, 1)            # aux: 0其他 / 1梯田 / 2坡地

    def _feat16(self, x):
        out = self.backbone(pixel_values=x)
        tok = out.last_hidden_state[:, 1 + self.nreg:, :]                 # drop CLS + register tokens
        B, N, D = tok.shape
        P = int(round(N ** 0.5))
        feat = tok.permute(0, 2, 1).reshape(B, D, P, P)
        h = self.proj(feat); h = self.up1(h); h = self.up2(h); h = self.up3(h); h = self.up4(h)
        return h

    def forward(self, x):
        h = self._feat16(x)
        return self.classifier(h), self.boundary_head(h), self.gdlx_head(h)


class ConvBNReLU(nn.Sequential):
    def __init__(self, ci, co, s=1, k=3):
        super().__init__(nn.Conv2d(ci, co, k, s, k // 2, bias=False), nn.BatchNorm2d(co), nn.ReLU(True))


class DinoV3FreqUNet(nn.Module):
    """DINOv3 lr grid + input multi-res hr branch, fused/upsampled by FreqFusion (frequency-aware,
    sharpens boundaries + small objects). Heads at crop/2; loss interpolates to full res."""
    def __init__(self, dinov3, num_classes=3, in_channels=6, unfreeze_last_n=4, n_register=4):
        super().__init__()
        from FreqFusion import FreqFusion
        self.backbone = dinov3; self.nreg = n_register
        vm = getattr(self.backbone, "vision_model", self.backbone); self.vm = vm
        emb = vm.embeddings.patch_embeddings; D = emb.out_channels
        new = nn.Conv2d(in_channels, D, kernel_size=emb.kernel_size, stride=emb.stride,
                        padding=emb.padding, bias=emb.bias is not None)
        with torch.no_grad():
            new.weight[:, :3] = emb.weight; m = emb.weight.mean(1, keepdim=True)
            for c in range(3, in_channels):
                new.weight[:, c:c + 1] = m / (in_channels / 3)
            if emb.bias is not None: new.bias.copy_(emb.bias)
        vm.embeddings.patch_embeddings = new
        for p in self.backbone.parameters(): p.requires_grad_(False)
        blocks = None
        for _, mod in vm.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 12: blocks = mod; break
        if blocks is not None and unfreeze_last_n > 0:
            for blk in blocks[-unfreeze_last_n:]:
                for p in blk.parameters(): p.requires_grad_(True)
        for p in new.parameters(): p.requires_grad_(True)
        C = 128
        self.proj = nn.Conv2d(D, C, 1)
        self.s1 = ConvBNReLU(in_channels, 64, s=2)   # input -> crop/2
        self.s2 = ConvBNReLU(64, C, s=2)             # -> crop/4
        self.s3 = ConvBNReLU(C, C, s=2)              # -> crop/8
        self.h_half = nn.Conv2d(64, C, 1)            # crop/2 hr -> C
        self.ff1 = FreqFusion(hr_channels=C, lr_channels=C)   # crop/16 -> crop/8
        self.ff2 = FreqFusion(hr_channels=C, lr_channels=C)   # crop/8  -> crop/4
        self.ff3 = FreqFusion(hr_channels=C, lr_channels=C)   # crop/4  -> crop/2
        self.classifier = nn.Conv2d(C, num_classes, 1)
        self.boundary_head = nn.Conv2d(C, 1, 1)
        self.gdlx_head = nn.Conv2d(C, 3, 1)

    def forward(self, x, return_feat=False):
        a = self.s1(x); b = self.s2(a); c = self.s3(b)        # crop/2, crop/4, crop/8
        hr8, hr4, hr2 = c, b, self.h_half(a)                  # all C channels
        out = self.backbone(pixel_values=x)
        tok = out.last_hidden_state[:, 1 + self.nreg:, :]
        B, N, Dd = tok.shape; P = int(round(N ** 0.5))
        lr = self.proj(tok.permute(0, 2, 1).reshape(B, Dd, P, P))   # C @ crop/16
        _, h, l = self.ff1(hr_feat=hr8, lr_feat=lr);  f = h + l     # crop/8
        _, h, l = self.ff2(hr_feat=hr4, lr_feat=f);   f = h + l     # crop/4
        _, h, l = self.ff3(hr_feat=hr2, lr_feat=f);   f = h + l     # crop/2
        cls, bnd, gd = self.classifier(f), self.boundary_head(f), self.gdlx_head(f)
        if return_feat:
            return cls, bnd, gd, tok.mean(1)                       # mean-pooled backbone feat for DANN domain head
        if hasattr(self, "frame_field_head"):                      # Frame Field Learning (4th output)
            return cls, bnd, gd, self.frame_field_head(f)
        return cls, bnd, gd


class DinoV3FreqUNetBD(DinoV3FreqUNet):
    """DinoV3FreqUNet + a dedicated higher-capacity BOUNDARY DECODER (2×ConvBNReLU before the edge head,
    vs the base 1×1 head) for sharper field delineation when trained on dense parcel edges
    (Gansu DLTB + Tibet FSDA). forward() is inherited (it just calls self.boundary_head)."""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        C = self.classifier.in_channels
        self.boundary_head = nn.Sequential(ConvBNReLU(C, C), ConvBNReLU(C, C), nn.Conv2d(C, 1, 1))


class DinoV3FreqUNetBDD(DinoV3FreqUNetBD):
    """BD + a DISTANCE-to-boundary head (ResUNet-a/BsiNet recipe). Reuses the 3rd output slot (gdlx_head)
    as a 1-channel distance regression -> forward returns (cls, bnd, dist). Distance-map peaks seed the
    watershed for clean dense-field instance separation (no instance cap, unlike YOLO)."""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        C = self.classifier.in_channels
        self.gdlx_head = nn.Sequential(ConvBNReLU(C, C), nn.Conv2d(C, 1, 1))   # 3rd output = distance


class DinoV3FreqUNetBDDF(DinoV3FreqUNetBDD):
    """BDD + a FRAME-FIELD head (Girard et al., CVPR'21 'Polygonal Building Extraction by Frame Field
    Learning'): per-pixel complex coeffs (c0, c2) of the frame-field polynomial f(z)=z^4 + c2·z^2 + c0,
    whose roots are the two orthogonal local edge directions. Supervised by DLTB polygon edge tangents,
    it guides polygonisation to REGULAR, wave-free, topology-clean polygons (vs raster marching-squares +
    Chaikin). forward -> (cls, bnd, dist, frame_field[B,4,H,W] = c0_re,c0_im,c2_re,c2_im)."""
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        C = self.classifier.in_channels
        self.frame_field_head = nn.Sequential(ConvBNReLU(C, C), nn.Conv2d(C, 4, 1), nn.Tanh())  # coeffs in [-1,1]


class DinoV3DySampleUNet(nn.Module):
    """DINOv3 grid -> DySample (point-sampling learnable upsampling, ICCV23) x4 -> full res.
    Single-input upsampler (no hr branch). Tests whether a newer/lighter upsampler beats FreqFusion."""
    def __init__(self, dinov3, num_classes=3, in_channels=6, unfreeze_last_n=4, n_register=4):
        super().__init__()
        from dysample import DySample
        self.backbone = dinov3; self.nreg = n_register
        vm = getattr(self.backbone, "vision_model", self.backbone); self.vm = vm
        emb = vm.embeddings.patch_embeddings; D = emb.out_channels
        new = nn.Conv2d(in_channels, D, kernel_size=emb.kernel_size, stride=emb.stride,
                        padding=emb.padding, bias=emb.bias is not None)
        with torch.no_grad():
            new.weight[:, :3] = emb.weight; m = emb.weight.mean(1, keepdim=True)
            for c in range(3, in_channels):
                new.weight[:, c:c + 1] = m / (in_channels / 3)
            if emb.bias is not None: new.bias.copy_(emb.bias)
        vm.embeddings.patch_embeddings = new
        for p in self.backbone.parameters(): p.requires_grad_(False)
        blocks = None
        for _, mod in vm.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 12: blocks = mod; break
        if blocks is not None and unfreeze_last_n > 0:
            for blk in blocks[-unfreeze_last_n:]:
                for p in blk.parameters(): p.requires_grad_(True)
        for p in new.parameters(): p.requires_grad_(True)
        self.proj = nn.Conv2d(D, 256, 1)
        chs = [256, 128, 64, 32]
        self.ups = nn.ModuleList([DySample(chs[i], scale=2, groups=4) for i in range(4)])
        self.convs = nn.ModuleList([ConvBNReLU(chs[i], chs[i + 1] if i < 3 else 16) for i in range(4)])
        self.classifier = nn.Conv2d(16, num_classes, 1)
        self.boundary_head = nn.Conv2d(16, 1, 1)
        self.gdlx_head = nn.Conv2d(16, 3, 1)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        tok = out.last_hidden_state[:, 1 + self.nreg:, :]
        B, N, Dd = tok.shape; P = int(round(N ** 0.5))
        h = self.proj(tok.permute(0, 2, 1).reshape(B, Dd, P, P))
        for up, cv in zip(self.ups, self.convs):
            h = cv(up(h))
        return self.classifier(h), self.boundary_head(h), self.gdlx_head(h)


class DinoV3PointRendUNet(nn.Module):
    """PointRend-lite: DINOv3 coarse decoder features + input-level fine features, fused by a per-pixel
    MLP ('point head' applied densely) to refine boundaries/small structures. Size-aware loss focuses
    the refinement on small parcels (approximating PointRend's hard-point sampling). Full point-sampling
    + iterative-subdivision version to follow once data is complete."""
    def __init__(self, dinov3, num_classes=3, in_channels=6, unfreeze_last_n=4, n_register=4):
        super().__init__()
        self.backbone = dinov3; self.nreg = n_register
        vm = getattr(self.backbone, "vision_model", self.backbone); self.vm = vm
        emb = vm.embeddings.patch_embeddings; D = emb.out_channels
        new = nn.Conv2d(in_channels, D, kernel_size=emb.kernel_size, stride=emb.stride,
                        padding=emb.padding, bias=emb.bias is not None)
        with torch.no_grad():
            new.weight[:, :3] = emb.weight; m = emb.weight.mean(1, keepdim=True)
            for c in range(3, in_channels):
                new.weight[:, c:c + 1] = m / (in_channels / 3)
            if emb.bias is not None: new.bias.copy_(emb.bias)
        vm.embeddings.patch_embeddings = new
        for p in self.backbone.parameters(): p.requires_grad_(False)
        blocks = None
        for _, mod in vm.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 12: blocks = mod; break
        if blocks is not None and unfreeze_last_n > 0:
            for blk in blocks[-unfreeze_last_n:]:
                for p in blk.parameters(): p.requires_grad_(True)
        for p in new.parameters(): p.requires_grad_(True)
        self.proj = nn.Conv2d(D, 256, 1)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True))
        self.fine = nn.Sequential(ConvBNReLU(in_channels, 32), ConvBNReLU(32, 32))   # full-res fine detail
        self.refine = nn.Sequential(nn.Conv2d(16 + 32, 64, 1), nn.ReLU(True), nn.Conv2d(64, 64, 1), nn.ReLU(True))
        self.classifier = nn.Conv2d(64, num_classes, 1)
        self.boundary_head = nn.Conv2d(64, 1, 1)
        self.gdlx_head = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        out = self.backbone(pixel_values=x)
        tok = out.last_hidden_state[:, 1 + self.nreg:, :]
        B, N, Dd = tok.shape; P = int(round(N ** 0.5))
        h = self.proj(tok.permute(0, 2, 1).reshape(B, Dd, P, P))
        h = self.up1(h); h = self.up2(h); h = self.up3(h); h = self.up4(h)   # coarse 16ch
        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)
        r = self.refine(torch.cat([h, self.fine(x)], 1))                     # per-pixel point MLP
        return self.classifier(r), self.boundary_head(r), self.gdlx_head(r)


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
                    lg = model(xb)[0]
                    if lg.shape[-2:] != (cs, cs):
                        lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                    pr = torch.softmax(lg.float(), 1)[0].cpu().numpy()
                acc[:, t:t + cs, l:l + cs] += pr; cnt[t:t + cs, l:l + cs] += 1
        pred = (acc / np.maximum(cnt, 1)).argmax(0)
        v = lbl > 0; ti = (lbl == 1) & v; pi = (pred == 1) & v
        tp += int((pi & ti).sum()); fp += int((pi & ~ti & v).sum()); fn += int((~pi & ti).sum())
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9), pr, rc


class C1mGdlxDS(C1mDSv2):
    """C1mDSv2 + a GDLX aux label (c_1m_gdlx/{name}.npy: 0其他/1梯田/2坡地), same crop & aug."""
    def __init__(self, names, dd, crop, training, gdlx_dir, multitemporal=False, enhance=False):
        super().__init__(names, dd, crop, training, multitemporal=multitemporal)
        self.gd = Path(gdlx_dir); self.enhance = enhance

    def __getitem__(self, i):
        import random
        nm = self.n[i]
        z = np.load(self.d / f"{nm}.npz")
        x6 = z["x6"]; lbl = z["label"].astype(np.int64); cs = self.c
        gf = self.gd / f"{nm}.npy"
        gl = np.load(gf).astype(np.int64) if gf.exists() else np.zeros(lbl.shape, np.int64)
        _, SZ, SZw = x6.shape
        if SZ < cs or SZw < cs:
            ph = max(0, cs - SZ); pw = max(0, cs - SZw)
            x6 = np.pad(x6, ((0, 0), (0, ph), (0, pw)), mode="edge")
            lbl = np.pad(lbl, ((0, ph), (0, pw))); gl = np.pad(gl, ((0, ph), (0, pw))); SZ, SZw = x6.shape[1:]
        if self.tr:
            t = random.randint(0, SZ - cs); l = random.randint(0, SZw - cs)
        else:
            t = (SZ - cs) // 2; l = (SZw - cs) // 2
        xc6 = x6[:, t:t + cs, l:l + cs]
        if getattr(self, "enhance", False): xc6 = enhance6(xc6)
        x = norm6(xc6)
        if self.mt:
            x = np.concatenate([x, self._ndvi_crop(nm, t, l, cs, SZ, SZw)], 0)
        lc = lbl[t:t + cs, l:l + cs]; gc = gl[t:t + cs, l:l + cs]
        if self.tr:
            if random.random() < 0.5: x = x[:, :, ::-1].copy(); lc = lc[:, ::-1].copy(); gc = gc[:, ::-1].copy()
            if random.random() < 0.5: x = x[:, ::-1, :].copy(); lc = lc[::-1, :].copy(); gc = gc[::-1, :].copy()
            k = random.randint(0, 3)
            if k: x = np.rot90(x, k, (1, 2)).copy(); lc = np.rot90(lc, k).copy(); gc = np.rot90(gc, k).copy()
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(lc)),
                torch.from_numpy(np.ascontiguousarray(gc)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone-dir", default=DINOV3_SAT)
    p.add_argument("--freqfusion", action="store_true", help="use FreqFusion decoder (DinoV3FreqUNet)")
    p.add_argument("--dysample", action="store_true", help="use DySample upsampling decoder (DinoV3DySampleUNet)")
    p.add_argument("--pointrend", action="store_true", help="use PointRend-lite refinement decoder (DinoV3PointRendUNet)")
    p.add_argument("--enhance", action="store_true", help="apply 1m RS image enhancement (CLAHE+unsharp) before norm")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/dino_1m_v3_sat")
    p.add_argument("--multitemporal", action="store_true")
    p.add_argument("--boundary-head", action="store_true")
    p.add_argument("--boundary-weight", type=float, default=0.3)
    p.add_argument("--small-weight", type=float, default=0.0)
    p.add_argument("--small-k", type=int, default=31)
    p.add_argument("--gdlx-head", action="store_true", help="add GDLX(梯田/坡地) multi-task aux head")
    p.add_argument("--gdlx-weight", type=float, default=0.3)
    p.add_argument("--gdlx-dir", default="/mnt/sda/zf/landform/data/c_1m_gdlx")
    p.add_argument("--crop", type=int, default=448)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--backbone-lr", type=float, default=5e-6)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--unfreeze", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--init-ckpt", default="", help="warm-start from a trained checkpoint (domain adaptation / fine-tune)")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    in_ch = 11 if a.multitemporal else 6

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    tr_names = man["train"][:a.max_train] if a.max_train else man["train"]; te_names = man["test"]
    print(f"[dino-v3-sat] in_ch={in_ch} train={len(tr_names)} test={len(te_names)}", flush=True)
    ds = (C1mGdlxDS(tr_names, a.data_dir, a.crop, True, a.gdlx_dir, multitemporal=a.multitemporal, enhance=a.enhance)
          if a.gdlx_head else C1mDSv2(tr_names, a.data_dir, a.crop, True, multitemporal=a.multitemporal))
    trl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                                      num_workers=a.workers, pin_memory=True, drop_last=True)

    from transformers import AutoModel
    dinov3 = AutoModel.from_pretrained(a.backbone_dir, local_files_only=True)
    Model = (DinoV3PointRendUNet if a.pointrend else DinoV3FreqUNet if a.freqfusion
             else DinoV3DySampleUNet if a.dysample else DinoV3UNet)
    model = Model(dinov3, num_classes=3, in_channels=in_ch, unfreeze_last_n=a.unfreeze).to(a.device)
    if a.init_ckpt:
        isd = torch.load(a.init_ckpt, map_location=a.device, weights_only=True); msd = model.state_dict()
        nload = model.load_state_dict({k: v for k, v in isd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
        print(f"  warm-start from {a.init_ckpt} (missing={len(nload.missing_keys)})", flush=True)
    print(f"  trainable={sum(q.numel() for q in model.parameters() if q.requires_grad)/1e6:.1f}M", flush=True)

    bc = np.zeros(3)
    for n in tr_names[:300]:
        bc += np.bincount(np.load(Path(a.data_dir) / f"{n}.npz")["label"].ravel(), minlength=3)
    cw = np.where(bc > 0, 1 / np.sqrt(bc), 0).astype(np.float32); cw[0] = 0; cw = cw / cw.sum() * 2
    cwt = torch.from_numpy(cw).to(a.device)
    gwt = None
    if a.gdlx_head:
        gc = np.zeros(3)
        for n in tr_names[:400]:
            gp = Path(a.gdlx_dir) / f"{n}.npy"
            if gp.exists(): gc += np.bincount(np.load(gp).ravel(), minlength=3)
        gw = np.where(gc > 0, 1 / np.sqrt(gc), 0).astype(np.float32); gw = gw / gw.sum() * 3
        gwt = torch.from_numpy(gw).to(a.device)
        print(f"  gdlx px%: 其他={gc[0]/gc.sum()*100:.1f} 梯田={gc[1]/gc.sum()*100:.1f} 坡地={gc[2]/gc.sum()*100:.1f}", flush=True)
    bb = [q for nm, q in model.named_parameters() if q.requires_grad and "backbone" in nm]
    hd = [q for nm, q in model.named_parameters() if q.requires_grad and "backbone" not in nm]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.backbone_lr}, {"params": hd, "lr": a.head_lr}], weight_decay=1e-4)
    total = a.epochs * len(trl); warm = max(50, total // 20)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    best = -1.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); el = 0.0; nb = 0
        for batch in trl:
            if a.gdlx_head:
                x, y, gy = batch; gy = gy.to(a.device)
            else:
                x, y = batch; gy = None
            x = x.to(a.device); y = y.to(a.device); opt.zero_grad()
            cropmask = (y == 1)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cls_lg, bnd_lg, gdlx_lg = model(x)
                if cls_lg.shape[-2:] != y.shape[-2:]:
                    cls_lg = F.interpolate(cls_lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
                    bnd_lg = F.interpolate(bnd_lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
                    gdlx_lg = F.interpolate(gdlx_lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
                if (y > 0).any():
                    if a.small_weight > 0:
                        ce = F.cross_entropy(cls_lg.float(), y, weight=cwt, ignore_index=0, reduction="none")
                        pw = small_feature_w(y, 3, a.small_k, a.small_weight)
                        loss = (ce * pw).sum() / ((pw * (y > 0).float()).sum() + 1e-6)
                    else:
                        loss = F.cross_entropy(cls_lg.float(), y, weight=cwt, ignore_index=0)
                else:
                    loss = cls_lg.float().sum() * 0.0
                if a.boundary_head:
                    bnd_t = edge_band(cropmask, 3).float(); vm = (y > 0).float()
                    bce = F.binary_cross_entropy_with_logits(bnd_lg[:, 0].float(), bnd_t, reduction="none")
                    loss = loss + a.boundary_weight * (bce * vm).sum() / (vm.sum() + 1e-6)
                if a.gdlx_head:
                    loss = loss + a.gdlx_weight * F.cross_entropy(gdlx_lg.float(), gy, weight=gwt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            el += loss.item(); nb += 1
        f1, pr, rc = full_eval(model, te_names, a.data_dir, a.device, a.crop, a.multitemporal)
        if f1 > best:
            best = f1; torch.save(model.state_dict(), out / "best.pt")
        print(f"  ep{ep+1}/{a.epochs} loss={el/nb:.4f} 1m-F1={f1:.4f}(P{pr:.3f}/R{rc:.3f}) best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), out / "last.pt")   # final ckpt too: pixel-F1-best != parcel-best
    print(f"\n[FINAL dino-v3-sat] best 1m-F1={best:.4f}  (DINOv2 smallw pixel 0.865 / parcel-area 0.929)", flush=True)
    json.dump({"best_1m_f1": best, "in_ch": in_ch}, open(out / "final.json", "w"))


if __name__ == "__main__":
    main()
