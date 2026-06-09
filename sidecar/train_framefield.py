"""Frame Field Learning (Girard et al., CVPR'21) on DINOv3-Sat, fine-tuned with DLTB polygon edges.

Adds a frame-field head to dino_v3_bdd (-> DinoV3FreqUNetBDDF) and trains it so the per-pixel complex
coeffs (c0, c2) of the polynomial f(z)=z^4 + c2 z^2 + c0 have the local DLTB EDGE DIRECTION as a root
(and its 90° rotation). The learned frame field then guides polygonisation to regular, wave-free,
topology-clean polygons — fixing the wavy-line / staircase artefact at its source (learned, not post-hoc).

GT frame direction = tangent of the DLTB boundary (c_1m_pbound): distance-transform gradient = normal,
rotate 90° = tangent; stored as û² = e^{i·2θ} (cos2θ, sin2θ). align loss only on a dilated edge band.

Warm-starts dino_v3_bdd; freezes everything but the frame-field head (fast). Run on .250 GPU."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from scipy import ndimage as ndi

HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6
from train_dino_1m_v2 import load_ndvi_full
from train_dino_1m_v3 import DinoV3FreqUNetBDDF, DINOV3_SAT


class FrameDS(torch.utils.data.Dataset):
    def __init__(self, names, data_dir, pbound_dir, crop=448, mt=True):
        self.n = names; self.d = Path(data_dir); self.pb = Path(pbound_dir); self.c = crop; self.mt = mt

    def __len__(self):
        return len(self.n)

    def __getitem__(self, i):
        nm = self.n[i]; cs = self.c
        x6 = np.load(self.d / f"{nm}.npz")["x6"]
        pf = self.pb / f"{nm}.npy"
        edge_full = (np.load(pf) > 0) if pf.exists() else np.zeros(x6.shape[1:], bool)
        _, H, W = x6.shape
        t = np.random.randint(0, max(1, H - cs)); l = np.random.randint(0, max(1, W - cs))
        x = norm6(x6[:, t:t + cs, l:l + cs])
        if self.mt:
            x = np.concatenate([x, np.zeros((5, cs, cs), np.float32)], 0)   # frame field needs no NDVI -> zero-fill
            # (matches parcel_dist deployment, which also zero-fills NDVI; removes the disk-IO bottleneck)
        edge = edge_full[t:t + cs, l:l + cs]
        # frame-field GT: tangent of the boundary -> û² = e^{i 2θ}
        dist = ndi.distance_transform_edt(~edge) if edge.any() else np.zeros((cs, cs), np.float32)
        gy, gx = np.gradient(dist.astype(np.float32))
        mag = np.hypot(gx, gy) + 1e-6
        tx, ty = -gy / mag, gx / mag                               # tangent = normal rotated 90°
        u2c = (tx * tx - ty * ty).astype(np.float32)               # cos 2θ
        u2s = (2 * tx * ty).astype(np.float32)                     # sin 2θ
        band = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # align only near edges
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(u2c), torch.from_numpy(u2s),
                torch.from_numpy(band.astype(np.float32)))


def ff_loss(ff, u2c, u2s, band, smooth_w=0.05):
    """FFL: f(z)=z^4+c2 z^2+c0 should vanish at û and iû (edge dir + 90°). align on edge band; TV smooth."""
    c0 = torch.complex(ff[:, 0].float(), ff[:, 1].float())
    c2 = torch.complex(ff[:, 2].float(), ff[:, 3].float())
    u2 = torch.complex(u2c, u2s)                                   # û²
    u4 = u2 * u2
    f_u = u4 + c2 * u2 + c0                                        # f(û)
    f_iu = u4 - c2 * u2 + c0                                       # f(iû): (iû)^4=û^4, (iû)^2=-û^2
    align = (f_u.abs() ** 2 + f_iu.abs() ** 2) * band
    align = align.sum() / (band.sum() + 1.0)
    sm = sum((t[:, :, 1:] - t[:, :, :-1]).abs().mean() + (t[:, 1:, :] - t[:, :-1, :]).abs().mean()
             for t in (ff[:, 0], ff[:, 1], ff[:, 2], ff[:, 3]))
    return align + smooth_w * sm, float(align)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-ckpt", default="/mnt/sda/zf/landform/results/dino_v3_bdd/best.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m_lc")
    p.add_argument("--pbound-dir", default="/mnt/sda/zf/landform/data/c_1m_pbound")
    p.add_argument("--crop", type=int, default=448)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--multitemporal", action="store_true")
    p.add_argument("--tune-decoder", action="store_true", help="also fine-tune the FreqFusion decoder (low lr)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/dino_v3_ff")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    in_ch = 11 if a.multitemporal else 6
    from transformers import AutoModel
    d3 = AutoModel.from_pretrained(DINOV3_SAT, local_files_only=True)
    model = DinoV3FreqUNetBDDF(d3, num_classes=9, in_channels=in_ch, unfreeze_last_n=4).to(a.device)
    sd = torch.load(a.init_ckpt, map_location=a.device, weights_only=True); msd = model.state_dict()
    miss = model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    print(f"[ff] warm-start {a.init_ckpt} (frame_field_head reinit) {time.time()-t0:.0f}s", flush=True)
    # freeze everything except frame-field head (+ optionally decoder)
    for prm in model.parameters():
        prm.requires_grad = False
    train_params = list(model.frame_field_head.parameters())
    for prm in model.frame_field_head.parameters():
        prm.requires_grad = True
    if a.tune_decoder:
        for nm, prm in model.named_parameters():
            if any(k in nm for k in ("ff1", "ff2", "ff3", "proj", "h_half")):
                prm.requires_grad = True; train_params.append(prm)
    print(f"[ff] trainable {sum(p.numel() for p in train_params)/1e6:.2f}M", flush=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    tr = [n for n in man["train"] if (Path(a.pbound_dir) / f"{n}.npy").exists()]
    ds = FrameDS(tr, a.data_dir, a.pbound_dir, a.crop, a.multitemporal)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=6,
                                     multiprocessing_context="spawn", persistent_workers=True, drop_last=True)
    # spawn (not fork): workers are fresh processes that DON'T inherit the main CUDA context -> no CUDA-fork
    # hang, while parallelising the CPU-heavy getitem (dist-transform GT) so the GPU isn't starved.
    print(f"[ff] train {len(tr)} cells", flush=True)
    opt = torch.optim.AdamW(train_params, lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(dl))
    for ep in range(a.epochs):
        model.train(); el = 0.0; al = 0.0; nb = 0; te = time.time()
        for x, u2c, u2s, band in dl:
            x = x.to(a.device); u2c = u2c.to(a.device); u2s = u2s.to(a.device); band = band.to(a.device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ff = model(x)[3]
                if ff.shape[-2:] != band.shape[-2:]:
                    ff = F.interpolate(ff, size=band.shape[-2:], mode="bilinear", align_corners=False)
            loss, alv = ff_loss(ff, u2c, u2s, band)
            loss.backward(); torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step(); sch.step()
            el += float(loss); al += alv; nb += 1
        torch.save(model.state_dict(), out / "last.pt")
        print(f"[ff] ep{ep+1}/{a.epochs} loss={el/max(nb,1):.4f} align={al/max(nb,1):.4f} ({time.time()-te:.0f}s)", flush=True)
    json.dump({"epochs": a.epochs, "final_align": al / max(nb, 1)}, open(out / "final.json", "w"))
    print(f"[ff] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
