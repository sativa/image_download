"""Route (c) — Stage-1 inference: run the 1m boundary/extent net over every fused cell,
then aggregate the 1m cropland probability down to that cell's 10m S2 grid as 2 channels:
  [0] mean cropland probability  (area-pooled -> 1m crop fraction per 10m pixel)
  [1] boundary density           (1m field-edge fraction per 10m pixel)

These feed stage-2 (concatenated to the 9-ch 10m spectral stack). The 1m and 10m grids
share the SAME cell bbox, so a bbox->bbox interpolate aligns them exactly.

Tiled sliding-window inference (512, stride 384) so b5@2200px never OOMs a 4090.
Out: c_stage1_feat/{name}.npz {feat2 (2,Hs,Ws) float16}.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")


def boundary(m, k=3):
    m = m.unsqueeze(1).float(); p = k // 2
    d = F.max_pool2d(m, k, 1, p); e = -F.max_pool2d(-m, k, 1, p)
    return (d - e).squeeze(1)


@torch.no_grad()
def infer_prob(model, x6, dev, tile=512, stride=384):
    """x6: (6,SZ,SZ) float32 [0,1] -> cropland prob (SZ,SZ) float32 via sliding window."""
    C, H, W = x6.shape
    x = torch.from_numpy(x6).unsqueeze(0).to(dev)
    acc = torch.zeros((1, 3, H, W), device=dev)
    cnt = torch.zeros((1, 1, H, W), device=dev)
    ys = list(range(0, max(1, H - tile + 1), stride)) or [0]
    xs = list(range(0, max(1, W - tile + 1), stride)) or [0]
    if ys[-1] != max(0, H - tile): ys.append(max(0, H - tile))
    if xs[-1] != max(0, W - tile): xs.append(max(0, W - tile))
    for t in ys:
        for l in xs:
            patch = x[:, :, t:t + tile, l:l + tile]
            ph, pw = patch.shape[2], patch.shape[3]
            if ph < tile or pw < tile:
                patch = F.pad(patch, (0, tile - pw, 0, tile - ph), mode="replicate")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = model(patch).float()
            lg = lg[:, :, :ph, :pw]
            acc[:, :, t:t + ph, l:l + pw] += torch.softmax(lg, 1)
            cnt[:, :, t:t + ph, l:l + pw] += 1
    prob = (acc / cnt.clamp(min=1))[:, 1]          # cropland prob (1,H,W)
    hard = (acc / cnt.clamp(min=1)).argmax(1)       # (1,H,W)
    bnd = boundary((hard == 1).float())             # (1,H,W) in {0,1}
    return prob, bnd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(HOME / "data/c_1m"))
    p.add_argument("--s2-dir", default=str(HOME / "data/v19_s2_raw"))
    p.add_argument("--ckpt", default=str(HOME / "results/c_stage1/best.pt"))
    p.add_argument("--out-dir", default=str(HOME / "data/c_stage1_feat"))
    p.add_argument("--backbone", default="efficientnet-b5")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sources", default="both", choices=["both", "esri", "google"],
                   help="must match the stage-1 ckpt: both=6ch, esri/google=3ch")
    a = p.parse_args()
    CH = {"both": [0, 1, 2, 3, 4, 5], "esri": [0, 1, 2], "google": [3, 4, 5]}[a.sources]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    names = man["train"] + man["test"]
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=a.backbone, encoder_weights=None, in_channels=len(CH), classes=3).to(a.device)
    model.load_state_dict(torch.load(a.ckpt, map_location=a.device, weights_only=True)); model.eval()
    print(f"[s1-infer] {len(names)} cells -> {out}", flush=True)
    ok = miss = 0
    for i, name in enumerate(names):
        fz = Path(a.data_dir) / f"{name}.npz"
        s2 = Path(a.s2_dir) / f"{name}.npz"
        if not fz.exists() or not s2.exists():
            miss += 1; continue
        x6 = np.load(fz)["x6"][CH].astype(np.float32) / 255.0
        Hs, Ws = np.load(s2)["rgbnir"].shape[1:3]
        prob, bnd = infer_prob(model, x6, a.device)
        fp = F.interpolate(prob.unsqueeze(1), size=(Hs, Ws), mode="area")[0, 0]
        fb = F.interpolate(bnd.unsqueeze(1), size=(Hs, Ws), mode="area")[0, 0]
        feat2 = torch.stack([fp, fb], 0).cpu().numpy().astype(np.float16)
        np.savez_compressed(out / f"{name}.npz", feat2=feat2)
        ok += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(names)} ok={ok} miss={miss}", flush=True)
    print(f"[done] s1-infer ok={ok} miss={miss} -> {out}", flush=True)


if __name__ == "__main__":
    main()
