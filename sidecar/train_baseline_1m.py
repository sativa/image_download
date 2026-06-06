"""Same-protocol architecture baselines for the benchmark table (vs our DINOv2-UNet).

Identical setup to train_dino_1m (6ch @ 1m Esri+Google RGB, ImageNet-norm, 5000 c_1m cells, binary
cropland, full-cell tiled 1m-F1) — ONLY the architecture changes, so the table isolates the backbone:
  --arch unet|deeplabv3plus|segformer   (smp 0.5.0; encoder ImageNet-pretrained, in_channels=6)
Reports pixel 1m-F1 directly comparable to DINOv2-UNet 0.866. Run parcel_eval afterwards for the
parcel-level row. Honest same-data, same-label, same-eval comparison.
"""
import argparse, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import C1mDS, norm6, full_eval
import segmentation_models_pytorch as smp


def build(arch, enc):
    k = dict(encoder_name=enc, encoder_weights="imagenet", in_channels=6, classes=3)
    if arch == "unet":
        return smp.Unet(**k)
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(**k)
    if arch == "segformer":
        return smp.Segformer(**k)
    raise ValueError(arch)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=["unet", "deeplabv3plus", "segformer"])
    p.add_argument("--encoder", default="efficientnet-b5")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/baseline")
    p.add_argument("--crop", type=int, default=448)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    tr_names = man["train"]; te_names = man["test"]
    print(f"[base:{a.arch}/{a.encoder}] train={len(tr_names)} test={len(te_names)}", flush=True)
    trl = torch.utils.data.DataLoader(C1mDS(tr_names, a.data_dir, a.crop, True),
                                      batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                                      pin_memory=True, drop_last=True)
    model = build(a.arch, a.encoder).to(a.device)
    print(f"  params={sum(q.numel() for q in model.parameters())/1e6:.1f}M", flush=True)

    bc = np.zeros(3)
    for n in tr_names[:300]:
        bc += np.bincount(np.load(Path(a.data_dir) / f"{n}.npz")["label"].ravel(), minlength=3)
    cw = np.where(bc > 0, 1 / np.sqrt(bc), 0).astype(np.float32); cw[0] = 0; cw = cw / cw.sum() * 2
    cwt = torch.from_numpy(cw).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda")
    best = -1.0
    for ep in range(a.epochs):
        model.train(); t0 = time.time()
        for x, y in trl:
            x = x.to(a.device); y = y.to(a.device); opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                lg = model(x)
                if lg.shape[-2:] != y.shape[-2:]:
                    lg = F.interpolate(lg, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(lg.float(), y, weight=cwt, ignore_index=0)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
        f1 = full_eval(model, te_names, a.data_dir, a.device, a.crop)
        if f1 > best:
            best = f1; torch.save(model.state_dict(), out / "best.pt")
        print(f"  ep{ep+1}/{a.epochs} 1m-F1={f1:.4f} (best {best:.4f}) ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n[FINAL {a.arch}/{a.encoder}] 1m-F1={best:.4f}  (DINOv2-UNet ours 0.866)", flush=True)
    json.dump({"arch": a.arch, "encoder": a.encoder, "best_1m_f1": best}, open(out / "final.json", "w"))


if __name__ == "__main__":
    main()
