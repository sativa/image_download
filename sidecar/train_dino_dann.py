"""Domain-Adversarial (DANN) cross-domain adaptation for the DINOv3-Sat cropland model — the FSDA-style
'adversarial domain adaptation' route, using UNLABELED target imagery (no target labels needed).

Source = labeled Gansu (cropland CE). Target = unlabeled Tibet imagery. A gradient-reversal layer +
domain discriminator on the pooled backbone feature pushes the backbone toward DOMAIN-INVARIANT
features, so the source-trained classifier transfers to the target. Warm-started from the best
in-domain model; evaluated on held-out Tibet test (FSDA labels)."""
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6
from train_dino_1m_v2 import C1mDSv2, small_feature_w
from train_dino_1m_v3 import DinoV3FreqUNet, DINOV3_SAT, full_eval


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd; return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", default="/mnt/sda/zf/landform/data/c_1m")        # labeled Gansu
    p.add_argument("--tgt-dir", default="/mnt/sda/zf/landform/data/c_1m_tibet")  # unlabeled target (eval on its test)
    p.add_argument("--init-ckpt", default="/mnt/sda/zf/landform/results/dino_1m_v3_gdlxff_max/last.pt")
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/dino_tibet_dann")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--backbone-lr", type=float, default=3e-6)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--lambd", type=float, default=0.5, help="max domain-adversarial weight (ramped)")
    p.add_argument("--crop", type=int, default=448)
    p.add_argument("--small-weight", type=float, default=4.0)
    p.add_argument("--small-k", type=int, default=31)
    p.add_argument("--max-src", type=int, default=1500, help="cap labeled source cells (speed)")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = a.device

    sman = json.loads((Path(a.src_dir) / "manifest.json").read_text())
    tman = json.loads((Path(a.tgt_dir) / "manifest.json").read_text())
    src = sman["train"][:a.max_src]; tgt = tman["train"]; te = tman["test"]
    print(f"[DANN] source(labeled)={len(src)} target(unlabeled)={len(tgt)} tibet-test={len(te)}", flush=True)
    sds = C1mDSv2(src, a.src_dir, a.crop, True, multitemporal=True)
    tds = C1mDSv2(tgt, a.tgt_dir, a.crop, True, multitemporal=True)
    sdl = torch.utils.data.DataLoader(sds, batch_size=a.batch_size, shuffle=True, num_workers=6, drop_last=True, pin_memory=True)
    tdl = torch.utils.data.DataLoader(tds, batch_size=a.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    from transformers import AutoModel
    d3 = AutoModel.from_pretrained(DINOV3_SAT, local_files_only=True)
    model = DinoV3FreqUNet(d3, num_classes=3, in_channels=11, unfreeze_last_n=4).to(dev)
    if a.init_ckpt:
        isd = torch.load(a.init_ckpt, map_location=dev, weights_only=True); msd = model.state_dict()
        model.load_state_dict({k: v for k, v in isd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
        print(f"  warm-start {a.init_ckpt}", flush=True)
    Dd = model.proj.in_channels                                   # backbone hidden dim (1024)
    dom = nn.Sequential(nn.Linear(Dd, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, 1)).to(dev)

    cwt = torch.tensor([0.0, 1.0, 1.0], device=dev)
    bb = [q for n, q in model.named_parameters() if q.requires_grad and "backbone" in n]
    hd = [q for n, q in model.named_parameters() if q.requires_grad and "backbone" not in n] + list(dom.parameters())
    opt = torch.optim.AdamW([{"params": bb, "lr": a.backbone_lr}, {"params": hd, "lr": a.head_lr}], weight_decay=1e-4)
    steps = a.epochs * len(sdl); gstep = 0

    # zero-shot baseline
    f1, pr, rc = full_eval(model, te, a.tgt_dir, dev, a.crop, multitemporal=True)
    print(f"  [tibet zero-shot] F1={f1:.4f} P{pr:.3f} R{rc:.3f}", flush=True)
    best = f1
    for ep in range(a.epochs):
        model.train(); dom.train(); t0 = time.time(); el = ed = 0.0; nb = 0
        tit = iter(tdl)
        for xs, ys in sdl:
            try:
                xt, _ = next(tit)
            except StopIteration:
                tit = iter(tdl); xt, _ = next(tit)
            xs = xs.to(dev); ys = ys.to(dev); xt = xt.to(dev); opt.zero_grad()
            lam = a.lambd * (2.0 / (1.0 + math.exp(-10.0 * gstep / steps)) - 1.0)   # ramp 0->lambd
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                cls_s, _, _, fs = model(xs, return_feat=True)
                if cls_s.shape[-2:] != ys.shape[-2:]:
                    cls_s = F.interpolate(cls_s, size=ys.shape[-2:], mode="bilinear", align_corners=False)
                ce = F.cross_entropy(cls_s.float(), ys, weight=cwt, ignore_index=0, reduction="none")
                pw = small_feature_w(ys, 3, a.small_k, a.small_weight)
                seg = (ce * pw).sum() / ((pw * (ys > 0).float()).sum() + 1e-6)
                _, _, _, ft = model(xt, return_feat=True)
                ds = dom(GRL.apply(fs, lam)); dt = dom(GRL.apply(ft, lam))
                dloss = F.binary_cross_entropy_with_logits(ds[:, 0], torch.zeros(ds.shape[0], device=dev)) \
                    + F.binary_cross_entropy_with_logits(dt[:, 0], torch.ones(dt.shape[0], device=dev))
                loss = seg + dloss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(dom.parameters()), 1.0)
            opt.step(); gstep += 1
            el += seg.item(); ed += dloss.item(); nb += 1
        f1, pr, rc = full_eval(model, te, a.tgt_dir, dev, a.crop, multitemporal=True)
        if f1 > best:
            best = f1; torch.save(model.state_dict(), out / "best.pt")
        print(f"  ep{ep+1}/{a.epochs} seg={el/nb:.4f} dom={ed/nb:.4f} lam={lam:.2f} "
              f"tibet-F1={f1:.4f}(P{pr:.3f}/R{rc:.3f}) best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), out / "last.pt")
    print(f"[FINAL DANN] best tibet-F1={best:.4f} (zero-shot was the ep0 line above)", flush=True)


if __name__ == "__main__":
    main()
