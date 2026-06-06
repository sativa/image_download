"""DINOv2-1m semi-supervised (FixMatch + EMA mean-teacher) — beat the LABELED-data plateau.

Saturation at ~0.86 is a labeled-data ceiling, not an imagery ceiling: the tile downloader gives
unlimited UNLABELED 1m imagery. This trains on labeled (CE) + unlabeled (weak->pseudo-label,
strong->consistency) jointly, the S5/SAMST family shown to keep improving as unlabeled data grows.

Two intended configs:
  (B-val)  --n-labeled 1000  (no warm-start): hold out 4000 train labels as UNLABELED. If
           FixMatch(1000L+4000U) approaches the 5000-labeled 0.860, the pipeline is validated and
           swapping in real downloaded tiles is pure upside.
  (B-full) --n-labeled 0 --warm-start .../dino_1m/best.pt --unlabeled-dir <new tiles>: all 5000
           labeled + NEW unlabeled tiles -> push above 0.860.

Plain DinoUNet5ch(in_channels=6) + full_eval from train_dino_1m -> 1m-F1 directly comparable.
"""
import argparse, copy, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6, C1mDS as LabeledDS
from train_v24_dino_s2 import DinoUNet5ch
from train_v12_unet import DEFAULT_DINOV2


@torch.no_grad()
def full_eval(model, names, dd, dev, cs=448):
    """1m-F1 over full cells (plain-logits model). Returns (f1, precision, recall)."""
    model.eval(); tp = fp = fn = 0
    for name in names:
        z = np.load(Path(dd) / f"{name}.npz"); x6 = z["x6"]; lbl = z["label"]
        _, SZ, SZw = x6.shape
        acc = np.zeros((3, SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
        ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
        if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
        if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
        for t in ys:
            for l in xs:
                xb = torch.from_numpy(norm6(x6[:, t:t + cs, l:l + cs])).unsqueeze(0).to(dev)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    lg = model(xb)
                    if lg.shape[-2:] != (cs, cs):
                        lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                    pr = torch.softmax(lg.float(), 1)[0].cpu().numpy()
                acc[:, t:t + cs, l:l + cs] += pr; cnt[t:t + cs, l:l + cs] += 1
        pred = (acc / np.maximum(cnt, 1)).argmax(0)
        v = lbl > 0; ti = (lbl == 1) & v; pi = (pred == 1) & v
        tp += int((pi & ti).sum()); fp += int((pi & ~ti & v).sum()); fn += int((~pi & ti).sum())
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9), pr, rc


class UnlabeledDS(torch.utils.data.Dataset):
    """Yield a normalized 6ch crop (geometry-augmented). No labels."""
    def __init__(self, names, dd, crop):
        self.n = names; self.d = Path(dd); self.c = crop

    def __len__(self):
        return len(self.n)

    def __getitem__(self, i):
        z = np.load(self.d / f"{self.n[i]}.npz"); x6 = z["x6"]; cs = self.c
        _, SZ, SZw = x6.shape
        if SZ < cs or SZw < cs:
            ph = max(0, cs - SZ); pw = max(0, cs - SZw)
            x6 = np.pad(x6, ((0, 0), (0, ph), (0, pw)), mode="edge"); SZ, SZw = x6.shape[1:]
        t = random.randint(0, SZ - cs); l = random.randint(0, SZw - cs)
        x = norm6(x6[:, t:t + cs, l:l + cs])
        if random.random() < 0.5: x = x[:, :, ::-1].copy()
        if random.random() < 0.5: x = x[:, ::-1, :].copy()
        k = random.randint(0, 3)
        if k: x = np.rot90(x, k, (1, 2)).copy()
        return torch.from_numpy(np.ascontiguousarray(x))


def strong_photometric(x):
    """Strong photometric perturbation in normalized space (geometry preserved -> pixel-aligned)."""
    B = x.shape[0]; dev = x.device
    br = (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.8         # brightness shift
    co = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.8   # contrast scale
    m = x.mean(dim=(2, 3), keepdim=True)
    xs = (x - m) * co + m + br
    xs = xs + torch.randn_like(xs) * 0.05                          # gaussian noise
    if random.random() < 0.3:                                     # random channel dropout
        c = random.randint(0, x.shape[1] - 1); xs[:, c] = 0
    return xs


@torch.no_grad()
def ema_update(teacher, student, m):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(m).add_(sp.detach(), alpha=1 - m)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        tb.copy_(sb)


def cycle(loader):
    while True:
        for b in loader:
            yield b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--unlabeled-dir", default="", help="dir of unlabeled .npz (default: data-dir train remainder)")
    p.add_argument("--unlabeled-list", default="", help="optional json list of unlabeled names")
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/dino_1m_semisup")
    p.add_argument("--warm-start", default="", help="ckpt to warm-start (empty = ImageNet DINOv2 init)")
    p.add_argument("--n-labeled", type=int, default=1000, help="labeled cells (rest of train -> unlabeled). 0=all")
    p.add_argument("--tau", type=float, default=0.95, help="pseudo-label confidence threshold")
    p.add_argument("--lambda-u", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--crop", type=int, default=448)
    p.add_argument("--steps", type=int, default=9000)
    p.add_argument("--eval-every", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--mu", type=int, default=1, help="unlabeled batch = mu * batch-size")
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--unfreeze", type=int, default=4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    tr = man["train"]; te = man["test"]
    if a.n_labeled and a.n_labeled < len(tr):
        lab = tr[:a.n_labeled]; rest = tr[a.n_labeled:]
    else:
        lab = tr; rest = tr  # all labeled; consistency-reg on same set unless --unlabeled-dir given
    un_dir = a.unlabeled_dir or a.data_dir
    if a.unlabeled_list:
        un_names = json.loads(Path(a.unlabeled_list).read_text())
    elif a.unlabeled_dir and a.unlabeled_dir != a.data_dir:
        un_names = sorted(p.stem for p in Path(a.unlabeled_dir).glob("*.npz"))  # cross-domain dir (e.g. Changzhi)
    else:
        un_names = rest
    un_names = [n for n in un_names if (Path(un_dir) / f"{n}.npz").exists()]
    print(f"[semisup] labeled={len(lab)} unlabeled={len(un_names)}@{Path(un_dir).name} "
          f"tau={a.tau} lambda_u={a.lambda_u} steps={a.steps}", flush=True)

    ldl = cycle(torch.utils.data.DataLoader(LabeledDS(lab, a.data_dir, a.crop, True),
                batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True))
    udl = cycle(torch.utils.data.DataLoader(UnlabeledDS(un_names, un_dir, a.crop),
                batch_size=a.batch_size * a.mu, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True))

    from transformers import AutoModel
    dinov2 = AutoModel.from_pretrained(str(DEFAULT_DINOV2), local_files_only=True)
    model = DinoUNet5ch(dinov2, num_classes=3, in_channels=6, unfreeze_last_n=a.unfreeze).to(a.device)
    if a.warm_start and Path(a.warm_start).exists():
        sd = torch.load(a.warm_start, map_location="cpu", weights_only=True)
        model.load_state_dict({k: v for k, v in sd.items() if k in model.state_dict()
                               and model.state_dict()[k].shape == v.shape}, strict=False)
        print(f"  warm-started from {Path(a.warm_start).name}", flush=True)
    teacher = copy.deepcopy(model)
    for q in teacher.parameters(): q.requires_grad_(False)
    teacher.eval()

    bc = np.zeros(3)
    for n in lab[:300]:
        bc += np.bincount(np.load(Path(a.data_dir) / f"{n}.npz")["label"].ravel(), minlength=3)
    cw = np.where(bc > 0, 1 / np.sqrt(bc), 0).astype(np.float32); cw[0] = 0; cw = cw / cw.sum() * 2
    cwt = torch.from_numpy(cw).to(a.device)

    bb = [q for nm, q in model.named_parameters() if q.requires_grad and "backbone" in nm]
    hd = [q for nm, q in model.named_parameters() if q.requires_grad and "backbone" not in nm]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.backbone_lr},
                             {"params": hd, "lr": a.head_lr}], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    scaler = torch.amp.GradScaler("cuda")

    best = -1.0; t0 = time.time(); run_sup = run_un = run_cov = 0.0
    for step in range(1, a.steps + 1):
        model.train()
        xl, yl = next(ldl); xl = xl.to(a.device); yl = yl.to(a.device)
        xu = next(udl).to(a.device)
        opt.zero_grad()
        # teacher pseudo-labels on WEAK (geometry-aug already applied in loader; no photometric)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            tl = teacher(xu)
            if tl.shape[-2:] != xu.shape[-2:]:
                tl = F.interpolate(tl, size=xu.shape[-2:], mode="bilinear", align_corners=False)
            prob = torch.softmax(tl.float(), 1); conf, pl = prob.max(1)
            keep = (conf > a.tau) & (pl > 0)                       # confident, non-nodata
        xs = strong_photometric(xu)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            lgl = model(xl)
            if lgl.shape[-2:] != yl.shape[-2:]:
                lgl = F.interpolate(lgl, size=yl.shape[-2:], mode="bilinear", align_corners=False)
            sup = F.cross_entropy(lgl.float(), yl, weight=cwt, ignore_index=0) if (yl > 0).any() else lgl.float().sum() * 0.0
            lgu = model(xs)
            if lgu.shape[-2:] != xs.shape[-2:]:
                lgu = F.interpolate(lgu, size=xs.shape[-2:], mode="bilinear", align_corners=False)
            plm = pl.clone(); plm[~keep] = 0                       # ignore_index=0 where not confident
            un = F.cross_entropy(lgu.float(), plm, ignore_index=0) if keep.any() else torch.zeros((), device=a.device)
            ramp = min(1.0, step / 1000.0)                         # ramp up unlabeled weight
            loss = sup + a.lambda_u * ramp * un
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        ema_update(teacher, model, a.ema)
        run_sup += float(sup); run_un += float(un); run_cov += float(keep.float().mean())
        if step % a.eval_every == 0:
            f1, pr, rc = full_eval(teacher, te, a.data_dir, a.device, a.crop)   # eval the TEACHER (EMA)
            if f1 > best:
                best = f1; torch.save(teacher.state_dict(), out / "best.pt")
            k = a.eval_every
            print(f"  step{step}/{a.steps} sup={run_sup/k:.3f} un={run_un/k:.3f} cov={run_cov/k:.2f} "
                  f"1m-F1={f1:.4f}(P{pr:.3f}/R{rc:.3f}) best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
            run_sup = run_un = run_cov = 0.0
    print(f"\n[FINAL semisup] best 1m-F1={best:.4f}  (baseline 5000-labeled 0.860)", flush=True)
    json.dump({"best_1m_f1": best, "n_labeled": len(lab), "n_unlabeled": len(un_names),
               "tau": a.tau}, open(out / "final.json", "w"))


if __name__ == "__main__":
    main()
