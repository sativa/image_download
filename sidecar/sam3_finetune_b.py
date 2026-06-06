"""SAM3 fine-tune — option B: custom loop (full device control, bypasses the official trainer).

Design (de-risked by sam3_ft_smoke.py: grad flows through forward_grounding):
  - Freeze the vision+language backbone; run forward_image under no_grad (saves memory AND sidesteps
    the inference-only fused MLP op). Train ONLY the detector: segmentation_head + transformer(decoder)
    + dot_prod_scoring (the "crop field" concept/mask predictor).
  - Data: the prebuilt COCO (sam3_coco) — crops (JPG) + per-parcel instance masks (DLTB cropland).
  - Per crop: manual set_image (grad-capable) + forward_grounding("crop field") -> 200 query masks+scores
    -> Hungarian match (scipy) pred<->GT parcels on dice cost -> dice+focal mask loss + presence BCE.
  - Eval: union of confident pred masks vs the union of GT parcels -> semantic cropland F1 (vs zero-shot ~0.55).

Run on .250 (GPU). Uses the inference fork (sam3-inference), whose vitdet MLP is eager (trainable).
"""
import argparse, json, sys, time, types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from scipy.optimize import linear_sum_assignment

SAM3_REPO = "/home/ps/sam3/sam3-inference"
sys.path.insert(0, SAM3_REPO)
try:
    import decord  # noqa
except Exception:
    m = types.ModuleType("decord"); m.cpu = m.gpu = lambda *a, **k: None
    m.VideoReader = object; m.bridge = types.SimpleNamespace(set_bridge=lambda *a, **k: None)
    sys.modules["decord"] = m
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEV = "cuda"


def forward_grad(model, proc, img_t, prompt):
    """Manual set_image+forward_grounding WITH grad on the detector (backbone frozen under no_grad)."""
    with torch.no_grad():
        backbone_out = model.backbone.forward_image(img_t)
        backbone_out.update(model.backbone.forward_text([prompt], device=DEV))
    geo = model._get_dummy_prompt()
    out = model.forward_grounding(backbone_out=backbone_out, find_input=proc.find_stage,
                                  find_target=None, geometric_prompt=geo)
    return out["pred_masks"][0], out["pred_logits"][0, :, 0], out["presence_logit_dec"].reshape(-1)[0]


def dice_cost(pred, gt):
    """pred [Q,k], gt [N,k] (flattened, in [0,1]) -> [Q,N] dice loss (1-dice)."""
    num = 2 * pred @ gt.t()
    den = pred.sum(1, keepdim=True) + gt.sum(1, keepdim=True).t()
    return 1 - (num + 1) / (den + 1)


def mask_losses(pm, gtm):
    """pm [h,w] logits, gtm [h,w] {0,1} -> dice + focal-bce (scalars)."""
    p = pm.sigmoid().flatten(); g = gtm.flatten()
    dice = 1 - (2 * (p * g).sum() + 1) / (p.sum() + g.sum() + 1)
    bce = F.binary_cross_entropy_with_logits(pm.flatten(), g, reduction="none")
    pt = p * g + (1 - p) * (1 - g)
    focal = (bce * (1 - pt).pow(2)).mean()
    return dice, focal


def focal_score(logits, tgt, alpha=0.75, gamma=2.0):
    """Sigmoid focal loss for the per-query score. alpha>0.5 favors the RARE matched positives so
    the ~196 unmatched queries don't collapse the model to all-negative (the ep1 recall-crash bug)."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    pt = p * tgt + (1 - p) * (1 - tgt)
    at = alpha * tgt + (1 - alpha) * (1 - tgt)
    return (at * (1 - pt).pow(gamma) * ce).mean()


class CocoParcels(torch.utils.data.Dataset):
    def __init__(self, root, split, max_crops=0, mask_res=288):
        from pycocotools.coco import COCO
        import contextlib, io
        self.root = Path(root) / split
        with contextlib.redirect_stdout(io.StringIO()):
            self.coco = COCO(str(self.root / "_annotations.coco.json"))
        self.ids = sorted(self.coco.imgs.keys())
        if max_crops:
            self.ids = self.ids[:max_crops]
        self.mres = mask_res

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        iid = self.ids[i]; info = self.coco.imgs[iid]
        img = Image.open(self.root / info["file_name"]).convert("RGB")
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=iid))
        masks = []
        for a in anns:
            m = self.coco.annToMask(a)
            if m.sum() > 0:
                masks.append(m)
        if masks:
            gt = torch.from_numpy(np.stack(masks)).float().unsqueeze(1)  # [N,1,H,W]
            gt = F.interpolate(gt, size=(self.mres, self.mres), mode="area")[:, 0]
            gt = (gt > 0.5).float()
            gt = gt[gt.flatten(1).sum(1) > 0]  # drop empties after downsample
        else:
            gt = torch.zeros(0, self.mres, self.mres)
        return img, gt


def run_eval(model, proc, ds, prompt, conf, n=40):
    model.eval(); tp = fp = fn = 0
    with torch.no_grad():
        for i in range(min(n, len(ds))):
            img, gt = ds[i]
            img_t = proc.transform(v2.functional.to_image(img).to(DEV)).unsqueeze(0)
            pm, pl, pres = forward_grad(model, proc, img_t, prompt)
            score = (pl.sigmoid() * pres.sigmoid())
            keep = score > conf
            H, W = gt.shape[-2:] if gt.numel() else pm.shape[-2:]
            pred = torch.zeros(H, W, device=DEV)
            if keep.any():
                pmk = F.interpolate(pm[keep].unsqueeze(1).sigmoid(), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
                pred = (pmk > 0.5).any(0).float()
            g = (gt.to(DEV).sum(0) > 0).float() if gt.numel() else torch.zeros(H, W, device=DEV)
            tp += float((pred * g).sum()); fp += float((pred * (1 - g)).sum()); fn += float(((1 - pred) * g).sum())
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9), pr, rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="/home/ps/sam3/sam3_weights/sam3.pt")
    p.add_argument("--coco", default="/mnt/sda/zf/landform/data/sam3_coco")
    p.add_argument("--prompt", default="crop field")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-crops", type=int, default=0)
    p.add_argument("--max-gt", type=int, default=120, help="cap parcels/crop for the matcher")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/sam3_ft_b")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    model = build_sam3_image_model(checkpoint_path=a.weights, load_from_HF=False, device=DEV)
    proc = Sam3Processor(model, device=DEV)
    for prm in model.parameters():
        prm.requires_grad = False
    trainable = []
    for nm in ["segmentation_head", "transformer", "dot_prod_scoring"]:
        mod = getattr(model, nm, None)
        if mod is not None and isinstance(mod, torch.nn.Module):
            mod.train()
            for prm in mod.parameters():
                prm.requires_grad = True; trainable.append(prm)
    print(f"[ft-b] trainable {sum(p.numel() for p in trainable)/1e6:.1f}M ({time.time()-t0:.0f}s)", flush=True)

    tr = CocoParcels(a.coco, "train", a.max_crops or (4 if a.smoke else 0))
    va = CocoParcels(a.coco, "valid", 4 if a.smoke else 40)
    print(f"[ft-b] train crops {len(tr)} | val {len(va)}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    if not a.smoke:
        f1, pr, rc = run_eval(model, proc, va, a.prompt, a.conf)
        print(f"[ft-b] zero-shot val F1={f1:.4f} (P{pr:.2f}/R{rc:.2f})", flush=True)

    epochs = 1 if a.smoke else a.epochs
    best = -1
    for ep in range(epochs):
        model.segmentation_head.train(); te = time.time(); el = 0; nb = 0
        order = np.random.permutation(len(tr))
        for idx in order:
            img, gt = tr[int(idx)]
            if gt.numel() == 0:
                continue
            if gt.shape[0] > a.max_gt:
                gt = gt[torch.randperm(gt.shape[0])[:a.max_gt]]
            gt = gt.to(DEV)
            img_t = proc.transform(v2.functional.to_image(img).to(DEV)).unsqueeze(0)
            pm, pl, pres = forward_grad(model, proc, img_t, a.prompt)  # [Q,h,w],[Q],[Q]
            Q = pm.shape[0]; mres = pm.shape[-1]
            if gt.shape[-1] != mres:
                gt = F.interpolate(gt.unsqueeze(1), size=(mres, mres), mode="area")[:, 0]; gt = (gt > 0.5).float()
            # match on a downsampled dice cost (fast), no grad
            with torch.no_grad():
                ps = F.interpolate(pm.unsqueeze(1).sigmoid(), size=(64, 64), mode="bilinear", align_corners=False)[:, 0].flatten(1)
                gs = F.interpolate(gt.unsqueeze(1), size=(64, 64), mode="area")[:, 0].flatten(1)
                cost = dice_cost(ps, gs).cpu().numpy()
            ri, ci = linear_sum_assignment(cost)  # matched query->gt pairs
            # loss
            ml = dl = torch.tensor(0.0, device=DEV)
            for q, n in zip(ri, ci):
                d, fcl = mask_losses(pm[q], gt[n]); dl = dl + d; ml = ml + fcl
            k = max(len(ri), 1); dl = dl / k; ml = ml / k
            tgt = torch.zeros(Q, device=DEV); tgt[ri] = 1.0
            score_loss = focal_score(pl, tgt)  # focal, favors matched positives (anti-collapse)
            pres_loss = F.binary_cross_entropy_with_logits(pres, torch.ones((), device=DEV))
            loss = 5 * dl + 5 * ml + 1 * score_loss + 0.5 * pres_loss
            opt.zero_grad(); loss.backward(); opt.step()
            el += float(loss); nb += 1
            if a.smoke and nb >= 2:
                print(f"[ft-b][smoke] step {nb} loss={float(loss):.3f} (dice {float(dl):.3f} mask {float(ml):.3f} pres {float(pres_loss):.3f}) matched {len(ri)} OK", flush=True)
                print(f"[ft-b][smoke] grad flows + loop runs ({time.time()-t0:.0f}s)", flush=True); return
        f1, pr, rc = run_eval(model, proc, va, a.prompt, a.conf)
        if f1 > best:
            best = f1
            torch.save({"seg": model.segmentation_head.state_dict(),
                        "trans": model.transformer.state_dict(),
                        "dps": (model.dot_prod_scoring.state_dict()
                                if getattr(model, "dot_prod_scoring", None) is not None else None)},
                       out / "ft_state.pt")
        print(f"[ft-b] ep{ep+1}/{epochs} loss={el/max(nb,1):.4f} val-F1={f1:.4f} (P{pr:.2f}/R{rc:.2f}) best={best:.4f} ({time.time()-te:.0f}s)", flush=True)
    print(f"\n[ft-b] done best val-F1={best:.4f} | compare zero-shot ~0.55", flush=True)
    json.dump({"best_f1": best}, open(out / "final.json", "w"))


if __name__ == "__main__":
    main()
