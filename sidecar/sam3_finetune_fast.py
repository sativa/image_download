"""SAM3 fine-tune — OPTIMIZED loop (for next runs; ~several× faster than sam3_finetune_b.py).

Speedups over the batch=1 / fp32 / Python-loop-loss baseline:
  1. BATCHED backbone: run the expensive ViT-Det @1008 on B crops in ONE forward_image (the bottleneck),
     then run the cheaper detector (forward_grounding) per image by indexing find_input.img_ids=[i].
  2. AMP bf16 autocast over backbone+detector (the baseline ran fp32) -> ~1.5-2x + halves memory.
  3. VECTORIZED mask loss: dice+focal for ALL matched pairs in one tensor op (no Python per-pair loop).
Same recipe otherwise: freeze backbone (no_grad), train seg_head+transformer+dot_prod_scoring; scipy
Hungarian match (downsampled dice cost); focal score loss (anti-collapse). COCO data from coco_convert.py.
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
from sam3.model.data_misc import FindStage
from sam3_finetune_b import CocoParcels, focal_score  # reuse dataset + focal score

DEV = "cuda"


def find_stage(i):
    return FindStage(img_ids=torch.tensor([i], device=DEV, dtype=torch.long),
                     text_ids=torch.tensor([0], device=DEV, dtype=torch.long),
                     input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                     input_points=None, input_points_mask=None)


def batched_backbone(model, imgs_t, prompt):
    """forward_image on B crops (no_grad, frozen) + forward_text -> shared backbone_out."""
    with torch.no_grad():
        bo = model.backbone.forward_image(imgs_t)
        bo.update(model.backbone.forward_text([prompt], device=DEV))
    return bo


def detector(model, bo, i):
    out = model.forward_grounding(backbone_out=bo, find_input=find_stage(i),
                                  find_target=None, geometric_prompt=model._get_dummy_prompt())
    return out["pred_masks"][0], out["pred_logits"][0, :, 0], out["presence_logit_dec"].reshape(-1)[0]


def vec_mask_loss(pm_sel, gt_sel):
    """pm_sel [M,h,w] logits, gt_sel [M,h,w] {0,1} -> (dice, focal) over all matched pairs at once."""
    p = pm_sel.sigmoid().flatten(1); g = gt_sel.flatten(1)
    dice = (1 - (2 * (p * g).sum(1) + 1) / (p.sum(1) + g.sum(1) + 1)).mean()
    bce = F.binary_cross_entropy_with_logits(pm_sel.flatten(1), g, reduction="none")
    pt = p * g + (1 - p) * (1 - g)
    focal = (bce * (1 - pt).pow(2)).mean()
    return dice, focal


def run_eval(model, proc, ds, prompt, conf, n=40):
    model.eval(); tp = fp = fn = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(min(n, len(ds))):
            img, gt = ds[i]
            img_t = proc.transform(v2.functional.to_image(img).to(DEV)).unsqueeze(0)
            bo = batched_backbone(model, img_t, prompt)
            pm, pl, pres = detector(model, bo, 0)
            score = pl.sigmoid() * pres.sigmoid(); keep = score > conf
            H, W = (gt.shape[-2:] if gt.numel() else pm.shape[-2:])
            pred = torch.zeros(H, W, device=DEV)
            if keep.any():
                pmk = F.interpolate(pm[keep].unsqueeze(1).float().sigmoid(), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
                pred = (pmk > 0.5).any(0).float()
            g = (gt.to(DEV).sum(0) > 0).float() if gt.numel() else torch.zeros(H, W, device=DEV)
            tp += float((pred * g).sum()); fp += float((pred * (1 - g)).sum()); fn += float(((1 - pred) * g).sum())
    pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
    return 2 * pr * rc / (pr + rc + 1e-9), pr, rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="/home/ps/sam3/sam3_weights/sam3.pt")
    p.add_argument("--coco", default="/mnt/sda/zf/landform/data/sam3_coco_dual")
    p.add_argument("--prompt", default="crop field")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-crops", type=int, default=0)
    p.add_argument("--max-gt", type=int, default=120)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--out", default="/mnt/sda/zf/landform/results/sam3_ft_fast")
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
    print(f"[fast] trainable {sum(x.numel() for x in trainable)/1e6:.1f}M ({time.time()-t0:.0f}s)", flush=True)

    tr = CocoParcels(a.coco, "train", a.max_crops or (8 if a.smoke else 0))
    va = CocoParcels(a.coco, "valid", 4 if a.smoke else 40)
    # num_workers=0: the model holds a CUDA context, so forked DataLoader workers deadlock
    # (CUDA-before-fork). Loading is light (JPG + pycocotools annToMask) so main-thread is fine.
    dl = torch.utils.data.DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=0,
                                     collate_fn=lambda b: b, drop_last=True)
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=1e-4)
    print(f"[fast] train {len(tr)} crops, batch={a.batch} | val {len(va)}", flush=True)

    epochs = 1 if a.smoke else a.epochs
    best = -1
    for ep in range(epochs):
        for m in (model.segmentation_head, model.transformer):
            m.train()
        te = time.time(); el = 0; nb = 0
        for batch in dl:
            imgs = torch.stack([proc.transform(v2.functional.to_image(img).to(DEV)) for img, _ in batch])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bo = batched_backbone(model, imgs, a.prompt)
                loss = torch.zeros((), device=DEV)
                for i, (_, gt) in enumerate(batch):
                    if gt.numel() == 0:
                        continue
                    gt = gt.to(DEV)
                    if gt.shape[0] > a.max_gt:
                        gt = gt[torch.randperm(gt.shape[0])[:a.max_gt]]
                    pm, pl, pres = detector(model, bo, i)
                    Q, mres = pm.shape[0], pm.shape[-1]
                    if gt.shape[-1] != mres:
                        gt = (F.interpolate(gt.unsqueeze(1), size=(mres, mres), mode="area")[:, 0] > 0.5).float()
                    with torch.no_grad():
                        ps = F.interpolate(pm.unsqueeze(1).float().sigmoid(), size=(64, 64), mode="bilinear", align_corners=False)[:, 0].flatten(1)
                        gs = F.interpolate(gt.unsqueeze(1), size=(64, 64), mode="area")[:, 0].flatten(1)
                        cost = (1 - (2 * ps @ gs.t() + 1) / (ps.sum(1, keepdim=True) + gs.sum(1, keepdim=True).t() + 1)).float().cpu().numpy()
                    ri, ci = linear_sum_assignment(cost)
                    dl_, fl_ = vec_mask_loss(pm[ri], gt[ci])      # vectorized over matched pairs
                    tgt = torch.zeros(Q, device=DEV); tgt[ri] = 1.0
                    sl = focal_score(pl, tgt)
                    pres_l = F.binary_cross_entropy_with_logits(pres, torch.ones((), device=DEV))
                    loss = loss + 5 * dl_ + 5 * fl_ + sl + 0.5 * pres_l
                loss = loss / len(batch)
            opt.zero_grad(); loss.backward(); opt.step()
            el += float(loss); nb += 1
            if a.smoke and nb >= 2:
                print(f"[fast][smoke] step {nb} loss={float(loss):.3f} batch={len(batch)} bf16+batched OK ({time.time()-t0:.0f}s)", flush=True)
                return
        f1, pr, rc = run_eval(model, proc, va, a.prompt, a.conf)
        if f1 > best:
            best = f1
            torch.save({"seg": model.segmentation_head.state_dict(), "trans": model.transformer.state_dict(),
                        "dps": (model.dot_prod_scoring.state_dict() if getattr(model, "dot_prod_scoring", None) is not None else None)},
                       out / "ft_state.pt")
        print(f"[fast] ep{ep+1}/{epochs} loss={el/max(nb,1):.4f} val-F1={f1:.4f} (P{pr:.2f}/R{rc:.2f}) best={best:.4f} ({time.time()-te:.0f}s)", flush=True)
    print(f"\n[fast] done best={best:.4f}", flush=True)
    json.dump({"best_f1": best}, open(out / "final.json", "w"))


if __name__ == "__main__":
    main()
