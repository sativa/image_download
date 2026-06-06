"""De-risk SAM3 fine-tuning: can gradients flow through forward_grounding?

The Sam3Processor wraps everything in @torch.inference_mode() (blocks grad). For fine-tuning we must
call the underlying model methods WITHOUT inference_mode, with the backbone frozen and the seg head /
decoder trainable. This smoke runs one c_1m crop through a manual forward, computes a dice loss of the
(soft) union of "crop field" instance masks vs the DLTB cropland mask, backprops, and checks grads
reach the segmentation head. If grad norm > 0, the custom fine-tune loop is feasible.
"""
import sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

SAM3_REPO = "/home/ps/sam3/sam3-inference"
sys.path.insert(0, SAM3_REPO)
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEV = "cuda"
CKPT = "/home/ps/sam3/sam3_weights/sam3.pt"
CELL = "/mnt/sda/zf/landform/data/c_1m/620822_498.npz"  # ~66% cropland


def main():
    t0 = time.time()
    model = build_sam3_image_model(checkpoint_path=CKPT, load_from_HF=False, device=DEV)
    proc = Sam3Processor(model, device=DEV)
    print(f"[smoke] loaded ({time.time()-t0:.0f}s)", flush=True)

    # Freeze everything, then unfreeze the segmentation head (+ decoder) for fine-tuning.
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for mod_name in ["segmentation_head", "transformer"]:
        mod = getattr(model, mod_name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True
            trainable += [p for p in mod.parameters()]
    model.segmentation_head.train()
    n_tr = sum(p.numel() for p in trainable if p.requires_grad)
    print(f"[smoke] trainable params: {n_tr/1e6:.1f}M", flush=True)

    # one crop -> image (resized to 1008 by the same transform) + GT cropland mask
    z = np.load(CELL); rgb = np.ascontiguousarray(z["x6"][0:3].transpose(1, 2, 0)); lbl = z["label"]
    img = proc.transform(v2.functional.to_image(Image.fromarray(rgb)).to(DEV)).unsqueeze(0)

    # manual forward WITH grad (NOT proc.set_image which is inference_mode)
    backbone_out = model.backbone.forward_image(img)
    backbone_out.update(model.backbone.forward_text(["crop field"], device=DEV))
    geo = model._get_dummy_prompt()
    out = model.forward_grounding(backbone_out=backbone_out, find_input=proc.find_stage,
                                  find_target=None, geometric_prompt=geo)
    pm = out["pred_masks"]  # [B,Q,h,w]
    print(f"[smoke] pred_masks {tuple(pm.shape)} pred_logits {tuple(out['pred_logits'].shape)}", flush=True)

    # soft union of instance masks -> semantic cropland prob; dice vs GT (recall-oriented smoke loss)
    prob = pm.sigmoid().squeeze(0)              # [Q,h,w]
    union = prob.max(0).values                  # [h,w]
    gt = torch.from_numpy((lbl == 1).astype(np.float32))[None, None].to(DEV)
    gt = F.interpolate(gt, size=union.shape[-2:], mode="area")[0, 0]
    inter = (union * gt).sum(); dice = 1 - (2 * inter + 1) / (union.sum() + gt.sum() + 1)
    print(f"[smoke] dice loss {dice.item():.4f}", flush=True)

    dice.backward()
    gn = sum(float(p.grad.norm()) for p in model.segmentation_head.parameters() if p.grad is not None)
    n_with_grad = sum(1 for p in model.segmentation_head.parameters() if p.grad is not None)
    print(f"[smoke] seg-head grad norm = {gn:.4e} ({n_with_grad} tensors got grad)", flush=True)
    print(f"[smoke] {'OK — gradients flow, SAM3 fine-tune is feasible' if gn > 0 else 'FAIL — no grad'} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
