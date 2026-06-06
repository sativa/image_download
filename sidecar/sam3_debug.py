"""Debug: pick the MOST cropland-dense test cell, then check SAM3 text-prompt output under
(a) full-cell resized to 1008 (SAM3 native res), (b) a cropland-dense 1008 tile. Prompts x conf."""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image

HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))
sys.path.insert(0, "/home/ps/sam3/sam3-inference")
import types
try:
    import decord  # noqa
except Exception:
    sys.modules["decord"] = types.ModuleType("decord")
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DD = Path("/mnt/sda/zf/landform/data/c_1m_lc")
man = json.loads((DD / "manifest.json").read_text())
te = [n for n in man["test"] if (DD / f"{n}.npz").exists()][:120]
# find most cropland-dense cell by label==1 fraction
best = None
for n in te:
    z = np.load(DD / f"{n}.npz")
    lbl = z["label"]; frac = float((lbl == 1).mean())
    if best is None or frac > best[1]:
        best = (n, frac)
n, frac = best
print(f"most cropland-dense test cell = {n}  cropland frac={frac:.2f}", flush=True)
z = np.load(DD / f"{n}.npz")
x6 = z["x6"]; lbl = z["label"]
rgb = np.ascontiguousarray(x6[:3].transpose(1, 2, 0)).astype(np.uint8)
H, Wd = rgb.shape[:2]

DEV = "cuda:0"; W = "/home/ps/sam3/sam3_weights/sam3.pt"
model = build_sam3_image_model(checkpoint_path=W, load_from_HF=False, device=DEV)
model.to(DEV).eval()

# (a) full cell down to ~1008, (b) cropland-dense 1008 tile (window with max label==1)
full = np.array(Image.fromarray(rgb).resize((1008, 1008), Image.BILINEAR))
crop_sum = lbl.astype(np.float32)
# integral-ish: find 1008 window with most cropland (coarse search)
by = bx = 0; bestc = -1
for y in range(0, max(1, H - 1008 + 1), 256):
    for x in range(0, max(1, Wd - 1008 + 1), 256):
        c = crop_sum[y:y + 1008, x:x + 1008].sum()
        if c > bestc:
            bestc = c; by, bx = y, x
tile = rgb[by:by + 1008, bx:bx + 1008]
print(f"dense tile @({by},{bx}) cropland frac={(lbl[by:by+1008,bx:bx+1008]==1).mean():.2f}", flush=True)

for tag, img in [("FULL@1008", full), ("DENSE-TILE", tile)]:
    for conf in [0.05, 0.25]:
        proc = Sam3Processor(model, device=DEV, confidence_threshold=conf)
        for prompt in ["farmland", "field", "a farm field", "agricultural land"]:
            st = proc.set_image(Image.fromarray(img))
            st = proc.set_text_prompt(prompt=prompt, state=st)
            m = st.get("masks")
            if m is None or (hasattr(m, "numel") and m.numel() == 0):
                print(f"  {tag} conf={conf} '{prompt}': EMPTY", flush=True)
                continue
            mm = m.squeeze(1).cpu().numpy().astype(bool)
            areas = [int(mm[i].sum()) for i in range(mm.shape[0])]
            cov = 100.0 * mm.any(0).sum() / (img.shape[0] * img.shape[1])
            print(f"  {tag} conf={conf} '{prompt}': N={mm.shape[0]} "
                  f"area[min/med/max]={min(areas)}/{int(np.median(areas))}/{max(areas)} cov={cov:.1f}%", flush=True)
