"""Pre-compute v27 (teacher) logits on every cell's S2 data.

Saves teacher_logits/{county}_{idx}.npy (uint8 quantized soft-prob, 3-class, H × W)
to be loaded during v29 distillation training.

Why quantize: full float32 logits = 22K × 240 × 240 × 3 × 4 bytes = 15 GB.
Quantized to uint8 probs = 4 GB. Loss in KL is negligible.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch


HOME = Path("/home/ps/landform")
sys.path.insert(0, str(HOME / "sidecar"))


S2_MEAN = np.array([400, 460, 320, 1800], dtype=np.float32)
S2_STD = np.array([200, 200, 200, 700], dtype=np.float32)
NDVI_MEAN = 0.5; NDVI_STD = 0.3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regions-json", type=Path, default=HOME / "data/v27_regions.json")
    p.add_argument("--s2-dir", type=Path, default=HOME / "data/v19_s2_raw")
    p.add_argument("--teacher-ckpt", type=Path, default=HOME / "results/v27/best.pt")
    p.add_argument("--out-dir", type=Path, default=HOME / "data/v29_teacher_logits")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"[1] load teacher (v27)", flush=True)
    import segmentation_models_pytorch as smp
    teacher = smp.Unet(encoder_name="efficientnet-b3", encoder_weights=None,
                       in_channels=5, classes=3).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device, weights_only=True))
    teacher = getattr(teacher, "eval")()

    regions = json.loads(args.regions_json.read_text())
    all_regions = regions["train"] + regions["test"]
    print(f"[2] {len(all_regions)} cells to process", flush=True)

    t0 = time.time(); done = 0; skipped = 0
    batch_inputs = []; batch_names = []

    def flush_batch():
        nonlocal batch_inputs, batch_names, done
        if not batch_inputs: return
        x = torch.from_numpy(np.stack(batch_inputs)).to(device)
        with torch.no_grad():
            logits = teacher(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        # Quantize prob → uint8 (×255)
        probs_q = np.clip(probs * 255, 0, 255).astype(np.uint8)
        for name, prob in zip(batch_names, probs_q):
            np.save(args.out_dir / f"{name}.npy", prob)
            done += 1
        batch_inputs = []; batch_names = []

    for r in all_regions:
        name = f"{r['county']}_{r['idx']}"
        out_path = args.out_dir / f"{name}.npy"
        if out_path.exists(): done += 1; continue
        s2_path = args.s2_dir / f"{name}.npz"
        if not s2_path.exists(): skipped += 1; continue
        data = np.load(s2_path)
        rgbnir = data["rgbnir"].astype(np.float32).copy()
        for b in range(4): rgbnir[b] = (rgbnir[b] - S2_MEAN[b]) / S2_STD[b]
        ndvi = (data["ndvi"].astype(np.float32) - NDVI_MEAN) / NDVI_STD
        x5 = np.concatenate([rgbnir, ndvi[None, ...]], axis=0).astype(np.float32)
        # Pad/crop to 224×224 (teacher's training size)
        H, W = x5.shape[1], x5.shape[2]
        sz = 224
        if H != sz or W != sz:
            # Pad to make at least sz×sz, then center crop
            ph = max(0, sz - H); pw = max(0, sz - W)
            if ph or pw:
                x5 = np.pad(x5, ((0,0),(0,ph),(0,pw)), mode="edge")
            x5 = x5[:, :sz, :sz]
        batch_inputs.append(x5); batch_names.append(name)
        if len(batch_inputs) >= args.batch_size:
            flush_batch()
            if done % 500 == 0:
                print(f"  {done}/{len(all_regions)} ({time.time()-t0:.0f}s)", flush=True)
    flush_batch()

    print(f"[done] {done} cells, {skipped} skipped in {time.time()-t0:.0f}s", flush=True)
    # Show disk usage
    sz = sum(p.stat().st_size for p in args.out_dir.glob("*.npy"))
    print(f"  disk: {sz/1024/1024:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
