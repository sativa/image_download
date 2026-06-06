"""Dump the full row-normalised confusion matrix for a 7-class land-cover checkpoint
on the 120-cell standard test (rows = true class, columns = predicted %, so the diagonal
is recall and off-diagonals show WHERE each class is confused)."""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HOME = Path("/home/ps/landform"); sys.path.insert(0, str(HOME / "sidecar"))
from train_dino_1m import norm6
from train_dino_1m_v2 import load_ndvi_full
from train_dino_1m_v3 import DinoV3FreqUNet, DINOV3_SAT

NAMES = ["nodata", "耕地", "园地", "林地", "草地", "水体", "建筑", "荒漠", "设施大棚"]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/mnt/sda/zf/landform/results/dino_v3_7class/best.pt")
    p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m")
    p.add_argument("--label-dir", default="/mnt/sda/zf/landform/data/c_1m_label7")
    p.add_argument("--ncls", type=int, default=8)
    p.add_argument("--n-cells", type=int, default=120)
    p.add_argument("--merge", default="", help="fold a class into another, e.g. 8:6 (设施大棚->建筑)")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    dev = a.device; NCLS = a.ncls; cs = 448

    from transformers import AutoModel
    d3 = AutoModel.from_pretrained(DINOV3_SAT, local_files_only=True)
    model = DinoV3FreqUNet(d3, num_classes=NCLS, in_channels=11, unfreeze_last_n=4).to(dev)
    sd = torch.load(a.ckpt, map_location=dev, weights_only=True)
    msd = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False)
    model.eval()

    import json
    man = json.loads((Path(a.data_dir) / "manifest.json").read_text())
    te = [n for n in man["test"] if (Path(a.label_dir) / f"{n}.npy").exists()][:a.n_cells]
    conf = np.zeros((NCLS, NCLS), np.int64)
    for nm in te:
        x6 = np.load(Path(a.data_dir) / f"{nm}.npz")["x6"]; lbl = np.load(Path(a.label_dir) / f"{nm}.npy")
        _, SZ, SZw = x6.shape
        ndvi = load_ndvi_full(nm, SZ, SZw)
        acc = np.zeros((NCLS, SZ, SZw), np.float32); cnt = np.zeros((SZ, SZw), np.float32)
        ys = list(range(0, max(1, SZ - cs + 1), cs)); xs = list(range(0, max(1, SZw - cs + 1), cs))
        if ys[-1] != SZ - cs: ys.append(max(0, SZ - cs))
        if xs[-1] != SZw - cs: xs.append(max(0, SZw - cs))
        for t in ys:
            for l in xs:
                xc = norm6(x6[:, t:t + cs, l:l + cs])
                nd = ndvi[:, t:t + cs, l:l + cs] if ndvi is not None else np.zeros((5, cs, cs), np.float32)
                xc = np.concatenate([xc, nd], 0)
                xb = torch.from_numpy(xc).unsqueeze(0).to(dev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(xb)[0]
                    if lg.shape[-2:] != (cs, cs):
                        lg = F.interpolate(lg, size=(cs, cs), mode="bilinear", align_corners=False)
                    pr = torch.softmax(lg.float(), 1)[0].cpu().numpy()
                acc[:, t:t + cs, l:l + cs] += pr; cnt[t:t + cs, l:l + cs] += 1
        pred = (acc / np.maximum(cnt, 1)).argmax(0); v = lbl > 0
        idx = np.ravel_multi_index((lbl[v].ravel().clip(0, NCLS - 1), pred[v].ravel().clip(0, NCLS - 1)), (NCLS, NCLS))
        conf += np.bincount(idx, minlength=NCLS * NCLS).reshape(NCLS, NCLS)

    present = list(range(1, NCLS))
    if a.merge:                                            # e.g. "8:6" -> fold class 8 into 6
        s, d = (int(x) for x in a.merge.split(":"))
        conf[d, :] += conf[s, :]; conf[:, d] += conf[:, s]; conf[s, :] = 0; conf[:, s] = 0
        present = [c for c in present if c != s]
        print(f"[merged class {s}({NAMES[s]}) -> {d}({NAMES[d]})]")

    rowsum = conf.sum(1, keepdims=True)
    rn = 100.0 * conf / np.maximum(rowsum, 1)
    oa = np.diag(conf)[1:].sum() / max(1, conf[1:, 1:].sum())
    print("\n=== 行归一化混淆矩阵 (行=真值, 列=预测%, 对角=召回) ===")
    hdr = "true\\pred  " + "".join("%7s" % NAMES[c] for c in range(1, NCLS))
    print(hdr)
    for r in range(1, NCLS):
        print("%-9s " % NAMES[r] + "".join("%6.1f " % rn[r, c] for c in range(1, NCLS)))
    print("\nOA(valid)=%.4f" % oa)
    f1s = []
    for r in present:
        tp = conf[r, r]; fp = conf[1:, r].sum() - tp; fn = conf[r, 1:].sum() - tp
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec); f1s.append(f1)
        conf_to = [(NAMES[c], rn[r, c]) for c in np.argsort(-rn[r]) if c != r and c >= 1 and rn[r, c] > 5][:3]
        conf_str = ", ".join("%s %.0f%%" % (nm, v) for nm, v in conf_to) or "-"
        print("  %-6s P=%.3f R=%.3f F1=%.3f  (main confusion: %s)" % (NAMES[r], prec, rec, f1, conf_str))
    print("  macro-F1 (%d classes) = %.4f" % (len(present), float(np.mean(f1s))))


if __name__ == "__main__":
    main()
