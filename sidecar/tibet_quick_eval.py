import sys, argparse, json
from pathlib import Path
sys.path.insert(0, "/home/ps/landform/sidecar")
import torch
from train_dino_1m_v3 import DinoV3FreqUNet, DINOV3_SAT, full_eval
from transformers import AutoModel
p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--data-dir", default="/mnt/sda/zf/landform/data/c_1m_tibet")
p.add_argument("--device", default="cuda:0")
a = p.parse_args()
d3 = AutoModel.from_pretrained(DINOV3_SAT, local_files_only=True)
m = DinoV3FreqUNet(d3, num_classes=3, in_channels=11, unfreeze_last_n=4).to(a.device)
sd = torch.load(a.ckpt, map_location=a.device, weights_only=True); msd = m.state_dict()
m.load_state_dict({k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}, strict=False); m.eval()
te = json.load(open(Path(a.data_dir) / "manifest.json"))["test"]
f1, pr, rc = full_eval(m, te, a.data_dir, a.device, 448, multitemporal=True)
print(f"TIBET test ({len(te)} cells) cropland pixel-F1={f1:.4f}  P{pr:.3f} R{rc:.3f}", flush=True)
