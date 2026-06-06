"""Publication figures for the 1m parcel-level cropland result (Gansu cross-county).

Fig1 parcel_size_sweep.png : count-/area-weighted parcel F1 vs minimum mapping unit (the MMU story:
      area-F1 stays >=0.93 everywhere; count-F1 crosses 0.9 at the standard 0.5 ha MMU).
Fig2 ablation_parcel.png   : parcel-level gain from the size-aware loss (+small-weight) over the
      boundary+multitemporal model, across area-weighted / 0.5ha-MMU / unfiltered metrics.
English labels (paper-ready; no CJK font dependency). Data = measured (parcel_eval.py / _fused.py).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/mnt/sda/zf/landform/results/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})

# ---- Fig 1: parcel-size sweep (best model: dino_1m_v2_smallw) ----
min_ha = np.array([0.0001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5])
count_f1 = np.array([0.732, 0.734, 0.744, 0.775, 0.821, 0.867, 0.917])
area_f1 = np.array([0.929, 0.929, 0.929, 0.930, 0.934, 0.939, 0.948])
fig, ax = plt.subplots(figsize=(7.2, 5))
ax.semilogx(min_ha, area_f1, "s-", color="#1f77b4", lw=2, ms=7, label="Area-weighted parcel F1")
ax.semilogx(min_ha, count_f1, "o-", color="#d62728", lw=2, ms=7, label="Count-weighted parcel F1")
ax.axhline(0.9, ls="--", c="gray", lw=1.5, label="0.9 target")
ax.axvline(0.5, ls=":", c="green", lw=1.5)
ax.annotate("0.5 ha MMU", xy=(0.5, 0.70), xytext=(0.12, 0.66), color="green",
            arrowprops=dict(arrowstyle="->", color="green"))
ax.set_xlabel("Minimum mapping unit (parcel area, ha)")
ax.set_ylabel("Cropland F1 (parcel level)")
ax.set_title("Gansu cross-county parcel-level cropland accuracy")
ax.set_ylim(0.65, 0.97); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig1_parcel_size_sweep.png", dpi=200); plt.close(fig)

# ---- Fig 2: size-aware-loss ablation at parcel level ----
metrics = ["Area-weighted", "0.5 ha MMU\n(count)", "Unfiltered\n(count)"]
bnd_mt = [0.916, 0.906, 0.678]      # boundary + multitemporal
smallw = [0.929, 0.917, 0.732]      # + size-aware loss
x = np.arange(len(metrics)); w = 0.36
fig, ax = plt.subplots(figsize=(7.2, 5))
b1 = ax.bar(x - w / 2, bnd_mt, w, label="boundary + multitemporal", color="#aec7e8")
b2 = ax.bar(x + w / 2, smallw, w, label="+ size-aware loss (ours)", color="#1f77b4")
ax.axhline(0.9, ls="--", c="gray", lw=1.5, label="0.9 target")
ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=10); ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylabel("Cropland F1 (parcel level)")
ax.set_title("Effect of size-aware loss (Gansu cross-county, 1m)")
ax.set_ylim(0.6, 0.99); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig2_ablation_parcel.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- Fig 3: pixel vs parcel (why the unit matters) ----
fig, ax = plt.subplots(figsize=(6.4, 5))
labels = ["Pixel\nF1", "Parcel\narea-wtd", "Parcel\n0.5ha-MMU"]
vals = [0.871, 0.929, 0.917]
bars = ax.bar(labels, vals, color=["#ff7f0e", "#1f77b4", "#2ca02c"], width=0.6)
ax.axhline(0.9, ls="--", c="gray", lw=1.5, label="0.9 target")
ax.bar_label(bars, fmt="%.3f", padding=3)
ax.set_ylabel("Cropland F1"); ax.set_ylim(0.7, 0.97)
ax.set_title("Pixel saturates ~0.87; parcel level clears 0.9"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "fig3_pixel_vs_parcel.png", dpi=200); plt.close(fig)

print(f"saved 3 figures to {OUT}", flush=True)
for f in sorted(OUT.glob("*.png")):
    print("  ", f.name, flush=True)
