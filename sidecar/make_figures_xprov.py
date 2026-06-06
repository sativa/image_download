"""Cross-province (Gansu vs Changzhi) figures — the domain-invariance headline.

fig4 generalization_gap.png : same base model (dino_1m, 6ch, zero adaptation) on Gansu (in-domain)
      vs Changzhi (cross-province), parcel-level. Area-weighted gap ~0 (0.903 -> 0.918) = 1m RGB
      texture is domain-invariant; the cross-province "collapse" was a pixel/10m-spectral artifact.
fig5 changzhi_size_sweep.png : Changzhi cross-province count-/area-F1 vs minimum mapping unit
      (mirror of the Gansu fig1) — area >=0.92 everywhere; count crosses 0.9 by ~1 ha.
English labels, paper-ready. Data = measured (changzhi_parcel_eval.py / parcel_eval.py --plain).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/mnt/sda/zf/landform/results/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})

# ---- fig4: Gansu vs Changzhi, same base model (apples-to-apples) ----
metrics = ["Pixel\nF1", "Parcel\narea-wtd", "Parcel\n0.5ha-MMU", "Parcel\nunfiltered"]
gansu = [0.860, 0.903, 0.901, 0.676]        # base dino_1m, Gansu cross-county
cz_base = [0.843, 0.918, 0.882, 0.761]      # base dino_1m, Changzhi cross-province (zero adaptation)
cz_adapt = [0.848, 0.928, 0.893, 0.750]     # + semi-sup domain adaptation on unlabeled Changzhi tiles
x = np.arange(len(metrics)); w = 0.27
fig, ax = plt.subplots(figsize=(8.4, 5))
b1 = ax.bar(x - w, gansu, w, label="Gansu (in-domain)", color="#1f77b4")
b2 = ax.bar(x, cz_base, w, label="Changzhi (cross-province, zero adapt)", color="#ff7f0e")
b3 = ax.bar(x + w, cz_adapt, w, label="Changzhi (+ semi-sup adaptation)", color="#2ca02c")
ax.axhline(0.9, ls="--", c="gray", lw=1.5, label="0.9 target")
for b in (b1, b2, b3): ax.bar_label(b, fmt="%.3f", padding=2, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylabel("Cropland F1")
ax.set_title("Cross-province generalization + semi-supervised adaptation")
ax.set_ylim(0.6, 0.99); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=9, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig4_generalization_gap.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- fig5: Changzhi cross-province size sweep ----
min_ha = np.array([0.005, 0.05, 0.1, 0.2, 0.5, 1.0])
count_f1 = np.array([0.761, 0.788, 0.811, 0.840, 0.882, 0.911])
area_f1 = np.array([0.918, 0.919, 0.921, 0.926, 0.937, 0.947])
fig, ax = plt.subplots(figsize=(7.2, 5))
ax.semilogx(min_ha, area_f1, "s-", color="#ff7f0e", lw=2, ms=7, label="Area-weighted parcel F1")
ax.semilogx(min_ha, count_f1, "o-", color="#d62728", lw=2, ms=7, label="Count-weighted parcel F1")
ax.axhline(0.9, ls="--", c="gray", lw=1.5, label="0.9 target")
ax.axvline(0.5, ls=":", c="green", lw=1.5); ax.annotate("0.5 ha MMU", xy=(0.5, 0.72), xytext=(0.04, 0.70),
            color="green", arrowprops=dict(arrowstyle="->", color="green"))
ax.set_xlabel("Minimum mapping unit (parcel area, ha)")
ax.set_ylabel("Cropland F1 (parcel level)")
ax.set_title("Changzhi (Shanxi) cross-province parcel-level accuracy")
ax.set_ylim(0.70, 0.97); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig5_changzhi_size_sweep.png", dpi=200); plt.close(fig)

print("saved fig4_generalization_gap.png + fig5_changzhi_size_sweep.png", flush=True)
