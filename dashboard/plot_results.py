import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[1] / "experiments" / "vision" / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

data = {
    "standard_small": {
        "train_loss": [2.0024,1.8603,1.7043,1.5853,1.5134,1.4502,1.4077,1.3740,1.3378,1.3011,1.2761,1.2565,1.2279,1.2029,1.1827,1.1545,1.1398,1.1145,1.0944,1.0757,1.0582,1.0392,1.0188,1.0058,0.9835,0.9724,0.9592,0.9435,0.9316,0.9157,0.9101,0.8990,0.8845,0.8791,0.8698,0.8678,0.8633,0.8568,0.8554,0.8530],
        "val_loss":   [1.9186,1.7606,1.5972,1.5103,1.4546,1.3755,1.3660,1.3099,1.2724,1.2635,1.2160,1.1985,1.1785,1.1589,1.1626,1.1172,1.1169,1.0766,1.0738,1.0754,1.0444,1.0373,1.0427,1.0076,0.9970,0.9875,0.9792,0.9889,0.9670,0.9574,0.9715,0.9594,0.9624,0.9510,0.9465,0.9400,0.9439,0.9437,0.9414,0.9401],
        "val_acc":    [33.67,41.32,49.80,53.71,55.35,60.06,60.18,63.50,64.78,64.92,67.06,67.68,69.28,70.03,70.29,71.94,72.05,73.75,73.81,74.12,75.52,75.78,75.51,77.40,77.91,78.13,78.46,78.34,79.11,79.76,79.36,79.75,80.12,80.28,80.41,81.08,80.59,80.67,80.79,80.88],
    },
    "standard_medium": {
        "train_loss": [1.9622,1.7857,1.6178,1.5254,1.4594,1.3921,1.3391,1.3022,1.2590,1.2218,1.1914,1.1532,1.1247,1.0968,1.0719,1.0442,1.0183,0.9878,0.9692,0.9475,0.9241,0.9035,0.8870,0.8682,0.8500,0.8364,0.8260,0.8152,0.8105,0.8105],
        "val_loss":   [1.9059,1.6558,1.6339,1.4698,1.4175,1.3548,1.2710,1.2206,1.2100,1.1620,1.1512,1.1316,1.0851,1.0825,1.0741,1.0195,1.0293,1.0044,0.9676,0.9880,0.9515,0.9447,0.9448,0.9387,0.9298,0.9270,0.9157,0.9160,0.9141,0.9138],
        "val_acc":    [33.16,46.15,48.69,55.20,58.55,60.83,65.08,66.82,67.83,70.19,70.47,71.58,73.41,73.45,73.89,76.63,76.29,77.63,79.17,78.29,79.95,80.27,80.44,80.71,81.30,81.57,81.62,81.88,81.80,81.78],
    },
    "gated_small": {
        "train_loss": [2.0176,1.8766,1.7109,1.5876,1.5104,1.4552,1.4032,1.3672,1.3325,1.3004,1.2767,1.2425,1.2144,1.1901,1.1666,1.1460,1.1233,1.1051,1.0889,1.0739,1.0507,1.0334,1.0137,1.0078,0.9818,0.9719,0.9530,0.9410,0.9314,0.9185,0.9064,0.8948,0.8881,0.8807,0.8718,0.8668,0.8631,0.8580,0.8585,0.8542],
        "val_loss":   [1.9716,1.8058,1.5952,1.4671,1.4672,1.3789,1.3490,1.3129,1.3073,1.2481,1.2527,1.1904,1.1981,1.1520,1.1282,1.1061,1.1304,1.1016,1.0896,1.0584,1.0397,1.0229,1.0363,1.0095,0.9988,0.9925,0.9777,0.9803,0.9630,0.9824,0.9563,0.9561,0.9636,0.9505,0.9424,0.9461,0.9456,0.9443,0.9397,0.9401],
        "val_acc":    [31.03,39.97,48.88,55.22,55.57,59.63,60.67,62.86,63.18,66.10,65.47,68.89,68.38,70.57,71.60,72.43,72.09,73.18,73.77,75.15,75.43,76.30,75.49,77.48,77.82,77.68,78.47,78.55,79.40,78.53,79.65,79.82,79.50,80.20,80.42,80.48,80.69,80.54,80.64,80.74],
    },
    "gated_medium": {
        "train_loss": [1.9754,1.7928,1.6293,1.5261,1.4583,1.3953,1.3481,1.3078,1.2697,1.2356,1.1981,1.1708,1.1369,1.1109,1.0825,1.0528,1.0224,1.0005,0.9787,0.9535,0.9314,0.9084,0.8890,0.8705,0.8579,0.8415,0.8332,0.8206,0.8173,0.8144],
        "val_loss":   [1.9219,1.6521,1.5459,1.4531,1.3720,1.3317,1.2708,1.2658,1.2360,1.1767,1.1567,1.1609,1.1301,1.0950,1.0512,1.0285,1.0526,1.0165,0.9979,0.9948,0.9632,0.9549,0.9470,0.9422,0.9420,0.9242,0.9291,0.9194,0.9186,0.9191],
        "val_acc":    [34.39,47.12,51.53,55.57,59.66,61.83,65.02,64.67,66.60,69.27,70.10,70.28,71.89,72.93,75.16,76.42,75.19,76.77,77.61,78.16,79.22,79.53,80.41,80.42,80.50,81.45,81.18,81.44,81.62,81.44],
    },
}

COLORS = {
    "standard_small":  "#4878CF",
    "standard_medium": "#6ACC65",
    "gated_small":     "#D65F5F",
    "gated_medium":    "#B47CC7",
}

LABELS = {
    "standard_small":  "Standard Small",
    "standard_medium": "Standard Medium",
    "gated_small":     "Gated Small",
    "gated_medium":    "Gated Medium",
}

plt.style.use("seaborn-v0_8-whitegrid")


# ── Figure 1: Validation accuracy curves ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for key, d in data.items():
    epochs = range(1, len(d["val_acc"]) + 1)
    ax.plot(epochs, d["val_acc"], color=COLORS[key], label=LABELS[key], linewidth=2)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
ax.set_title("Validation Accuracy over Training", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim(1, None)
fig.tight_layout()
out = FIGURES_DIR / "val_accuracy_curves.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")


# ── Figure 2: Loss curves (2×2 subplots) ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (key, d) in zip(axes.flatten(), data.items()):
    epochs = range(1, len(d["train_loss"]) + 1)
    ax.plot(epochs, d["train_loss"], color="#555555", label="Train loss", linewidth=1.8, linestyle="--")
    ax.plot(epochs, d["val_loss"],   color=COLORS[key], label="Val loss",   linewidth=2)
    ax.set_title(LABELS[key], fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim(1, None)
fig.suptitle("Training & Validation Loss", fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
out = FIGURES_DIR / "loss_curves.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")


# ── Figure 3: Final validation accuracy bar chart ───────────────────────────
keys   = list(data.keys())
finals = [data[k]["val_acc"][-1] for k in keys]
labels = [LABELS[k] for k in keys]
colors = [COLORS[k] for k in keys]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, finals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, finals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("Final Validation Accuracy (%)", fontsize=12)
ax.set_title("Final Validation Accuracy by Model", fontsize=14, fontweight="bold")
ax.set_ylim(0, max(finals) + 4)
ax.tick_params(axis="x", labelsize=11)
fig.tight_layout()
out = FIGURES_DIR / "final_accuracy_bar.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved: {out}")
