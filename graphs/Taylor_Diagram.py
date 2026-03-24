"""
Taylor Diagram - Clean, correct implementation using matplotlib.
A Taylor diagram summarizes model performance using:
  - Radial axis    → Standard Deviation (normalized to obs = 1)
  - Angular axis   → Correlation Coefficient  (θ = arccos(R))
  - Green contours → Centered RMSD from the reference point
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "figure.dpi": 150,
})

# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLE DATA  (replace with your own or load from Excel)
# ══════════════════════════════════════════════════════════════════════════════
# Each entry: (model_name, normalized_std, correlation)
# normalized_std = SD_predicted / SD_observed   → observed point sits at (r=1, θ=0)
# correlation    = R  (Pearson)

MODELS = [
    # name           norm-std   R
    ("Observed",      1.00,    1.000),   # reference – always plotted as a star
    ("Model A",       0.85,    0.920),
    ("Model B",       1.10,    0.880),
    ("Model C",       0.72,    0.750),
    ("Model D",       1.30,    0.960),
    ("Model E",       0.60,    0.650),
    ("Model F",       1.05,    0.990),
    ("Model G",       0.95,    0.830),
]

# ══════════════════════════════════════════════════════════════════════════════
#  CORE DRAWING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def draw_taylor_diagram(models, title="Taylor Diagram", output=None):
    """
    Parameters
    ----------
    models : list of (name, norm_std, correlation)
        First entry should be the observed/reference point.
    title  : str
    output : str or None  – file path to save; None → show interactively
    """
    ref_name, ref_std, _ = models[0]          # reference (observed)
    plot_models = models[1:]                   # all non-reference models

    # ── figure & polar axes (only first quadrant: 0 → π/2) ─────────────────
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, polar=True)

    # Restrict to first quadrant
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    # Radial extent
    all_stds = [s for _, s, _ in models]
    r_max    = max(all_stds) * 1.30
    r_max    = max(r_max, 1.5)               # at least 1.5
    ax.set_rlim(0, r_max)

    # ── correlation labels on the arc (top) ─────────────────────────────────
    corr_ticks = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                           0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    theta_ticks = np.arccos(corr_ticks)           # radians

    ax.set_thetagrids(np.degrees(theta_ticks),
                      labels=[str(c) for c in corr_ticks],
                      fontsize=9)
    ax.set_xlabel("Correlation Coefficient  $R$", labelpad=15, fontsize=12)

    # ── radial (std) grid & labels ───────────────────────────────────────────
    r_ticks = np.arange(0, r_max + 0.01, 0.25)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{v:.2f}" for v in r_ticks], fontsize=8)
    ax.yaxis.set_tick_params(labelcolor="grey")

    # Radial axis label (rotated along the left side)
    ax.text(np.radians(90), r_max * 1.15, "Normalised Standard Deviation  $\\sigma_n$",
            ha="center", va="center", fontsize=12, rotation=0)

    # ── reference std arc (dashed) ───────────────────────────────────────────
    theta_arc = np.linspace(0, np.pi / 2, 300)
    ax.plot(theta_arc, np.full_like(theta_arc, ref_std),
            color="black", linestyle="--", linewidth=1.2, alpha=0.6, zorder=2)

    # ── centred-RMSD contours (green dashed circles centred on reference) ────
    crmsd_levels = np.arange(0.25, r_max + 0.01, 0.25)
    for crmsd in crmsd_levels:
        # arc of points at fixed centred-RMSD from (θ=0, r=ref_std)
        # CRMSD² = ref² + r² - 2·ref·r·cos(θ)  →  solve for allowed (θ,r) pairs
        theta_c = np.linspace(0, np.pi / 2, 500)
        r_c     = (ref_std * np.cos(theta_c) +
                   np.sqrt(np.maximum(0, crmsd**2 - ref_std**2 * np.sin(theta_c)**2)))
        # keep only points inside the plot
        mask = (r_c >= 0) & (r_c <= r_max)
        if mask.sum() < 2:
            continue
        ax.plot(theta_c[mask], r_c[mask],
                color="forestgreen", linestyle=":", linewidth=0.9, alpha=0.55, zorder=1)
        # label on the bottom (θ ≈ 0 side) near r_max
        idx = mask.nonzero()[0][-1]
        ax.text(theta_c[idx], r_c[idx],
                f" {crmsd:.2f}", color="forestgreen", fontsize=7, va="center")

    # ── small legend for RMSD label ──────────────────────────────────────────
    ax.text(np.radians(55), r_max * 0.92,
            "— Centred RMSD", color="forestgreen",
            fontsize=8, ha="center", style="italic")

    # ── plot reference / observed point ─────────────────────────────────────
    ax.plot(0, ref_std, marker="*", ms=16, color="black",
            ls="none", label=ref_name, zorder=10, clip_on=False)

    # ── plot model points ────────────────────────────────────────────────────
    markers  = ["o", "s", "^", "D", "p", "P", "X", "v", "h", "<"]
    palette  = plt.cm.tab10(np.linspace(0, 0.9, len(plot_models)))

    for i, (name, std, corr) in enumerate(plot_models):
        theta = np.arccos(np.clip(corr, -1, 1))
        ax.plot(theta, std,
                marker=markers[i % len(markers)],
                ms=9, ls="none",
                color=palette[i],
                markeredgecolor="black", markeredgewidth=0.6,
                label=name, zorder=8)

    # ── legend ───────────────────────────────────────────────────────────────
    ax.legend(loc="upper right",
              bbox_to_anchor=(1.35, 1.10),
              frameon=True, fontsize=9,
              title="Models", title_fontsize=10)

    # ── title ────────────────────────────────────────────────────────────────
    ax.set_title(title, pad=20, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output:
        plt.savefig(output, bbox_inches="tight")
        print(f"Saved → {output}")
    else:
        plt.show()

    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIONAL: load from Excel  (uncomment & adapt as needed)
# ══════════════════════════════════════════════════════════════════════════════
# import pandas as pd
#
# def load_from_excel(path):
#     df = pd.read_excel(path)
#     # Expected columns: Model, SD_Observed, SD_Predicted, R2
#     obs_std = df["SD_Observed"].iloc[0]
#     models  = [("Observed", 1.0, 1.0)]
#     for _, row in df.iterrows():
#         norm_std = row["SD_Predicted"] / obs_std
#         corr     = np.sqrt(np.clip(row["R2"], 0, 1))   # R2 → R
#         models.append((row["Model"], norm_std, corr))
#     return models


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── use sample data above ────────────────────────────────────────────────
    draw_taylor_diagram(
        MODELS,
        title="Taylor Diagram – Suspended Sediment Concentration",
        output="/mnt/user-data/outputs/taylor_diagram.png",
    )