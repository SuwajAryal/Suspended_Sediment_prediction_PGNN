#!/usr/bin/env python
"""
Advanced Taylor Diagram Generator for Thesis Publication.
Uses axisartist for professional, publication-ready geometry.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.grid_finder as grid_finder
from pathlib import Path

# --- Publication Settings ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300
})

BASE_DIR = Path(__file__).resolve().parent
INPUT_XLSX = BASE_DIR / "model_performance_Comparison.xlsx"

class TaylorDiagram:
    """
    Implementation of a Taylor Diagram using axisartist for curved axes.
    """
    def __init__(self, ref_std, fig=None, rect=111, label='Observed', srange=(0, 1.5)):
        self.ref_std = ref_std
        tr = PolarAxes.PolarTransform()

        # Correlation labels (the arc)
        r_ticks = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        t_locs = np.arccos(r_ticks)
        grid_loc = grid_finder.FixedLocator(t_locs)
        grid_form = grid_finder.DictFormatter(dict(zip(t_locs, map(str, r_ticks))))

        # Set up the floating axis helper
        gh = floating_axes.GridHelperCurveLinear(
            tr, extremes=(0, np.pi/2, srange[0], srange[1]),
            grid_locator1=grid_loc, tick_formatter1=grid_form
        )

        if fig is None:
            fig = plt.figure(figsize=(8, 8))

        ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=gh)
        fig.add_subplot(ax)

        # Style the labels and axes
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation Coefficient ($R$)")

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text(r"Standard Deviation ($\sigma$)")

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("left")
        ax.axis["bottom"].set_visible(False)
        
        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        # Reference point (The Gold Standard)
        self.ax.plot([0], self.ref_std, 'k*', ls='', ms=12, label=label, zorder=10)
        
        # Reference Standard Deviation Arc
        t = np.linspace(0, np.pi/2)
        r = np.zeros_like(t) + self.ref_std
        self.ax.plot(t, r, 'k--', alpha=0.5, lw=1)

        # Centered RMSD Contours
        rs, ts = np.meshgrid(np.linspace(srange[0], srange[1]), np.linspace(0, np.pi/2))
        rms = np.sqrt(self.ref_std**2 + rs**2 - 2 * self.ref_std * rs * np.cos(ts))
        contours = self.ax.contour(ts, rs, rms, colors='green', linestyles=':', alpha=0.4)
        self.ax.clabel(contours, inline=1, fontsize=8, fmt='RMSD=%.1f')

        self.points = []

    def add_model(self, std, corr, *args, **kwargs):
        p, = self.ax.plot(np.arccos(corr), std, *args, **kwargs)
        self.points.append(p)
        return p

def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Flexible column mapping
    def n(s): return "".join(filter(str.isalnum, str(s).lower()))
    cmap = {n(c): c for c in df.columns}
    
    mapping = {
        "model": cmap["model"],
        "ssc_r2": cmap["sscr2"], "ssc_obs": cmap["sdobservedssc"], "ssc_pred": cmap["sdpredictedssc"],
        "ssl_r2": cmap["sslr2"], "ssl_obs": cmap["sdobservedssl"], "ssl_pred": cmap["sdpredictedssl"],
    }
    res = df[list(mapping.values())].copy()
    res.columns = mapping.keys()
    return res.dropna()


def _prepare_plot_values(data: pd.DataFrame, var_prefix: str) -> pd.DataFrame:
    obs_series = pd.to_numeric(data[f"{var_prefix}_obs"], errors="coerce")
    obs_std = float(obs_series.iloc[0])
    if obs_std <= 0:
        raise ValueError(f"Observed SD for {var_prefix.upper()} must be positive.")

    out = pd.DataFrame(
        {
            "model": data["model"].astype(str),
            "r2": pd.to_numeric(data[f"{var_prefix}_r2"], errors="coerce"),
            "sd_observed_raw": obs_series,
            "sd_predicted_raw": pd.to_numeric(data[f"{var_prefix}_pred"], errors="coerce"),
        }
    ).dropna()

    # Observed SD should be the same for all models; enforce one reference value.
    out["sd_observed_used"] = obs_std

    # Workbook contains R2, so correlation uses the positive branch.
    out["corr_used"] = np.sqrt(np.clip(out["r2"].to_numpy(dtype=float), 0.0, 1.0))

    # Normalize SD to make the reference point exactly 1.0.
    out["sd_reference_used"] = 1.0
    out["sd_model_used"] = out["sd_predicted_raw"] / out["sd_observed_used"]

    # Centered RMSD in normalized space.
    out["crmsd_used"] = np.sqrt(
        np.maximum(
            0.0,
            1.0
            + out["sd_model_used"].to_numpy(dtype=float) ** 2
            - 2.0 * out["sd_model_used"].to_numpy(dtype=float) * out["corr_used"].to_numpy(dtype=float),
        )
    )
    return out

def generate_plot(data, var_prefix, title, output_file):
    values = _prepare_plot_values(data, var_prefix)
    stds = values["sd_model_used"].to_numpy(dtype=float)
    corrs = values["corr_used"].to_numpy(dtype=float)
    models = values["model"].to_numpy()

    # Auto-scale in normalized SD space.
    max_val = max(np.max(stds), 1.0) * 1.25
    
    fig = plt.figure(figsize=(10, 8))
    dia = TaylorDiagram(1.0, fig=fig, srange=(0, max_val))
    
    markers = ['o', 's', '^', 'D', 'p', 'P', '*', 'X', 'h', 'v']
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, (m, s, c) in enumerate(zip(models, stds, corrs)):
        dia.add_model(s, c, marker=markers[i % 10], ms=9, ls='',
                      mfc=colors[i], mec='k', label=m, mew=0.7)

    # Place legend to the right of the plot
    fig.legend(loc='center right', bbox_to_anchor=(0.98, 0.5), frameon=True, title="Models")
    
    plt.suptitle(title + " (Normalized SD)", y=0.95, fontsize=16, fontweight='bold')
    dia._ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    return values

def main():
    try:
        df = _load_data(INPUT_XLSX)
        
        # SSC Diagram
        ssc_values = generate_plot(df, "ssc", "Taylor Diagram: Suspended Sediment Concentration", 
                                   BASE_DIR / "Taylor_Diagram_SSC.png")
        
        # SSL Diagram
        ssl_values = generate_plot(df, "ssl", "Taylor Diagram: Suspended Sediment Load", 
                                   BASE_DIR / "Taylor_Diagram_SSL.png")

        # Export value audit for transparency and reproducibility.
        ssc_values.to_csv(BASE_DIR / "Taylor_Diagram_SSC_values_used.csv", index=False)
        ssl_values.to_csv(BASE_DIR / "Taylor_Diagram_SSL_values_used.csv", index=False)

        print("\n[SSC values used]\n" + ssc_values.to_string(index=False))
        print("\n[SSL values used]\n" + ssl_values.to_string(index=False))
        
        print(f"Success! Diagrams saved to {BASE_DIR}")
    except Exception as e:
        print(f"Error processing workbook: {e}")

if __name__ == "__main__":
    main()