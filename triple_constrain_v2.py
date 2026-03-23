#!/usr/bin/env python
"""
Dual-Regime Physics-Informed Neural Network (PINN) for Suspended Sediment
==============================================================================
Description:
This script models the daily Suspended Sediment Concentration (SSC) and 
calculates the Suspended Sediment Load (SSL) in complex, monsoon-driven catchments.
Standard data-driven models suffer from hysteresis, extreme peak-underprediction, 
and dry-season overprediction. This architecture solves these using a 3-part 
regime-switching physics constraint.

Updates: 
Now includes advanced hydrological analytics:
- Hysteresis-Aware SRC Baseline (Rising vs Falling Limb)
- Observed vs Predicted Scatter Plots
- Taylor Diagrams (Correlation, RMSE, Standard Deviation)
- SHAP Feature Attributions
"""

import os
import sys
import warnings
import pathlib
from datetime import datetime, timedelta

SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent)
sys.path = [
    p for p in sys.path
    if os.path.abspath(p or os.getcwd()) != SCRIPT_DIR
]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
import shap

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# --- CONFIG ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

def _locate_data_file(filename="data.xlsx"):
    base = pathlib.Path(__file__).resolve().parent
    for p in [base] + list(base.parents):
        candidate = p / "data" / filename
        if candidate.exists():
            return candidate
    return base / "data" / filename

DATA_PATH = _locate_data_file("data.xlsx")
RUN_TAG    = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs" / f"PINN_DualRegime_Daily_{RUN_TAG}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHAP_DIR = OUTPUT_DIR / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS       = 800
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE     = 80
GRAD_CLIP    = 1.0
SHAP_BACKGROUND_SIZE = 128
SHAP_EXPLAIN_SIZE = 256
PLOT_DPI = 300

COLORS = {
    "obs": "#1f1f1f",
    "pinn": "#0f766e",
    "yang": "#b45309",
    "wash": "#7c3aed",
    "residual": "#b91c1c",
    "grid": "#d1d5db",
    "accent": "#1d4ed8",
}

HEXBIN_CMAP = LinearSegmentedColormap.from_list(
    "teal_orange",
    ["#0f766e", "#f97316"],
)

# --- THE 3-PART REGIME CONSTRAINTS ---
WASH_LOAD_MIN  = 3
LAMBDA_FLOOR   = 0.001
LAMBDA_CEILING = 0.005
LAMBDA_MASS    = 0.05

BASE_FEATURES = [
    "Q_m3s", "d50_mm", "rainfall_mm", "temp_C", "VS", "Soil_Moisture", "NDVI",
    "dQ", "Q_ratio", "Q_lag1", "Q_lag2", "Q_lag3", "Q_max_7d", "HI_proxy",
    "API", "P_MA3", "P_MA7", "P_MA14", "P_MA30", "SPI_30"
]
TARGET_COL = "SSC_mgL"
RAW_PHYS_COLS = ["Q_m3s", "d50_mm", "temp_C", "VS", "depth_m", "slope"]

def apply_publication_style():
    plt.rcParams.update({
        "figure.dpi": PLOT_DPI,
        "savefig.dpi": PLOT_DPI,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "grid.linestyle": "--",
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def nse_metric(obs, pred):
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    den = np.sum((obs - np.mean(obs)) ** 2)
    if den <= 0:
        return np.nan
    return 1.0 - (np.sum((pred - obs) ** 2) / den)


def pbias_metric(obs, pred):
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    den = np.sum(obs)
    if np.isclose(den, 0.0):
        return np.nan
    return 100.0 * np.sum(pred - obs) / den


def get_yang_capacity_np(phys_array, alpha=1.0, batch_size=4096):
    if len(phys_array) == 0:
        return np.array([], dtype=np.float64)

    outputs = []
    with torch.no_grad():
        for start in range(0, len(phys_array), batch_size):
            batch = torch.tensor(phys_array[start:start + batch_size], dtype=torch.float32, device=DEVICE)
            outputs.append((get_yang_capacity(batch) * alpha).cpu().numpy())
    return np.concatenate(outputs)


def build_phase_diagnostics(phase_df, phase_results, physics_meta):
    q_vals = phase_df["Q_m3s"].values
    observed_ssc = phase_df[TARGET_COL].values
    pinn_ssc = phase_results["preds_ssc"]
    observed_load = observed_ssc * q_vals * 0.0864
    pinn_load = phase_results["preds_load"]
    yang_ssc = get_yang_capacity_np(phase_df[RAW_PHYS_COLS].values, alpha=physics_meta["alpha"])
    wash_load = np.where(q_vals < physics_meta["global_q_mean"], WASH_LOAD_MIN, np.nan)
    ssc_residual = pinn_ssc - observed_ssc
    load_residual = pinn_load - observed_load

    return pd.DataFrame({
        "Date": pd.to_datetime(phase_df["date_dt"]),
        "Q_m3s": q_vals,
        "Observed_SSC_mgL": observed_ssc,
        "PINN_SSC_mgL": pinn_ssc,
        "Yang_Predicted_SSC_mgL": yang_ssc,
        "Wash_Load_Floor_SSC_mgL": wash_load,
        "SSC_Residual_mgL": ssc_residual,
        "Observed_Load_ton_day": observed_load,
        "PINN_Load_ton_day": pinn_load,
        "Mass_Balance_Diff_ton_day": load_residual,
        "Cumulative_Observed_Load_ton": np.cumsum(observed_load),
        "Cumulative_PINN_Load_ton": np.cumsum(pinn_load),
        "Cumulative_Mass_Balance_Diff_ton": np.cumsum(load_residual),
    })


def build_phase_diagnostics_all_days(phase_df, phase_all_day_results, physics_meta):
    q_vals = phase_df["Q_m3s"].values
    observed_ssc = phase_df[TARGET_COL].values if TARGET_COL in phase_df.columns else np.full(len(phase_df), np.nan)
    pinn_ssc = phase_all_day_results["preds_ssc"]
    observed_load = observed_ssc * q_vals * 0.0864
    pinn_load = phase_all_day_results["preds_load"]
    yang_ssc = get_yang_capacity_np(phase_df[RAW_PHYS_COLS].values, alpha=physics_meta["alpha"])
    wash_load = np.where(q_vals < physics_meta["global_q_mean"], WASH_LOAD_MIN, np.nan)
    ssc_residual = pinn_ssc - observed_ssc
    load_residual = pinn_load - observed_load

    obs_load_for_cumsum = np.nan_to_num(observed_load, nan=0.0)
    load_diff_for_cumsum = np.nan_to_num(load_residual, nan=0.0)

    return pd.DataFrame({
        "Date": pd.to_datetime(phase_df["date_dt"]),
        "Q_m3s": q_vals,
        "Observed_SSC_mgL": observed_ssc,
        "PINN_SSC_mgL": pinn_ssc,
        "Yang_Predicted_SSC_mgL": yang_ssc,
        "Wash_Load_Floor_SSC_mgL": wash_load,
        "SSC_Residual_mgL": ssc_residual,
        "Observed_Load_ton_day": observed_load,
        "PINN_Load_ton_day": pinn_load,
        "Mass_Balance_Diff_ton_day": load_residual,
        "Cumulative_Observed_Load_ton": np.cumsum(obs_load_for_cumsum),
        "Cumulative_PINN_Load_ton": np.cumsum(pinn_load),
        "Cumulative_Mass_Balance_Diff_ton": np.cumsum(load_diff_for_cumsum),
        "Has_Observed_SSC": ~pd.isna(observed_ssc),
    })


def export_diagnostics_workbook(rep_df, dfs, results, physics_meta, all_dfs, all_day_results):
    workbook_path = OUTPUT_DIR / "PINN_Diagnostics_Workbook.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        rep_df.to_excel(writer, sheet_name="Metrics", index=False)

        pd.DataFrame([{
            "Alpha": physics_meta["alpha"],
            "Global_Q_Mean_m3s": physics_meta["global_q_mean"],
            "Q_95th_m3s": physics_meta["q_95th"],
            "Wash_Load_Floor_mgL": WASH_LOAD_MIN,
        }]).to_excel(writer, sheet_name="Physics_Summary", index=False)

        for phase in ["train", "val", "test"]:
            diag_df = build_phase_diagnostics(dfs[phase], results[phase], physics_meta)
            diag_df.to_excel(writer, sheet_name=f"{phase}_diagnostics", index=False)

            all_days_diag_df = build_phase_diagnostics_all_days(all_dfs[phase], all_day_results[phase], physics_meta)
            all_days_diag_df.to_excel(writer, sheet_name=f"{phase}_all_days", index=False)

    return workbook_path

# --- DATA PROCESSOR ---
class DataProcessor:
    def __init__(self, path: pathlib.Path):
        self.raw = pd.read_excel(path)
        self._prepare()

    @staticmethod
    def _excel_to_dt(serial):
        return datetime(1899, 12, 30) + timedelta(days=int(serial))

    @staticmethod
    def _nepali_wy(dt):
        return dt.year if dt.month >= 4 else dt.year - 1

    def _prepare(self):
        df = self.raw.copy()
        df["date_dt"] = df["date"].apply(self._excel_to_dt)
        df = df.sort_values("date_dt").reset_index(drop=True)

        df["VS"] = (df["Q_m3s"] * df["slope"]) / (df["width_m"] * df["depth_m"])
        for col in ("Soil_Moisture", "NDVI"):
            if col not in df.columns: df[col] = 0.0

        Q = df["Q_m3s"]
        df["dQ"] = Q.diff().fillna(0.0)
        q7 = Q.rolling(7, min_periods=1).mean()
        df["Q_ratio"] = (Q / q7.replace(0, np.nan)).fillna(1.0)
        
        df["Q_lag1"] = Q.shift(1).ffill().fillna(0.0)
        df["Q_lag2"] = Q.shift(2).ffill().fillna(0.0)
        df["Q_lag3"] = Q.shift(3).ffill().fillna(0.0)
        df["Q_max_7d"] = Q.rolling(7, min_periods=1).max()

        P = df["rainfall_mm"]
        p_arr = P.values
        api_vals = np.zeros(len(df))
        for k in range(1, 31):
            shifted = np.concatenate([np.full(k, p_arr[:k].mean()), p_arr[:-k]])
            api_vals += (0.9 ** k) * shifted
        df["API"] = api_vals
        df["P_MA3"] = P.rolling(3, min_periods=1).mean()
        df["P_MA7"] = P.rolling(7, min_periods=1).mean()
        df["P_MA14"] = P.rolling(14, min_periods=1).mean()
        df["P_MA30"] = P.rolling(30, min_periods=1).mean()
        
        p30 = P.rolling(30, min_periods=5).sum()
        p30mu = p30.expanding(min_periods=5).mean()
        p30std = p30.expanding(min_periods=5).std().replace(0, np.nan)
        df["SPI_30"] = ((p30 - p30mu) / p30std).fillna(0.0)

        df["SSC_LP_lag1"] = df[TARGET_COL].shift(1).ffill().fillna(0.0)
        df["SSC_LP_lag2"] = df[TARGET_COL].shift(2).ffill().fillna(0.0)

        q_val = df["Q_m3s"]
        q_delta = q_val.diff().fillna(0.0)
        q_maxroll = q_val.rolling(7, min_periods=1).max().replace(0, np.nan)
        df["HI_proxy"] = (q_delta / q_maxroll).fillna(0.0)

        df["WY"] = df["date_dt"].apply(self._nepali_wy)
        self.full_df = df.copy()

        df_obs = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        self.df = df_obs
        self.wys = sorted(df_obs["WY"].unique())

    def split_and_scale(self, features):
        wys = self.wys
        test_wy, val_wy, train_wys = wys[-1], wys[-2], wys[:-2]
        train = self.df[self.df["WY"].isin(train_wys)].copy()
        val = self.df[self.df["WY"] == val_wy].copy()
        test = self.df[self.df["WY"] == test_wy].copy()

        train_all = self.full_df[self.full_df["WY"].isin(train_wys)].copy()
        val_all = self.full_df[self.full_df["WY"] == val_wy].copy()
        test_all = self.full_df[self.full_df["WY"] == test_wy].copy()

        x_scaler, y_scaler = RobustScaler(), RobustScaler()
        X_tr = x_scaler.fit_transform(train[features].values)
        X_va = x_scaler.transform(val[features].values)
        X_te = x_scaler.transform(test[features].values)
        y_tr = y_scaler.fit_transform(train[[TARGET_COL]].values)
        y_va = y_scaler.transform(val[[TARGET_COL]].values)
        y_te = y_scaler.transform(test[[TARGET_COL]].values)

        r_tr, r_va, r_te = train[RAW_PHYS_COLS].values, val[RAW_PHYS_COLS].values, test[RAW_PHYS_COLS].values
        
        return {
            "loaders": {
                "train": self.make_loader(X_tr, y_tr, r_tr, shuffle=True),
                "val": self.make_loader(X_va, y_va, r_va),
                "test": self.make_loader(X_te, y_te, r_te)
            },
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "dfs": {"train": train, "val": val, "test": test},
            "all_dfs": {"train": train_all, "val": val_all, "test": test_all},
        }

    @staticmethod
    def make_loader(X, y, r, batch_size=64, shuffle=False):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), 
                           torch.tensor(y, dtype=torch.float32), 
                           torch.tensor(r, dtype=torch.float32))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# --- MODEL ---
class _ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.LeakyReLU(0.1), nn.Dropout(0.1))
    def forward(self, x): return x + self.block(x)

class ResidualMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(in_dim, 128), nn.LeakyReLU(0.1))
        self.blocks  = nn.Sequential(*[_ResBlock(128) for _ in range(6)])
        self.head    = nn.Linear(128, 1)
    def forward(self, x): return self.head(self.blocks(self.project(x)))

# --- PHYSICS EQUATION (Yang) ---
def get_yang_capacity(phys_tensor):
    a1, a2, a3, a4, a5, a6 = 7.1178, -1.3908, -3.3899, 5.5007, -1.6111, -3.9525
    d50_m, temp, VS, depth, slope = phys_tensor[:, 1]*1e-3, phys_tensor[:, 2], phys_tensor[:, 3], phys_tensor[:, 4], phys_tensor[:, 5]
    nu = torch.clamp(1.14e-6 - 3.0e-9 * temp, min=1e-7)
    d = torch.clamp(d50_m, min=1e-6)
    w = torch.sqrt(torch.clamp(1.65 * 9.81 * d, min=1e-12))
    w = w * (2.0/3.0 + 36.0 * nu / (9.81 * d**2 * 1.65 + 1e-12))
    u_s = torch.sqrt(torch.clamp(9.81 * depth * slope, min=1e-12))
    
    l_wd = torch.log10(torch.clamp(w*d50_m/1e-6, min=1e-8))
    l_us = torch.log10(torch.clamp(u_s/torch.clamp(w, min=1e-8), min=1e-8))
    l_VS = torch.log10(torch.clamp(VS/torch.clamp(w, min=1e-8), min=1e-8))
    
    l_Ct_phys = (a1 + a2*l_wd + a3*l_us + (a4 + a5*l_wd + a6*l_us) * l_VS)
    return torch.pow(10.0, l_Ct_phys)

# --- TRAINING WITH REGIME-SWITCHING ---
def train_model(model, loaders, y_scaler):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    amp_scaler = GradScaler(enabled=USE_AMP)
    y_center = torch.tensor(y_scaler.center_[0], dtype=torch.float32, device=DEVICE)
    y_scale = torch.tensor(y_scaler.scale_[0], dtype=torch.float32, device=DEVICE)

    print("[TRAIN] Calibrating Yang's Capacity on Top 5% Extreme Flows...")
    all_actuals, all_yangs, all_qs = [], [], []
    for batch in loaders["train"]:
        yb, pb = batch[1].to(DEVICE), batch[2].to(DEVICE)
        actual_mgl = yb.squeeze() * y_scale + y_center
        yang_cap = get_yang_capacity(pb)
        q_val = pb[:, 0]
        
        all_actuals.append(actual_mgl)
        all_yangs.append(yang_cap)
        all_qs.append(q_val)

    all_actuals = torch.cat(all_actuals)
    all_yangs = torch.cat(all_yangs)
    all_qs = torch.cat(all_qs)

    q_95th = torch.quantile(all_qs, 0.95)
    top_5_mask = all_qs >= q_95th

    extreme_actuals = all_actuals[top_5_mask]
    extreme_yangs = all_yangs[top_5_mask]
    alpha = torch.median(extreme_actuals / torch.clamp(extreme_yangs, min=1e-6)).item()
    global_q_mean = torch.mean(all_qs)
    
    q_95th_value = q_95th.item()
    global_q_mean_value = global_q_mean.item()

    print(f"        -> Top 5% Flow Threshold: {q_95th_value:.2f} m³/s")
    print(f"        -> Extreme Flood Multiplier (Alpha): {alpha:.4f}")

    best_val, wait, best_state = float("inf"), 0, None
    loss_tracker = {"Data": 0, "Floor": 0, "Ceiling": 0, "Mass": 0}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_data, ep_down, ep_up, ep_mass = 0.0, 0.0, 0.0, 0.0
        
        for batch in loaders["train"]:
            xb, yb, pb = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=USE_AMP):
                pred = model(xb)
                data_loss = nn.functional.mse_loss(pred, yb)
                
                pr_mgl = pred.squeeze() * y_scale + y_center
                yb_mgl = yb.squeeze() * y_scale + y_center
                q_vals = pb[:, 0] 
                
                low_mask = q_vals < global_q_mean
                floor_loss = torch.tensor(0.0, device=DEVICE)
                if low_mask.any():
                    floor_penalty = torch.nn.functional.relu(pr_mgl[low_mask] - WASH_LOAD_MIN)
                    floor_loss = torch.mean((floor_penalty / y_scale)**2)
                
                high_mask = q_vals >= global_q_mean
                ceiling_loss = torch.tensor(0.0, device=DEVICE)
                if high_mask.any():
                    yang_cap_calibrated = get_yang_capacity(pb[high_mask]) * alpha
                    ceiling_penalty = torch.nn.functional.relu(yang_cap_calibrated - pr_mgl[high_mask])
                    ceiling_loss = torch.mean((ceiling_penalty / y_scale)**2)
                
                mass_diff = torch.abs(torch.sum(pr_mgl) - torch.sum(yb_mgl))
                mass_loss = (mass_diff / len(yb)) / y_scale 
                
                weighted_floor   = LAMBDA_FLOOR * floor_loss
                weighted_ceiling = LAMBDA_CEILING * ceiling_loss
                weighted_mass    = LAMBDA_MASS * mass_loss
                
                loss = data_loss + weighted_floor + weighted_ceiling + weighted_mass
                
                ep_data += data_loss.item()
                ep_down += weighted_floor.item()
                ep_up += weighted_ceiling.item()
                ep_mass += weighted_mass.item()
                
            amp_scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in loaders["val"]:
                val_loss += nn.functional.mse_loss(model(b[0].to(DEVICE)), b[1].to(DEVICE), reduction="sum").item()
        val_loss /= len(loaders["val"].dataset)
        scheduler.step(val_loss)

        if val_loss < best_val: 
            # Clone tensors to snapshot weights without relying on the stdlib copy module.
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val, wait = val_loss, 0
            loss_tracker = {"Data": ep_data, "Floor": ep_down, "Ceiling": ep_up, "Mass": ep_mass}
        else:
            wait += 1
            if wait >= PATIENCE: break
    
    if best_state: model.load_state_dict(best_state)
    physics_meta = {
        "alpha": alpha,
        "q_95th": q_95th_value,
        "global_q_mean": global_q_mean_value,
    }
    return model, loss_tracker, physics_meta

# --- ADVANCED ANALYTICS & PLOTTING ---
def run_shap_analysis(model, data, feature_names):
    print("\n[SHAP] Computing feature attributions on the test split...")
    train_x = data["loaders"]["train"].dataset.tensors[0]
    test_x = data["loaders"]["test"].dataset.tensors[0]

    background = train_x[:min(SHAP_BACKGROUND_SIZE, len(train_x))].to(DEVICE)
    explain_x = test_x[:min(SHAP_EXPLAIN_SIZE, len(test_x))].to(DEVICE)

    model.eval()
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(explain_x)
    shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap_array = np.asarray(shap_array)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    explain_np = explain_x.detach().cpu().numpy()
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.mean(np.abs(shap_array), axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(SHAP_DIR / "shap_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_array, explain_np, feature_names=feature_names, show=False)
    plt.title("SHAP Summary: PINN SSC Prediction", pad=12)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_array, explain_np, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance", pad=12)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary_bar.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

def plot_scatter(obs, pred, phase, title_prefix, unit):
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    hb = ax.hexbin(obs, pred, gridsize=34, cmap=HEXBIN_CMAP, mincnt=1, linewidths=0.2)
    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("Sample Count")

    max_val = max(np.max(obs), np.max(pred))
    min_val = min(np.min(obs), np.min(pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color=COLORS["residual"], lw=1.8, label="1:1 line")

    slope, intercept = np.polyfit(obs, pred, 1)
    fit_x = np.linspace(min_val, max_val, 200)
    ax.plot(fit_x, slope * fit_x + intercept, color=COLORS["accent"], lw=1.6, label="Best-fit line")

    r2_val = r2_score(obs, pred)
    rmse_val = np.sqrt(mean_squared_error(obs, pred))
    ax.text(
        0.04,
        0.96,
        f"R² = {r2_val:.3f}\nRMSE = {rmse_val:.2f} {unit}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.95},
    )

    style_axis(ax, f"{title_prefix} Observed vs PINN Prediction ({phase.capitalize()})", f"Observed ({unit})", f"Predicted ({unit})")
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, PLOTS_DIR / f"{phase.capitalize()}_{title_prefix}_Scatter.png")

def draw_taylor_axis(ax, obs, pred, title):
    """Draws a simplified Taylor diagram on a provided polar axis."""
    std_obs = np.std(obs)
    std_pred = np.std(pred)
    corr, _ = stats.pearsonr(obs, pred)
    corr = max(0.0, min(corr, 1.0)) # Clamp correlation for arccos
    
    # Polar coordinates: theta = arccos(R), radius = std
    theta = np.arccos(corr)
    
    # Setup polar axes for Taylor
    ax.set_theta_direction(-1) # Clockwise
    ax.set_theta_zero_location("N") # 0 at top
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    
    # Add correlation grid lines
    rs, ts = np.meshgrid(np.linspace(0, max(std_obs, std_pred)*1.2, 100), 
                         np.arccos(np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99])))
    ax.plot(ts.T, rs.T, color='#9ca3af', linestyle=':', alpha=0.8)

    # Plot Observation Reference
    ax.plot(0, std_obs, marker='*', color=COLORS["obs"], markersize=13, label="Observed")
    ax.plot(np.linspace(0, np.pi/2, 100), np.full(100, std_obs), linestyle='--', color=COLORS["obs"], alpha=0.45)
    
    # Plot Prediction
    ax.plot(theta, std_pred, marker='o', color=COLORS["pinn"], markersize=10, label="PINN")
    
    ax.set_title(title, va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=False)


def plot_taylor(obs, pred, phase, title_prefix):
    """Generates a simplified Polar Taylor Diagram"""
    fig = plt.figure(figsize=(7.2, 7.0))
    ax = fig.add_subplot(111, polar=True)
    draw_taylor_axis(
        ax,
        obs,
        pred,
        f"{title_prefix} Taylor Diagram ({phase.capitalize()})\nCorrelation as angle, standard deviation as radius",
    )
    save_figure(fig, PLOTS_DIR / f"{phase.capitalize()}_{title_prefix}_Taylor.png")


def plot_hydrograph(dates, obs, pred, phase, title_prefix, unit, metric_line, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.4), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.0]})
    axes[0].scatter(dates, obs, color=COLORS["obs"], s=18, alpha=0.75, edgecolors="none", label="Observed")
    axes[0].plot(dates, pred, color=COLORS["pinn"], lw=2.0, label="PINN")
    axes[0].fill_between(dates, obs, pred, color=COLORS["pinn"], alpha=0.10)
    style_axis(axes[0], f"{title_prefix} Hydrograph ({phase.capitalize()})", "", f"{title_prefix} ({unit})")
    axes[0].text(
        0.01,
        0.97,
        metric_line,
        transform=axes[0].transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.95},
    )
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    residual = pred - obs
    axes[1].axhline(0.0, color="#6b7280", lw=1.0)
    axes[1].fill_between(dates, residual, 0.0, color=COLORS["residual"], alpha=0.20)
    axes[1].plot(dates, residual, color=COLORS["residual"], lw=1.1)
    style_axis(axes[1], "Residual Structure", "Date", f"Residual ({unit})")
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator()))
    save_figure(fig, output_path)


def plot_summary_plate(dates, obs, pred, phase, title_prefix, unit, metric_line, output_path):
    fig = plt.figure(figsize=(14.2, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.28)

    ax_h = fig.add_subplot(gs[0, :])
    ax_s = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[1, 1], polar=True)

    ax_h.scatter(dates, obs, color=COLORS["obs"], s=16, alpha=0.72, edgecolors="none", label="Observed")
    ax_h.plot(dates, pred, color=COLORS["pinn"], lw=2.0, label="PINN")
    ax_h.fill_between(dates, obs, pred, color=COLORS["pinn"], alpha=0.10)
    style_axis(ax_h, f"{title_prefix} Summary Plate ({phase.capitalize()})", "Date", f"{title_prefix} ({unit})")
    ax_h.text(
        0.01,
        0.97,
        metric_line,
        transform=ax_h.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.95},
    )
    ax_h.legend(frameon=False, ncol=2, loc="upper right")
    ax_h.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_h.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_h.xaxis.get_major_locator()))

    hb = ax_s.hexbin(obs, pred, gridsize=30, cmap=HEXBIN_CMAP, mincnt=1, linewidths=0.2)
    cbar = fig.colorbar(hb, ax=ax_s, pad=0.02)
    cbar.set_label("Sample Count")
    max_val = max(np.max(obs), np.max(pred))
    min_val = min(np.min(obs), np.min(pred))
    ax_s.plot([min_val, max_val], [min_val, max_val], linestyle="--", color=COLORS["residual"], lw=1.8, label="1:1 line")
    slope, intercept = np.polyfit(obs, pred, 1)
    fit_x = np.linspace(min_val, max_val, 200)
    ax_s.plot(fit_x, slope * fit_x + intercept, color=COLORS["accent"], lw=1.6, label="Best-fit line")
    r2_val = r2_score(obs, pred)
    rmse_val = np.sqrt(mean_squared_error(obs, pred))
    ax_s.text(
        0.04,
        0.96,
        f"R² = {r2_val:.3f}\nRMSE = {rmse_val:.2f} {unit}",
        transform=ax_s.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.95},
    )
    style_axis(ax_s, "Observed vs Predicted", f"Observed ({unit})", f"Predicted ({unit})")
    ax_s.legend(frameon=False, loc="lower right")
    ax_s.set_aspect("equal", adjustable="box")

    draw_taylor_axis(ax_t, obs, pred, "Taylor Diagnostic\nCorrelation as angle, standard deviation as radius")
    save_figure(fig, output_path)

def run_src_hysteresis_baseline(dfs):
    print("\n[BASELINES] Fitting Hysteresis-Aware Sediment Rating Curve (SRC)...")
    train_df = dfs["train"]
    
    # Ensure no zeros/negatives for log transform
    train_df = train_df[(train_df["Q_m3s"] > 1e-6) & (train_df[TARGET_COL] > 1e-6)]
    
    # Split training into Rising and Falling Limbs
    rising_mask = train_df["dQ"] > 0
    falling_mask = train_df["dQ"] <= 0
    
    train_rising = train_df[rising_mask]
    train_falling = train_df[falling_mask]
    
    # Fit Rising Curve
    reg_rising = LinearRegression()
    reg_rising.fit(np.log10(train_rising["Q_m3s"].values).reshape(-1, 1), np.log10(train_rising[TARGET_COL].values))
    a_ris, b_ris = 10**reg_rising.intercept_, reg_rising.coef_[0]
    
    # Fit Falling Curve
    reg_falling = LinearRegression()
    reg_falling.fit(np.log10(train_falling["Q_m3s"].values).reshape(-1, 1), np.log10(train_falling[TARGET_COL].values))
    a_fal, b_fal = 10**reg_falling.intercept_, reg_falling.coef_[0]
    
    print(f"        -> Rising Limb Equation : SSC = {a_ris:.4f} * Q^{b_ris:.4f}")
    print(f"        -> Falling Limb Equation: SSC = {a_fal:.4f} * Q^{b_fal:.4f}")
    
    results_src = {}
    for phase in ["train", "val", "test"]:
        df_phase = dfs[phase]
        q_safe = np.maximum(df_phase["Q_m3s"].values, 1e-6)
        a_ssc = df_phase[TARGET_COL].values
        
        # Apply regime-specific curves
        is_rising = df_phase["dQ"].values > 0
        p_ssc = np.zeros_like(q_safe)
        p_ssc[is_rising] = a_ris * (q_safe[is_rising] ** b_ris)
        p_ssc[~is_rising] = a_fal * (q_safe[~is_rising] ** b_fal)
        
        r2_src = r2_score(a_ssc, p_ssc)
        rmse_src = np.sqrt(mean_squared_error(a_ssc, p_ssc))
        nse_src = nse_metric(a_ssc, p_ssc)
        pbias_src = pbias_metric(a_ssc, p_ssc)
        results_src[phase] = round(r2_src, 4)
        
        # Plot Hysteresis Curve overlay for this phase
        fig, ax = plt.subplots(figsize=(7.6, 6.4))
        ax.scatter(q_safe[is_rising], a_ssc[is_rising], c="#2563eb", alpha=0.42, s=28, edgecolors="white", linewidths=0.3, label="Observed Rising")
        ax.scatter(q_safe[~is_rising], a_ssc[~is_rising], c="#ea580c", alpha=0.42, s=28, edgecolors="white", linewidths=0.3, label="Observed Falling")
        
        # Plot the theoretical fitted lines
        q_sorted = np.sort(q_safe)
        ax.plot(q_sorted, a_ris * (q_sorted ** b_ris), linestyle='--', color=COLORS["obs"], lw=2.0, label="SRC Rising Fit")
        ax.plot(q_sorted, a_fal * (q_sorted ** b_fal), linestyle='-', color=COLORS["residual"], lw=2.0, label="SRC Falling Fit")

        ax.text(
            0.03,
            0.97,
            f"R² = {r2_src:.3f}\nNSE = {nse_src:.3f}\nRMSE = {rmse_src:.2f} mg/L\nPBIAS = {pbias_src:.2f}%",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.95},
        )
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        style_axis(ax, f"Hysteresis-Aware Rating Curve ({phase.capitalize()})", "Discharge Q (m3/s)", "SSC (mg/L)")
        ax.legend(frameon=False)
        save_figure(fig, PLOTS_DIR / f"{phase.capitalize()}_Hysteresis_SRC.png")
        
    print(f"        -> SRC Hysteresis Baseline R2 - Train: {results_src['train']}, Val: {results_src['val']}, Test: {results_src['test']}")

def main():
    print(f"[INIT] Dual-Regime PINN initializing. Saving to {OUTPUT_DIR}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}. Looked for data/data.xlsx in current and parent directories.")
    apply_publication_style()
    proc = DataProcessor(DATA_PATH)
    
    feats = BASE_FEATURES + ["SSC_LP_lag1", "SSC_LP_lag2"]
    data = proc.split_and_scale(feats)
    
    # Run Baseline SRC Hysteresis before training
    run_src_hysteresis_baseline(data["dfs"])
    
    print(f"\n[RUN] Training Model with {len(feats)} features on RTX 4060...")
    model = ResidualMLP(len(feats))
    model, loss_tracker, physics_meta = train_model(model, data["loaders"], data["y_scaler"])
    
    total_loss = sum(loss_tracker.values())
    if total_loss > 0:
        print("\n[IMPACT OF EQUATIONS] Loss Component Breakdown (Best Epoch):")
        print(f"  - Data (Observations) : {loss_tracker['Data'] / total_loss * 100:.2f}%")
        print(f"  - Floor (Wash Load)   : {loss_tracker['Floor'] / total_loss * 100:.2f}%")
        print(f"  - Ceiling (Yang Cap)  : {loss_tracker['Ceiling'] / total_loss * 100:.2f}%")
        print(f"  - Driver (Mass Bal)   : {loss_tracker['Mass'] / total_loss * 100:.2f}%")

    model.eval()
    results = {}
    for phase in ["train", "val", "test"]:
        eval_loader = proc.make_loader(
            data["loaders"][phase].dataset.tensors[0].numpy(),
            data["loaders"][phase].dataset.tensors[1].numpy(),
            data["loaders"][phase].dataset.tensors[2].numpy(),
            shuffle=False
        )
        preds = []
        with torch.no_grad():
            for b in eval_loader:
                preds.append(model(b[0].to(DEVICE)).cpu())
        
        # Calculate SSC predictions
        p_ssc = data["y_scaler"].inverse_transform(torch.cat(preds).numpy()).flatten()
        a_ssc = data["dfs"][phase][TARGET_COL].values
        
        # Fetch Discharge (Q) to calculate Load
        q_m3s = data["dfs"][phase]["Q_m3s"].values
        
        # Calculate Load in tons/day (SSC * Q * 0.0864)
        p_load = p_ssc * q_m3s * 0.0864
        a_load = a_ssc * q_m3s * 0.0864

        # Metrics for SSC
        r2_ssc = r2_score(a_ssc, p_ssc)
        rmse_ssc = np.sqrt(mean_squared_error(a_ssc, p_ssc))
        nse_ssc = nse_metric(a_ssc, p_ssc)
        pbias_ssc = pbias_metric(a_ssc, p_ssc)
        sd_obs_ssc = np.std(a_ssc)
        sd_pred_ssc = np.std(p_ssc)
        mass_error_ssc = pbias_ssc
        
        # Metrics for Load
        r2_load = r2_score(a_load, p_load)
        rmse_load = np.sqrt(mean_squared_error(a_load, p_load))
        nse_load = nse_metric(a_load, p_load)
        pbias_load = pbias_metric(a_load, p_load)
        sd_obs_load = np.std(a_load)
        sd_pred_load = np.std(p_load)
        mass_error_load = pbias_load
        
        results[phase] = {
            "R2_SSC": round(r2_ssc, 4), "RMSE_SSC": round(rmse_ssc, 2), "NSE_SSC": round(nse_ssc, 4), "PBIAS_SSC_%": round(pbias_ssc, 2), "SD_Observed_SSC": round(sd_obs_ssc, 2), "SD_Predicted_SSC": round(sd_pred_ssc, 2), "Mass_Error_SSC_%": round(mass_error_ssc, 2),
            "R2_Load": round(r2_load, 4), "RMSE_Load": round(rmse_load, 2), "NSE_Load": round(nse_load, 4), "PBIAS_Load_%": round(pbias_load, 2), "SD_Observed_Load": round(sd_obs_load, 2), "SD_Predicted_Load": round(sd_pred_load, 2), "Mass_Error_Load_%": round(mass_error_load, 2),
            "preds_ssc": p_ssc, "preds_load": p_load, "actuals_load": a_load
        }
        
    # Clean up the reporting dataframe so it doesn't print the raw arrays
    clean_rows = []
    for k, v in results.items():
        row = {"Phase": k}
        for metric, val in v.items():
            if metric not in ["preds_ssc", "preds_load", "actuals_load"]:
                row[metric] = val
        clean_rows.append(row)
        
    rep_df = pd.DataFrame(clean_rows)
    print("\n" + "="*80)
    print(rep_df.to_string(index=False))
    print("="*80)

    all_day_results = {}
    for phase in ["train", "val", "test"]:
        all_df = data["all_dfs"][phase]
        X_all = data["x_scaler"].transform(all_df[feats].values)
        r_all = all_df[RAW_PHYS_COLS].values
        y_dummy = np.zeros((len(all_df), 1), dtype=np.float32)

        all_loader = proc.make_loader(X_all, y_dummy, r_all, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for b in all_loader:
                all_preds.append(model(b[0].to(DEVICE)).cpu())

        p_ssc_all = data["y_scaler"].inverse_transform(torch.cat(all_preds).numpy()).flatten()
        q_all = all_df["Q_m3s"].values
        p_load_all = p_ssc_all * q_all * 0.0864
        all_day_results[phase] = {
            "preds_ssc": p_ssc_all,
            "preds_load": p_load_all,
        }
    
    rep_df.to_csv(OUTPUT_DIR / "PINN_Metrics_SSC_and_Load.csv", index=False)
    diagnostics_workbook = export_diagnostics_workbook(
        rep_df,
        data["dfs"],
        results,
        physics_meta,
        data["all_dfs"],
        all_day_results,
    )
    
    # Generate Plots for both SSC and Load
    for phase in ["train", "val", "test"]:
        dates = data["dfs"][phase]["date_dt"].values
        a_ssc = data["dfs"][phase][TARGET_COL].values
        p_ssc = results[phase]["preds_ssc"]
        a_load = results[phase]["actuals_load"]
        p_load = results[phase]["preds_load"]
        
        phase_title = f"{phase.capitalize()} Phase"
        
        # 1. SSC Hydrograph
        plot_hydrograph(
            dates,
            a_ssc,
            p_ssc,
            phase,
            "SSC",
            "mg/L",
            f"R² = {results[phase]['R2_SSC']:.3f} | NSE = {results[phase]['NSE_SSC']:.3f} | RMSE = {results[phase]['RMSE_SSC']:.2f} mg/L | PBIAS = {results[phase]['PBIAS_SSC_%']:.2f}%",
            PLOTS_DIR / f"{phase_title.replace(' ', '_')}_SSC_Hydrograph.png",
        )
        
        # 2. Load Hydrograph
        plot_hydrograph(
            dates,
            a_load,
            p_load,
            phase,
            "Sediment Load",
            "ton/day",
            f"R² = {results[phase]['R2_Load']:.3f} | NSE = {results[phase]['NSE_Load']:.3f} | RMSE = {results[phase]['RMSE_Load']:.2f} ton/day | PBIAS = {results[phase]['PBIAS_Load_%']:.2f}%",
            PLOTS_DIR / f"{phase_title.replace(' ', '_')}_Load_Hydrograph.png",
        )
        
        # 3. Scatter Plots
        plot_scatter(a_ssc, p_ssc, phase, "SSC", "mg/L")
        plot_scatter(a_load, p_load, phase, "Load", "tons/day")
        
        # 4. Taylor Diagrams
        plot_taylor(a_ssc, p_ssc, phase, "SSC")
        plot_taylor(a_load, p_load, phase, "Load")

        # 5. Combined journal-style summary plates
        plot_summary_plate(
            dates,
            a_ssc,
            p_ssc,
            phase,
            "SSC",
            "mg/L",
            f"R² = {results[phase]['R2_SSC']:.3f} | NSE = {results[phase]['NSE_SSC']:.3f} | RMSE = {results[phase]['RMSE_SSC']:.2f} mg/L | PBIAS = {results[phase]['PBIAS_SSC_%']:.2f}%",
            PLOTS_DIR / f"{phase_title.replace(' ', '_')}_SSC_Summary_Plate.png",
        )
        plot_summary_plate(
            dates,
            a_load,
            p_load,
            phase,
            "Sediment Load",
            "ton/day",
            f"R² = {results[phase]['R2_Load']:.3f} | NSE = {results[phase]['NSE_Load']:.3f} | RMSE = {results[phase]['RMSE_Load']:.2f} ton/day | PBIAS = {results[phase]['PBIAS_Load_%']:.2f}%",
            PLOTS_DIR / f"{phase_title.replace(' ', '_')}_Load_Summary_Plate.png",
        )

    run_shap_analysis(model, data, feats)
    print(f"[EXPORT] Diagnostics workbook saved to {diagnostics_workbook}")
    print(f"\n[DONE] Execution complete. Analytics and plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()