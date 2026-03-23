#!/usr/bin/env python
"""
Pure Data-Driven Baseline (Residual MLP) for Daily Suspended Sediment and Load
==============================================================================
Description:
This script models the daily Suspended Sediment Concentration (SSC) and 
calculates the Suspended Sediment Load (SSL) using a purely data-driven approach. 
All physical constraints (Yang's capacity, wash load floor, mass balance) have 
been completely removed to serve as a control baseline.

Updates: 
Now includes advanced hydrological analytics (Scatter, Taylor, Hysteresis SRC)
while maintaining the original flat output pathing structure.
"""

import warnings
import copy
import pathlib
from datetime import datetime, timedelta

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
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs" / f"Pure_MLP_Daily_Load_{RUN_TAG}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR = OUTPUT_DIR / "shap"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS       = 800
BATCH_SIZE   = 64
LR           = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE     = 80
GRAD_CLIP    = 1.0
SHAP_BACKGROUND_SIZE = 128
SHAP_EXPLAIN_SIZE = 256

BASE_FEATURES = [
    "Q_m3s", "d50_mm", "rainfall_mm", "temp_C", "VS", "Soil_Moisture", "NDVI",
    "dQ", "Q_ratio", "Q_lag1", "Q_lag2", "Q_lag3", "Q_max_7d", "HI_proxy",
    "API", "P_MA3", "P_MA7", "P_MA14", "P_MA30", "SPI_30"
]
TARGET_COL = "SSC_mgL"

HEXBIN_CMAP = LinearSegmentedColormap.from_list(
    "teal_orange",
    ["#0f766e", "#f97316"],
)


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

# --- DATA PROCESSOR (Leak-Proof) ---
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

        # Leak-Proof Memory (Shifted for pure DAILY prediction: t-1 and t-2)
        df["SSC_LP_lag1"] = df[TARGET_COL].shift(1).ffill().fillna(0.0)
        df["SSC_LP_lag2"] = df[TARGET_COL].shift(2).ffill().fillna(0.0)

        # HI_proxy using Discharge (No Leaks)
        q_val = df["Q_m3s"]
        q_delta = q_val.diff().fillna(0.0)
        q_maxroll = q_val.rolling(7, min_periods=1).max().replace(0, np.nan)
        df["HI_proxy"] = (q_delta / q_maxroll).fillna(0.0)

        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        df["WY"] = df["date_dt"].apply(self._nepali_wy)
        self.df = df
        self.wys = sorted(df["WY"].unique())

    def split_and_scale(self, features):
        wys = self.wys
        test_wy, val_wy, train_wys = wys[-1], wys[-2], wys[:-2]
        train = self.df[self.df["WY"].isin(train_wys)].copy()
        val = self.df[self.df["WY"] == val_wy].copy()
        test = self.df[self.df["WY"] == test_wy].copy()

        x_scaler, y_scaler = RobustScaler(), RobustScaler()
        X_tr = x_scaler.fit_transform(train[features].values)
        X_va = x_scaler.transform(val[features].values)
        X_te = x_scaler.transform(test[features].values)
        y_tr = y_scaler.fit_transform(train[[TARGET_COL]].values)
        y_va = y_scaler.transform(val[[TARGET_COL]].values)
        y_te = y_scaler.transform(test[[TARGET_COL]].values)

        return {
            "loaders": {
                "train": self.make_loader(X_tr, y_tr, shuffle=True),
                "val": self.make_loader(X_va, y_va),
                "test": self.make_loader(X_te, y_te)
            },
            "y_scaler": y_scaler,
            "dfs": {"train": train, "val": val, "test": test}
        }

    @staticmethod
    def make_loader(X, y, batch_size=64, shuffle=False):
        # Physics tensor completely removed
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), 
                           torch.tensor(y, dtype=torch.float32))
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

# --- PURE DATA-DRIVEN TRAINING ---
def train_model(model, loaders, y_scaler):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    amp_scaler = GradScaler(enabled=USE_AMP)

    best_val, wait, best_state = float("inf"), 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        for batch in loaders["train"]:
            xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=USE_AMP):
                pred = model(xb)
                # Standard MSE Loss - no physical penalties
                loss = nn.functional.mse_loss(pred, yb)
                
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
            best_val, wait, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= PATIENCE: break
    
    if best_state: model.load_state_dict(best_state)
    return model

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
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_array, explain_np, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_scatter(obs, pred, phase, title_prefix, unit):
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    hb = ax.hexbin(obs, pred, gridsize=30, cmap=HEXBIN_CMAP, mincnt=1, linewidths=0.2)
    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("Sample Count")

    max_val = max(np.max(obs), np.max(pred))
    min_val = min(np.min(obs), np.min(pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#b91c1c", lw=1.8, label="1:1 line")

    slope, intercept = np.polyfit(obs, pred, 1)
    fit_x = np.linspace(min_val, max_val, 200)
    ax.plot(fit_x, slope * fit_x + intercept, color="#1d4ed8", lw=1.6, label="Best-fit line")

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

    ax.set_title(f"{title_prefix} - {phase.capitalize()} (Scatter)")
    ax.set_xlabel(f"Observed ({unit})")
    ax.set_ylabel(f"Predicted ({unit})")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{phase.capitalize()}_{title_prefix}_Scatter.png", dpi=150)
    plt.close(fig)

def plot_taylor(obs, pred, phase, title_prefix):
    """Generates a simplified Polar Taylor Diagram"""
    std_obs = np.std(obs)
    std_pred = np.std(pred)
    corr, _ = stats.pearsonr(obs, pred)
    corr = max(0.0, min(corr, 1.0)) 
    
    theta = np.arccos(corr)
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    
    ax.set_theta_direction(-1) 
    ax.set_theta_zero_location("N") 
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    
    rs, ts = np.meshgrid(np.linspace(0, max(std_obs, std_pred)*1.2, 100), 
                         np.arccos(np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99])))
    ax.plot(ts.T, rs.T, color='gray', linestyle=':', alpha=0.5)

    ax.plot(0, std_obs, 'k*', markersize=12, label="Observation Ref")
    ax.plot(np.linspace(0, np.pi/2, 100), np.full(100, std_obs), 'k--', alpha=0.5) 
    
    ax.plot(theta, std_pred, 'ro', markersize=10, label="MLP Prediction")
    
    ax.set_title(f"{title_prefix} - {phase.capitalize()} Taylor Diagram\nCorrelation (Angle) & Std Dev (Radius)", va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{phase.capitalize()}_{title_prefix}_Taylor.png", dpi=150)
    plt.close()

def run_src_hysteresis_baseline(dfs):
    print("\n[BASELINES] Fitting Hysteresis-Aware Sediment Rating Curve (SRC)...")
    train_df = dfs["train"]
    
    train_df = train_df[(train_df["Q_m3s"] > 1e-6) & (train_df[TARGET_COL] > 1e-6)]
    
    rising_mask = train_df["dQ"] > 0
    falling_mask = train_df["dQ"] <= 0
    
    train_rising = train_df[rising_mask]
    train_falling = train_df[falling_mask]
    
    reg_rising = LinearRegression()
    reg_rising.fit(np.log10(train_rising["Q_m3s"].values).reshape(-1, 1), np.log10(train_rising[TARGET_COL].values))
    a_ris, b_ris = 10**reg_rising.intercept_, reg_rising.coef_[0]
    
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
        
        is_rising = df_phase["dQ"].values > 0
        p_ssc = np.zeros_like(q_safe)
        p_ssc[is_rising] = a_ris * (q_safe[is_rising] ** b_ris)
        p_ssc[~is_rising] = a_fal * (q_safe[~is_rising] ** b_fal)
        
        r2_src = r2_score(a_ssc, p_ssc)
        rmse_src = np.sqrt(mean_squared_error(a_ssc, p_ssc))
        nse_src = nse_metric(a_ssc, p_ssc)
        pbias_src = pbias_metric(a_ssc, p_ssc)
        results_src[phase] = round(r2_src, 4)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(q_safe[is_rising], a_ssc[is_rising], c="#2563eb", alpha=0.3, label="Observed Rising")
        ax.scatter(q_safe[~is_rising], a_ssc[~is_rising], c="#ea580c", alpha=0.3, label="Observed Falling")
        
        q_sorted = np.sort(q_safe)
        ax.plot(q_sorted, a_ris * (q_sorted ** b_ris), "k--", lw=2, label="SRC Rising Fit")
        ax.plot(q_sorted, a_fal * (q_sorted ** b_fal), "r-", lw=2, label="SRC Falling Fit")

        ax.text(
            0.03,
            0.97,
            f"R² = {r2_src:.3f}\nNSE = {nse_src:.3f}\nRMSE = {rmse_src:.2f} mg/L\nPBIAS = {pbias_src:.2f}%",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.95},
        )
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Hysteresis Rating Curve - {phase.capitalize()} Phase")
        ax.set_xlabel("Discharge Q (m3/s)")
        ax.set_ylabel("SSC (mg/L)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{phase.capitalize()}_Hysteresis_SRC.png", dpi=150)
        plt.close(fig)
        
    print(f"        -> SRC Hysteresis Baseline R2 - Train: {results_src['train']}, Val: {results_src['val']}, Test: {results_src['test']}")


def main():
    print(f"[INIT] Pure Data-Driven MLP initializing. Saving to {OUTPUT_DIR}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}. Looked for data/data.xlsx in current and parent directories.")
    proc = DataProcessor(DATA_PATH)
    
    feats = BASE_FEATURES + ["SSC_LP_lag1", "SSC_LP_lag2"]
    data = proc.split_and_scale(feats)
    
    # Run Baseline SRC Hysteresis before training
    run_src_hysteresis_baseline(data["dfs"])
    
    print(f"\n[RUN] Training Baseline Model with {len(feats)} features on RTX 4060...")
    model = ResidualMLP(len(feats))
    model = train_model(model, data["loaders"], data["y_scaler"])
    
    model.eval()
    results = {}
    for phase in ["train", "val", "test"]:
        eval_loader = proc.make_loader(
            data["loaders"][phase].dataset.tensors[0].numpy(),
            data["loaders"][phase].dataset.tensors[1].numpy(),
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
    
    rep_df.to_csv(OUTPUT_DIR / "Baseline_Metrics_SSC_and_Load.csv", index=False)
    
    # Generate Plots for both SSC and Load
    for phase in ["train", "val", "test"]:
        dates = data["dfs"][phase]["date_dt"].values
        a_ssc = data["dfs"][phase][TARGET_COL].values
        p_ssc = results[phase]["preds_ssc"]
        a_load = results[phase]["actuals_load"]
        p_load = results[phase]["preds_load"]
        
        phase_title = f"{phase.capitalize()} Phase"
        
        # 1. SSC Hydrograph
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(dates, a_ssc, s=16, alpha=0.75, color="#111111", edgecolors="none", label="Observed SSC")
        ax.plot(dates, p_ssc, color="#0f766e", lw=2.0, label="MLP Predicted SSC")
        ax.set_title(
            f"{phase_title} SSC: R²={results[phase]['R2_SSC']:.3f} | NSE={results[phase]['NSE_SSC']:.3f} | "
            f"RMSE={results[phase]['RMSE_SSC']:.2f} | PBIAS={results[phase]['PBIAS_SSC_%']:.2f}%"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("SSC (mg/L)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{phase_title.replace(' ', '_')}_SSC_Hydrograph.png", dpi=150)
        plt.close(fig)
        
        # 2. Load Hydrograph
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(dates, a_load, s=16, alpha=0.75, color="#1d4ed8", edgecolors="none", label="Observed Load (tons/day)")
        ax.plot(dates, p_load, color="#166534", lw=2.0, label="MLP Predicted Load (tons/day)")
        ax.set_title(
            f"{phase_title} Load: R²={results[phase]['R2_Load']:.3f} | NSE={results[phase]['NSE_Load']:.3f} | "
            f"RMSE={results[phase]['RMSE_Load']:.2f} | PBIAS={results[phase]['PBIAS_Load_%']:.2f}%"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Sediment Load (tons/day)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{phase_title.replace(' ', '_')}_Load_Hydrograph.png", dpi=150)
        plt.close(fig)

        # 3. Scatter Plots
        plot_scatter(a_ssc, p_ssc, phase, "SSC", "mg/L")
        plot_scatter(a_load, p_load, phase, "Load", "tons/day")
        
        # 4. Taylor Diagrams
        plot_taylor(a_ssc, p_ssc, phase, "SSC")
        plot_taylor(a_load, p_load, phase, "Load")

    run_shap_analysis(model, data, feats)
    print(f"\n[DONE] Execution complete. Results and plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()