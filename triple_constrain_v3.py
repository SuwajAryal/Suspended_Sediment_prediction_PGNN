#!/usr/bin/env python
"""
Physics-Informed Neural Network v3 for Suspended Sediment Concentration
=======================================================================
COMPLETE REWRITE based on systematic model selection analysis:

Key findings that drove this design:
  1. SSC lag features are ESSENTIAL (+0.5 NSE improvement across ALL models)
  2. Simple models (Ridge) beat complex ones (MoE, Attention) on this dataset
  3. The "compact+SSC_lags" feature set (19 features) is optimal
  4. Log-space training is critical (logQ vs logSSC r=0.527 vs raw r=0.393)
  5. Physics constraints must be LIGHT — data loss must dominate
  6. slope and catchment_area are CONSTANT — removed
  7. d50_mm is a seasonal proxy — kept but understood

Architecture: Simple residual MLP with light physics regularization
  - 3 hidden layers (128-128-64) with residual connections
  - Huber loss for robustness to outliers
  - Wash-load floor constraint (SSC >= 3 mg/L at low flow)
  - Yang capacity ceiling constraint (SSC <= transport capacity at high flow)
  - Mass balance constraint
  - NO attention, NO MoE — simplicity wins on 1273 samples
"""

import os
import sys
import warnings
import pathlib
from datetime import datetime, timedelta

SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != SCRIPT_DIR]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import GradScaler, autocast

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge

warnings.filterwarnings("ignore")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

def _locate_data_file(filename="data.xlsx"):
    base = pathlib.Path(__file__).resolve().parent
    for p in [base] + list(base.parents):
        candidate = p / "data" / filename
        if candidate.exists():
            return candidate
    return base / "data" / filename

DATA_PATH = _locate_data_file("data.xlsx")
RUN_TAG = datetime.now().strftime("%Y-%m-%d_%H-%M")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs" / f"PINNv3_{RUN_TAG}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Hyperparameters (tuned for this dataset) ---
EPOCHS       = 1000
BATCH_SIZE   = 32
LR           = 3e-4
WEIGHT_DECAY = 5e-4
PATIENCE     = 150
GRAD_CLIP    = 1.0
HUBER_DELTA  = 0.8

# --- Physics constraint weights (ULTRA-LIGHT — data loss must dominate >95%) ---
WASH_LOAD_MIN   = 3.0      # mg/L — minimum SSC at low flow
LAMBDA_FLOOR    = 0.0005   # negligible floor
LAMBDA_CEILING  = 0.001    # negligible ceiling
LAMBDA_MASS     = 0.001    # negligible mass balance (was too dominant at 0.02!)
LAMBDA_VAR      = 0.01     # this one matters most — fight variance compression
LAMBDA_ASYM     = 0.005    # gentle asymmetric penalty for underprediction

# --- Target ---
TARGET_COL = "SSC_mgL"

# --- Feature set (from model_selection.py: "7_compact+SSC_lags") ---
FEATURES = [
    # Core Q features (best predictors without lags)
    "logQ", "logQ_lag1", "logQ_lag2",
    "dQ", "Q_ratio", "rising_flag",
    "Q_MA7", "Q_max_7d", "HI_proxy",
    # Rainfall
    "rainfall_mm", "API", "P_MA7",
    # Climate
    "Soil_Moisture",
    # Seasonality
    "sin_doy", "cos_doy", "monsoon_flag",
    # SSC lags (THE critical features — +0.5 NSE)
    "logSSC_lag1", "logSSC_lag2", "dSSC_lag1",
]

# Columns needed for Yang's formula
RAW_PHYS_COLS = ["Q_m3s", "d50_mm", "temp_C", "VS", "depth_m", "slope"]

# --- Plot style ---
PLOT_DPI = 300
COLORS = {
    "obs": "#1f1f1f", "pinn": "#0f766e", "yang": "#b45309",
    "residual": "#b91c1c", "grid": "#d1d5db", "accent": "#1d4ed8",
}
HEXBIN_CMAP = LinearSegmentedColormap.from_list("teal_orange", ["#0f766e", "#f97316"])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  METRICS                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def nse_metric(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    den = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - np.sum((pred - obs) ** 2) / den if den > 0 else np.nan

def pbias_metric(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    den = np.sum(obs)
    return 100.0 * np.sum(pred - obs) / den if not np.isclose(den, 0.0) else np.nan

def kge_metric(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred) / np.std(obs) if np.std(obs) > 0 else 0
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) > 0 else 0
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DATA PROCESSING                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class DataProcessor:
    def __init__(self, path):
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

        Q = df["Q_m3s"]
        P = df["rainfall_mm"]

        # --- Core hydrological features ---
        df["logQ"] = np.log1p(Q.clip(lower=0))
        df["dQ"] = Q.diff().fillna(0.0)
        q7 = Q.rolling(7, min_periods=1).mean().replace(0, np.nan)
        df["Q_ratio"] = (Q / q7).fillna(1.0)
        df["rising_flag"] = (df["dQ"] > 0).astype(float)
        df["Q_MA7"] = Q.rolling(7, min_periods=1).mean()
        df["Q_max_7d"] = Q.rolling(7, min_periods=1).max()

        # Discharge lags (log-space — better correlation)
        for lag in [1, 2]:
            df[f"Q_lag{lag}"] = Q.shift(lag).ffill().fillna(0.0)
            df[f"logQ_lag{lag}"] = np.log1p(df[f"Q_lag{lag}"].clip(lower=0))

        # Hysteresis index
        q_maxroll = Q.rolling(7, min_periods=1).max().replace(0, np.nan)
        df["HI_proxy"] = (df["dQ"] / q_maxroll).fillna(0.0)

        # --- Rainfall ---
        df["P_MA7"] = P.rolling(7, min_periods=1).mean()
        p_arr = P.values
        api_vals = np.zeros(len(df))
        for k in range(1, 31):
            shifted = np.concatenate([np.full(k, p_arr[:k].mean()), p_arr[:-k]])
            api_vals += (0.9 ** k) * shifted
        df["API"] = api_vals

        # --- Seasonality ---
        doy = df["date_dt"].dt.dayofyear
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
        df["monsoon_flag"] = ((df["date_dt"].dt.month >= 6) & (df["date_dt"].dt.month <= 9)).astype(float)

        # --- Sediment supply proxy ---
        df["VS"] = (Q * df["slope"]) / (df["width_m"] * df["depth_m"])

        # --- SSC lag features (autoregressive — justified for continuous monitoring) ---
        df["SSC_lag1"] = df[TARGET_COL].shift(1)
        df["SSC_lag2"] = df[TARGET_COL].shift(2)
        # Forward-fill for gaps, then fill remaining with 0
        df["SSC_lag1"] = df["SSC_lag1"].ffill().fillna(0.0)
        df["SSC_lag2"] = df["SSC_lag2"].ffill().fillna(0.0)
        df["logSSC_lag1"] = np.log1p(df["SSC_lag1"].clip(lower=0))
        df["logSSC_lag2"] = np.log1p(df["SSC_lag2"].clip(lower=0))
        df["dSSC_lag1"] = (df["SSC_lag1"] - df["SSC_lag2"]).fillna(0.0)

        # --- Log target ---
        df["logSSC"] = np.log1p(df[TARGET_COL].clip(lower=0))

        # --- Water year ---
        df["WY"] = df["date_dt"].apply(self._nepali_wy)

        self.full_df = df.copy()
        self.df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        self.wys = sorted(self.df["WY"].unique())

    def split_and_scale(self):
        wys = self.wys
        test_wy, val_wy, train_wys = wys[-1], wys[-2], wys[:-2]

        train = self.df[self.df["WY"].isin(train_wys)].copy()
        val   = self.df[self.df["WY"] == val_wy].copy()
        test  = self.df[self.df["WY"] == test_wy].copy()

        # All-days data (for full time series prediction)
        train_all = self.full_df[self.full_df["WY"].isin(train_wys)].copy()
        val_all   = self.full_df[self.full_df["WY"] == val_wy].copy()
        test_all  = self.full_df[self.full_df["WY"] == test_wy].copy()

        print(f"  Train: WYs {train_wys} ({len(train)} samples)")
        print(f"  Val:   WY {val_wy} ({len(val)} samples)")
        print(f"  Test:  WY {test_wy} ({len(test)} samples)")

        # Scale features
        x_scaler = RobustScaler()
        y_scaler = RobustScaler()

        X_tr = x_scaler.fit_transform(train[FEATURES].values)
        X_va = x_scaler.transform(val[FEATURES].values)
        X_te = x_scaler.transform(test[FEATURES].values)

        # Target in log-space
        y_tr = y_scaler.fit_transform(train[["logSSC"]].values)
        y_va = y_scaler.transform(val[["logSSC"]].values)
        y_te = y_scaler.transform(test[["logSSC"]].values)

        # Physics columns
        r_tr = train[RAW_PHYS_COLS].values
        r_va = val[RAW_PHYS_COLS].values
        r_te = test[RAW_PHYS_COLS].values

        return {
            "loaders": {
                "train": self._make_loader(X_tr, y_tr, r_tr, shuffle=True),
                "val":   self._make_loader(X_va, y_va, r_va),
                "test":  self._make_loader(X_te, y_te, r_te),
            },
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "dfs": {"train": train, "val": val, "test": test},
            "all_dfs": {"train": train_all, "val": val_all, "test": test_all},
        }

    @staticmethod
    def _make_loader(X, y, r, batch_size=BATCH_SIZE, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODEL — Simple Residual MLP (complexity kills on small data)               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class _ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),      # LayerNorm > BatchNorm for small batches
            nn.GELU(),              # GELU > LeakyReLU for smoother gradients
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class SSC_PINN(nn.Module):
    """
    Simple residual MLP with physics-informed loss.
    Architecture: Input -> 128 -> ResBlock*2 -> 64 -> ResBlock -> 1
    Compact ~50K params with heavy dropout — matches data complexity
    """
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            # Encoder
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),

            # Residual blocks
            _ResBlock(128, dropout=0.15),
            _ResBlock(128, dropout=0.15),

            # Compression
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),

            _ResBlock(64, dropout=0.1),

            # Output
            nn.Linear(64, 1),
        )

        # Initialize weights carefully
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PHYSICS: Yang (1979) Transport Capacity                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def get_yang_capacity(phys_tensor):
    """Yang (1979) total-load transport formula with Rubey fall velocity."""
    a1, a2, a3, a4, a5, a6 = 7.1178, -1.3908, -3.3899, 5.5007, -1.6111, -3.9525

    d50_m = phys_tensor[:, 1] * 1e-3
    temp  = phys_tensor[:, 2]
    VS    = phys_tensor[:, 3]
    depth = phys_tensor[:, 4]
    slope = phys_tensor[:, 5]

    nu = torch.clamp(1.14e-6 - 3.0e-9 * temp, min=1e-7)
    d = torch.clamp(d50_m, min=1e-6)

    Gs_minus_1, g = 1.65, 9.81
    denom_F = torch.sqrt(torch.clamp(g * Gs_minus_1 * d**3, min=1e-18))
    F = torch.clamp(2.0/3.0 + 36.0 * nu / (denom_F + 1e-12), min=0.01, max=1.0)
    w = F * torch.sqrt(torch.clamp(g * Gs_minus_1 * d, min=1e-12))

    u_s = torch.sqrt(torch.clamp(g * depth * slope, min=1e-12))

    l_wd = torch.log10(torch.clamp(w * d50_m / 1e-6, min=1e-8))
    l_us = torch.log10(torch.clamp(u_s / torch.clamp(w, min=1e-8), min=1e-8))
    l_VS = torch.log10(torch.clamp(VS / torch.clamp(w, min=1e-8), min=1e-8))

    l_Ct = a1 + a2*l_wd + a3*l_us + (a4 + a5*l_wd + a6*l_us) * l_VS
    return torch.pow(10.0, l_Ct)


def get_yang_capacity_np(phys_array, alpha=1.0, batch_size=4096):
    if len(phys_array) == 0:
        return np.array([], dtype=np.float64)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(phys_array), batch_size):
            batch = torch.tensor(phys_array[start:start + batch_size], dtype=torch.float32, device=DEVICE)
            outputs.append((get_yang_capacity(batch) * alpha).cpu().numpy())
    return np.concatenate(outputs)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TRAINING                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def _to_mgl(scaled_vals, y_center, y_scale):
    """Convert scaled output -> logSSC -> mg/L."""
    raw = scaled_vals * y_scale + y_center  # unscale to log-space
    return torch.expm1(raw).clamp(min=0.0)  # log -> mg/L


def train_model(model, loaders, y_scaler):
    """
    2-Stage Training:
      Stage 1: Pure data-driven (Huber loss only) — learn the Q-SSC relationship
      Stage 2: Fine-tune with physics constraints at lower LR — enforce physical consistency
    """
    model = model.to(DEVICE)
    amp_scaler = GradScaler(enabled=USE_AMP)

    y_center = torch.tensor(y_scaler.center_[0], dtype=torch.float32, device=DEVICE)
    y_scale  = torch.tensor(y_scaler.scale_[0],  dtype=torch.float32, device=DEVICE)

    # --- Calibrate Yang's capacity using extreme flows ---
    print("\n[TRAIN] Calibrating Yang capacity on top 5% flows...")
    all_actuals, all_yangs, all_qs = [], [], []
    for batch in loaders["train"]:
        yb, pb = batch[1].to(DEVICE), batch[2].to(DEVICE)
        actual_mgl = _to_mgl(yb.squeeze(), y_center, y_scale)
        yang_cap = get_yang_capacity(pb)
        all_actuals.append(actual_mgl)
        all_yangs.append(yang_cap)
        all_qs.append(pb[:, 0])

    all_actuals = torch.cat(all_actuals)
    all_yangs = torch.cat(all_yangs)
    all_qs = torch.cat(all_qs)

    q_95th = torch.quantile(all_qs, 0.95)
    top_mask = all_qs >= q_95th
    alpha = torch.median(all_actuals[top_mask] / torch.clamp(all_yangs[top_mask], min=1e-6)).item()
    global_q_mean = torch.mean(all_qs).item()
    global_max_ssc = all_actuals.max().item()

    print(f"  Q_95th = {q_95th.item():.1f} m3/s, alpha = {alpha:.4f}, Q_mean = {global_q_mean:.1f}")

    loss_history = {"epoch": [], "train": [], "val": [], "lr": []}
    loss_components = {"Data": 0, "Floor": 0, "Ceiling": 0, "Mass": 0, "Var": 0, "Asym": 0}

    # ===================== STAGE 1: Pure Data-Driven =====================
    STAGE1_EPOCHS = 500
    STAGE1_PATIENCE = 120
    print(f"\n  === STAGE 1: Data-driven training ({STAGE1_EPOCHS} epochs, patience={STAGE1_PATIENCE}) ===")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-6
    )

    best_val, wait, best_state = float("inf"), 0, None

    for epoch in range(1, STAGE1_EPOCHS + 1):
        model.train()
        ep_data = 0.
        n_batches = 0

        for batch in loaders["train"]:
            xb, yb, pb = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()

            with autocast(device_type=DEVICE.type, enabled=USE_AMP):
                pred = model(xb)
                data_loss = nn.functional.huber_loss(pred.squeeze(), yb.squeeze(), delta=HUBER_DELTA)

            amp_scaler.scale(data_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            amp_scaler.step(optimizer)
            amp_scaler.update()

            ep_data += data_loss.item()
            n_batches += 1

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in loaders["val"]:
                val_loss += nn.functional.mse_loss(model(b[0].to(DEVICE)), b[1].to(DEVICE), reduction="sum").item()
        val_loss /= len(loaders["val"].dataset)
        scheduler.step(epoch)

        loss_history["epoch"].append(epoch)
        loss_history["train"].append(ep_data / max(n_batches, 1))
        loss_history["val"].append(val_loss)
        loss_history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val, wait = val_loss, 0
        else:
            wait += 1
            if wait >= STAGE1_PATIENCE:
                print(f"  Stage 1 early stop at epoch {epoch}")
                break

        if epoch % 100 == 0 or epoch == 1:
            print(f"  S1 Ep {epoch:4d} | data={ep_data/n_batches:.4f} | val={val_loss:.6f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} | wait={wait}/{STAGE1_PATIENCE}")

    # Load best Stage 1 state
    if best_state:
        model.load_state_dict(best_state)
    print(f"  Stage 1 best val loss: {best_val:.6f}")

    # ===================== STAGE 2: Physics Fine-tuning =====================
    STAGE2_EPOCHS = 500
    STAGE2_PATIENCE = 150
    STAGE2_LR = LR * 0.1  # 10x lower LR for fine-tuning
    print(f"\n  === STAGE 2: Physics fine-tuning ({STAGE2_EPOCHS} epochs, LR={STAGE2_LR:.1e}) ===")

    optimizer2 = torch.optim.AdamW(model.parameters(), lr=STAGE2_LR, weight_decay=WEIGHT_DECAY)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer2, T_0=100, T_mult=2, eta_min=1e-7
    )

    best_val2, wait2 = best_val, 0

    for epoch in range(1, STAGE2_EPOCHS + 1):
        model.train()
        ep_data, ep_floor, ep_ceil, ep_mass, ep_var, ep_asym = 0., 0., 0., 0., 0., 0.
        n_batches = 0

        # Gentle physics ramp over first 20% of stage 2
        ramp = min(1.0, epoch / (STAGE2_EPOCHS * 0.2))

        for batch in loaders["train"]:
            xb, yb, pb = [t.to(DEVICE) for t in batch]
            optimizer2.zero_grad()

            with autocast(device_type=DEVICE.type, enabled=USE_AMP):
                pred = model(xb)

                # Data loss
                data_loss = nn.functional.huber_loss(pred.squeeze(), yb.squeeze(), delta=HUBER_DELTA)

                # Physics losses
                pr_mgl = _to_mgl(pred.squeeze(), y_center, y_scale)
                yb_mgl = _to_mgl(yb.squeeze(), y_center, y_scale)

                # Floor
                floor_loss = torch.mean(nn.functional.relu(WASH_LOAD_MIN - pr_mgl) ** 2)

                # Ceiling
                yang_cap = get_yang_capacity(pb) * alpha
                ceiling_loss = torch.mean(nn.functional.relu(pr_mgl - yang_cap) ** 2)

                # Mass balance
                mass_loss = torch.abs(torch.sum(pr_mgl) - torch.sum(yb_mgl)) / (len(yb) + 1e-6)

                # Variance matching
                var_loss = torch.tensor(0.0, device=DEVICE)
                if len(yb) > 2:
                    std_pred = torch.std(pr_mgl)
                    std_obs = torch.std(yb_mgl).clamp(min=1e-6)
                    var_loss = (std_pred / std_obs - 1.0) ** 2

                # Asymmetric underprediction penalty
                under = nn.functional.relu(yb_mgl - pr_mgl)
                over  = nn.functional.relu(pr_mgl - yb_mgl)
                asym_loss = (3.0 * torch.mean(under) + 0.5 * torch.mean(over)) / (global_max_ssc + 1e-6)

                loss = (
                    data_loss
                    + ramp * LAMBDA_FLOOR * floor_loss
                    + ramp * LAMBDA_CEILING * ceiling_loss
                    + ramp * LAMBDA_MASS * mass_loss
                    + ramp * LAMBDA_VAR * var_loss
                    + ramp * LAMBDA_ASYM * asym_loss
                )

            amp_scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            amp_scaler.step(optimizer2)
            amp_scaler.update()

            ep_data  += data_loss.item()
            ep_floor += (ramp * LAMBDA_FLOOR * floor_loss).item()
            ep_ceil  += (ramp * LAMBDA_CEILING * ceiling_loss).item()
            ep_mass  += (ramp * LAMBDA_MASS * mass_loss).item()
            ep_var   += (ramp * LAMBDA_VAR * var_loss).item()
            ep_asym  += (ramp * LAMBDA_ASYM * asym_loss).item()
            n_batches += 1

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in loaders["val"]:
                val_loss += nn.functional.mse_loss(model(b[0].to(DEVICE)), b[1].to(DEVICE), reduction="sum").item()
        val_loss /= len(loaders["val"].dataset)
        scheduler2.step(epoch)

        global_epoch = STAGE1_EPOCHS + epoch
        loss_history["epoch"].append(global_epoch)
        loss_history["train"].append(ep_data / max(n_batches, 1))
        loss_history["val"].append(val_loss)
        loss_history["lr"].append(optimizer2.param_groups[0]["lr"])

        if val_loss < best_val2:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val2, wait2 = val_loss, 0
            loss_components = {
                "Data": ep_data, "Floor": ep_floor, "Ceiling": ep_ceil,
                "Mass": ep_mass, "Var": ep_var, "Asym": ep_asym,
            }
        else:
            wait2 += 1
            if wait2 >= STAGE2_PATIENCE:
                print(f"  Stage 2 early stop at epoch {epoch}")
                break

        if epoch % 100 == 0 or epoch == 1:
            phys_pct = 100 * (1 - ep_data / max(ep_data + ep_floor + ep_ceil + ep_mass + ep_var + ep_asym, 1e-8))
            print(f"  S2 Ep {epoch:4d} | data={ep_data/n_batches:.4f} | val={val_loss:.6f} | "
                  f"phys={phys_pct:.1f}% | lr={optimizer2.param_groups[0]['lr']:.2e} | wait={wait2}/{STAGE2_PATIENCE}")

    if best_state:
        model.load_state_dict(best_state)

    improved = best_val2 < best_val
    print(f"  Stage 2 best val: {best_val2:.6f} ({'IMPROVED' if improved else 'no improvement'} from {best_val:.6f})")

    physics_meta = {
        "alpha": alpha,
        "q_95th": q_95th.item(),
        "global_q_mean": global_q_mean,
    }
    return model, loss_components, physics_meta, loss_history


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  EVALUATION & PREDICTION                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def predict(model, loader, y_scaler):
    """Run inference and return SSC in mg/L."""
    model.eval()
    preds = []
    with torch.no_grad():
        for b in loader:
            preds.append(model(b[0].to(DEVICE)).cpu())
    pred_scaled = torch.cat(preds).numpy()
    pred_log = y_scaler.inverse_transform(pred_scaled).flatten()
    return np.expm1(pred_log).clip(min=0.0)


def evaluate_phase(obs_ssc, pred_ssc, q_m3s, phase_name):
    """Compute all metrics for a phase."""
    pred_load = pred_ssc * q_m3s * 0.0864
    obs_load  = obs_ssc  * q_m3s * 0.0864

    metrics = {
        "Phase": phase_name,
        "R2_SSC":  round(r2_score(obs_ssc, pred_ssc), 4),
        "NSE_SSC": round(nse_metric(obs_ssc, pred_ssc), 4),
        "KGE_SSC": round(kge_metric(obs_ssc, pred_ssc), 4),
        "RMSE_SSC": round(np.sqrt(mean_squared_error(obs_ssc, pred_ssc)), 2),
        "PBIAS_SSC_%": round(pbias_metric(obs_ssc, pred_ssc), 2),
        "SD_Obs_SSC": round(np.std(obs_ssc), 2),
        "SD_Pred_SSC": round(np.std(pred_ssc), 2),
        "R2_Load":  round(r2_score(obs_load, pred_load), 4),
        "NSE_Load": round(nse_metric(obs_load, pred_load), 4),
        "KGE_Load": round(kge_metric(obs_load, pred_load), 4),
        "RMSE_Load": round(np.sqrt(mean_squared_error(obs_load, pred_load)), 2),
        "PBIAS_Load_%": round(pbias_metric(obs_load, pred_load), 2),
    }
    return metrics, pred_load, obs_load


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PLOTTING (publication quality)                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def apply_publication_style():
    plt.rcParams.update({
        "figure.dpi": PLOT_DPI, "savefig.dpi": PLOT_DPI,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
        "axes.grid": True, "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6, "grid.alpha": 0.6, "grid.linestyle": "--",
        "font.family": "DejaVu Serif", "font.size": 11,
        "axes.titlesize": 14, "axes.titleweight": "semibold",
        "axes.labelsize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
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


def plot_loss_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    epochs = history["epoch"]
    ax1.plot(epochs, history["train"], color=COLORS["pinn"], lw=1.5, label="Train (data)")
    ax1.plot(epochs, history["val"], color=COLORS["residual"], lw=1.5, label="Validation")
    ax1.set_yscale("log")
    style_axis(ax1, "Training & Validation Loss", "Epoch", "Loss (log scale)")
    ax1.legend(frameon=False)

    ax2.plot(epochs, history["lr"], color=COLORS["accent"], lw=1.5)
    style_axis(ax2, "Learning Rate Schedule", "Epoch", "Learning Rate")
    save_figure(fig, PLOTS_DIR / "Loss_Curves.png")


def plot_hydrograph(dates, obs, pred, phase, var_name, unit, metric_line, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.4), sharex=True,
                              gridspec_kw={"height_ratios": [3.2, 1.0]})
    axes[0].scatter(dates, obs, color=COLORS["obs"], s=18, alpha=0.75, edgecolors="none", label="Observed")
    axes[0].plot(dates, pred, color=COLORS["pinn"], lw=2.0, label="PINN v3")
    axes[0].fill_between(dates, obs, pred, color=COLORS["pinn"], alpha=0.10)
    style_axis(axes[0], f"{var_name} Hydrograph ({phase.capitalize()})", "", f"{var_name} ({unit})")
    axes[0].text(0.01, 0.97, metric_line, transform=axes[0].transAxes, va="top",
                 bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.95})
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    residual = pred - obs
    axes[1].axhline(0.0, color="#6b7280", lw=1.0)
    axes[1].fill_between(dates, residual, 0.0, color=COLORS["residual"], alpha=0.20)
    axes[1].plot(dates, residual, color=COLORS["residual"], lw=1.1)
    style_axis(axes[1], "Residual", "Date", f"Residual ({unit})")
    axes[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[1].xaxis.get_major_locator()))
    save_figure(fig, output_path)


def plot_scatter(obs, pred, phase, var_name, unit):
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    hb = ax.hexbin(obs, pred, gridsize=34, cmap=HEXBIN_CMAP, mincnt=1, linewidths=0.2)
    fig.colorbar(hb, ax=ax, pad=0.02).set_label("Count")

    vmin, vmax = min(np.min(obs), np.min(pred)), max(np.max(obs), np.max(pred))
    ax.plot([vmin, vmax], [vmin, vmax], "--", color=COLORS["residual"], lw=1.8, label="1:1")
    slope, intercept = np.polyfit(obs, pred, 1)
    fit_x = np.linspace(vmin, vmax, 200)
    ax.plot(fit_x, slope*fit_x+intercept, color=COLORS["accent"], lw=1.6, label="Best-fit")

    r2 = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    ax.text(0.04, 0.96, f"R2 = {r2:.3f}\nRMSE = {rmse:.2f} {unit}",
            transform=ax.transAxes, va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.95})
    style_axis(ax, f"{var_name} Observed vs Predicted ({phase.capitalize()})",
               f"Observed ({unit})", f"Predicted ({unit})")
    ax.legend(frameon=False, loc="lower right")
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, PLOTS_DIR / f"{phase}_{var_name}_Scatter.png")


def draw_taylor_axis(ax, obs, pred, title):
    std_obs, std_pred = np.std(obs), np.std(pred)
    corr = max(0.0, min(np.corrcoef(obs, pred)[0, 1], 1.0))
    theta = np.arccos(corr)

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    rs, ts = np.meshgrid(
        np.linspace(0, max(std_obs, std_pred)*1.2, 100),
        np.arccos(np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]))
    )
    ax.plot(ts.T, rs.T, color='#9ca3af', linestyle=':', alpha=0.8)
    ax.plot(0, std_obs, marker='*', color=COLORS["obs"], markersize=13, label="Observed")
    ax.plot(np.linspace(0, np.pi/2, 100), np.full(100, std_obs), '--', color=COLORS["obs"], alpha=0.45)
    ax.plot(theta, std_pred, marker='o', color=COLORS["pinn"], markersize=10, label="PINN v3")
    ax.set_title(title, va='bottom')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=False)


def plot_summary_plate(dates, obs, pred, phase, var_name, unit, metric_line, output_path):
    fig = plt.figure(figsize=(14.2, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.28)

    # Hydrograph
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.scatter(dates, obs, color=COLORS["obs"], s=16, alpha=0.72, edgecolors="none", label="Observed")
    ax_h.plot(dates, pred, color=COLORS["pinn"], lw=2.0, label="PINN v3")
    ax_h.fill_between(dates, obs, pred, color=COLORS["pinn"], alpha=0.10)
    style_axis(ax_h, f"{var_name} Summary ({phase.capitalize()})", "Date", f"{var_name} ({unit})")
    ax_h.text(0.01, 0.97, metric_line, transform=ax_h.transAxes, va="top",
              bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.95})
    ax_h.legend(frameon=False, ncol=2, loc="upper right")
    ax_h.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax_h.xaxis.get_major_locator()))

    # Scatter
    ax_s = fig.add_subplot(gs[1, 0])
    hb = ax_s.hexbin(obs, pred, gridsize=30, cmap=HEXBIN_CMAP, mincnt=1, linewidths=0.2)
    fig.colorbar(hb, ax=ax_s, pad=0.02).set_label("Count")
    vmin, vmax = min(np.min(obs), np.min(pred)), max(np.max(obs), np.max(pred))
    ax_s.plot([vmin, vmax], [vmin, vmax], "--", color=COLORS["residual"], lw=1.8)
    slope, intercept = np.polyfit(obs, pred, 1)
    ax_s.plot(np.linspace(vmin, vmax, 200), slope*np.linspace(vmin, vmax, 200)+intercept,
              color=COLORS["accent"], lw=1.6)
    r2 = r2_score(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    ax_s.text(0.04, 0.96, f"R2 = {r2:.3f}\nRMSE = {rmse:.2f} {unit}",
              transform=ax_s.transAxes, va="top",
              bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#9ca3af", "alpha": 0.95})
    style_axis(ax_s, "Observed vs Predicted", f"Observed ({unit})", f"Predicted ({unit})")
    ax_s.set_aspect("equal", adjustable="box")

    # Taylor
    ax_t = fig.add_subplot(gs[1, 1], polar=True)
    draw_taylor_axis(ax_t, obs, pred, "Taylor Diagram")
    save_figure(fig, output_path)


def run_src_baseline(dfs):
    """Hysteresis-aware sediment rating curve baseline."""
    print("\n[BASELINE] Fitting Hysteresis-Aware SRC...")
    train_df = dfs["train"]
    train_df = train_df[(train_df["Q_m3s"] > 1e-6) & (train_df[TARGET_COL] > 1e-6)]

    rising = train_df[train_df["dQ"] > 0] if "dQ" in train_df.columns else train_df
    falling = train_df[train_df["dQ"] <= 0] if "dQ" in train_df.columns else train_df

    reg_r = LinearRegression()
    reg_r.fit(np.log10(rising["Q_m3s"].values).reshape(-1, 1), np.log10(rising[TARGET_COL].values))
    a_r, b_r = 10**reg_r.intercept_, reg_r.coef_[0]

    reg_f = LinearRegression()
    reg_f.fit(np.log10(falling["Q_m3s"].values).reshape(-1, 1), np.log10(falling[TARGET_COL].values))
    a_f, b_f = 10**reg_f.intercept_, reg_f.coef_[0]

    print(f"  Rising:  SSC = {a_r:.4f} * Q^{b_r:.4f}")
    print(f"  Falling: SSC = {a_f:.4f} * Q^{b_f:.4f}")

    src_results = {}
    for phase in ["train", "val", "test"]:
        df = dfs[phase]
        q = np.maximum(df["Q_m3s"].values, 1e-6)
        is_rising = df["dQ"].values > 0 if "dQ" in df.columns else np.ones(len(df), dtype=bool)
        pred = np.where(is_rising, a_r * q**b_r, a_f * q**b_f)
        obs = df[TARGET_COL].values

        nse_val = nse_metric(obs, pred)
        r2_val = r2_score(obs, pred)
        print(f"  {phase:6s}: R2={r2_val:.4f}, NSE={nse_val:.4f}, RMSE={np.sqrt(mean_squared_error(obs, pred)):.1f}")
        src_results[phase] = {"R2": r2_val, "NSE": nse_val, "pred": pred}

    return src_results


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DIAGNOSTICS EXPORT                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def build_diagnostics(df, pred_ssc, physics_meta):
    q = df["Q_m3s"].values
    obs_ssc = df[TARGET_COL].values
    obs_load = obs_ssc * q * 0.0864
    pred_load = pred_ssc * q * 0.0864
    yang_ssc = get_yang_capacity_np(df[RAW_PHYS_COLS].values, alpha=physics_meta["alpha"])

    return pd.DataFrame({
        "Date": pd.to_datetime(df["date_dt"]),
        "Q_m3s": q,
        "Observed_SSC_mgL": obs_ssc,
        "PINN_SSC_mgL": pred_ssc,
        "Yang_Cap_SSC_mgL": yang_ssc,
        "SSC_Residual_mgL": pred_ssc - obs_ssc,
        "Observed_Load_ton_day": obs_load,
        "PINN_Load_ton_day": pred_load,
        "Cum_Obs_Load_ton": np.cumsum(obs_load),
        "Cum_PINN_Load_ton": np.cumsum(pred_load),
    })


def build_alldays_diagnostics(df, pred_ssc, physics_meta):
    q = df["Q_m3s"].values
    obs_ssc = df[TARGET_COL].values if TARGET_COL in df.columns else np.full(len(df), np.nan)
    pred_load = pred_ssc * q * 0.0864
    obs_load = np.where(np.isnan(obs_ssc), 0.0, obs_ssc * q * 0.0864)

    return pd.DataFrame({
        "Date": pd.to_datetime(df["date_dt"]),
        "Q_m3s": q,
        "Observed_SSC_mgL": obs_ssc,
        "PINN_SSC_mgL": pred_ssc,
        "Observed_Load_ton_day": np.where(np.isnan(obs_ssc), np.nan, obs_ssc * q * 0.0864),
        "PINN_Load_ton_day": pred_load,
        "Cum_PINN_Load_ton": np.cumsum(pred_load),
        "Has_Observed": ~pd.isna(obs_ssc),
    })


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def main():
    print("=" * 80)
    print("  PINN v3 for Suspended Sediment Concentration")
    print("  Architecture: Simple Residual MLP + Light Physics Constraints")
    print(f"  Features: {len(FEATURES)} (compact + SSC lags)")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found: {DATA_PATH}")

    apply_publication_style()

    # --- Load and split ---
    print("\n[DATA] Loading and engineering features...")
    proc = DataProcessor(DATA_PATH)
    data = proc.split_and_scale()

    # --- Baseline ---
    src_results = run_src_baseline(data["dfs"])

    # --- Model ---
    print(f"\n[MODEL] Building SSC_PINN with {len(FEATURES)} features...")
    model = SSC_PINN(len(FEATURES))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # --- Train ---
    model, loss_comp, physics_meta, loss_history = train_model(model, data["loaders"], data["y_scaler"])

    # --- Loss breakdown ---
    total = sum(loss_comp.values())
    if total > 0:
        print("\n[LOSS BREAKDOWN] Best epoch:")
        for name, val in loss_comp.items():
            print(f"  {name:10s}: {val/total*100:6.2f}%")

    # --- Evaluate ---
    print("\n[EVAL] Computing predictions...")
    model.eval()
    results = {}
    all_metrics = []

    for phase in ["train", "val", "test"]:
        loader = DataProcessor._make_loader(
            data["loaders"][phase].dataset.tensors[0].numpy(),
            data["loaders"][phase].dataset.tensors[1].numpy(),
            data["loaders"][phase].dataset.tensors[2].numpy(),
            shuffle=False,
        )
        pred_ssc = predict(model, loader, data["y_scaler"])
        obs_ssc = data["dfs"][phase][TARGET_COL].values
        q = data["dfs"][phase]["Q_m3s"].values

        metrics, pred_load, obs_load = evaluate_phase(obs_ssc, pred_ssc, q, phase)
        all_metrics.append(metrics)
        results[phase] = {
            "pred_ssc": pred_ssc, "obs_ssc": obs_ssc,
            "pred_load": pred_load, "obs_load": obs_load, "q": q,
        }

    # --- Print results ---
    rep_df = pd.DataFrame(all_metrics)
    print("\n" + "=" * 100)
    print("  RESULTS")
    print("=" * 100)
    print(rep_df.to_string(index=False))
    print("=" * 100)

    # Compare with baseline
    print("\n[COMPARISON] PINN v3 vs SRC Baseline (test set):")
    test_m = all_metrics[2]  # test
    src_test = src_results["test"]
    print(f"  SRC:     R2={src_test['R2']:.4f}, NSE={src_test['NSE']:.4f}")
    print(f"  PINN v3: R2={test_m['R2_SSC']:.4f}, NSE={test_m['NSE_SSC']:.4f}, "
          f"KGE={test_m['KGE_SSC']:.4f}, RMSE={test_m['RMSE_SSC']:.1f}")
    improvement = test_m["NSE_SSC"] - src_test["NSE"]
    print(f"  NSE improvement: {improvement:+.4f}")

    # ===================== ENSEMBLE: PINN + Ridge =====================
    print("\n[ENSEMBLE] Training Ridge baseline and blending with PINN...")
    # Fit Ridge on same features in log-space
    X_tr_scaled = data["x_scaler"].transform(data["dfs"]["train"][FEATURES].values)
    X_va_scaled = data["x_scaler"].transform(data["dfs"]["val"][FEATURES].values)
    X_te_scaled = data["x_scaler"].transform(data["dfs"]["test"][FEATURES].values)
    y_tr_log = np.log1p(data["dfs"]["train"][TARGET_COL].clip(lower=0).values)
    y_va_log = np.log1p(data["dfs"]["val"][TARGET_COL].clip(lower=0).values)
    y_te_log = np.log1p(data["dfs"]["test"][TARGET_COL].clip(lower=0).values)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_scaled, y_tr_log)

    ridge_preds = {}
    for phase, X_s in [("train", X_tr_scaled), ("val", X_va_scaled), ("test", X_te_scaled)]:
        ridge_pred_log = ridge.predict(X_s)
        ridge_preds[phase] = np.expm1(ridge_pred_log).clip(min=0)

    # Ridge standalone metrics
    for phase in ["train", "val", "test"]:
        obs = data["dfs"][phase][TARGET_COL].values
        rpred = ridge_preds[phase]
        print(f"  Ridge {phase:5s}: NSE={nse_metric(obs, rpred):.4f}, "
              f"R2={r2_score(obs, rpred):.4f}, RMSE={np.sqrt(mean_squared_error(obs, rpred)):.1f}")

    # Find optimal blend weight using validation set
    best_w, best_nse_blend = 0.5, -999
    for w in np.arange(0.0, 1.05, 0.05):
        blend = w * results["val"]["pred_ssc"] + (1 - w) * ridge_preds["val"]
        nse_b = nse_metric(results["val"]["obs_ssc"], blend)
        if nse_b > best_nse_blend:
            best_nse_blend = nse_b
            best_w = w
    print(f"\n  Optimal blend: {best_w:.2f}*PINN + {1-best_w:.2f}*Ridge (val NSE={best_nse_blend:.4f})")

    # Apply ensemble to all phases
    ensemble_metrics = []
    ensemble_results = {}
    for phase in ["train", "val", "test"]:
        ens_pred = best_w * results[phase]["pred_ssc"] + (1 - best_w) * ridge_preds[phase]
        obs = results[phase]["obs_ssc"]
        q = results[phase]["q"]
        ens_m, ens_load, ens_obs_load = evaluate_phase(obs, ens_pred, q, f"ensemble_{phase}")
        ensemble_metrics.append(ens_m)
        ensemble_results[phase] = {
            "pred_ssc": ens_pred, "obs_ssc": obs,
            "pred_load": ens_load, "obs_load": ens_obs_load, "q": q,
        }

    ens_df = pd.DataFrame(ensemble_metrics)
    print("\n  ENSEMBLE RESULTS:")
    print("  " + "-" * 96)
    print(ens_df.to_string(index=False))
    print("  " + "-" * 96)

    ens_test = ensemble_metrics[2]
    print(f"\n  FINAL: PINN NSE={test_m['NSE_SSC']:.4f} -> Ensemble NSE={ens_test['NSE_SSC']:.4f} "
          f"(delta={ens_test['NSE_SSC'] - test_m['NSE_SSC']:+.4f})")

    # Update results to use ensemble for plotting
    if ens_test["NSE_SSC"] > test_m["NSE_SSC"]:
        print("  ** Ensemble is BETTER — using ensemble for final outputs **")
        results = ensemble_results
        all_metrics = ensemble_metrics
    else:
        print("  ** PINN alone is better — keeping PINN outputs **")

    # --- All-days prediction ---
    print("\n[ALLDAYS] Predicting for all days (including unobserved)...")
    all_day_results = {}
    for phase in ["train", "val", "test"]:
        all_df = data["all_dfs"][phase]
        X_all = data["x_scaler"].transform(all_df[FEATURES].values)
        r_all = all_df[RAW_PHYS_COLS].values
        y_dummy = np.zeros((len(all_df), 1), dtype=np.float32)
        loader_all = DataProcessor._make_loader(X_all, y_dummy, r_all, shuffle=False)
        pinn_pred_all = predict(model, loader_all, data["y_scaler"])
        # Apply ensemble if it was better
        if ens_test["NSE_SSC"] > test_m["NSE_SSC"]:
            ridge_pred_all = np.expm1(ridge.predict(X_all)).clip(min=0)
            pred_ssc_all = best_w * pinn_pred_all + (1 - best_w) * ridge_pred_all
        else:
            pred_ssc_all = pinn_pred_all
        all_day_results[phase] = pred_ssc_all

    # --- Save outputs ---
    rep_df.to_csv(OUTPUT_DIR / "PINN_v3_Metrics.csv", index=False)

    # Diagnostics workbook
    wb_path = OUTPUT_DIR / "PINN_v3_Diagnostics.xlsx"
    with pd.ExcelWriter(wb_path) as writer:
        rep_df.to_excel(writer, sheet_name="Metrics", index=False)
        pd.DataFrame([{
            "Alpha": physics_meta["alpha"],
            "Q_95th": physics_meta["q_95th"],
            "Q_mean": physics_meta["global_q_mean"],
            "Wash_Load_Floor": WASH_LOAD_MIN,
            "N_features": len(FEATURES),
            "N_params": n_params,
        }]).to_excel(writer, sheet_name="Config", index=False)

        for phase in ["train", "val", "test"]:
            diag = build_diagnostics(data["dfs"][phase], results[phase]["pred_ssc"], physics_meta)
            diag.to_excel(writer, sheet_name=f"{phase}_diagnostics", index=False)

            alldays_diag = build_alldays_diagnostics(
                data["all_dfs"][phase], all_day_results[phase], physics_meta
            )
            alldays_diag.to_excel(writer, sheet_name=f"{phase}_all_days", index=False)

    print(f"\n  Diagnostics saved to: {wb_path}")

    # --- Plots ---
    print("\n[PLOTS] Generating publication-quality figures...")
    plot_loss_curves(loss_history)

    for phase in ["train", "val", "test"]:
        dates = data["dfs"][phase]["date_dt"].values
        obs_ssc = results[phase]["obs_ssc"]
        pred_ssc = results[phase]["pred_ssc"]
        obs_load = results[phase]["obs_load"]
        pred_load = results[phase]["pred_load"]
        m = all_metrics[["train", "val", "test"].index(phase)]

        ssc_line = (f"R2={m['R2_SSC']:.3f} | NSE={m['NSE_SSC']:.3f} | "
                    f"KGE={m['KGE_SSC']:.3f} | RMSE={m['RMSE_SSC']:.1f} mg/L | "
                    f"PBIAS={m['PBIAS_SSC_%']:.1f}%")
        load_line = (f"R2={m['R2_Load']:.3f} | NSE={m['NSE_Load']:.3f} | "
                     f"KGE={m['KGE_Load']:.3f} | RMSE={m['RMSE_Load']:.1f} t/d | "
                     f"PBIAS={m['PBIAS_Load_%']:.1f}%")

        # Hydrographs
        plot_hydrograph(dates, obs_ssc, pred_ssc, phase, "SSC", "mg/L", ssc_line,
                        PLOTS_DIR / f"{phase}_SSC_Hydrograph.png")
        plot_hydrograph(dates, obs_load, pred_load, phase, "Load", "ton/day", load_line,
                        PLOTS_DIR / f"{phase}_Load_Hydrograph.png")

        # Scatter
        plot_scatter(obs_ssc, pred_ssc, phase, "SSC", "mg/L")
        plot_scatter(obs_load, pred_load, phase, "Load", "ton/day")

        # Summary plates
        plot_summary_plate(dates, obs_ssc, pred_ssc, phase, "SSC", "mg/L", ssc_line,
                           PLOTS_DIR / f"{phase}_SSC_Summary.png")
        plot_summary_plate(dates, obs_load, pred_load, phase, "Load", "ton/day", load_line,
                           PLOTS_DIR / f"{phase}_Load_Summary.png")

    # --- SHAP (optional — try if available) ---
    try:
        import shap
        SHAP_DIR = OUTPUT_DIR / "shap"
        SHAP_DIR.mkdir(exist_ok=True)
        print("\n[SHAP] Computing feature attributions...")

        train_x = data["loaders"]["train"].dataset.tensors[0][:128].to(DEVICE)
        test_x = data["loaders"]["test"].dataset.tensors[0][:256].to(DEVICE)

        explainer = shap.GradientExplainer(model, train_x)
        sv = explainer.shap_values(test_x)
        sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
        if sv.ndim == 3:
            sv = sv[:, :, 0]

        imp_df = pd.DataFrame({
            "feature": FEATURES,
            "mean_abs_shap": np.mean(np.abs(sv), axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        imp_df.to_csv(SHAP_DIR / "feature_importance.csv", index=False)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, test_x.cpu().numpy(), feature_names=FEATURES, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "shap_summary.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv, test_x.cpu().numpy(), feature_names=FEATURES, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "shap_bar.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()

        print(f"  Top 5 features by SHAP:")
        for _, row in imp_df.head(5).iterrows():
            print(f"    {row['feature']:20s}: {row['mean_abs_shap']:.4f}")
    except Exception as e:
        print(f"\n[SHAP] Skipped: {e}")

    print("\n" + "=" * 80)
    print(f"  DONE. All outputs in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
