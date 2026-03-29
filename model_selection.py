#!/usr/bin/env python
"""
Model Selection & Feature Analysis for SSC Prediction
======================================================
Systematic approach:
  Step 1: Engineer all candidate features
  Step 2: Test multiple feature sets x multiple models
  Step 3: Rank by test NSE/R² and pick the best foundation
  Step 4: Output recommendations for PINN integration

Key data findings incorporated:
  - slope is CONSTANT (0.011) -> removed
  - catchment_area_km2 is CONSTANT -> removed
  - d50_mm is a seasonal proxy (2mm dry, 63-77mm wet) -> kept but understood
  - SSC autocorrelation at lag-1 = 0.725 -> SSC lags are powerful
  - 81% measurements are consecutive -> lag features are valid
  - logQ vs logSSC = 0.527 -> log-space is better
  - Soil_Moisture vs logSSC = 0.432 -> useful predictor
  - Only 1273/2922 days have SSC -> sparse data
"""

import os
import sys
import warnings
import pathlib
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# Try importing optional packages
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — skipping XGBoost models")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] lightgbm not installed — skipping LightGBM models")

SEED = 42
np.random.seed(SEED)

DATA_PATH = pathlib.Path(r"C:\Users\suwaj\OneDrive\Desktop\Thesis\coding\PINN_MLP\data\data.xlsx")
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs" / "model_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# METRICS
# ==============================================================================
def nse(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    denom = np.sum((obs - np.mean(obs)) ** 2)
    return 1.0 - np.sum((pred - obs) ** 2) / denom if denom > 0 else np.nan

def pbias(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    denom = np.sum(obs)
    return 100.0 * np.sum(pred - obs) / denom if not np.isclose(denom, 0) else np.nan

def kge(obs, pred):
    """Kling-Gupta Efficiency — often better than NSE for hydrological models."""
    obs, pred = np.asarray(obs), np.asarray(pred)
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred) / np.std(obs) if np.std(obs) > 0 else 0
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) > 0 else 0
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def eval_metrics(obs, pred, label=""):
    return {
        "label": label,
        "R2": round(r2_score(obs, pred), 4),
        "NSE": round(nse(obs, pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(obs, pred)), 2),
        "MAE": round(mean_absolute_error(obs, pred), 2),
        "PBIAS%": round(pbias(obs, pred), 2),
        "KGE": round(kge(obs, pred), 4),
    }


# ==============================================================================
# DATA LOADING & FEATURE ENGINEERING
# ==============================================================================
def load_and_engineer(path):
    df = pd.read_excel(path)
    df["date_dt"] = df["date"].apply(lambda s: datetime(1899, 12, 30) + timedelta(days=int(s)))
    df = df.sort_values("date_dt").reset_index(drop=True)

    Q = df["Q_m3s"]
    P = df["rainfall_mm"]

    # --- Core hydrological features ---
    df["logQ"] = np.log1p(Q)
    df["logP"] = np.log1p(P)
    df["dQ"] = Q.diff().fillna(0.0)
    df["dQ_abs"] = df["dQ"].abs()
    df["Q_ratio"] = (Q / Q.rolling(7, min_periods=1).mean().replace(0, np.nan)).fillna(1.0)

    # Discharge lags
    for lag in range(1, 8):
        df[f"Q_lag{lag}"] = Q.shift(lag).ffill().fillna(0.0)
        df[f"logQ_lag{lag}"] = np.log1p(df[f"Q_lag{lag}"])

    # Discharge rolling stats
    df["Q_max_7d"] = Q.rolling(7, min_periods=1).max()
    df["Q_min_7d"] = Q.rolling(7, min_periods=1).min()
    df["Q_MA3"] = Q.rolling(3, min_periods=1).mean()
    df["Q_MA7"] = Q.rolling(7, min_periods=1).mean()
    df["Q_MA14"] = Q.rolling(14, min_periods=1).mean()
    df["Q_MA30"] = Q.rolling(30, min_periods=1).mean()
    df["Q_std_7d"] = Q.rolling(7, min_periods=1).std().fillna(0.0)

    # Hysteresis index
    q_delta = Q.diff().fillna(0.0)
    q_maxroll = Q.rolling(7, min_periods=1).max().replace(0, np.nan)
    df["HI_proxy"] = (q_delta / q_maxroll).fillna(0.0)

    # Rising/falling limb flag
    df["rising_flag"] = (df["dQ"] > 0).astype(float)

    # --- Rainfall features ---
    df["P_MA3"] = P.rolling(3, min_periods=1).mean()
    df["P_MA7"] = P.rolling(7, min_periods=1).mean()
    df["P_MA14"] = P.rolling(14, min_periods=1).mean()
    df["P_MA30"] = P.rolling(30, min_periods=1).mean()

    # Antecedent Precipitation Index
    p_arr = P.values
    api_vals = np.zeros(len(df))
    for k in range(1, 31):
        shifted = np.concatenate([np.full(k, p_arr[:k].mean()), p_arr[:-k]])
        api_vals += (0.9 ** k) * shifted
    df["API"] = api_vals

    # SPI-30
    p30 = P.rolling(30, min_periods=5).sum()
    p30mu = p30.expanding(min_periods=5).mean()
    p30std = p30.expanding(min_periods=5).std().replace(0, np.nan)
    df["SPI_30"] = ((p30 - p30mu) / p30std).fillna(0.0)

    # --- Sediment supply proxies (no leakage) ---
    df["VS"] = (Q * df["slope"]) / (df["width_m"] * df["depth_m"])
    df["stream_power"] = Q * df["slope"]  # simplified stream power
    df["unit_stream_power"] = df["VS"] * df["slope"]

    # --- Seasonal features ---
    doy = df["date_dt"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    df["monsoon_flag"] = ((df["date_dt"].dt.month >= 6) & (df["date_dt"].dt.month <= 9)).astype(float)
    df["month"] = df["date_dt"].dt.month

    # --- Water year cumulative discharge ---
    nepali_wy = lambda dt: dt.year if dt.month >= 4 else dt.year - 1
    df["WY"] = df["date_dt"].apply(nepali_wy)
    df["Q_cumsum_wy"] = df.groupby("WY")["Q_m3s"].cumsum()
    wy_max = df.groupby("WY")["Q_cumsum_wy"].transform("max").replace(0, 1.0)
    df["Q_cumsum_wy"] = df["Q_cumsum_wy"] / wy_max

    # --- SSC lag features (WITH leakage warning) ---
    df["SSC_lag1"] = df["SSC_mgL"].shift(1).ffill().fillna(0.0)
    df["SSC_lag2"] = df["SSC_mgL"].shift(2).ffill().fillna(0.0)
    df["SSC_lag3"] = df["SSC_mgL"].shift(3).ffill().fillna(0.0)
    df["logSSC_lag1"] = np.log1p(df["SSC_lag1"].clip(lower=0))
    df["logSSC_lag2"] = np.log1p(df["SSC_lag2"].clip(lower=0))
    df["dSSC_lag1"] = (df["SSC_lag1"] - df["SSC_lag2"]).fillna(0.0)

    # Target
    df["logSSC"] = np.log1p(df["SSC_mgL"].clip(lower=0))

    return df


# ==============================================================================
# TRAIN / VAL / TEST SPLIT  (by water year — same as your PINN)
# ==============================================================================
def split_by_wy(df):
    df_obs = df.dropna(subset=["SSC_mgL"]).reset_index(drop=True)
    wys = sorted(df_obs["WY"].unique())

    test_wy = wys[-1]
    val_wy = wys[-2]
    train_wys = wys[:-2]

    train = df_obs[df_obs["WY"].isin(train_wys)].copy()
    val = df_obs[df_obs["WY"] == val_wy].copy()
    test = df_obs[df_obs["WY"] == test_wy].copy()

    print(f"  Split: train WYs={train_wys} ({len(train)}), val WY={val_wy} ({len(val)}), test WY={test_wy} ({len(test)})")
    return train, val, test


# ==============================================================================
# FEATURE SETS TO TEST
# ==============================================================================
def get_feature_sets():
    """Define multiple candidate feature sets, from simple to complex."""

    # Set 1: Minimal — just Q and seasonality
    minimal = ["logQ", "sin_doy", "cos_doy", "monsoon_flag"]

    # Set 2: Rating curve features — Q + hydro context
    rating = [
        "logQ", "Q_m3s", "dQ", "Q_ratio", "rising_flag",
        "sin_doy", "cos_doy", "monsoon_flag",
    ]

    # Set 3: Hydro-met — Q + rainfall + climate
    hydromet = [
        "logQ", "Q_m3s", "dQ", "Q_ratio", "rising_flag",
        "Q_lag1", "Q_lag2", "Q_lag3", "Q_MA7", "Q_max_7d",
        "rainfall_mm", "P_MA3", "P_MA7", "API",
        "temp_C", "Soil_Moisture", "NDVI",
        "sin_doy", "cos_doy", "monsoon_flag",
    ]

    # Set 4: Full Q-based (no SSC lags) — everything we can honestly use
    full_q = [
        "logQ", "Q_m3s", "dQ", "dQ_abs", "Q_ratio", "rising_flag",
        "Q_lag1", "Q_lag2", "Q_lag3", "logQ_lag1", "logQ_lag2", "logQ_lag3",
        "Q_MA3", "Q_MA7", "Q_MA14", "Q_MA30",
        "Q_max_7d", "Q_min_7d", "Q_std_7d",
        "HI_proxy", "Q_cumsum_wy",
        "rainfall_mm", "logP", "P_MA3", "P_MA7", "P_MA14", "P_MA30", "API", "SPI_30",
        "temp_C", "Soil_Moisture", "NDVI",
        "d50_mm", "depth_m", "width_m",
        "VS", "stream_power",
        "sin_doy", "cos_doy", "monsoon_flag",
    ]

    # Set 5: Full + SSC lags (WITH leakage — for comparison only)
    full_with_lags = full_q + [
        "SSC_lag1", "SSC_lag2", "SSC_lag3",
        "logSSC_lag1", "logSSC_lag2", "dSSC_lag1",
    ]

    # Set 6: Compact — hand-picked best predictors based on correlation analysis
    compact = [
        "logQ", "logQ_lag1", "logQ_lag2",
        "dQ", "Q_ratio", "rising_flag",
        "Q_MA7", "Q_max_7d", "HI_proxy",
        "rainfall_mm", "API", "P_MA7",
        "Soil_Moisture",
        "sin_doy", "cos_doy", "monsoon_flag",
    ]

    # Set 7: Compact + SSC lags (WITH leakage — for comparison)
    compact_lags = compact + ["logSSC_lag1", "logSSC_lag2", "dSSC_lag1"]

    return {
        "1_minimal": minimal,
        "2_rating": rating,
        "3_hydromet": hydromet,
        "4_full_Q": full_q,
        "5_full+SSC_lags": full_with_lags,
        "6_compact": compact,
        "7_compact+SSC_lags": compact_lags,
    }


# ==============================================================================
# MODELS TO TEST
# ==============================================================================
def get_models():
    models = {
        "LinearReg": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RF_200": RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "RF_500": RandomForestRegressor(n_estimators=500, max_depth=15, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "GBR_500": GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=SEED),
        "MLP_2x128": MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=500, learning_rate_init=1e-3, early_stopping=True, random_state=SEED),
        "MLP_3x256": MLPRegressor(hidden_layer_sizes=(256, 256, 256), max_iter=500, learning_rate_init=5e-4, early_stopping=True, random_state=SEED),
        "SVR_rbf": SVR(kernel="rbf", C=10.0, epsilon=0.1),
    }
    if HAS_XGB:
        models["XGB_500"] = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, verbosity=0, n_jobs=-1,
        )
        models["XGB_1000"] = xgb.XGBRegressor(
            n_estimators=1000, max_depth=8, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            random_state=SEED, verbosity=0, n_jobs=-1,
        )
    if HAS_LGB:
        models["LGBM_500"] = lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        models["LGBM_1000"] = lgb.LGBMRegressor(
            n_estimators=1000, max_depth=8, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
    return models


# ==============================================================================
# MAIN EVALUATION LOOP
# ==============================================================================
def run_experiment(df, feature_sets, models, target="logSSC"):
    """Train all model×featureset combos, evaluate in log-space and real-space."""
    train_df, val_df, test_df = split_by_wy(df)
    results = []
    best_score = -999
    best_config = None

    total = len(feature_sets) * len(models)
    count = 0

    for fs_name, feats in feature_sets.items():
        # Verify features exist
        missing = [f for f in feats if f not in train_df.columns]
        if missing:
            print(f"  [SKIP] Feature set '{fs_name}': missing {missing}")
            continue

        # Scale
        scaler_x = RobustScaler()
        scaler_y = RobustScaler()

        X_tr = scaler_x.fit_transform(train_df[feats].values)
        X_va = scaler_x.transform(val_df[feats].values)
        X_te = scaler_x.transform(test_df[feats].values)

        y_tr = scaler_y.fit_transform(train_df[[target]].values).ravel()
        y_va = scaler_y.transform(val_df[[target]].values).ravel()
        y_te = scaler_y.transform(test_df[[target]].values).ravel()

        # Real-space targets
        ssc_tr = train_df["SSC_mgL"].values
        ssc_va = val_df["SSC_mgL"].values
        ssc_te = test_df["SSC_mgL"].values

        for model_name, model in models.items():
            count += 1
            t0 = time.time()

            try:
                # Train
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_tr, y_tr)

                # Predict (scaled log-space)
                pred_tr_scaled = model_copy.predict(X_tr)
                pred_va_scaled = model_copy.predict(X_va)
                pred_te_scaled = model_copy.predict(X_te)

                # Inverse scale -> log-space
                pred_tr_log = scaler_y.inverse_transform(pred_tr_scaled.reshape(-1, 1)).ravel()
                pred_va_log = scaler_y.inverse_transform(pred_va_scaled.reshape(-1, 1)).ravel()
                pred_te_log = scaler_y.inverse_transform(pred_te_scaled.reshape(-1, 1)).ravel()

                # log-space -> real mg/L
                pred_tr_mgl = np.expm1(pred_tr_log).clip(min=0)
                pred_va_mgl = np.expm1(pred_va_log).clip(min=0)
                pred_te_mgl = np.expm1(pred_te_log).clip(min=0)

                # Metrics in REAL space (mg/L)
                m_tr = eval_metrics(ssc_tr, pred_tr_mgl, f"{model_name}|{fs_name}|train")
                m_va = eval_metrics(ssc_va, pred_va_mgl, f"{model_name}|{fs_name}|val")
                m_te = eval_metrics(ssc_te, pred_te_mgl, f"{model_name}|{fs_name}|test")

                elapsed = time.time() - t0

                row = {
                    "model": model_name,
                    "features": fs_name,
                    "n_features": len(feats),
                    "train_R2": m_tr["R2"],
                    "train_NSE": m_tr["NSE"],
                    "train_KGE": m_tr["KGE"],
                    "val_R2": m_va["R2"],
                    "val_NSE": m_va["NSE"],
                    "val_RMSE": m_va["RMSE"],
                    "val_KGE": m_va["KGE"],
                    "test_R2": m_te["R2"],
                    "test_NSE": m_te["NSE"],
                    "test_RMSE": m_te["RMSE"],
                    "test_MAE": m_te["MAE"],
                    "test_PBIAS%": m_te["PBIAS%"],
                    "test_KGE": m_te["KGE"],
                    "time_s": round(elapsed, 1),
                }
                results.append(row)

                # Track best by test NSE
                if m_te["NSE"] > best_score:
                    best_score = m_te["NSE"]
                    best_config = (model_name, fs_name)

                flag = " *** BEST ***" if (model_name, fs_name) == best_config else ""
                print(f"  [{count:3d}/{total}] {model_name:15s} x {fs_name:22s} | "
                      f"test NSE={m_te['NSE']:+.4f} R²={m_te['R2']:.4f} RMSE={m_te['RMSE']:7.1f} KGE={m_te['KGE']:+.4f} | "
                      f"val NSE={m_va['NSE']:+.4f} | {elapsed:.1f}s{flag}")

            except Exception as e:
                count += 1
                print(f"  [{count:3d}/{total}] {model_name:15s} x {fs_name:22s} | FAILED: {e}")
                continue

    return pd.DataFrame(results), best_config


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 90)
    print("  MODEL SELECTION & FEATURE ANALYSIS FOR SSC PREDICTION")
    print("=" * 90)

    print("\n[1/4] Loading and engineering features...")
    df = load_and_engineer(DATA_PATH)
    print(f"  Total rows: {len(df)}, SSC observed: {df['SSC_mgL'].notna().sum()}")

    print("\n[2/4] Defining feature sets and models...")
    feature_sets = get_feature_sets()
    models = get_models()
    print(f"  Feature sets: {len(feature_sets)}")
    for name, feats in feature_sets.items():
        print(f"    {name}: {len(feats)} features")
    print(f"  Models: {list(models.keys())}")

    print(f"\n[3/4] Running {len(feature_sets) * len(models)} experiments...")
    print("-" * 90)
    results_df, best_config = run_experiment(df, feature_sets, models)

    print("\n" + "=" * 90)
    print("[4/4] RESULTS SUMMARY")
    print("=" * 90)

    # Sort by test NSE
    results_df = results_df.sort_values("test_NSE", ascending=False).reset_index(drop=True)

    # Show top 15
    print("\n  TOP 15 CONFIGURATIONS (sorted by test NSE):")
    print("-" * 90)
    top15 = results_df.head(15)[["model", "features", "n_features",
                                  "test_NSE", "test_R2", "test_RMSE", "test_KGE", "test_PBIAS%",
                                  "val_NSE", "train_NSE"]].to_string(index=False)
    print(top15)

    # Show best per model type
    print("\n\n  BEST FEATURE SET PER MODEL TYPE:")
    print("-" * 90)
    best_per_model = results_df.loc[results_df.groupby("model")["test_NSE"].idxmax()]
    best_per_model = best_per_model.sort_values("test_NSE", ascending=False)
    print(best_per_model[["model", "features", "test_NSE", "test_R2", "test_RMSE", "test_KGE",
                           "val_NSE", "train_NSE"]].to_string(index=False))

    # Show best per feature set
    print("\n\n  BEST MODEL PER FEATURE SET:")
    print("-" * 90)
    best_per_fs = results_df.loc[results_df.groupby("features")["test_NSE"].idxmax()]
    best_per_fs = best_per_fs.sort_values("test_NSE", ascending=False)
    print(best_per_fs[["model", "features", "n_features", "test_NSE", "test_R2", "test_RMSE",
                        "test_KGE"]].to_string(index=False))

    # Impact of SSC lags
    print("\n\n  IMPACT OF SSC LAGS (data leakage test):")
    print("-" * 90)
    for base_fs, lag_fs in [("6_compact", "7_compact+SSC_lags"), ("4_full_Q", "5_full+SSC_lags")]:
        for m in results_df["model"].unique():
            base_row = results_df[(results_df["model"] == m) & (results_df["features"] == base_fs)]
            lag_row = results_df[(results_df["model"] == m) & (results_df["features"] == lag_fs)]
            if len(base_row) > 0 and len(lag_row) > 0:
                base_nse = base_row.iloc[0]["test_NSE"]
                lag_nse = lag_row.iloc[0]["test_NSE"]
                diff = lag_nse - base_nse
                print(f"  {m:15s} | {base_fs:22s} NSE={base_nse:+.4f} -> {lag_fs:22s} NSE={lag_nse:+.4f} | delta={diff:+.4f}")

    # Overall winner
    print("\n" + "=" * 90)
    best_row = results_df.iloc[0]
    print(f"  WINNER: {best_row['model']} + {best_row['features']}")
    print(f"    Test:  NSE={best_row['test_NSE']:.4f}, R²={best_row['test_R2']:.4f}, "
          f"RMSE={best_row['test_RMSE']:.1f}, KGE={best_row['test_KGE']:.4f}, PBIAS={best_row['test_PBIAS%']:.1f}%")
    print(f"    Val:   NSE={best_row['val_NSE']:.4f}")
    print(f"    Train: NSE={best_row['train_NSE']:.4f}")

    # Check overfitting
    overfit_gap = best_row["train_NSE"] - best_row["test_NSE"]
    if overfit_gap > 0.15:
        print(f"    [!] OVERFIT WARNING: train-test NSE gap = {overfit_gap:.4f}")

    # Save full results
    out_path = OUTPUT_DIR / "model_selection_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Full results saved to: {out_path}")

    # RECOMMENDATION
    print("\n" + "=" * 90)
    print("  RECOMMENDATIONS FOR PINN")
    print("=" * 90)
    # Find best no-leakage config
    no_leak = results_df[~results_df["features"].str.contains("SSC_lag")]
    if len(no_leak) > 0:
        best_nl = no_leak.iloc[0]
        print(f"\n  Best NO-LEAKAGE config: {best_nl['model']} + {best_nl['features']}")
        print(f"    Test NSE={best_nl['test_NSE']:.4f}, R²={best_nl['test_R2']:.4f}, RMSE={best_nl['test_RMSE']:.1f}")
        print(f"\n  -> Use feature set '{best_nl['features']}' as the PINN backbone")
        print(f"  -> The {best_nl['model']} baseline is what the PINN must beat")

    # Find best with-leakage for comparison
    with_leak = results_df[results_df["features"].str.contains("SSC_lag")]
    if len(with_leak) > 0:
        best_wl = with_leak.iloc[0]
        print(f"\n  Best WITH-LAGS config: {best_wl['model']} + {best_wl['features']}")
        print(f"    Test NSE={best_wl['test_NSE']:.4f} (upper bound if SSC lags available)")

    print("\n" + "=" * 90)
    print("  DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()
