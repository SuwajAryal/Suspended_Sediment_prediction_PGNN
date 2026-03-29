#!/usr/bin/env python
"""
Optuna Hyperparameter Search for PINN v2
=========================================
Run overnight on RTX 4060. Searches across architecture, loss weights,
and training hyperparameters. Objective: minimize validation RMSE of SSC.

Usage:
    python optuna_search.py            # runs 150 trials
    python optuna_search.py --trials 50  # custom trial count
"""

import argparse
import pathlib
import warnings

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore")

# Import core components from main script
import triple_constrain_v2 as T

DEVICE = T.DEVICE
USE_AMP = T.USE_AMP


def objective(trial):
    """Single Optuna trial — train a model with sampled hyperparameters."""

    # --- Sample hyperparameters ---
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    n_blocks = trial.suggest_int("n_blocks", 3, 8)
    dropout = trial.suggest_float("dropout", 0.05, 0.30, step=0.05)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    lambda_floor = trial.suggest_float("lambda_floor", 0.0, 0.02)
    lambda_ceiling = trial.suggest_float("lambda_ceiling", 0.0, 0.05)
    lambda_mass = trial.suggest_float("lambda_mass", 0.0, 0.15)
    lambda_var = trial.suggest_float("lambda_var", 0.0, 0.10)

    # --- Data ---
    proc = T.DataProcessor(T.DATA_PATH)
    feats = T.BASE_FEATURES + [f"SSC_LP_lag{k}" for k in range(1, 8)]
    data = proc.split_and_scale(feats)

    # Rebuild loaders with sampled batch_size
    for phase in ["train", "val", "test"]:
        ds = data["loaders"][phase].dataset
        data["loaders"][phase] = DataLoader(
            ds, batch_size=batch_size, shuffle=(phase == "train")
        )

    y_scaler = data["y_scaler"]
    y_center = torch.tensor(y_scaler.center_[0], dtype=torch.float32, device=DEVICE)
    y_scale = torch.tensor(y_scaler.scale_[0], dtype=torch.float32, device=DEVICE)

    # --- Model ---
    temporal_indices = [feats.index(f) for f in T.TEMPORAL_FEAT_NAMES if f in feats]

    # Build a custom-sized MoE model
    class _TrialExpert(nn.Module):
        def __init__(self, in_dim, hid, nb, drop):
            super().__init__()
            self.project = nn.Sequential(nn.Linear(in_dim, hid), nn.LeakyReLU(0.1))
            self.blocks = nn.Sequential(*[T._ResBlock(hid, drop) for _ in range(nb)])
            self.head = nn.Linear(hid, 1)
        def forward(self, x):
            return self.head(self.blocks(self.project(x)))

    # Simplified single-expert model for faster search
    n_temporal = len(temporal_indices)
    n_static = len(feats) - n_temporal
    attn_out = n_temporal * 32 if n_temporal > 0 else 0
    static_out = 64 if n_static > 0 else 0
    fused_dim = attn_out + static_out

    model = T.ResidualMLP(len(feats), temporal_indices=temporal_indices).to(DEVICE)

    # Override expert sizes based on trial params
    model.expert_dry = _TrialExpert(fused_dim, hidden_dim, max(2, n_blocks - 1), dropout).to(DEVICE)
    model.expert_normal = _TrialExpert(fused_dim, hidden_dim, n_blocks, dropout).to(DEVICE)
    model.expert_flood = _TrialExpert(fused_dim, hidden_dim, n_blocks, dropout).to(DEVICE)
    model.gate = nn.Sequential(nn.Linear(fused_dim, 3), nn.Softmax(dim=-1)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    amp_scaler = GradScaler(enabled=USE_AMP)

    # --- Calibrate alpha ---
    all_actuals, all_yangs, all_qs = [], [], []
    for batch in data["loaders"]["train"]:
        yb, pb = batch[1].to(DEVICE), batch[2].to(DEVICE)
        actual_mgl = T._to_mgl(yb.squeeze(), y_center, y_scale)
        all_actuals.append(actual_mgl)
        all_yangs.append(T.get_yang_capacity(pb))
        all_qs.append(pb[:, 0])

    all_actuals = torch.cat(all_actuals)
    all_yangs = torch.cat(all_yangs)
    all_qs = torch.cat(all_qs)

    q_95th = torch.quantile(all_qs, 0.95)
    top_mask = all_qs >= q_95th
    alpha = torch.median(all_actuals[top_mask] / torch.clamp(all_yangs[top_mask], min=1e-6)).item()
    global_q_mean = torch.mean(all_qs)
    global_max_ssc = all_actuals.max().item()

    # --- Training loop (reduced epochs for search) ---
    SEARCH_EPOCHS = 300
    PATIENCE = 40
    best_val, wait = float("inf"), 0

    for epoch in range(1, SEARCH_EPOCHS + 1):
        model.train()
        progress = epoch / SEARCH_EPOCHS
        ramp = min(1.0, max(0.0, (progress - 0.125) / 0.25))

        for batch in data["loaders"]["train"]:
            xb, yb, pb = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=USE_AMP):
                pred = model(xb)
                pr_mgl = T._to_mgl(pred.squeeze(), y_center, y_scale)
                yb_mgl = T._to_mgl(yb.squeeze(), y_center, y_scale)

                peak_weights = 1.0 + 3.0 * (yb_mgl / (global_max_ssc + 1e-6))
                data_loss = torch.mean(peak_weights * (pred.squeeze() - yb.squeeze()) ** 2)

                q_vals = pb[:, 0]
                low_mask = q_vals < global_q_mean
                floor_loss = torch.tensor(0.0, device=DEVICE)
                if low_mask.any():
                    floor_loss = torch.mean(torch.nn.functional.relu(pr_mgl[low_mask] - T.WASH_LOAD_MIN) ** 2)

                high_mask = q_vals >= global_q_mean
                ceiling_loss = torch.tensor(0.0, device=DEVICE)
                if high_mask.any():
                    yang_cap = T.get_yang_capacity(pb[high_mask]) * alpha
                    ceiling_loss = torch.mean(torch.nn.functional.relu(yang_cap - pr_mgl[high_mask]) ** 2)

                mass_loss = torch.abs(torch.sum(pr_mgl) - torch.sum(yb_mgl)) / (len(yb) + 1e-6)

                var_loss = torch.tensor(0.0, device=DEVICE)
                if len(yb) > 2:
                    var_loss = (torch.std(pr_mgl) / torch.std(yb_mgl).clamp(min=1e-6) - 1.0) ** 2

                loss = (data_loss
                        + ramp * lambda_floor * floor_loss
                        + ramp * lambda_ceiling * ceiling_loss
                        + ramp * lambda_mass * mass_loss
                        + ramp * lambda_var * var_loss)

            amp_scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in data["loaders"]["val"]:
                val_loss += nn.functional.mse_loss(
                    model(b[0].to(DEVICE)), b[1].to(DEVICE), reduction="sum"
                ).item()
        val_loss /= len(data["loaders"]["val"].dataset)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

        # Optuna pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # --- Evaluate on validation set in original SSC space ---
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for b in data["loaders"]["val"]:
            preds.append(model(b[0].to(DEVICE)).cpu())
    p_raw = y_scaler.inverse_transform(torch.cat(preds).numpy()).flatten()
    if T.USE_LOG_TARGET:
        p_ssc = np.expm1(p_raw).clip(min=0.0)
    else:
        p_ssc = p_raw
    a_ssc = data["dfs"]["val"][T.TARGET_COL].values

    rmse = np.sqrt(np.mean((a_ssc - p_ssc) ** 2))
    return rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=150)
    args = parser.parse_args()

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50),
        study_name="PINN_SSC_HyperSearch",
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("BEST TRIAL:")
    print(f"  Val RMSE: {study.best_trial.value:.2f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    print("=" * 60)

    # Save results
    out_path = pathlib.Path(__file__).resolve().parent / "outputs" / "optuna_results.csv"
    df = study.trials_dataframe()
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
