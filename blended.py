#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

# ---------- Optional: CatBoost ----------
try:
    from catboost import CatBoostRegressor, Pool
    _HAVE_CATBOOST = True
except Exception:
    _HAVE_CATBOOST = False


# =========================
# ======  HELPERS  ========
# =========================
def align_features(df_pred: pd.DataFrame, columns_train: List[str], id_col: Optional[str], target_col: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Align df_pred to the training feature set:
      - drop id/target if present
      - add any missing columns as 0.0
      - drop extra columns
      - order exactly like columns_train
    Returns (X_aligned, id_series_or_None)
    """
    id_series = None
    if id_col is not None and id_col in df_pred.columns:
        id_series = df_pred[id_col].copy()

    drop_cols = []
    if id_col is not None and id_col in df_pred.columns:
        drop_cols.append(id_col)
    if target_col is not None and target_col in df_pred.columns:
        drop_cols.append(target_col)

    Xp = df_pred.drop(columns=drop_cols, errors="ignore").copy()

    # add missing as 0.0
    for c in columns_train:
        if c not in Xp.columns:
            Xp[c] = 0.0

    # keep only known training columns, in the same order
    Xp = Xp[columns_train]

    return safe_numeric_df(Xp), id_series


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics_block(y_true, y_pred, label: str):
    _rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    _mae = float(mean_absolute_error(y_true, y_pred))
    _r2 = float(r2_score(y_true, y_pred))
    print(f"{label} ({len(y_true)} samples): "
          f"RMSE={_rmse:,.2f} | MAE={_mae:,.2f} | R²={_r2:,.4f}")
    return _rmse, _mae, _r2

def price_bins_for_stratify(y: pd.Series, q: int = 10) -> Optional[pd.Series]:
    try:
        b = pd.qcut(y, q=q, duplicates="drop")
        return b if b.nunique() >= 2 else None
    except Exception:
        return None

def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0)


def train_catboost_weighted_rmse(
    X: pd.DataFrame, y: pd.Series,
    eval_size: float = 0.1,
    random_state: int = 0,
    iterations: int = 1500,
    learning_rate: float = 0.03,
    depth: int = 8,
    l2_leaf_reg: float = 6.0,
    early_stopping_rounds: int = 100,
    verbose: int = 200,
    weight_alpha: float = 2.0,   # ↑ to punish big-$ errors more (e.g., 1.5–2.0)
):
    """
    Baseline model that optimizes dollar-scale RMSE but upweights expensive homes,
    so large absolute mistakes are penalized more strongly.
    """
    Xn = safe_numeric_df(X)
    yv = pd.to_numeric(y, errors="coerce").astype(float).values

    # weights: relative to median price
    med = np.median(yv) if np.isfinite(np.median(yv)) else 1.0
    w_full = np.power(np.clip(yv / max(med, 1e-9), 1e-9, None), weight_alpha)

    X_tr, X_ev, y_tr, y_ev, w_tr, w_ev = train_test_split(
        Xn, yv, w_full,
        test_size=eval_size,
        random_state=random_state,
        stratify=price_bins_for_stratify(pd.Series(yv), q=10)
    )

    train_pool = Pool(X_tr, y_tr, weight=w_tr)
    eval_pool  = Pool(X_ev, y_ev, weight=w_ev)

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_state,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

    pred_tr = model.predict(Xn)  # in dollars
    return model, pred_tr


# =========================
# ======  CATBOOST B  =====
# =========================

def train_catboost_rmse(
    X: pd.DataFrame, y: pd.Series,
    eval_size: float = 0.1,
    random_state: int = 0,
    iterations: int = 1500,
    learning_rate: float = 0.03,
    depth: int = 8,
    l2_leaf_reg: float = 6.0,
    early_stopping_rounds: int = 100,
    verbose: int = 200,
) -> Tuple[CatBoostRegressor, np.ndarray, np.ndarray]:
    if not _HAVE_CATBOOST:
        raise RuntimeError("CatBoost is not installed. Run: pip install catboost")

    Xn = safe_numeric_df(X)
    yv = y.values.astype(float)

    X_tr, X_ev, y_tr, y_ev = train_test_split(
        Xn, yv, test_size=eval_size, random_state=random_state,
        stratify=price_bins_for_stratify(pd.Series(yv), q=10)
    )

    train_pool = Pool(X_tr, y_tr)
    eval_pool  = Pool(X_ev, y_ev)

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_state,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )
    model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

    pred_tr = model.predict(Xn)
    return model, pred_tr, model.get_feature_importance(train_pool)


# =========================
# ======  BLENDING  =======
# =========================

def best_weight_grid(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                     step: float = 0.01) -> float:
    best_w = 0.5
    best_val = float("inf")
    for w in np.arange(0.0, 1.0 + 1e-12, step):
        pred = (1.0 - w) * p_a + w * p_b
        v = rmse(y_true, pred)
        if v < best_val:
            best_val = v
            best_w = float(w)
    return best_w

def gated_blend(
    p_a: np.ndarray, p_b: np.ndarray,
    w_b: float,
    log_diff_gate: float = 0.0
) -> np.ndarray:
    """
    If |log(pb) - log(pa)| > gate → fall back to baseline (A). Else blend.
    """
    if log_diff_gate <= 0:
        return (1.0 - w_b) * p_a + w_b * p_b
    la = np.log(np.clip(p_a, 1e-9, None))
    lb = np.log(np.clip(p_b, 1e-9, None))
    mask_disagree = np.abs(lb - la) > log_diff_gate
    blend = (1.0 - w_b) * p_a + w_b * p_b
    blend[mask_disagree] = p_a[mask_disagree]
    return blend


# =========================
# ======  DRIVER  =========
# =========================

def main(
    csv_path: str = "out/train_preprocessed.csv",
    id_col: str = "Id",
    target_col: str = "SalePrice",
    calib_size: float = 0.2,
    seed: int = 0,
    out_dir: str = "out",
    # Baseline CatBoost (weighted RMSE on $)
    cbA_iterations: int = 1500,
    cbA_lr: float = 0.03,
    cbA_depth: int = 8,
    cbA_l2: float = 6.0,
    cbA_early_stop: int = 100,
    cbA_verbose: int = 200,
    cbA_weight_alpha: float = 1.5,
    cb_iterations: int = 1500,
    cb_lr: float = 0.03,
    cb_depth: int = 8,
    cb_l2: float = 6.0,
    cb_early_stop: int = 100,
    cb_verbose: int = 200,
    # Blending
    gate_logdiff: float = 0.35,
    test_size: float = 0.15,
    predict_csv: Optional[str] = None,
    pred_out: str = "out/ensemble_test_predictions.csv"
):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # ---------- Load ----------
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")
    y = pd.to_numeric(df[target_col], errors="coerce").astype(float)
    X = df.drop(columns=[target_col] + ([id_col] if id_col in df.columns else []))

    # ---------- Split: TEST (final holdout) first ----------
    inference_mode = predict_csv is not None

    # ---------- Split data ----------
    if inference_mode:
        # No final test holdout. Use all data for train+calibration.
        bins_all = price_bins_for_stratify(y, q=10)
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y,
            test_size=calib_size,
            random_state=seed,
            stratify=bins_all
        )
    else:
        # Your existing: final TEST first, then TRAIN vs CALIBRATION
        bins_all = price_bins_for_stratify(y, q=10)
        X_rem, X_test, y_rem, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            stratify=bins_all
        )

        bins_rem = price_bins_for_stratify(y_rem, q=10)
        calib_frac_of_rem = calib_size / (1.0 - test_size)
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X_rem, y_rem,
            test_size=calib_frac_of_rem,
            random_state=seed,
            stratify=bins_rem
        )

    # ---------- Model A: CatBoost baseline (weighted RMSE on $) ----------
    print("\n=== CatBoost A (dollar RMSE, weighted) ===")
    cbA_model, yhatA_tr_all = train_catboost_weighted_rmse(
        X_tr, y_tr,
        eval_size=0.10,
        random_state=seed,
        iterations=cbA_iterations,
        learning_rate=cbA_lr,
        depth=cbA_depth,
        l2_leaf_reg=cbA_l2,
        early_stopping_rounds=cbA_early_stop,
        verbose=cbA_verbose,
        weight_alpha=cbA_weight_alpha,
    )
    yhatA_cal = cbA_model.predict(safe_numeric_df(X_cal))
    if not inference_mode:
        yhatA_test = cbA_model.predict(safe_numeric_df(X_test))

    metrics_block(y_tr.values, yhatA_tr_all, "Train (CatBoost A)")
    metrics_block(y_cal.values, yhatA_cal, "Calibration (CatBoost A)")

    # ---------- Model B: CatBoost (unweighted RMSE on $) ----------
    print("\n=== CatBoost B (RMSE on $) ===")
    cb_model, yhatB_tr_all, importances = train_catboost_rmse(
        X_tr, y_tr,
        eval_size=0.10,
        random_state=seed,
        iterations=cb_iterations,
        learning_rate=cb_lr,
        depth=cb_depth,
        l2_leaf_reg=cb_l2,
        early_stopping_rounds=cb_early_stop,
        verbose=cb_verbose,
    )
    yhatB_cal = cb_model.predict(safe_numeric_df(X_cal))
    if not inference_mode:
        yhatB_test = cb_model.predict(safe_numeric_df(X_test))

    metrics_block(y_tr.values, yhatB_tr_all, "Train (CatBoost B)")
    metrics_block(y_cal.values, yhatB_cal, "Calibration (CatBoost B)")

    # ---------- Learn blend weight on CALIBRATION ----------
    wB = best_weight_grid(y_cal.values, yhatA_cal, yhatB_cal, step=0.01)

    # Optional: log-space linear stacker (still fit ONLY on calibration)
    Z_cal = np.column_stack([
        np.log(np.clip(yhatA_cal, 1e-9, None)),
        np.log(np.clip(yhatB_cal, 1e-9, None)),
    ])
    t_cal = np.log(np.clip(y_cal.values, 1e-9, None))
    reg = Ridge(alpha=1.0, fit_intercept=True).fit(Z_cal, t_cal)

    # Blended predictions
    yhat_blend_cal = np.exp(reg.predict(Z_cal))

    # If we're in inference mode, load the external file and predict
    if inference_mode:
        df_pred = pd.read_csv(predict_csv)
        Xp, id_series = align_features(df_pred, X.columns.tolist(), id_col, target_col)

        # Predict with both models
        pA_pred = cbA_model.predict(Xp)
        pB_pred = cb_model.predict(Xp)

        # Blend in log space using the stacker trained on CAL
        Z_pred = np.column_stack([
            np.log(np.clip(pA_pred, 1e-9, None)),
            np.log(np.clip(pB_pred, 1e-9, None)),
        ])
        p_blend = np.exp(reg.predict(Z_pred))

        # Build submission-style frame
        out_df = pd.DataFrame({
            (id_col if id_col in df_pred.columns else "Id"): (
                id_series.values if id_series is not None else np.arange(len(p_blend))
            ),
            "SalePrice": p_blend.astype(float),
        })

        out_path = Path(pred_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved predictions → {out_path}")
        return


    Z_test = np.column_stack([
        np.log(np.clip(yhatA_test, 1e-9, None)),
        np.log(np.clip(yhatB_test, 1e-9, None)),
    ])
    yhat_blend_test = np.exp(reg.predict(Z_test))

    print("Stacking weights (log-space):",
          dict(wA=float(reg.coef_[0]), wB=float(reg.coef_[1]), bias=float(reg.intercept_)))

    print("\n=== Blended on Calibration ===")
    print(f"  Learned weight for CatBoost (wB): {wB:.2f}  (Baseline weight = {1.0 - wB:.2f})")
    if gate_logdiff > 0:
        la = np.log(np.clip(yhatA_cal, 1e-9, None))
        lb = np.log(np.clip(yhatB_cal, 1e-9, None))
        frac_gated = float(np.mean(np.abs(lb - la) > gate_logdiff))
        print(f"  Gate active (|Δlog|>{gate_logdiff:.2f}) on {frac_gated*100:.1f}% of calibration rows")
    metrics_block(y_cal.values, yhat_blend_cal, "Blended (Calibration)")

    # ---------- Final evaluation on TEST ----------
    print("\n=== Final evaluation on TEST (never seen by stacker) ===")
    metrics_block(y_test.values, yhatA_test,      "Test (CatBoost A weighted)")
    metrics_block(y_test.values, yhatB_test,      "Test (CatBoost B RMSE)")
    metrics_block(y_test.values, yhat_blend_test, "Test (Blended)")

    pred_cal = pd.DataFrame({
        (id_col if id_col in df.columns else "row_idx"): (
            df.loc[X_cal.index, id_col].values if id_col in df.columns else X_cal.index.values
        ),
        "y_true": y_cal.values,
        "p_baseline": yhatA_cal,
        "p_catboost": yhatB_cal,
        "p_blend": yhat_blend_cal,
    })
    pred_cal.to_csv(out / "ensemble_calibration_preds.csv", index=False)
    print(f"Saved calibration preds → {out / 'ensemble_calibration_preds.csv'}")


# =========================
# ========  CLI  ==========
# =========================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/train_preprocessed.csv")
    ap.add_argument("--id-col", default="Id")
    ap.add_argument("--target-col", default="SalePrice")
    ap.add_argument("--calib-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out")

    ap.add_argument("--gate-logdiff", type=float, default=0.5)

    ap.add_argument("--predict-csv", default=None)
    ap.add_argument("--pred-out", default="out/ensemble_test_predictions.csv")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        csv_path=args.csv,
        id_col=args.id_col,
        target_col=args.target_col,
        calib_size=args.calib_size,
        test_size=args.test_size,
        seed=args.seed,
        out_dir=args.out,
        gate_logdiff=args.gate_logdiff,
        predict_csv=args.predict_csv,
        pred_out=args.pred_out,
    )
