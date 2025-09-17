from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from catboost import CatBoostRegressor, Pool


# ======  HELPERS  ========

def rmse(y_true, y_pred) -> float:
    # compatible with older sklearn (no squared=False)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics_block(y_true, y_pred, label: str):
    _rmse = rmse(y_true, y_pred)
    _mae = float(mean_absolute_error(y_true, y_pred))
    _r2 = float(r2_score(y_true, y_pred))
    print(f"{label} ({len(y_true)} samples): RMSE={_rmse:,.2f} | MAE={_mae:,.2f} | R²={_r2:,.4f}")
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

def gated_blend_pct(
    p_a: np.ndarray,   # baseline predictions (choose these when disagreement is large)
    p_b: np.ndarray,   # catboost predictions
    w_b: float,        # weight for catboost when not gated (0..1)
    pct_gate: float = 0.35,   # e.g., 0.35 = 35% threshold
    mode: str = "mean"        # "mean" => 2|a-b|/(|a|+|b|); "a" => |a-b|/|a|; "max" => |a-b|/max(|a|,|b|)
) -> tuple[np.ndarray, np.ndarray]:
    """
    If relative difference > pct_gate -> use baseline p_a.
    Else use linear blend: (1-w_b)*p_a + w_b*p_b.

    Returns:
        blend : np.ndarray
        rel   : np.ndarray  (per-row relative differences used for gating)
    """
    a = np.asarray(p_a, dtype=float)
    b = np.asarray(p_b, dtype=float)
    eps = 1e-9

    if mode == "mean":
        # symmetric % difference, scale-invariant
        rel = 2.0 * np.abs(b - a) / (np.abs(a) + np.abs(b) + eps)
    elif mode == "a":
        # percent error relative to baseline
        rel = np.abs(b - a) / (np.abs(a) + eps)
    else:  # "max"
        rel = np.abs(b - a) / (np.maximum(np.abs(a), np.abs(b)) + eps)

    blend = (1.0 - w_b) * a + w_b * b
    mask = rel > pct_gate
    blend[mask] = a[mask]        # when they strongly disagree, trust the baseline

    return blend, rel


def default_simple_features(candidates: List[str]) -> List[str]:
    """
    A compact, robust set of features most homes have after your preprocessing.
    Keep this small to avoid overfit and to be reliably available.
    """
    pref = [
        "OverallQual", "OverallCond",
        "GrLivArea", "TotalBsmtSF",
        "GarageArea", "GarageCars",
        "FullBath", "HalfBath",
        "TotRmsAbvGrd", "Bedroom", "Kitchen",
        "LotArea", "LotFrontage",
        "HouseAge", "SinceRemod",
        "GarageAge", "GarageAgeMissing",
        "HasVeneer",
        # Common ordinal flags that survived preprocessing:
        "PavedDrive", "ExterQual", "KitchenQual",
    ]
    use = [c for c in pref if c in candidates]
    # Fallback safety: if too few, just take numerics
    return use if len(use) >= 8 else candidates


# =========================
# == MODEL A: TREE, L^p ==
# =========================

def train_tree_Lp(
    X: pd.DataFrame, y: pd.Series,
    feature_list: Optional[List[str]] = None,
    p: float = 2.0,                 # >2 → harsher on large errors, gentler on small
    iters: int = 3,                 # IRLS iterations
    max_depth: int = 8,
    min_samples_leaf: int = 50,
    random_state: int = 0,
    weight_clip: Tuple[float, float] = (0.1, 50.0),  # stabilize weights
) -> Tuple[DecisionTreeRegressor, np.ndarray]:
    """
    Fit a DecisionTreeRegressor with an approximate L^p loss (p>2) via IRLS:
      minimize sum_i |e_i|^p  ≈  minimize sum_i w_i * e_i^2, with w_i ∝ |e_i|^(p-2)
    This makes small errors matter less (vs MSE) and large errors matter more.

    Returns: (fitted_tree, in_sample_predictions)
    """
    if feature_list is None:
        feature_list = default_simple_features(X.columns.tolist())
    Xs = safe_numeric_df(X[feature_list])
    yv = pd.to_numeric(y, errors="coerce").astype(float).values

    # Start with uniform weights
    w = np.ones_like(yv, dtype=float)
    eps = 1e-6

    tree = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    yhat = np.full_like(yv, yv.mean())
    for t in range(iters):
        tree.fit(Xs, yv, sample_weight=w)
        yhat = tree.predict(Xs)
        res = yv - yhat

        # Robust scale (median absolute residual) to normalize error
        scale = np.median(np.abs(res)) + eps
        # IRLS weight update for L^p: w = (|e|/scale)^(p-2)
        w = np.power(np.abs(res) / scale, max(p - 2.0, 0.0))
        # Clip to keep the tree stable
        w = np.clip(w, weight_clip[0], weight_clip[1])

    return tree, yhat


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
) -> Tuple[CatBoostRegressor, np.ndarray]:


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
    return model, pred_tr


# =========================
# ======  BLENDING  =======
# =========================

def best_weight_grid(y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                     step: float = 0.01) -> float:
    """
    Simple convex blend: argmin_w RMSE( (1-w)*A + w*B ) on calibration.
    """
    best_w = 0.5
    best_rmse = float("inf")
    for w in np.arange(0.0, 1.0 + 1e-12, step):
        pred = (1.0 - w) * p_a + w * p_b
        r = rmse(y_true, pred)
        if r < best_rmse:
            best_rmse = r
            best_w = float(w)
    return best_w


# =========================
# ======  DRIVER  =========
# =========================

def main(
    csv_path: str = "out/train_preprocessed.csv",
    id_col: str = "Id",
    target_col: str = "SalePrice",
    calib_size: float = 0.2,
    test_size: float = 0.15,
    seed: int = 42,
    out_dir: str = "out",
    # Baseline TREE (L^p)
    bl_p: float = 3.0,
    bl_iters: int = 3,
    bl_max_depth: int = 8,
    bl_min_samples_leaf: int = 50,
    # CatBoost B
    cb_iterations: int = 1500,
    cb_lr: float = 0.03,
    cb_depth: int = 8,
    cb_l2: float = 6.0,
    cb_early_stop: int = 100,
    cb_verbose: int = 200,
):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # ---------- Load ----------
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")
    y = pd.to_numeric(df[target_col], errors="coerce").astype(float)
    X = df.drop(columns=[target_col] + ([id_col] if id_col in df.columns else []))

    # ---------- Split: TEST first ----------
    bins_all = price_bins_for_stratify(y, q=10)
    X_rem, X_test, y_rem, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=bins_all
    )

    # ---------- Split remaining: TRAIN vs CALIBRATION ----------
    bins_rem = price_bins_for_stratify(y_rem, q=10)
    calib_frac_of_rem = calib_size / (1.0 - test_size)
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_rem, y_rem,
        test_size=calib_frac_of_rem,
        random_state=seed,
        stratify=bins_rem
    )

    # ---------- Model A: Simple TREE with L^p loss (p>2) ----------
    base_feats = default_simple_features(X_tr.columns.tolist())
    print("\n=== Baseline A: Decision Tree (approx L^p, p={:.2f}) ===".format(bl_p))
    treeA, yhatA_tr = train_tree_Lp(
        X_tr, y_tr,
        feature_list=base_feats,
        p=bl_p,
        iters=bl_iters,
        max_depth=bl_max_depth,
        min_samples_leaf=bl_min_samples_leaf,
        random_state=seed,
    )
    yhatA_cal  = treeA.predict(safe_numeric_df(X_cal[base_feats]))
    yhatA_test = treeA.predict(safe_numeric_df(X_test[base_feats]))

    metrics_block(y_tr.values,  yhatA_tr,  "Train (Tree L^p)")
    metrics_block(y_cal.values, yhatA_cal, "Calibration (Tree L^p)")

    # ---------- Model B: CatBoost RMSE (full features) ----------
    print("\n=== Model B: CatBoost (RMSE on $) ===")

    cb_model, yhatB_tr = train_catboost_rmse(
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
    yhatB_cal  = cb_model.predict(safe_numeric_df(X_cal))
    yhatB_test = cb_model.predict(safe_numeric_df(X_test))

    metrics_block(y_tr.values,  yhatB_tr,  "Train (CatBoost)")
    metrics_block(y_cal.values, yhatB_cal, "Calibration (CatBoost)")

    # ---------- Blend learned ONLY on CAL ----------
    wB = best_weight_grid(y_cal.values, yhatA_cal, yhatB_cal, step=0.01)
    yhat_blend_cal = (1.0 - wB) * yhatA_cal + wB * yhatB_cal
    yhat_blend_test = (1.0 - wB) * yhatA_test + wB * yhatB_test

    # Calibration blend with gate
    yhat_blend_cal, rel_cal = gated_blend_pct(
        yhatA_cal, yhatB_cal, w_b=wB, pct_gate=0.4, mode="mean"
    )
    print(f"  Gate active (>35% rel diff) on {100.0 * np.mean(rel_cal > 0.4):.1f}% of CAL rows")

    # Test blend with the SAME gate and weight learned on CAL
    yhat_blend_test, rel_test = gated_blend_pct(
        yhatA_test, yhatB_test, w_b=wB, pct_gate=0.6, mode="mean"
    )
    print(f"  Gate active (>35% rel diff) on {100.0 * np.mean(rel_test > 0.4):.1f}% of TEST rows")

    print("\n=== Blended ===")
    print(f"  Learned blend weight for CatBoost (wB): {wB:.2f}  (Tree weight = {1.0 - wB:.2f})")
    metrics_block(y_cal.values,  yhat_blend_cal,  "Calibration (Blended)")
    print("\n=== Final evaluation on TEST (never seen by blender) ===")
    metrics_block(y_test.values, yhatA_test,      "Test (Tree L^p)")
    metrics_block(y_test.values, yhatB_test,      "Test (CatBoost)")
    metrics_block(y_test.values, yhat_blend_test, "Test (Blended)")

    # ---------- Save artifacts ----------
    try:
        import joblib
        joblib.dump({
            "tree_model": treeA,
            "tree_features": base_feats,
            "tree_p": bl_p,
            "tree_iters": bl_iters,
            "cb_model": cb_model,
            "blend_wB": wB,
            "id_col": id_col,
            "target_col": target_col,
            "columns_train": X.columns.tolist(),
        }, out / "ensemble_artifacts.joblib")
        print(f"\nSaved artifacts → {out / 'ensemble_artifacts.joblib'}")
    except Exception:
        print("\n(joblib not available; skipping artifact save)")

    # Save calibration preds for inspection
    pred_cal = pd.DataFrame({
        (id_col if id_col in df.columns else "row_idx"): (
            df.loc[X_cal.index, id_col].values if id_col in df.columns else X_cal.index.values
        ),
        "y_true": y_cal.values,
        "p_tree": yhatA_cal,
        "p_cb":   yhatB_cal,
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

    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--calib-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out")

    # Baseline TREE (L^p) params
    ap.add_argument("--bl-p", type=float, default=2.0)
    ap.add_argument("--bl-iters", type=int, default=3)
    ap.add_argument("--bl-max-depth", type=int, default=8)
    ap.add_argument("--bl-min-samples-leaf", type=int, default=50)


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
        bl_p=args.bl_p,
        bl_iters=args.bl_iters,
        bl_max_depth=args.bl_max_depth,
        bl_min_samples_leaf=args.bl_min_samples_leaf
    )
