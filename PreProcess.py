from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import re

# =========================
# ======  CONFIG  =========
# =========================
TARGET_COL = "SalePrice"
ID_COL = "Id"  # set to None if you don't have one

ORDINAL_MAPS = {
    # access / terrain
    "Street":        ["Grvl", "Pave"],
    "LandContour":   ["Low", "HLS", "Bnk", "Lvl"],           # worse → better terrain
    "Utilities":     ["ELO", "NoSeWa", "NoSewr", "AllPub"],  # fewest → most services
    "LandSlope":     ["Gtl", "Mod", "Sev"],

    # exterior quality/condition
    "ExterQual":     ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond":     ["Po", "Fa", "TA", "Gd", "Ex"],

    # basement
    "BsmtQual":      ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond":      ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure":  ["NA", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1":  ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2":  ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],

    # heating / kitchen
    "HeatingQC":     ["Po", "Fa", "TA", "Gd", "Ex"],
    "CentralAir":    ["N", "Y"],
    "KitchenQual":   ["Po", "Fa", "TA", "Gd", "Ex"],

    # functionality
    "Functional":    ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],

    # fireplaces
    "FireplaceQu":   ["NA", "Po", "Fa", "TA", "Gd", "Ex"],

    # garage (quality/cond & finish only are truly ordered)
    "GarageFinish":  ["NA", "Unf", "RFn", "Fin"],
    "GarageQual":    ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond":    ["NA", "Po", "Fa", "TA", "Gd", "Ex"],

    # driveway / outdoor
    "PavedDrive":    ["N", "P", "Y"],
    "PoolQC":        ["NA", "Fa", "TA", "Gd", "Ex"],
    "Fence":         ["NA", "MnWw", "MnPrv", "GdWo", "GdPrv"],
}

ONEHOT_SMALL = [
    "MSSubClass",
    "MSZoning",
    "Alley",
    "LotShape",
    "LotConfig",
    "BldgType",
    "RoofStyle",
    "RoofMatl",
    "MasVnrType",
    "Foundation",
    "Heating",
    "Electrical",
    "MiscFeature",
    "SaleType",
    "SaleCondition",
    "HouseStyle",
    "GarageType",
]

# Union multi-hot pairs (handled with train-frozen label sets)
UNION_MULTI_HOT = [
    {"cols": ("Condition1", "Condition2"), "prefix": "Cond_"},
    {"cols": ("Exterior1st", "Exterior2nd"), "prefix": "Ext_"},
]

# Categorical columns we NEVER drop for low-variance (they may be predictive even if imbalanced)
LOW_VAR_WHITELIST = {'Street'}

# Drop categorical columns with a single majority level above this fraction (train-based)
LOW_VARIANCE_CUTOFF = 0.98

# Rare-level cutoff (per one-hot column; train-based)
RARE_MIN_COUNT = 8

# =========================
# ======  HELPERS  ========
# =========================

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _split_numeric_categorical(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols

def _sanitize_cat(s: str) -> str:
    """Safe token for column names (values like 'Wd Sdng', 'Tar&Grv')."""
    if s is None:
        return "Missing"
    s = str(s)
    s = s.strip()
    s = re.sub(r"[^\w]+", "_", s, flags=re.U)  # non-word → _
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "Missing"

def normalize_masvnr(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize MasVnrType with MasVnrArea, add HasVeneer."""
    if "MasVnrType" in df.columns:
        df["MasVnrType"] = df["MasVnrType"].astype("string")
    if "MasVnrArea" in df.columns:
        area = pd.to_numeric(df["MasVnrArea"], errors="coerce").fillna(0)
        if "MasVnrType" in df.columns:
            typ = df["MasVnrType"].copy()
            typ = np.where(area == 0, "None", typ)
            df["MasVnrType"] = pd.Series(typ, index=df.index).fillna("Missing").astype("string")
        df["HasVeneer"] = (area > 0).astype(int)
    return df

# ---------- Fix (1)+(2): union multi-hot with train-frozen labels & sanitized names ----------
@dataclass
class UnionSpec:
    cols: Tuple[str, str]
    prefix: str
    labels: List[str]  # sanitized label tokens to generate columns for

def fit_union_multi_hot(train: pd.DataFrame, specs: List[dict]) -> List[UnionSpec]:
    fitted: List[UnionSpec] = []
    for sp in specs:
        c1, c2 = sp["cols"]
        if c1 not in train.columns or c2 not in train.columns:
            continue
        raw = pd.unique(train[[c1, c2]].values.ravel("K"))
        labels = [x for x in raw if pd.notna(x)]
        # sanitize names once; we still match by raw values when creating flags
        labels_sanitized = [_sanitize_cat(x) for x in labels]
        fitted.append(UnionSpec(cols=(c1, c2), prefix=sp["prefix"], labels=labels_sanitized))
    return fitted

def transform_union_multi_hot(df: pd.DataFrame, fitted_specs: List[UnionSpec]) -> pd.DataFrame:
    out = df.copy()
    for spec in fitted_specs:
        c1, c2 = spec.cols
        # create flags for the train-frozen label set
        if c1 in out.columns and c2 in out.columns:
            # work with raw strings to compare; ignore NaN (treated as absence)
            s1 = out[c1].astype("string")
            s2 = out[c2].astype("string")
            # For each sanitized label, we need the corresponding raw label set.
            # Since we sanitized with a 1-1 on each original string, we will check membership
            # by sanitizing row values and comparing tokens.
            tok1 = s1.fillna("Missing").map(_sanitize_cat)
            tok2 = s2.fillna("Missing").map(_sanitize_cat)
            for lab in spec.labels:
                out[f"{spec.prefix}{lab}"] = ((tok1 == lab) | (tok2 == lab)).astype(int)
            out = out.drop(columns=[c1, c2])
        else:
            # columns missing: still ensure the flag columns exist (all zeros)
            for lab in spec.labels:
                colname = f"{spec.prefix}{lab}"
                if colname not in out.columns:
                    out[colname] = 0
    return out

# ---------- Fix (4): low-variance drop with whitelist ----------
def drop_low_variance_cats(df_train: pd.DataFrame, df_test: pd.DataFrame,
                           cutoff: float = LOW_VARIANCE_CUTOFF,
                           whitelist: set[str] = {}
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    dropped = []
    for col in df_train.columns:
        if col in whitelist:
            continue
        if df_train[col].dtype == "object" or str(df_train[col].dtype) == "string":
            freq = df_train[col].value_counts(normalize=True, dropna=False)
            if not freq.empty and freq.iloc[0] >= cutoff:
                df_train = df_train.drop(columns=[col])
                if col in df_test.columns:
                    df_test = df_test.drop(columns=[col])
                dropped.append(col)
    return df_train, df_test, dropped

# ---------- Numeric imputation ----------
def impute_numeric(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    train = df_train.copy()
    test = df_test.copy()
    num_cols, _ = _split_numeric_categorical(train)
    medians = {}
    for c in num_cols:
        if c == TARGET_COL:
            continue
        med = pd.to_numeric(train[c], errors="coerce").median()
        medians[c] = med
        train[c] = pd.to_numeric(train[c], errors="coerce").fillna(med)
        if c in test.columns:
            test[c] = pd.to_numeric(test[c], errors="coerce").fillna(med)
    return train, test, medians

# ---------- Fix (3)+(2): one-hot with explicit Missing and sanitized names ----------
@dataclass
class OneHotSpec:
    categories: Dict[str, List[str]]   # raw categories to keep (strings incl. "Missing")
    other_label: str = "OTHER"

def fit_onehot(train: pd.DataFrame, cols: List[str],
               rare_min_count: int = RARE_MIN_COUNT,
               other_label: str = "OTHER") -> OneHotSpec:
    cats = {}
    for c in cols:
        s = train[c].astype("string").fillna("Missing")
        vc = s.value_counts(dropna=False)
        keep = vc[vc >= rare_min_count].index.astype("string").tolist()
        if "Missing" not in keep and ("Missing" in vc.index):
            keep.append("Missing")  # ensure Missing is explicit if present
        if len(keep) == 0:
            keep = ["Missing"]
        cats[c] = keep
    return OneHotSpec(categories=cats, other_label=other_label)

def transform_onehot(df: pd.DataFrame, spec: OneHotSpec) -> pd.DataFrame:
    out = df.copy()
    for c, keep in spec.categories.items():
        if c not in out.columns:
            # still emit columns (all zeros)
            for val in keep + [spec.other_label]:
                out[f"{c}__{_sanitize_cat(val)}"] = 0
            continue
        s = out[c].astype("string").fillna("Missing")
        mask_keep = s.isin(keep)
        s_mapped = s.where(mask_keep, spec.other_label)
        # Build indicators for kept values (+ OTHER if used)
        cats_out = keep.copy()
        if (~mask_keep).any() and spec.other_label not in cats_out:
            cats_out.append(spec.other_label)
        for val in cats_out:
            out[f"{c}__{_sanitize_cat(val)}"] = (s_mapped == val).astype(int)
        out = out.drop(columns=[c])
    return out

# ---------- Target/Mean OOF ----------


def _smoothed_mean(counts: pd.Series, means: pd.Series, global_mean: float, m: int) -> pd.Series:
    return (means * counts + global_mean * m) / (counts + m)

def apply_ordinal_maps(df_train: pd.DataFrame, df_test: pd.DataFrame,
                       maps: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = df_train.copy()
    te = df_test.copy()
    for c, order in maps.items():
        lut = {lab: i for i, lab in enumerate(order)}
        if c in tr.columns:
            tr[c] = tr[c].map(lut).fillna(-1).astype(int)
        if c in te.columns:
            te[c] = te[c].map(lut).fillna(-1).astype(int)
    return tr, te


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add age-like features using YrSold as the reference year."""
    out = df.copy()
    # coerce to numeric once
    yr_sold      = pd.to_numeric(out.get("YrSold"), errors="coerce")
    year_built   = pd.to_numeric(out.get("YearBuilt"), errors="coerce")
    year_remod   = pd.to_numeric(out.get("YearRemodAdd"), errors="coerce")
    garage_yb    = pd.to_numeric(out.get("GarageYrBlt"), errors="coerce")

    # base ages (clipped at 0; NaNs stay NaN and will be imputed later)
    house_age    = (yr_sold - year_built).clip(lower=0)
    since_remod  = (yr_sold - year_remod).clip(lower=0)
    garage_age   = (yr_sold - garage_yb).clip(lower=0)

    out["HouseAge"]          = house_age
    out["SinceRemod"]        = since_remod
    out["IsRemodeled"]       = ((year_remod > year_built).astype("Int64")).fillna(0).astype(int)
    out["IsNew"]             = ((house_age <= 1).astype("Int64")).fillna(0).astype(int)
    out["GarageAge"]         = garage_age
    out["GarageAgeMissing"]  = garage_yb.isna().astype(int)

    return out


# =========================
# ======  PIPELINE  =======
# =========================
def preprocess(train_path: Path, test_path: Path, out_dir: Path):
    _ensure_dir(out_dir)
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)

    # Keep target aside
    y = df_tr[TARGET_COL] if TARGET_COL in df_tr.columns else None

    # ---- Domain harmonization (no target) ----
    df_tr = normalize_masvnr(df_tr)
    df_te = normalize_masvnr(df_te)

    # ---- Union multi-hot (train-frozen labels)  [Fix #1] ----
    fitted_union = fit_union_multi_hot(df_tr, UNION_MULTI_HOT)
    df_tr = transform_union_multi_hot(df_tr, fitted_union)
    df_te = transform_union_multi_hot(df_te, fitted_union)

    df_tr = add_age_features(df_tr)
    df_te = add_age_features(df_te)

    # ---- Drop ultra low-variance categoricals (train-based, whitelist where the variance is small but there is still noticeable relation (street )
    df_tr, df_te, dropped = drop_low_variance_cats(df_tr, df_te, LOW_VARIANCE_CUTOFF, LOW_VAR_WHITELIST)
    if dropped:
        print(f"Dropped low-variance categoricals: {dropped}")

    # ---- Ordinal encode true-ordered categoricals
    df_tr, df_te = apply_ordinal_maps(df_tr, df_te, ORDINAL_MAPS)

    # ---- Impute numeric with TRAIN medians ----
    df_tr, df_te, med = impute_numeric(df_tr, df_te)

    # ---- One-Hot (small nominal)
    exclude = {c for c in (TARGET_COL, ID_COL) if c}
    onehot_cols = [c for c in ONEHOT_SMALL if (c in df_tr.columns and c not in exclude)]
    if onehot_cols:
        spec_oh = fit_onehot(df_tr, onehot_cols, RARE_MIN_COUNT)
        df_tr = transform_onehot(df_tr, spec_oh)
        df_te = transform_onehot(df_te, spec_oh)

    # ---- Reorder / keep ID & TARGET ----
    if ID_COL and ID_COL in df_tr.columns:
        cols = [ID_COL] + [c for c in df_tr.columns if c != ID_COL]
        df_tr = df_tr[cols]
    if ID_COL and ID_COL in df_te.columns:
        cols = [ID_COL] + [c for c in df_te.columns if c != ID_COL]
        df_te = df_te[cols]
    if TARGET_COL in df_tr.columns:
        cols = [c for c in df_tr.columns if c != TARGET_COL] + [TARGET_COL]
        df_tr = df_tr[cols]

    # ---- Align schemas (exclude target) ----
    tr_cols = set(df_tr.columns) - {TARGET_COL}
    te_cols = set(df_te.columns)
    union_cols = sorted(tr_cols | te_cols)

    for col in union_cols:
        if col not in df_tr.columns:
            df_tr[col] = 0
        if col not in df_te.columns:
            df_te[col] = 0

    # ---- Save ----
    out_train = out_dir / "train_preprocessed.csv"
    out_test = out_dir / "test_preprocessed.csv"
    df_tr.to_csv(out_train, index=False)
    df_te.to_csv(out_test, index=False)

    # ---- Report ----
    print(f"Saved: {out_train}  (shape={df_tr.shape})")
    print(f"Saved: {out_test}  (shape={df_te.shape})")



    # Alignment sanity
    tr_feats = [c for c in df_tr.columns if c != TARGET_COL]
    te_feats = df_te.columns.tolist()
    mis = [c for c in tr_feats if c not in te_feats] + [c for c in te_feats if c not in tr_feats]
    if mis:
        print("WARNING: Train/Test feature columns not perfectly aligned:", mis)

# =========================
# ========  CLI  ==========
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess Ames-style train/test -> preprocessed CSVs.")
    ap.add_argument("--train", default="data/train.csv", type=str, help="Path to train.csv")
    ap.add_argument("--test", default="data/test.csv", type=str, help="Path to test.csv")
    ap.add_argument("--out", default="out", type=str, help="Output directory")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    preprocess(Path(args.train), Path(args.test), Path(args.out))
