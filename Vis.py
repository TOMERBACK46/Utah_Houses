from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr, spearmanr


def _sanitize(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.U)

def _numeric_pair(df: pd.DataFrame, x_col: str, y_col: str):
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    m = np.isfinite(x) & np.isfinite(y)
    return x[m].to_numpy(), y[m].to_numpy()

def _corr(x: np.ndarray, y: np.ndarray, method: str):
    """Return (r, p or None, n)."""
    n = x.size
    if n < 2:
        return np.nan, None, n
    if method == "spearman":
        r, p = spearmanr(x, y)
    else:  # pearson
        r, p = pearsonr(x, y)
    return float(r), (None if (p is None or pd.isna(p)) else float(p)), n

def correlate_and_plot(
    df: pd.DataFrame,
    price_col: str = "price",
    cols="*",
    out_dir: str = "charts",
    method: str = "pearson",
    show_line: bool = True,
    dpi: int = 150,
    write_txt: bool = False,                # NEW
    txt_filename: str | None = None,        # NEW
):
    """
    For each column in `cols`, save a scatter of price vs column with correlation in the title.
    Returns a DataFrame with columns=[column, r, p, n, path].

    cols: list[str] or "*" to auto-use all other columns that can be coerced to numeric.
    method: "pearson" or "spearman".
    If write_txt=True, also saves a TXT file: "column, correlation" ordered by |r| desc.
    """
    if cols == "*":
        candidates = [c for c in df.columns if c != price_col]
    else:
        candidates = list(cols)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for col in candidates:
        try:
            x, y = _numeric_pair(df, col, price_col)
            if x.size < 2 or np.unique(x).size < 2 or np.unique(y).size < 2:
                results.append({"column": col, "r": np.nan, "p": None, "n": x.size, "path": None, "status": "insufficient data"})
                continue

            r, p, n = _corr(x, y, method)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(x, y, alpha=0.7, s=20)
            if show_line:
                m, b = np.polyfit(x, y, 1)
                xline = np.linspace(x.min(), x.max(), 100)
                ax.plot(xline, m * xline + b, linewidth=2)

            ax.set_xlabel(col)
            ax.set_ylabel(price_col)
            title = f"{price_col} vs {col} — {method.title()} r={r:.3f}"
            if p is not None:
                title += f" (p={p:.2g})"
            title += f", n={n}"
            ax.set_title(title)
            ax.grid(True, alpha=0.25)

            fname = f"{_sanitize(price_col)}_vs_{_sanitize(col)}_{method}.png"
            fpath = out_dir / fname
            fig.tight_layout()
            fig.savefig(fpath, dpi=dpi)
            plt.close(fig)

            results.append({"column": col, "r": r, "p": p, "n": n, "path": str(fpath), "status": "ok"})
        except Exception as e:
            results.append({"column": col, "r": np.nan, "p": None, "n": 0, "path": None, "status": f"error: {e}"})

    # Sort by absolute correlation descending (NaNs go last)
    res_df = pd.DataFrame(results).sort_values(by="r", key=lambda s: s.abs(), ascending=False)

    # === NEW: write TXT summary if requested ===
    if write_txt:
        txt_path = out_dir / (
            txt_filename if txt_filename
            else f"correlations_{_sanitize(price_col)}_{method}.txt"
        )
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("column, correlation\n")
            for _, row in res_df.dropna(subset=["r"]).iterrows():
                f.write(f"{row['column']}, {row['r']:.6f}\n")
        # stash path on the result for convenience
        res_df.attrs["txt_path"] = str(txt_path)

    return res_df


# Convenience loader if you prefer passing a CSV path:
def correlate_and_plot_from_csv(csv_path: str, price_col="price", cols="*", **kwargs):
    df = pd.read_csv(csv_path)
    return correlate_and_plot(df, price_col=price_col, cols=cols, **kwargs)


def save_pca_2d_png_and_3d_html(csv_path: str,
                                label_col: str,
                                out_dir: str = "pca_plots",
                                id_col: str = "Id",
                                dpi: int = 150) -> dict:
    """
    PCA on numeric features (excluding `label_col` and `id_col`).
    Saves FOUR plots:
      1) 2D PNG:       PC1 (x) vs label_col (y)
      2) 3D HTML:      PC1/PC2/PC3 colored by label_col
      3) 2D PNG (LOG): PC1 (x) vs log1p(label_col) (y)
      4) 3D HTML(LOG): PC1/PC2/PC3 colored by log1p(label_col)

    Returns a dict of file paths.
    """
    # Plotly import (for HTML)
    try:
        import plotly.express as px
    except Exception as e:
        raise RuntimeError("This function needs Plotly for interactive HTML. "
                           "Install it with: pip install plotly") from e

    def _sanitize(s: str) -> str:
        return re.sub(r"[^\w\-]+", "_", str(s).strip())

    csv_path = Path(csv_path)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- load & basic checks
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"label_col '{label_col}' not found in CSV.")
    if id_col is not None and id_col not in df.columns:
        id_col = None  # ignore silently if absent

    # --- split label and features
    y_raw = df[label_col]
    feat_df = df.drop(columns=[c for c in (label_col, id_col) if c])

    # --- keep numeric features only, coerce & drop all-NaN cols
    for c in feat_df.columns:
        feat_df[c] = pd.to_numeric(feat_df[c], errors="coerce")
    feat_df = feat_df.dropna(axis=1, how="all")
    if feat_df.shape[1] < 2:
        raise ValueError(f"Need at least 2 numeric feature columns; got {feat_df.shape[1]}.")

    # --- impute & scale
    feat_df = feat_df.fillna(feat_df.median(numeric_only=True))
    X = feat_df.to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)

    # --- PCA up to 3 components
    n_comps = min(3, Xs.shape[1])
    pca = PCA(n_components=n_comps)
    PCs = pca.fit_transform(Xs)   # (n_samples, n_comps)
    evr = pca.explained_variance_ratio_
    if n_comps < 3:
        PCs = np.c_[PCs, np.zeros((PCs.shape[0], 3 - n_comps))]
        evr = np.r_[evr, np.zeros(3 - n_comps)]

    # --- label as numeric + its log
    y_num_all = pd.to_numeric(y_raw, errors="coerce")  # may contain NaN
    # log1p requires values >= -1; for prices this will be fine. Filter non-finite later.
    y_log_all = np.log1p(y_num_all)

    base = f"PCA_by_{_sanitize(label_col)}"
    path2d        = out_dir / f"{base}_2D.png"
    path3d_html   = out_dir / f"{base}_3D.html"
    path2d_log    = out_dir / f"{base}_LOG_2D.png"
    path3d_log    = out_dir / f"{base}_LOG_3D.html"

    # ---------------- 2D PNG: PC1 vs PRICE ----------------
    y_axis = y_num_all
    mask = y_axis.notna()
    if mask.sum() < 2:
        raise ValueError(f"`{label_col}` must be numeric (or coercible) for 2D plot.")
    x_pc1 = PCs[mask.to_numpy(), 0]
    y_price = y_axis[mask].to_numpy()
    r = np.corrcoef(x_pc1, y_price)[0, 1] if len(x_pc1) > 1 else np.nan

    fig2d, ax2d = plt.subplots(figsize=(7, 5))
    ax2d.scatter(x_pc1, y_price, alpha=0.85, s=16)
    ax2d.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax2d.set_ylabel(label_col)
    title = f"PCA 1D — PC1 vs {label_col}"
    if np.isfinite(r):
        title += f"  (r = {r:.3f})"
    ax2d.set_title(title)
    ax2d.grid(True, alpha=0.25)
    fig2d.tight_layout()
    fig2d.savefig(path2d, dpi=dpi)
    plt.close(fig2d)

    # ---------------- 3D HTML: colored by PRICE ----------------
    numeric_like = (y_num_all.notna().mean() > 0.9) and (y_raw.nunique(dropna=True) > 10)
    df_plot = pd.DataFrame({
        "PC1": PCs[:, 0],
        "PC2": PCs[:, 1],
        "PC3": PCs[:, 2],
        "__label_num__": y_num_all,
        "__label_str__": y_raw.fillna("Missing").astype(str),
    })
    hover_cols = []
    if id_col is not None:
        df_plot[id_col] = df[id_col]; hover_cols.append(id_col)
    df_plot[label_col] = y_raw; hover_cols.append(label_col)

    if numeric_like:
        fig = px.scatter_3d(
            df_plot, x="PC1", y="PC2", z="PC3",
            color="__label_num__", color_continuous_scale="Viridis",
            hover_data=hover_cols, opacity=0.9
        )
    else:
        fig = px.scatter_3d(
            df_plot, x="PC1", y="PC2", z="PC3",
            color="__label_str__", hover_data=hover_cols, opacity=0.9
        )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        title=f"PCA 3D — colored by {label_col}",
        scene=dict(
            xaxis_title=f"PC1 ({evr[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({evr[1]*100:.1f}%)",
            zaxis_title=f"PC3 ({evr[2]*100:.1f}%)",
        ),
        legend_title_text=label_col if not numeric_like else None,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(path3d_html, include_plotlyjs="cdn", full_html=True)

    # ---------------- 2D PNG (LOG): PC1 vs log1p(PRICE) ----------------
    mask_log = y_log_all.notna() & np.isfinite(y_log_all)
    if mask_log.sum() >= 2:
        x_pc1_log = PCs[mask_log.to_numpy(), 0]
        y_price_log = y_log_all[mask_log].to_numpy()
        r_log = np.corrcoef(x_pc1_log, y_price_log)[0, 1] if len(x_pc1_log) > 1 else np.nan

        fig2dL, ax2dL = plt.subplots(figsize=(7, 5))
        ax2dL.scatter(x_pc1_log, y_price_log, alpha=0.85, s=16)
        ax2dL.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
        ax2dL.set_ylabel(f"log1p({label_col})")
        titleL = f"PCA 1D — PC1 vs log1p({label_col})"
        if np.isfinite(r_log):
            titleL += f"  (r = {r_log:.3f})"
        ax2dL.set_title(titleL)
        ax2dL.grid(True, alpha=0.25)
        fig2dL.tight_layout()
        fig2dL.savefig(path2d_log, dpi=dpi)
        plt.close(fig2dL)
    else:
        # still return a path string for consistency, but note it's not created
        path2d_log = None

    # ---------------- 3D HTML (LOG): colored by log1p(PRICE) ----------------
    df_plot_log = pd.DataFrame({
        "PC1": PCs[:, 0],
        "PC2": PCs[:, 1],
        "PC3": PCs[:, 2],
        "__label_num_log__": y_log_all,
        "__label_str__": y_raw.fillna("Missing").astype(str),  # kept for fallback, though we use numeric
    })
    hover_cols_log = []
    if id_col is not None:
        df_plot_log[id_col] = df[id_col]; hover_cols_log.append(id_col)
    df_plot_log[label_col] = y_raw; hover_cols_log.append(label_col)

    # Always numeric coloring for log (if NaN, Plotly will ignore those points’ color)
    figL = px.scatter_3d(
        df_plot_log, x="PC1", y="PC2", z="PC3",
        color="__label_num_log__", color_continuous_scale="Viridis",
        hover_data=hover_cols_log, opacity=0.9
    )
    figL.update_traces(marker=dict(size=4))
    figL.update_layout(
        title=f"PCA 3D — colored by log1p({label_col})",
        scene=dict(
            xaxis_title=f"PC1 ({evr[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({evr[1]*100:.1f}%)",
            zaxis_title=f"PC3 ({evr[2]*100:.1f}%)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    figL.write_html(path3d_log, include_plotlyjs="cdn", full_html=True)

    return {
        "pca2d_path": str(path2d),
        "pca3d_html_path": str(path3d_html),
        "pca2d_log_path": (str(path2d_log) if path2d_log is not None else None),
        "pca3d_log_html_path": str(path3d_log),
    }

if __name__ == "__main__":
    df = pd.read_csv("out/train_preprocessed.csv")
    summary_all = correlate_and_plot(
        df, price_col="SalePrice", cols="*", out_dir="charts",
        method="spearman", write_txt=True
    )
    print(summary_all[["column", "r", "p", "n", "path"]])
    print("TXT saved to:", summary_all.attrs.get("txt_path"))

    paths = save_pca_2d_png_and_3d_html("out/train_preprocessed.csv", label_col="SalePrice")
