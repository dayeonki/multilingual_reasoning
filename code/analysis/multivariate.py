import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FormatStrFormatter

from sklearn.utils import resample


figtree_candidates = [
    Path.home() / ".local/share/fonts/Figtree-Regular.ttf",
    Path.home() / ".local/share/fonts/Figtree-Medium.ttf",
    Path.home() / ".local/share/fonts/Figtree-Bold.ttf",
]
for font_path in figtree_candidates:
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))

try:
    fm.findfont("Figtree", fallback_to_default=False)
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Figtree", "DejaVu Sans"]
except Exception:
    print("[Warning] Figtree font not found. Falling back to default sans-serif.")
    mpl.rcParams["font.family"] = "sans-serif"

mpl.rcParams["font.weight"] = "normal"
mpl.rcParams["font.size"] = 10

# ---------------------------------------------------------
DATASET = "aime"
MODELS = ["distill_qwen1.5b", "distill_qwen7b", "qwen4b", "qwen8b"]
# ---------------------------------------------------------

MODEL_TITLES = {
    "distill_qwen7b": "DeepSeek-Distill-Qwen-7B",
    "distill_qwen1.5b": "DeepSeek-Distill-Qwen-1.5B",
    "qwen4b": "Qwen3-4B",
    "qwen8b": "Qwen3-8B",
}
# ---------------------------------------------------------
LANGS = ["en", "bn", "de", "es", "fr", "ja", "ko", "ru", "sw", "te", "th", "zh"]
# ---------------------------------------------------------

# ---------------------------------------------------------
INDICATOR_COLS = [
    "q_comet",
    "r_length",
    "r_numsteps",
    "r_enalign",
    "r_ensimilar",
    "r_validity",
    "r_direct_utility",
    "r_indirect_utility",
    "r_vi",
    "r_selfcheck",
    "r_active",
    "r_problem",
    "r_plan",
    "r_final",
    "r_fact",
    "r_result",
    "r_uncertainty",
]

INDICATOR_RENAME = {
    "q_comet": "COMET-QE",
    "r_enalign": "Structural Similarity",
    "r_ensimilar": "Semantic Similarity",
    "r_numsteps": "Num. Steps",
    "r_validity": "Validity",
    "r_direct_utility": "Direct Utility",
    "r_indirect_utility": "Indirect Utility",
    "r_vi": "V-Information",
    "r_selfcheck": "Self-Checking",
    "r_active": "Active Computation",
    "r_problem": "Problem Setup",
    "r_plan": "Plan Generation",
    "r_final": "Final Answer Emission",
    "r_fact": "Fact Retrieval",
    "r_result": "Result Consolidation",
    "r_uncertainty": "Uncertainty Management",
}

INDICATOR_ORDER = [
    "COMET-QE",
    "Structural Similarity",
    "Semantic Similarity",
    "Num. Steps",
    "Validity",
    "Direct Utility",
    "Indirect Utility",
    "V-Information",
    "Self-Checking",
    "Active Computation",
    "Problem Setup",
    "Plan Generation",
    "Final Answer Emission",
    "Fact Retrieval",
    "Result Consolidation",
    "Uncertainty Management",
]

INDICATOR_STYLE = {
    "q_comet": {"prefix": "♥", "color": "#a53860"},
    "r_enalign": {"prefix": "♥", "color": "#a53860"},
    "r_ensimilar": {"prefix": "♥", "color": "#a53860"},
    "r_numsteps": {"prefix": "◆︎", "color": "#3a5a40"},
    "r_validity": {"prefix": "◆︎", "color": "#3a5a40"},
    "r_direct_utility": {"prefix": "◆︎", "color": "#3a5a40"},
    "r_indirect_utility": {"prefix": "◆︎", "color": "#3a5a40"},
    "r_vi": {"prefix": "◆︎", "color": "#3a5a40"},
    "r_selfcheck": {"prefix": "◼", "color": "#2a6f97"},
    "r_active": {"prefix": "◼", "color": "#2a6f97"},
    "r_problem": {"prefix": "◼", "color": "#2a6f97"},
    "r_plan": {"prefix": "◼", "color": "#2a6f97"},
    "r_final": {"prefix": "◼", "color": "#2a6f97"},
    "r_fact": {"prefix": "◼", "color": "#2a6f97"},
    "r_result": {"prefix": "◼", "color": "#2a6f97"},
    "r_uncertainty": {"prefix": "◼", "color": "#2a6f97"},
}


def bootstrap_ci(data: pd.Series, n_boot: int = 1000, ci: int = 95) -> tuple[float, float]:
    data = data.dropna()
    if len(data) < 2:
        mean_val = float(data.mean()) if len(data) else 0.0
        return mean_val, mean_val
    boot_means = [resample(data, replace=True, n_samples=len(data)).mean() for _ in range(n_boot)]
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return float(lower), float(upper)


def _standardize_within_language(df: pd.DataFrame) -> None:
    """In-place: z-score each indicator column per language (μ, σ within ℓ)."""
    cols = [c for c in INDICATOR_COLS if c in df.columns]
    for lang in LANGS:
        mask = df["language"] == lang
        for c in cols:
            vals = df.loc[mask, c].to_numpy(dtype=float)
            mu = np.nanmean(vals)
            sig = np.nanstd(vals, ddof=0)
            if sig > 0 and np.isfinite(sig):
                df.loc[mask, c] = (df.loc[mask, c] - mu) / sig
            else:
                df.loc[mask, c] = 0.0


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    eta = np.clip(eta, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-eta))


def load_model_df(model: str) -> pd.DataFrame:
    all_rows = []
    for lang in LANGS:
        fpath = Path(f"{DATASET}/{model}/{DATASET}_{lang}.jsonl")
        df_lang = pd.read_json(fpath, lines=True)
        df_lang["language"] = lang
        all_rows.append(df_lang)
    df = pd.concat(all_rows, ignore_index=True)
    _standardize_within_language(df)
    return df


def compute_model_agg(df: pd.DataFrame) -> pd.DataFrame:
    delta_rows = []
    for lang in LANGS:
        df_lang = df[df["language"] == lang]
        lang_indicator_cols = [c for c in INDICATOR_COLS if c in df_lang.columns]
        if lang == "en":
            lang_indicator_cols = [c for c in lang_indicator_cols if c != "q_comet"]

        varying = [c for c in lang_indicator_cols if df_lang[c].dropna().nunique() > 1]
        if not varying:
            continue

        sub = df_lang[varying + ["acc"]].dropna()
        if len(sub) <= len(varying) + 1:
            continue

        X = sm.add_constant(sub[varying], has_constant="add")
        y = sub["acc"]
        try:
            logit = sm.Logit(y, X, missing="drop").fit(disp=0)
        except Exception:
            continue

        Z = sub[varying].to_numpy(dtype=float)
        beta = np.array([logit.params[c] for c in varying], dtype=float)
        const = float(logit.params["const"])
        eta = const + Z @ beta

        for j_idx, j_col in enumerate(varying):
            bj = float(beta[j_idx])
            zj = Z[:, j_idx]
            eta_base = eta - bj * zj
            delta_ame = float((_sigmoid(eta_base + bj) - _sigmoid(eta_base - bj)).mean())
            delta_rows.append(
                {
                    "language": lang,
                    "indicator": j_col,
                    "delta_accuracy": delta_ame,
                }
            )

    df_delta = pd.DataFrame(delta_rows).dropna()
    if df_delta.empty:
        return pd.DataFrame()

    df_delta["indicator_readable"] = df_delta["indicator"].map(INDICATOR_RENAME)
    df_delta = df_delta.dropna(subset=["indicator_readable"])
    df_delta["is_en"] = df_delta["language"] == "en"

    df_agg = (
        df_delta.groupby(["indicator_readable", "is_en"])
        .agg(delta_mean=("delta_accuracy", "mean"), n=("delta_accuracy", "count"))
        .reset_index()
    )
    df_agg["group"] = df_agg["is_en"].map({True: "English", False: "Non-English"})

    # Pooled multivariate interaction: acc ~ is_en + all z + (is_en × z_j) for tested j
    df_for_logit = df.copy()
    df_for_logit["is_en"] = (df_for_logit["language"] == "en").astype(int)
    feat_cols = [c for c in INDICATOR_COLS if c in df_for_logit.columns]
    logit_rows = []
    for ind_key, ind_readable in INDICATOR_RENAME.items():
        if ind_key not in df_for_logit.columns:
            logit_rows.append({"indicator_readable": ind_readable, "p_en_vs_non_en": np.nan})
            continue
        sub = df_for_logit[["acc", "is_en"] + feat_cols].dropna()
        if sub["acc"].nunique() < 2 or sub["is_en"].nunique() < 2:
            logit_rows.append({"indicator_readable": ind_readable, "p_en_vs_non_en": np.nan})
            continue

        X = sm.add_constant(sub[["is_en"] + feat_cols].copy())
        interaction_col = f"is_en_x_{ind_key}"
        X[interaction_col] = sub["is_en"] * sub[ind_key]
        y = sub["acc"]
        try:
            model = sm.Logit(y, X, missing="drop").fit(disp=0)
            p_diff = model.pvalues.get(interaction_col, np.nan)
        except Exception:
            p_diff = np.nan
        logit_rows.append({"indicator_readable": ind_readable, "p_en_vs_non_en": p_diff})

    df_agg = df_agg.merge(pd.DataFrame(logit_rows), on="indicator_readable", how="left")
    df_agg.loc[df_agg["indicator_readable"] == "Structural Similarity", "p_en_vs_non_en"] = np.nan

    # Bootstrap CI around means
    err_low, err_high = [], []
    for _, row in df_agg.iterrows():
        subset = df_delta[
            (df_delta["indicator_readable"] == row["indicator_readable"])
            & (df_delta["is_en"] == row["is_en"])
        ]["delta_accuracy"]
        lower, upper = bootstrap_ci(subset)
        err_low.append(row["delta_mean"] - lower)
        err_high.append(upper - row["delta_mean"])

    df_agg["err_low"] = err_low
    df_agg["err_high"] = err_high

    # Fix plotting order
    order = INDICATOR_ORDER
    df_agg["indicator_readable"] = pd.Categorical(df_agg["indicator_readable"], categories=order, ordered=True)
    df_agg = df_agg.sort_values("indicator_readable")
    return df_agg


def p_to_star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_p_with_star(p: float) -> str:
    if pd.isna(p):
        return "p=n/a"
    p_text = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    stars = p_to_star(p)
    return f"{p_text} {stars}".strip()


def plot_one_model(ax: plt.Axes, df_agg: pd.DataFrame, title: str, show_y_labels: bool) -> None:
    legend_groups = set()
    order = INDICATOR_ORDER
    readable_to_indicator = {v: k for k, v in INDICATOR_RENAME.items()}
    y_pos_map = {name: idx for idx, name in enumerate(order)}

    for _, row in df_agg.iterrows():
        group = row["group"]
        color = {"English": "#212529", "Non-English": "#adb5bd"}[group]
        label = group if group not in legend_groups else "_nolegend_"
        legend_groups.add(group)
        y_pos = y_pos_map.get(row["indicator_readable"])
        if y_pos is None:
            continue
        ax.errorbar(
            x=row["delta_mean"],
            y=y_pos,
            xerr=[[row["err_low"]], [row["err_high"]]],
            fmt="o",
            color=color,
            markersize=4.5,
            zorder=2,
            capsize=2.5,
            label=label,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=10, pad=7)
    ax.set_xlabel(r"$\Delta$ Accuracy")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_yticks(range(len(order)))
    ax.set_ylim(len(order) - 0.5, -0.5)

    if show_y_labels:
        ax.set_ylabel("")
        ax.set_yticklabels(order)
        ax.tick_params(axis="y", pad=10)
        for tick in ax.get_yticklabels():
            label = tick.get_text()
            key = readable_to_indicator.get(label)
            color = INDICATOR_STYLE.get(key, {}).get("color", "black")
            tick.set_color(color)

        symbol_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for readable in order:
            key = readable_to_indicator[readable]
            prefix = INDICATOR_STYLE.get(key, {}).get("prefix", "")
            color = INDICATOR_STYLE.get(key, {}).get("color", "black")
            if not prefix:
                continue
            ax.text(
                -0.08,
                y_pos_map[readable],
                prefix,
                va="center",
                ha="left",
                fontsize=10,
                color=color,
                transform=symbol_transform,
                fontfamily="DejaVu Sans",
                clip_on=False,
            )
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    x_min, x_max = ax.get_xlim()
    x_offset = x_max + 0.03 * (x_max - x_min)
    for ind_readable in order:
        sub = df_agg[df_agg["indicator_readable"] == ind_readable]
        if len(sub) == 0:
            continue
        p_val = sub["p_en_vs_non_en"].iloc[0]
        ax.text(
            x_offset,
            y_pos_map[ind_readable],
            fmt_p_with_star(p_val),
            va="center",
            ha="left",
            fontsize=7.5,
            color="gray",
        )

    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)



def main() -> None:
    sns.set_theme(
        style="whitegrid",
        font_scale=0.9,
        rc={"font.family": "sans-serif", "font.sans-serif": ["Figtree", "DejaVu Sans"]},
    )

    model_aggs = {}
    for model in MODELS:
        df_model = load_model_df(model)
        model_aggs[model] = compute_model_agg(df_model)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        df_agg = model_aggs[model]
        title = MODEL_TITLES.get(model, model)
        plot_one_model(ax, df_agg, title, show_y_labels=(idx == 0))

    fig.tight_layout(rect=[0.06, 0.08, 0.98, 1.0], w_pad=0.6)
    fig.savefig(f"{DATASET}_multi.png", dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    main()
