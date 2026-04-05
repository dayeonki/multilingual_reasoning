import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import font_manager as fm
from matplotlib import transforms as mtransforms
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


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
LANGS = ["bn", "en", "de", "es", "fr", "ru", "sw", "te", "th", "zh"]
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

CONTINUOUS_COLS = [
    "q_comet",
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


def load_model_df(model: str) -> pd.DataFrame:
    all_rows = []
    for lang in LANGS:
        fpath = Path(f"{DATASET}/{model}/{DATASET}_{lang}.jsonl")
        df_lang = pd.read_json(fpath, lines=True)
        df_lang["language"] = lang
        all_rows.append(df_lang)
    df = pd.concat(all_rows, ignore_index=True)
    scaler = StandardScaler()
    cols = [c for c in CONTINUOUS_COLS if c in df.columns]
    df[cols] = scaler.fit_transform(df[cols])
    return df


_DELTA_COLS = ["indicator", "language", "delta_accuracy"]


def compute_model_perlang_delta(df: pd.DataFrame) -> pd.DataFrame:

    def sigma_stable(x):
        x = np.clip(np.asarray(x, dtype=float), -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    rows = []
    for lang in LANGS:
        df_lang = df[df["language"] == lang]
        lang_indicator_cols = [c for c in INDICATOR_COLS if c in df_lang.columns]
        if lang == "en":
            lang_indicator_cols = [c for c in lang_indicator_cols if c != "q_comet"]

        lang_indicator_cols = [c for c in lang_indicator_cols if df_lang[c].nunique() > 1]
        if not lang_indicator_cols:
            continue

        sub = df_lang[lang_indicator_cols + ["acc"]].dropna()
        if len(sub) < 2:
            continue

        y = sub["acc"].astype(int)
        if y.nunique() < 2:
            continue

        X = sub[lang_indicator_cols].to_numpy(dtype=float)
        try:
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000)
            clf.fit(X, y)
        except Exception:
            continue

        intercept = float(clf.intercept_[0])
        coef_vec = clf.coef_[0]

        for j, col in enumerate(lang_indicator_cols):
            coef = float(coef_vec[j])
            readable = INDICATOR_RENAME.get(col)
            if readable is None:
                continue
            # ±1 SD on j, all other predictors at 0 (their standardized means)
            p_pos = float(sigma_stable(intercept + coef))
            p_neg = float(sigma_stable(intercept - coef))
            rows.append(
                {
                    "indicator": readable,
                    "language": lang,
                    "delta_accuracy": p_pos - p_neg,
                }
            )
    return pd.DataFrame(rows, columns=_DELTA_COLS)


def build_heatmap_matrix(df_delta: pd.DataFrame) -> pd.DataFrame:
    if df_delta.empty:
        return pd.DataFrame(index=INDICATOR_ORDER, columns=LANGS, dtype=float)
    mat = df_delta.pivot(index="indicator", columns="language", values="delta_accuracy")
    return mat.reindex(index=INDICATOR_ORDER, columns=LANGS)


def main() -> None:
    sns.set_theme(
        style="white",
        font_scale=0.85,
        rc={"font.family": "sans-serif", "font.sans-serif": ["Figtree", "DejaVu Sans"]},
    )

    model_mats = {}
    all_values = []
    for model in MODELS:
        df_model = load_model_df(model)
        df_delta = compute_model_perlang_delta(df_model)
        mat = build_heatmap_matrix(df_delta)
        model_mats[model] = mat
        vals = mat.to_numpy().ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            all_values.append(vals)

    if not all_values:
        print("No plottable delta values were computed.")
        return

    all_values = np.concatenate(all_values)
    vmin = -1.0
    vmax = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    cbar_ax = fig.add_axes([0.92, 0.20, 0.012, 0.62])

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        mat = model_mats[model]
        sns.heatmap(
            mat,
            ax=ax,
            cmap="RdBu",
            center=0.0,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="#f0f0f0",
            cbar=(idx == len(MODELS) - 1),
            cbar_ax=cbar_ax if idx == len(MODELS) - 1 else None,
            cbar_kws={"label": r"$\Delta$ Accuracy"},
        )
        ax.set_title(MODEL_TITLES.get(model, model), fontsize=10, pad=8)
        ax.set_xlabel("Language")
        if idx == 0:
            ax.set_ylabel("")
            readable_to_indicator = {v: k for k, v in INDICATOR_RENAME.items()}
            for tick in ax.get_yticklabels():
                label = tick.get_text()
                key = readable_to_indicator.get(label)
                tick.set_color(INDICATOR_STYLE.get(key, {}).get("color", "black"))

            symbol_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            for i, readable in enumerate(INDICATOR_ORDER):
                key = readable_to_indicator.get(readable)
                prefix = INDICATOR_STYLE.get(key, {}).get("prefix", "")
                color = INDICATOR_STYLE.get(key, {}).get("color", "black")
                if not prefix:
                    continue
                ax.text(
                    -0.07,
                    i + 0.5,
                    prefix,
                    va="center",
                    ha="left",
                    fontsize=10,
                    color=color,
                    transform=symbol_transform,
                    fontfamily="DejaVu Sans",
                    clip_on=False,
                )
            ax.tick_params(axis="y", pad=10)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    fig.tight_layout(rect=[0.02, 0.03, 0.91, 0.98], w_pad=0.6)
    out_name = f"{DATASET}_perlang_multi.png"
    fig.savefig(out_name, dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    main()
