import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# -------------------------------
dataset = "mgsm_revised"
model = "distill_qwen7b"
mode = "q2"
# -------------------------------

FUNCTION_TAGS = [
    "problem_setup",
    "plan_generation",
    "fact_retrieval",
    "active_computation",
    "result_consolidation",
    "uncertainty_management",
    "final_answer_emission",
    "self_checking",
    "unknown"
]
colors = sns.color_palette("Set3", n_colors=len(FUNCTION_TAGS))

def pretty(name):
    return " ".join(w.capitalize() for w in name.split("_"))

pretty_tags = [pretty(t) for t in FUNCTION_TAGS]

def parse_classified_steps(raw):
    if raw is None:
        return {}

    if isinstance(raw, dict):
        return raw

    if isinstance(raw, list):
        return {str(i): step for i, step in enumerate(raw)}

    s = str(raw).strip()
    s = re.sub(r"^```(?:json|JSON)?", "", s)
    s = re.sub(r"```$", "", s).strip()

    match = re.search(r"\{[\s\S]*\}", s)
    if not match:
        return {}
    s = match.group(0)

    try:
        return json.loads(s)
    except:
        s2 = re.sub(r",\s*}", "}", s)
        s2 = re.sub(r",\s*]", "]", s2)
        try:
            return json.loads(s2)
        except:
            return {}


def extract_tag_frequencies(item):
    cls = parse_classified_steps(item.get("classified_steps", None))
    steps = item.get("reasoning_steps", [])
    if not steps:
        return None

    counts = {tag: 0 for tag in FUNCTION_TAGS}

    # Count occurrences
    for i in range(len(steps)):
        entry = cls.get(str(i), {})
        tags = entry.get("function_tags", [])
        if tags:
            tag = tags[0]
            if tag in counts:
                counts[tag] += 1

    total = len(steps)

    # Normalize known tags
    for tag in FUNCTION_TAGS:
        if tag != "unknown":
            counts[tag] /= total

    # Fill missing probability mass into "unknown"
    known_sum = sum(counts[tag] for tag in FUNCTION_TAGS if tag != "unknown")
    counts["unknown"] = max(0.0, 1.0 - known_sum)

    counts["acc"] = item.get("acc", None)
    counts["id"] = item.get("id", None)
    return counts


def load_df(lang):
    rows = []
    with open(f"classification/{dataset}/{model}/{lang}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            row = extract_tag_frequencies(item)
            if row:
                rows.append(row)
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["acc"])
    return df


if __name__ == "__main__":
    languages = ["en", "bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
    mean_table = {}

    df_en = load_df("en")
    df_en = df_en[df_en["acc"].isin([0,1])]
    en_acc_map = dict(zip(df_en["id"], df_en["acc"]))

    for lang in languages:
        df = load_df(lang)
        df = df[df["acc"].isin([0,1])]

        # Q2 mode
        if mode == "q2" and lang != "en":
            df = df.merge(
                df_en.set_index("id")["acc"].rename("acc_en"),
                left_on="id", right_index=True, how="inner"
            )
            df = df[(df["acc_en"] == 1) & (df["acc"] == 0)]

            if df.empty:
                print(f"[WARNING] No Q2 samples for", lang)
                mean_table[lang] = pd.Series({tag: 0.0 for tag in FUNCTION_TAGS})
                continue

        # Q2 mode: EN should include only correct (acc = 1)
        if mode == "q2" and lang == "en":
            df = df[df["acc"] == 1]

        mean_table[lang] = df[FUNCTION_TAGS].mean().fillna(0.0)

    mean_df = pd.DataFrame(mean_table).T  # lang × tag matrix
    mean_df.index.name = "language"

    plt.figure(figsize=(12, 5))
    bottom = np.zeros(len(mean_df))

    for tag, color in zip(FUNCTION_TAGS, colors):
        plt.bar(
            mean_df.index,
            mean_df[tag],
            bottom=bottom,
            label=pretty(tag),
            width=0.75,
            color=color
        )
        bottom += mean_df[tag].values

    plt.xticks(rotation=45)
    plt.ylabel("Mean Ratio of Steps")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),    # move below plot
        ncol=5,                         # 4 columns looks nice for 8 tags
        frameon=True,
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(f"distribution/{dataset}_{model}_{mode}.png", dpi=300, bbox_inches="tight")
