import numpy as np
import json
import re
import os
import pandas as pd
from scipy.stats import mannwhitneyu

# -------------------------------
dataset = "mgsm_revised"
model = "distill_qwen7b"
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
]
TAG2ID = {t: i for i, t in enumerate(FUNCTION_TAGS)}


def seq_from_classified_steps(item):
    raw = item.get("classified_steps", None)
    if raw is None:
        return []

    s = str(raw).strip()
    s = re.sub(r"^```(?:json|JSON)?", "", s)
    s = re.sub(r"```$", "", s).strip()

    match = re.search(r"\{[\s\S]*\}", s)
    if not match:
        return []   # nothing usable
    s = match.group(0)

    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        s_fixed = re.sub(r",\s*}", "}", s)
        s_fixed = re.sub(r",\s*]", "]", s_fixed)
        try:
            obj = json.loads(s_fixed)
        except:
            return []

    seq = []
    steps = item.get("reasoning_steps", [])
    for i in range(len(steps)):
        entry = obj.get(str(i), {})
        tags = entry.get("function_tags", [])
        tag = tags[0] if tags else ""
        seq.append(TAG2ID.get(tag, ""))

    return seq


# For Smith-Waterman
def smith_waterman(a, b, match=2, mismatch=-1, gap=-1):
    n, m = len(a), len(b)
    H = np.zeros((n+1, m+1), dtype=int)
    best = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            score = match if a[i-1] == b[j-1] else mismatch
            H[i,j] = max(
                0,
                H[i-1,j-1] + score,
                H[i-1,j] + gap,
                H[i,j-1] + gap,
            )
            best = max(best, H[i,j])
    return best


def behavioral_alignment_ratio(item_en, item_other):
    seq_en = seq_from_classified_steps(item_en)
    seq_ot = seq_from_classified_steps(item_other)

    if not seq_en or not seq_ot:
        return None

    score = smith_waterman(seq_en, seq_ot)
    max_score = 2 * min(len(seq_en), len(seq_ot))
    ratio = score / max_score
    return ratio # in [0, 1]
    # higher => greater alignment to English sequence (seq_en)


def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


if __name__ == "__main__":
    languages = ["en", "bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
    results_rows = []

    for lang in languages:
        print(f"======================= Processing {lang} =======================")
        # ---- Quadrant collectors ----
        Q1 = []  # en=1, non-en=1
        Q2 = []  # en=1, non-en=0
        Q3 = []  # en=0, non-en=1
        Q4 = []  # en=0, non-en=0
        with open(f"classification/{dataset}/{model}/{lang}.jsonl", "r", encoding="utf-8") as f, \
            open(f"classification/{dataset}/{model}/en.jsonl", "r", encoding="utf-8") as f_en:

            for line, en_line in zip(f, f_en):
                if not line.strip():
                    continue

                tgt_item = json.loads(line)
                en_item = json.loads(en_line)

                tgt_acc = tgt_item.get("acc", None)
                en_acc = en_item.get("acc", None)

                if tgt_acc is None or en_acc is None:
                    continue

                ratio = behavioral_alignment_ratio(en_item, tgt_item)

                if ratio is None:
                    continue

                # Bucket into quadrants
                if en_acc == 1 and tgt_acc == 1:
                    Q1.append(ratio)
                elif en_acc == 1 and tgt_acc == 0:
                    Q2.append(ratio)
                elif en_acc == 0 and tgt_acc == 1:
                    Q3.append(ratio)
                elif en_acc == 0 and tgt_acc == 0:
                    Q4.append(ratio)

        def avg(x):
            return np.mean(x) if len(x) > 0 else None

        print(f"Avg. alignment ratio (Q1):", avg(Q1))
        print(f"Avg. alignment ratio (Q2):", avg(Q2))
        print(f"Avg. alignment ratio (Q3):", avg(Q3))
        print(f"Avg. alignment ratio (Q4):", avg(Q4))

        q1_mean = avg(Q1); q2_mean = avg(Q2)
        q3_mean = avg(Q3); q4_mean = avg(Q4)

        print("[Count] Q1:", len(Q1), "Q2:", len(Q2), "Q3:", len(Q3), "Q4:", len(Q4))

        if len(Q1) > 1 and len(Q2) > 1:
            U, p = mannwhitneyu(Q1, Q2, alternative="two-sided")
            star = p_to_stars(p)
            print(f"[Q1 vs. Q2] U={U:.2f}, p={p:.4e} {star}")
        else:
            U = None; p = None; star = ""
            print("⚠️ Not enough Q1/Q2 samples.")

        results_rows.append({
            "language": lang,
            "q1_mean": q1_mean,
            "q2_mean": q2_mean,
            "q3_mean": q3_mean,
            "q4_mean": q4_mean,
            "count_q1": len(Q1),
            "count_q2": len(Q2),
            "count_q3": len(Q3),
            "count_q4": len(Q4),
            "U_q1_q2": U,
            "p_q1_q2": p,
            "p_star_q1_q2": star
        })
        print("\n", "="*60, "\n")

    out_csv = f"alignment/{dataset}_{model}.csv"
    os.makedirs("alignment", exist_ok=True)
    df_out = pd.DataFrame(results_rows)
    df_out.to_csv(out_csv, index=False)
