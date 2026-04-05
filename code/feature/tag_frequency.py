import json
import re
import pandas as pd

# -------------------------------
data = "mgsm_revised"
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

ORDERED_TAGS = [
    "self_checking",
    "active_computation",
    "problem_setup",
    "plan_generation",
    "final_answer_emission",
    "fact_retrieval",
    "result_consolidation",
    "uncertainty_management",
]


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
    for i in range(len(steps)):
        entry = cls.get(str(i), {})
        tags = entry.get("function_tags", [])
        if tags:
            tag = tags[0]
            if tag in counts:
                counts[tag] += 1

    total = len(steps)
    for tag in counts:
        counts[tag] /= total

    counts["acc"] = item.get("acc", None)
    counts["id"] = item.get("id", None)
    return counts


if __name__ == "__main__":
    languages = ["en", "bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
    rows_for_csv = []

    for lang in languages:
        input_jsonl = f"classification/{data}/{model}/{lang}.jsonl"
        rows = []

        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                row = extract_tag_frequencies(item)
                if row:
                    rows.append(row)

        df = pd.DataFrame(rows)
        df = df[df["acc"].isin([0,1])]

        corr = df[FUNCTION_TAGS + ["acc"]].corr(numeric_only=True)["acc"]
        corr = corr.drop("acc").sort_values(ascending=False)

        row_dict = {"language": lang}
        for tag in ORDERED_TAGS:
            row_dict[tag] = round(corr[tag], 3)

        rows_for_csv.append(row_dict)

    output_csv = f"behavior/{data}_{model}.csv"
    df_out = pd.DataFrame(rows_for_csv)
    df_out = df_out[["language"] + ORDERED_TAGS] 
    df_out.to_csv(output_csv, index=False)

    print(df_out)
