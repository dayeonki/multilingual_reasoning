import json
import torch
import argparse
import os
from typing import Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# (1) Validity
label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

def nli_relation(premise: str, hypothesis: str):
    inputs = nli_tokenizer.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True, max_length=2048).to(device)
    logits = nli_model(**inputs).logits
    pred = logits.argmax(dim=-1).item()
    return label_map[pred]


def step_validity(step_text: str, dependency_texts: list):
    if not dependency_texts:
        return {"entail": 1.0, "neutral": 0.0, "contradiction": 0.0, "validity": 1.0}

    entail = neutral = contra = 0

    for dep in dependency_texts:
        rel = nli_relation(dep, step_text)
        if rel == "entailment":
            entail += 1
        elif rel == "neutral":
            neutral += 1
        else:
            contra += 1

    total = max(1, len(dependency_texts))
    entail_r = entail / total
    neutral_r = neutral / total
    contra_r = contra / total

    # validity rule: if contradiction exists → 0, otherwise = entailment_rate
    validity_score = 0.0 if contra > 0 else entail_r

    return {
        "entail": entail_r,
        "neutral": neutral_r,
        "contradiction": contra_r,
        "validity": validity_score
    }


# (2) Utility
def find_final_step_id(classified_steps: Dict[str, Any]) -> str | None:
    try:
        for sid, meta in classified_steps.items():
            if "final_answer_emission" in meta.get("function_tags", []):
                return sid
        return None
    except:
        return None


def collect_ancestor_steps(classified_steps: Dict[str, Any], final_id: str):
    ancestors = set()
    stack = [final_id]
    while stack:
        cur = stack.pop()
        for parent in classified_steps[str(cur)]["depends_on"]:
            if parent not in ancestors:
                ancestors.add(parent)
                stack.append(parent)
    return ancestors


def compute_step_utilities(
        classified_steps: Dict[str, Any]
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], float]:
    """
    direct_utility[i] = 1 if i directly contributes to final
    indirect_utility[i] = 1 if i is a dependency of a direct ancestor
    """
    try:
        classified_steps = {str(k): v for k, v in classified_steps.items()}
        final_id = find_final_step_id(classified_steps)
        print("Final ID: ", final_id)
        
        if final_id: # If there is final answer emission
            ancestors = collect_ancestor_steps(classified_steps, final_id)

            step_ids = list(classified_steps.keys())
            num_steps = len(step_ids)

            direct_utility = {sid: 0 for sid in step_ids}
            for sid in ancestors:
                direct_utility[sid] = 1

            direct_utility[final_id] = 1
            indirect_utility = {sid: 0 for sid in step_ids}
            for sid in step_ids:
                if direct_utility[sid] == 1:
                    deps = classified_steps[sid].get("depends_on", [])
                    for d in deps:
                        indirect_utility[str(d)] = 1

            total_direct_utility = {
                sid: 1 if (direct_utility[sid] == 1) else 0
                for sid in step_ids
            }
            total_indirect_utility = {
                sid: 1 if (indirect_utility[sid] == 1) else 0
                for sid in step_ids
            }

            num_useful_steps_direct = sum(total_direct_utility.values())
            num_useful_steps_indirect = sum(total_indirect_utility.values())
            direct_utility_score = num_useful_steps_direct / num_steps if num_steps > 0 else 0.0
            indirect_utility_score = num_useful_steps_indirect / num_steps if num_steps > 0 else 0.0
        else:
            direct_utility, indirect_utility, direct_utility_score, indirect_utility_score = 0.0, 0.0, 0.0, 0.0
    except:
        direct_utility, indirect_utility, direct_utility_score, indirect_utility_score = 0.0, 0.0, 0.0, 0.0

    return direct_utility, indirect_utility, direct_utility_score, indirect_utility_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    args = parser.parse_args()

    # Load models/helpers
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NLI model (for validity)
        # https://huggingface.co/potsawee/deberta-v3-large-mnli
    nli_name = "potsawee/deberta-v3-large-mnli"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device)

    languages = ["en", "bn", "de", "es", "fr", "ja", "ko", "ru", "sw", "te", "th", "zh"]
    
    for language in languages:
        try:
            input_file = f"{args.input_path}/{language}.jsonl"
            output_file = f"{args.output_path}/{language}.jsonl"

            print(f"Processing language: {language}")
            print(f"  Input:  {input_file}")
            print(f"  Output: {output_file}")

            entries = []
            with open(input_file, "r") as f:
                for line in f:
                    entries.append(json.loads(line))

            results: list[dict] = []        # all results to write back
            processed_ids: set[str] = set() # ids we already processed

            if os.path.exists(output_file):
                with open(output_file, "r") as f_out:
                    for line in f_out:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        results.append(obj)
                        eid = obj.get("id")
                        if eid is not None:
                            processed_ids.add(eid)

            print(f"  Already processed {len(processed_ids)} ids.")

            results = []  # reset per-language

            for entry in entries:
                raw_cs = entry["classified_steps"]

                if isinstance(raw_cs, str):
                    try:
                        raw_cs = raw_cs.strip()
                        if raw_cs.startswith("```"):
                            lines = raw_cs.splitlines()
                            inner = "\n".join(lines[1:-1])  # drop first and last line
                        else:
                            inner = raw_cs
                        classified_steps = json.loads(inner)
                    except:
                        classified_steps = raw_cs
                else:
                    classified_steps = raw_cs

                reasoning_steps = entry["reasoning_steps"]
                step_texts = {str(i): reasoning_steps[i] for i in range(len(reasoning_steps))}

                # ========== A. Validity ==========
                all_validities = {}
                try:
                    for sid, meta in classified_steps.items():
                        try:
                            deps = meta.get("depends_on", [])
                            dependency_texts = [step_texts[d] for d in deps] if deps else []
                            v = step_validity(step_texts[sid], dependency_texts)
                            all_validities[sid] = v
                        except:
                            pass
                except:
                    pass

                avg_validity = (
                    sum(v["validity"] for v in all_validities.values()) / len(all_validities)
                    if all_validities
                    else 0.0
                )

                # ========== B. Utility ==========
                direct_u, indirect_u, direct_utility, indirect_utility = compute_step_utilities(classified_steps)

                result_entry = {
                    "id": entry.get("id"),
                    "query": entry.get("problem") or entry.get("question"),
                    "validity": all_validities,
                    "validity_score": avg_validity,
                    "direct_utility": direct_u,
                    "indirect_utility": indirect_u,
                    "direct_utility_score": direct_utility,
                    "indirect_utility_score": indirect_utility
                }

                results.append(result_entry)
                print(result_entry)

            with open(output_file, "w") as out:
                for r in results:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
        except:
            pass
