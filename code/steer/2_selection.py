import os
import json
import re
import glob
import random
import csv
import argparse
import time
from typing import Dict, Any, Tuple, List, Optional
from lingua import Language, LanguageDetectorBuilder
from math_verify import parse, verify
from collections import Counter

import numpy as np
from prompt import classification_prompt, classification_prompt_en
from openai import OpenAI
from openai import BadRequestError, APIError, APIConnectionError, RateLimitError
from sentence_transformers import SentenceTransformer


# Token pricing ($ per token)
PRICE_INPUT = 2.50 / 1_000_000           # non-cached prompt tokens
PRICE_INPUT_CACHED = 1.25 / 1_000_000    # cached prompt tokens
PRICE_OUTPUT = 10.00 / 1_000_000         # output tokens

_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost_usd": 0.0}


# Load OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
_openai_client = OpenAI()


# Load LaBSE model for cosine similarity
LABSE = SentenceTransformer("sentence-transformers/LaBSE")

def cosine_sim(a, b):
    if a is None or b is None:
        return None
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _get_candidates(data: dict) -> list:
    return data.get("candidates")


# Segment steps
def segment_steps():
    SRC_DIR = f"res/{DATASET}/{DATASET_NAME}_{MODEL_NAME}_{TMP_SUFFIX}/"
    OUT_DIR = f"segmentation/{DATASET}/{MODEL_NAME}_{TMP_SUFFIX}/"

    if os.path.exists(OUT_DIR) and glob.glob(os.path.join(OUT_DIR, "*.jsonl")):
        print(f"[Skip] {OUT_DIR} already has JSONL files")
        return

    with open("separators.json", "r", encoding="utf-8") as f:
        SEPARATORS = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True) 

    for file_path in glob.glob(os.path.join(SRC_DIR, "*.jsonl")):
        out_path = os.path.join(OUT_DIR, os.path.basename(file_path))

        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    candidates = _get_candidates(data)
                    agnostic_seps = SEPARATORS.get("agnostic", ["\n\n"])
                    pattern = re.compile("|".join(re.escape(s) for s in agnostic_seps))

                    reasoning_steps_list = []
                    num_steps_list = []
                    for cand in candidates:
                        trace = cand.get("reasoning_trace", "").replace("<think>\n", "").replace("\n</think>", "").replace("<think>", "").replace("</think>", "")
                        steps = [
                            s.strip()
                            for s in re.split(pattern, trace)
                            if s.strip()
                        ]
                        reasoning_steps_list.append(steps)
                        num_steps_list.append(len(steps))

                    data["reasoning_steps"] = reasoning_steps_list
                    data["num_steps"] = num_steps_list

                    json.dump(data, f_out, ensure_ascii=False)
                    f_out.write("\n")


LANG_CODE_TO_NAME = {
    "bn": "Bengali", "de": "German", "en": "English", "es": "Spanish",
    "fr": "French", "ja": "Japanese", "ko": "Korean", "ru": "Russian",
    "sw": "Swahili", "te": "Telugu", "th": "Thai", "zh": "Chinese",
    "ar": "Arabic", "pt": "Portuguese",
}

FUNCTION_TAGS = [
    "problem_setup", "plan_generation", "fact_retrieval", "active_computation",
    "result_consolidation", "uncertainty_management", "final_answer_emission", "self_checking",
]
TAG2ID = {t: i for i, t in enumerate(FUNCTION_TAGS)}


def _seq_from_classified_steps(item: dict) -> list:
    raw = item.get("classified_steps", None)
    if raw is None:
        return []
    s = str(raw).strip()
    s = re.sub(r"^```(?:json|JSON)?", "", s)
    s = re.sub(r"```$", "", s).strip()
    match = re.search(r"\{[\s\S]*\}", s)
    if not match:
        return []
    s = match.group(0)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        s_fixed = re.sub(r",\s*}", "}", s)
        s_fixed = re.sub(r",\s*]", "]", s_fixed)
        try:
            obj = json.loads(s_fixed)
        except Exception:
            return []
    seq = []
    steps = item.get("reasoning_steps", [])
    for i in range(len(steps)):
        entry = obj.get(str(i), {})
        tags = entry.get("function_tags", [])
        tag = tags[0] if tags else ""
        seq.append(TAG2ID.get(tag, ""))
    return seq


def _smith_waterman(a: list, b: list, match: int = 2, mismatch: int = -1, gap: int = -1) -> int:
    n, m = len(a), len(b)
    H = np.zeros((n + 1, m + 1), dtype=int)
    best = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match if a[i - 1] == b[j - 1] else mismatch
            H[i, j] = max(0, H[i - 1, j - 1] + score, H[i - 1, j] + gap, H[i, j - 1] + gap)
            best = max(best, H[i, j])
    return best


def _behavioral_alignment_ratio(item_en: dict, item_other: dict) -> Optional[float]:
    seq_en = _seq_from_classified_steps(item_en)
    seq_ot = _seq_from_classified_steps(item_other)
    if not seq_en or not seq_ot:
        return None
    score = _smith_waterman(seq_en, seq_ot)
    max_score = 2 * min(len(seq_en), len(seq_ot))
    return score / max_score if max_score > 0 else None


# Semantic similarity
def _normalize_trace(txt):
    if not txt:
        return ""
    if isinstance(txt, list):
        return " ".join(str(s) for s in txt)
    return str(txt)


def compute_cosine_similarities(item_en, item_other):
    if item_en is None or item_other is None:
        return None

    txt_en = item_en.get("reasoning_trace")
    if not txt_en:
        candidates_en = _get_candidates(item_en)
        if candidates_en:
            txt_en = candidates_en[0].get("reasoning_trace", "")
    txt_en = _normalize_trace(txt_en)
    if not txt_en:
        return None

    emb_en = LABSE.encode(txt_en, convert_to_numpy=True, normalize_embeddings=False)
    similarities = []
    for cand in _get_candidates(item_other):
        txt_ot = _normalize_trace(cand.get("reasoning_trace", ""))
        if not txt_ot:
            similarities.append(None)
            continue
        emb_ot = LABSE.encode(txt_ot, convert_to_numpy=True, normalize_embeddings=False)
        similarities.append(cosine_sim(emb_en, emb_ot))
    
    return similarities


def semantic_similarity():
    SRC_DIR = f"segmentation/{DATASET}/{MODEL_NAME}_{TMP_SUFFIX}/"

    en_path = os.path.join(SRC_DIR, "en.jsonl")
    if not os.path.exists(en_path):
        print(f"[Skip] {en_path} not found")
        return

    en_by_id = {}
    with open(en_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            en_by_id[item.get("id")] = item

    for file_path in glob.glob(os.path.join(SRC_DIR, "*.jsonl")):
        if os.path.basename(file_path) == "en.jsonl":
            continue

        lines_out = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                item_en = en_by_id.get(data.get("id"))
                sims = compute_cosine_similarities(item_en, data)
                data["semantic_similarities"] = sims if sims is not None else []
                lines_out.append(json.dumps(data, ensure_ascii=False) + "\n")

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines_out)


def structural_similarity():
    SRC_DIR = f"segmentation/{DATASET}/{MODEL_NAME}_{TMP_SUFFIX}/"
    en_path = os.path.join(SRC_DIR, "en.jsonl")
    if not os.path.exists(en_path):
        print(f"[Skip] {en_path} not found")
        return

    en_by_id = {}
    with open(en_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            en_by_id[item.get("id")] = item

    for file_path in glob.glob(os.path.join(SRC_DIR, "*.jsonl")):
        if os.path.basename(file_path) == "en.jsonl":
            continue

        lines_out = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                reasoning_steps_list = data.get("reasoning_steps", [])
                classified_list = data.get("classified_steps", [])
                if not reasoning_steps_list or not classified_list:
                    data["structural_similarities"] = []
                    lines_out.append(json.dumps(data, ensure_ascii=False) + "\n")
                    continue

                en_item = en_by_id.get(data.get("id"))
                if en_item is None:
                    data["structural_similarities"] = [None] * len(reasoning_steps_list)
                    lines_out.append(json.dumps(data, ensure_ascii=False) + "\n")
                    continue

                en_steps_list = en_item.get("reasoning_steps", [])
                en_classified_list = en_item.get("classified_steps", [])
                en_ref = {
                    "reasoning_steps": en_steps_list[0] if en_steps_list else [],
                    "classified_steps": en_classified_list[0] if en_classified_list else None,
                }

                structural_list = []
                for cand_idx, cand_steps in enumerate(reasoning_steps_list):
                    cand_classified = (
                        classified_list[cand_idx]
                        if cand_idx < len(classified_list)
                        else None
                    )
                    other_ref = {
                        "reasoning_steps": cand_steps,
                        "classified_steps": cand_classified,
                    }
                    ratio = _behavioral_alignment_ratio(en_ref, other_ref)
                    structural_list.append(ratio)

                data["structural_similarities"] = structural_list
                lines_out.append(json.dumps(data, ensure_ascii=False) + "\n")

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines_out)


# Step-level quality measures
def _find_final_step_id(classified_steps: Dict[str, Any]) -> Optional[str]:
    try:
        for sid, meta in classified_steps.items():
            if "final_answer_emission" in meta.get("function_tags", []):
                return sid
    except Exception:
        pass
    return None


def _collect_ancestor_steps(classified_steps: Dict[str, Any], final_id: str) -> set:
    ancestors = set()
    stack = [final_id]
    while stack:
        cur = stack.pop()
        for parent in classified_steps.get(str(cur), {}).get("depends_on", []):
            if str(parent) not in ancestors:
                ancestors.add(str(parent))
                stack.append(str(parent))
    return ancestors


def compute_step_utilities(
    classified_steps: Dict[str, Any],
) -> Tuple[Dict[str, int], Dict[str, int], float, float]:
    try:
        classified_steps = {str(k): v for k, v in classified_steps.items()}
        final_id = _find_final_step_id(classified_steps)
        if not final_id:
            return {}, {}, 0.0, 0.0
        ancestors = _collect_ancestor_steps(classified_steps, final_id)
        step_ids = list(classified_steps.keys())
        num_steps = len(step_ids)
        direct_utility = {sid: 1 if sid in ancestors or sid == final_id else 0 for sid in step_ids}
        indirect_utility = {sid: 0 for sid in step_ids}
        for sid in step_ids:
            if direct_utility.get(sid):
                for d in classified_steps[sid].get("depends_on", []):
                    indirect_utility[str(d)] = 1
        d_score = sum(direct_utility.values()) / num_steps if num_steps else 0.0
        i_score = sum(indirect_utility.values()) / num_steps if num_steps else 0.0
        return direct_utility, indirect_utility, d_score, i_score
    except Exception:
        return {}, {}, 0.0, 0.0


def _parse_classified_steps_behavior(raw) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {str(i): s for i, s in enumerate(raw)}
    s = str(raw).strip()
    s = re.sub(r"^```(?:json|JSON)?", "", s)
    s = re.sub(r"```$", "", s).strip()
    match = re.search(r"\{[\s\S]*\}", s)
    if not match:
        return {}
    s = match.group(0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s2 = re.sub(r",\s*}", "}", s)
        s2 = re.sub(r",\s*]", "]", s2)
        try:
            return json.loads(s2)
        except Exception:
            return {}


def _extract_tag_counts_and_frequencies(
    reasoning_steps: List[str],
    classified_raw: Any,
) -> Tuple[Dict[str, int], Dict[str, float]]:
    counts = {tag: 0 for tag in FUNCTION_TAGS}
    cls = _parse_classified_steps_behavior(classified_raw)
    for i in range(len(reasoning_steps)):
        entry = cls.get(str(i), {})
        tags = entry.get("function_tags", [])
        if tags and tags[0] in counts:
            counts[tags[0]] += 1
    total = len(reasoning_steps)
    frequencies = {tag: counts[tag] / total if total > 0 else 0.0 for tag in FUNCTION_TAGS}
    return counts, frequencies


def _parse_classified_steps(raw_cs) -> Optional[Dict]:
    if raw_cs is None:
        return None
    if isinstance(raw_cs, dict):
        return raw_cs
    if isinstance(raw_cs, str):
        try:
            s = raw_cs.strip()
            inner = "\n".join(s.splitlines()[1:-1]) if s.startswith("```") else s
            return json.loads(inner)
        except Exception:
            return None
    return None


def process_one_candidate(
    reasoning_steps: List[str],
    classified_steps: Optional[Dict],
) -> Dict[str, Any]:
    if classified_steps and isinstance(classified_steps, dict):
        direct_u, indirect_u, d_score, i_score = compute_step_utilities(classified_steps)
    else:
        direct_u = {str(i): 0 for i in range(len(reasoning_steps))}
        indirect_u = {str(i): 0 for i in range(len(reasoning_steps))}
        d_score = i_score = 0.0
    return {
        "direct_utility": direct_u,
        "indirect_utility": indirect_u,
        "direct_utility_score": d_score,
        "indirect_utility_score": i_score,
    }


def _sanitize_prompt(text: str, max_chars: int = 100_000) -> str:
    if not text:
        return ""
    sanitized = "".join(c for c in text if c == "\n" or c == "\t" or ord(c) >= 32)
    return sanitized[:max_chars] if len(sanitized) > max_chars else sanitized


def _prompt_gpt4(prompt: str, model: str = "gpt-4o", max_retries: int = 3) -> str:
    prompt = _sanitize_prompt(prompt)
    last_err = None
    for attempt in range(max_retries):
        try:
            print(f"\n{'='*60} Prompt {'='*60}\n{prompt[:500]}...\n" if len(prompt) > 500 else f"\n{'='*60} Prompt {'='*60}\n{prompt[:100]}...\n")
            response = _openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()
            print(f"\n{'='*60} Response {'='*60}\n{content[:500]}...\n" if len(content) > 500 else f"\n{'='*60} Response {'='*60}\n{content[:100]}\n")

            # Track token usage and cost
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            cost = prompt_tokens * PRICE_INPUT + completion_tokens * PRICE_OUTPUT
            _token_usage["prompt_tokens"] += prompt_tokens
            _token_usage["completion_tokens"] += completion_tokens
            _token_usage["total_cost_usd"] += cost
            print(f"[Tokens] input={prompt_tokens}, output={completion_tokens}, cost=${cost:.6f}")

            return content
        except (BadRequestError, APIError, APIConnectionError, RateLimitError) as e:
            last_err = e
            wait = 2 ** attempt  # 1, 2, 4 seconds
            print(f"[API Error] {type(e).__name__}: {e}. Retry {attempt + 1}/{max_retries} in {wait}s...")
            time.sleep(wait)
    print(f"[Skip] Failed after {max_retries} retries. Last error: {last_err}. Returning empty.")
    return ""


def _classify_one_candidate_en(
    problem: str,
    reasoning_steps: List[str],
    language: str = "English",
) -> str:
    if not reasoning_steps:
        return ""
    all_steps_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(reasoning_steps))
    prompt = (
        classification_prompt_en
        .replace("{{language}}", language)
        .replace("{{problem}}", problem)
        .replace("{{reasoning_steps}}", all_steps_text)
    )
    return _prompt_gpt4(prompt)


def _classify_one_candidate_nonen(
    problem: str,
    en_problem: str,
    reasoning_steps: List[str],
    language: str,
) -> str:
    if not reasoning_steps:
        return ""
    all_steps_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(reasoning_steps))
    prompt = (
        classification_prompt
        .replace("{{problem}}", problem)
        .replace("{{language}}", language)
        .replace("{{en_problem}}", en_problem)
        .replace("{{reasoning_steps}}", all_steps_text)
    )
    return _prompt_gpt4(prompt)


def classify_steps():
    global _token_usage
    _token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_cost_usd": 0.0}

    SRC_DIR = f"segmentation/{DATASET}/{MODEL_NAME}_{TMP_SUFFIX}/"
    en_path = os.path.join(SRC_DIR, "en.jsonl")
    en_by_id = {}
    if os.path.exists(en_path):
        with open(en_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                en_by_id[item.get("id")] = item

    for file_path in glob.glob(os.path.join(SRC_DIR, "*.jsonl")):
        lang_code = os.path.basename(file_path).replace(".jsonl", "")
        is_en = lang_code == "en"
        with open(file_path, "r", encoding="utf-8") as f:
            input_lines = [line for line in f if line.strip()]
        with open(file_path, "w", encoding="utf-8") as f_out:
            for line in input_lines:
                if not line.strip():
                    continue
                data = json.loads(line)
                reasoning_steps_list = data.get("reasoning_steps", [])
                if not reasoning_steps_list:
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    continue

                existing_classified = data.get("classified_steps", [])
                if existing_classified and len(existing_classified) == len(reasoning_steps_list):
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    continue

                problem = data.get("question") or data.get("problem", "")
                en_item = en_by_id.get(data.get("id")) if not is_en else None
                en_problem = (en_item.get("question") or en_item.get("problem", "")) if en_item else ""

                start_idx = len(existing_classified)
                classified_list = list(existing_classified)
                cost_per_candidate = list(data.get("cost_per_candidate", []))
                cost_input_per_candidate = list(data.get("cost_input_per_candidate", []))
                cost_output_per_candidate = list(data.get("cost_output_per_candidate", []))

                fname = os.path.basename(file_path)
                item_id = data.get("id")
                for cand_idx, cand_steps in enumerate(reasoning_steps_list[start_idx:], start=start_idx):
                    pt_before = _token_usage["prompt_tokens"]
                    ct_before = _token_usage["completion_tokens"]
                    cost_before = _token_usage["total_cost_usd"]
                    if is_en:
                        gen = _classify_one_candidate_en(problem, cand_steps, "English")
                    else:
                        language = LANG_CODE_TO_NAME.get(lang_code, lang_code)
                        gen = _classify_one_candidate_nonen(
                            problem, en_problem, cand_steps, language
                        )
                    pt_delta = _token_usage["prompt_tokens"] - pt_before
                    ct_delta = _token_usage["completion_tokens"] - ct_before
                    cost_input = pt_delta * PRICE_INPUT
                    cost_output = ct_delta * PRICE_OUTPUT
                    cost_total = _token_usage["total_cost_usd"] - cost_before
                    cost_per_candidate.append(cost_total)
                    cost_input_per_candidate.append(cost_input)
                    cost_output_per_candidate.append(cost_output)
                    classified_list.append(gen)
                    print(f"{fname} id={item_id} candidate={cand_idx}: cost=${cost_total:.6f} input=${cost_input:.6f} output=${cost_output:.6f} [saved]")

                data["classified_steps"] = classified_list
                data["cost_per_candidate"] = cost_per_candidate
                data["cost_input_per_candidate"] = cost_input_per_candidate
                data["cost_output_per_candidate"] = cost_output_per_candidate
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"\n[Total usage] prompt_tokens={_token_usage['prompt_tokens']}, "
          f"completion_tokens={_token_usage['completion_tokens']}, "
          f"cost=${_token_usage['total_cost_usd']:.6f}")


def add_intermediate_measures():
    SRC_DIR = f"segmentation/{DATASET}/{MODEL_NAME}_{TMP_SUFFIX}/"
    for file_path in glob.glob(os.path.join(SRC_DIR, "*.jsonl")):
        lines_out = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                reasoning_steps_list = data.get("reasoning_steps", [])

                if not reasoning_steps_list:
                    lines_out.append(json.dumps(data, ensure_ascii=False) + "\n")
                    continue

                classified_raw = data.get("classified_steps")
                classified_list = (
                    classified_raw
                    if isinstance(classified_raw, list)
                    else [classified_raw] * len(reasoning_steps_list)
                )
                direct_utility_list = []
                indirect_utility_list = []
                direct_utility_score_list = []
                indirect_utility_score_list = []
                tag_counts_list = []
                tag_frequencies_list = []

                for cand_idx, cand_steps in enumerate(reasoning_steps_list):
                    classified = _parse_classified_steps(
                        classified_list[cand_idx] if cand_idx < len(classified_list) else None
                    )
                    out = process_one_candidate(cand_steps, classified)
                    direct_utility_list.append(out["direct_utility"])
                    indirect_utility_list.append(out["indirect_utility"])
                    direct_utility_score_list.append(out["direct_utility_score"])
                    indirect_utility_score_list.append(out["indirect_utility_score"])

                    cand_classified = (
                        classified_list[cand_idx] if cand_idx < len(classified_list) else None
                    )
                    counts, freqs = _extract_tag_counts_and_frequencies(
                        cand_steps, cand_classified
                    )
                    tag_counts_list.append(counts)
                    tag_frequencies_list.append(freqs)

                data["direct_utilities"] = direct_utility_list
                data["indirect_utilities"] = indirect_utility_list
                data["direct_utility_scores"] = direct_utility_score_list
                data["indirect_utility_scores"] = indirect_utility_score_list
                data["tag_counts"] = tag_counts_list
                data["tag_frequencies"] = tag_frequencies_list
                lines_out.append(json.dumps(data, ensure_ascii=False) + "\n")

        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines_out)


# Selection metrics for greedy candidate choice
SELECTION_METRICS = [
    "random",
    "num_steps",
    "semantic_similarity",
    "structural_similarity",
    "direct_utility",
    "indirect_utility",
    "result_consolidation",
    "uncertainty_management",
]


def _select_candidate_idx(
    seg_data: dict,
    metric: str,
    rng: random.Random,
) -> int:
    n = len(seg_data.get("reasoning_steps", []))
    if n == 0:
        return 0
    if metric == "random":
        return rng.randint(0, n - 1)

    if metric == "num_steps":
        scores = seg_data.get("num_steps", [0] * n)
        return int(np.argmax(scores))

    if metric == "semantic_similarity":
        scores = seg_data.get("semantic_similarities", [])
        if not scores or all(s is None for s in scores):
            return 0
        vals = [(s if s is not None else -1.0) for s in scores]
        return int(np.argmax(vals))

    if metric == "structural_similarity":
        scores = seg_data.get("structural_similarities", [])
        if not scores or all(s is None for s in scores):
            return 0
        vals = [(s if s is not None else -1.0) for s in scores]
        return int(np.argmax(vals))

    if metric == "direct_utility":
        scores = seg_data.get("direct_utility_scores", seg_data.get("direct_utilities", []))
        if isinstance(scores, list) and scores and isinstance(scores[0], (int, float)):
            return int(np.argmax(scores))
        return 0

    if metric == "indirect_utility":
        scores = seg_data.get("indirect_utility_scores", seg_data.get("indirect_utilities", []))
        if isinstance(scores, list) and scores and isinstance(scores[0], (int, float)):
            return int(np.argmax(scores))
        return 0

    if metric == "result_consolidation":
        freqs = seg_data.get("tag_frequencies", [])
        vals = [
            f.get("result_consolidation", 0.0) if isinstance(f, dict) else 0.0
            for f in freqs
        ]
        return int(np.argmax(vals)) if vals else 0

    if metric == "uncertainty_management":
        freqs = seg_data.get("tag_frequencies", [])
        vals = [
            f.get("uncertainty_management", 0.0) if isinstance(f, dict) else 0.0
            for f in freqs
        ]
        return int(np.argmax(vals)) if vals else 0

    return 0


def select_and_write_selections(seed: int = 42):
    rng = random.Random(seed)
    SRC_DIR = f"res/{DATASET}/{DATASET_NAME}_{MODEL_NAME}_{TMP_SUFFIX}/"
    SEG_DIR = f"segmentation/{DATASET}/{MODEL_NAME}_{TMP_SUFFIX}/"
    for metric in SELECTION_METRICS:
        out_dir = f"selected/{DATASET}/{MODEL_NAME}_{TMP_SUFFIX}/{metric}"
        os.makedirs(out_dir, exist_ok=True)
        for file_path in glob.glob(os.path.join(SEG_DIR, "*.jsonl")):
            lang = os.path.basename(file_path).replace(".jsonl", "")
            res_path = os.path.join(SRC_DIR, f"{lang}.jsonl")
            if not os.path.exists(res_path):
                continue
            seg_by_id = {}
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    seg_by_id[d.get("id")] = d
            lines_out = []
            with open(res_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    res_item = json.loads(line)
                    seg_data = seg_by_id.get(res_item.get("id"))
                    if seg_data is None:
                        lines_out.append(line)
                        continue
                    candidates = _get_candidates(res_item)
                    if not candidates:
                        lines_out.append(line)
                        continue
                    sel_idx = _select_candidate_idx(seg_data, metric, rng)
                    sel_idx = min(sel_idx, len(candidates) - 1)
                    cand = candidates[sel_idx]
                    out_item = {
                        **res_item,
                        "reasoning_trace": cand.get("reasoning_trace", ""),
                        "response": cand.get("response", ""),
                    }
                    lines_out.append(json.dumps(out_item, ensure_ascii=False) + "\n")
            out_path = os.path.join(out_dir, f"{lang}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(lines_out)
        print(f"[Done] selected {metric}")


def _run_eval_on_file(input_path: str) -> Tuple[float, float]:
    
    detector = LanguageDetectorBuilder.from_languages(
        Language.BENGALI, Language.GERMAN, Language.ENGLISH, Language.SPANISH,
        Language.FRENCH, Language.JAPANESE, Language.KOREAN, Language.RUSSIAN,
        Language.SWAHILI, Language.TELUGU, Language.THAI, Language.CHINESE,
        Language.ARABIC, Language.PORTUGUESE,
    ).build()
    
    lang_map = {
        "Language.BENGALI": "bn", "Language.GERMAN": "de", "Language.ENGLISH": "en",
        "Language.SPANISH": "es", "Language.FRENCH": "fr", "Language.JAPANESE": "ja",
        "Language.KOREAN": "ko", "Language.RUSSIAN": "ru", "Language.SWAHILI": "sw",
        "Language.TELUGU": "te", "Language.THAI": "th", "Language.CHINESE": "zh",
        "Language.ARABIC": "ar", "Language.PORTUGUESE": "pt",
    }

    def _detect(txt):
        return lang_map.get(str(detector.detect_language_of(txt)), None)

    def _acc(gold, trace, resp):
        try:
            g = parse(gold, raise_on_error=True)
            t = parse(trace, raise_on_error=True)
            r = parse(resp, raise_on_error=True)
            return 1 if verify(g, r) or verify(g, t) else 0
        except Exception:
            return 0

    def _boxed(txt):
        m = re.search(r"\\boxed\{(.+?)\}", str(txt))
        return m.group(1).strip() if m else str(txt).strip()

    lang_code = os.path.basename(input_path).replace(".jsonl", "")
    acc_list = []
    lang_counter = Counter()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            gold = item.get("answer", "")
            resp = str(item.get("response", ""))
            trace = str(item.get("reasoning_trace", ""))
            lang_counter[_detect(trace)] += 1
            acc_list.append(_acc(gold, _boxed(trace), resp))
    avg_acc = np.mean(acc_list) if acc_list else 0.0
    total = sum(lang_counter.values())
    lang_cons = (lang_counter.get(lang_code, 0) / total) if total else 0.0
    return avg_acc, lang_cons


def run_eval_on_selections():
    results = []
    for metric in SELECTION_METRICS:
        sel_dir = f"selected/{DATASET}/{MODEL_NAME}_{TMP_SUFFIX}/{metric}"
        if not os.path.exists(sel_dir):
            continue
        for file_path in glob.glob(os.path.join(sel_dir, "*.jsonl")):
            lang = os.path.basename(file_path).replace(".jsonl", "")
            acc, lang_cons = _run_eval_on_file(file_path)
            results.append({
                "data": DATASET,
                "model": MODEL_NAME,
                "language": lang,
                "metric": metric,
                "accuracy": round(acc, 4),
                "lang_consistency": round(lang_cons, 4),
            })
    out_csv = f"eval_selection/{DATASET}/selection_results.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if results:
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["data", "model", "language", "metric", "accuracy", "lang_consistency"],
            )
            w.writeheader()
            w.writerows(results)
        print(f"[Done] Wrote {out_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="distill_1.5b", help="Model name (e.g. distill_1.5b)")
    parser.add_argument("--temp", type=float, default="1", help="Temperature name")
    parser.add_argument("--dataset", type=str, default="aime", help="Dataset name (e.g. aime, mgsm_revised)")
    args = parser.parse_args()

    MODEL_NAME = args.model
    DATASET = args.dataset
    DATASET_NAME = "mgsmv2" if DATASET == "mgsm_revised" else DATASET
    TMP = args.temp
    TMP_SUFFIX = f"temp{int(TMP) if TMP == int(TMP) else TMP}"

    segment_steps()
    classify_steps()
    semantic_similarity()
    structural_similarity()
    add_intermediate_measures()
    select_and_write_selections(seed=42)
    run_eval_on_selections()
