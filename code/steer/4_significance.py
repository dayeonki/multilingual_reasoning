import os
import re
import json
import glob
import argparse
import numpy as np
from math_verify import parse, verify


SELECTION_METRICS = [
    "random", "num_steps", "semantic_similarity", "structural_similarity",
    "direct_utility", "indirect_utility", "result_consolidation", "uncertainty_management",
]
N_BOOTSTRAP = 1000  # bootstrap iterations for 95% CI
MATH_VERIFY_TIMEOUT_SECONDS = 15


def _get_instance_outcomes(filepath: str) -> dict[int, int]:

    def _acc(gold, trace, resp):
        try:
            g = parse(gold, raise_on_error=True)
            t = parse(trace, raise_on_error=True)
            r = parse(resp, raise_on_error=True)
            return 1 if verify(g, r, timeout_seconds=MATH_VERIFY_TIMEOUT_SECONDS) or verify(
                g, t, timeout_seconds=MATH_VERIFY_TIMEOUT_SECONDS
            ) else 0
        except BaseException:
            return 0

    def _boxed(txt):
        m = re.search(r"\\boxed\{(.+?)\}", str(txt))
        return m.group(1).strip() if m else str(txt).strip()

    out = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                tid = item.get("id")
                gold = item.get("answer", "")
                resp = str(item.get("response", ""))
                trace = str(item.get("reasoning_trace", ""))
                out[tid] = _acc(gold, _boxed(trace), resp)
            except BaseException:
                continue  # skip this entry (timeout, parse error, etc.) and move on
    return out


def _load_outcomes_by_lang(sel_base: str, metric: str) -> dict[str, dict[int, int]]:
    metric_dir = os.path.join(sel_base, metric)
    result = {}
    for fp in glob.glob(os.path.join(metric_dir, "*.jsonl")):
        lang = os.path.basename(fp).replace(".jsonl", "")
        result[lang] = _get_instance_outcomes(fp)
    return result


def _bootstrap_acc(outcomes: list[int], n_boot: int = N_BOOTSTRAP, rng: np.random.Generator = None) -> tuple[float, float, float, float]:
    rng = rng or np.random.default_rng(42)
    arr = np.array(outcomes, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(arr[idx].mean())
    boots = np.array(boots)
    boot_std = float(np.std(boots))
    return float(arr.mean()), boot_std, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _bootstrap_significance(
    random_outcomes: list[int],
    metric_outcomes: list[int],
    n_boot: int = N_BOOTSTRAP,
    rng: np.random.Generator = None,
) -> tuple[float, float, float, float]:
    rng = rng or np.random.default_rng(42)
    n = min(len(random_outcomes), len(metric_outcomes))
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0
    rnd = np.array(random_outcomes[:n], dtype=float)
    mtr = np.array(metric_outcomes[:n], dtype=float)
    diff_obs = mtr.mean() - rnd.mean()
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = mtr[idx].mean() - rnd[idx].mean()
        boots.append(d)
    boots = np.array(boots)
    ci_low = float(np.percentile(boots, 2.5))
    ci_high = float(np.percentile(boots, 97.5))

    p_low = float(np.mean(boots <= 0))
    p_high = float(np.mean(boots >= 0))
    p_val = 2 * min(p_low, p_high)
    return diff_obs, ci_low, ci_high, p_val


def run_bootstrap_analysis(
    dataset: str,
    model_base: str,
    sel_dir: str = "selected_combined",
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
) -> None:
    sel_base = f"{sel_dir}/{dataset}/{model_base}"
    if not os.path.isdir(sel_base):
        print(f"[Error] {sel_base} not found")
        return
    rng = np.random.default_rng(seed)
    print(f"\nBootstrap analysis: {dataset} / {model_base} (n={n_boot})")
    print("=" * 95)
    metrics_to_compare = [m for m in SELECTION_METRICS if m != "random"]

    random_by_lang = _load_outcomes_by_lang(sel_base, "random")
    if not random_by_lang:
        print("[Error] No random outcomes loaded (check lingua, math_verify)")
        return
    en_langs = {"en"}
    non_en_langs = {lang for lang in random_by_lang if lang != "en"}
    
    print(f"\n{'metric':<22} {'pool':<6} {'acc':>8} {'±bootstrap_std':>14} {'95% CI':>22} {'vs_random':>10} {'95% CI diff':>22} {'p_value':>10} {'sig':>4}")
    print("-" * 105)
    for metric in SELECTION_METRICS:
        m_by_lang = _load_outcomes_by_lang(sel_base, metric)
        if not m_by_lang:
            continue
        for pool_name, lang_set in [("en", en_langs), ("non-en", non_en_langs)]:
            rnd_list = []
            mtr_list = []
            for lang in lang_set:
                if lang not in random_by_lang or lang not in m_by_lang:
                    continue
                rnd = random_by_lang[lang]
                mtr = m_by_lang[lang]
                common_ids = sorted(set(rnd.keys()) & set(mtr.keys()))
                for iid in common_ids:
                    rnd_list.append(rnd[iid])
                    mtr_list.append(mtr[iid])
            if not rnd_list:
                continue
            diff_obs, ci_lo, ci_hi, p_val = _bootstrap_significance(rnd_list, mtr_list, n_boot, rng)
            m_acc, m_std, m_lo, m_hi = _bootstrap_acc(mtr_list, n_boot, rng)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{metric:<22} {pool_name:<6} {m_acc:>8.4f} ±{m_std:>6.4f}   [{m_lo:.4f}, {m_hi:.4f}]   "
                  f"{diff_obs:>+10.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]   {p_val:>10.4f} {sig:>4}")
    print("-" * 105)
    print("sig: *** p<0.001, ** p<0.01, * p<0.05 (two-tailed vs random)\n")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap 95% CI and significance vs random")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_base", type=str, required=True)
    parser.add_argument("--sel_dir", type=str, default="selected_combined", help="selected or selected_combined")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_bootstrap_analysis(args.dataset, args.model_base, args.sel_dir, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
