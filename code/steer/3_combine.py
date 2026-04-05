import os
import json
import glob
import random
import argparse
from stat_significance import run_bootstrap_analysis


SELECTION_METRICS = [
    "random", "num_steps", "semantic_similarity", "structural_similarity",
    "direct_utility", "indirect_utility", "result_consolidation", "uncertainty_management",
]
DEFAULT_TEMPS = [0.3, 0.6, 0.8, 1.0]


def combine_temperature_selections(
    dataset: str,
    model_base: str,
    temps: list = None,
    seed: int = 42,
    out_subdir: str = "selected_combined",
) -> None:
    temps = temps or DEFAULT_TEMPS
    rng = random.Random(seed)
    for metric in SELECTION_METRICS:
        out_dir = f"{out_subdir}/{dataset}/{model_base}/{metric}"
        os.makedirs(out_dir, exist_ok=True)
        first_temp_dir = f"selected/{dataset}/{model_base}_temp{temps[0]}/{metric}"
        if not os.path.isdir(first_temp_dir):
            print(f"[Skip] {first_temp_dir} not found")
            continue
        for fp in sorted(glob.glob(os.path.join(first_temp_dir, "*.jsonl"))):
            lang = os.path.basename(fp).replace(".jsonl", "")
            id_to_items = {}
            for t in temps:
                temp_dir = f"selected/{dataset}/{model_base}_temp{t}/{metric}"
                src = os.path.join(temp_dir, f"{lang}.jsonl")
                if not os.path.exists(src):
                    continue
                with open(src, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        tid = item.get("id")
                        if tid not in id_to_items:
                            id_to_items[tid] = []
                        id_to_items[tid].append(item)
            if not id_to_items:
                continue
            lines_out = []
            for tid in sorted(id_to_items.keys()):
                items = id_to_items[tid]
                chosen = rng.choice(items)
                lines_out.append(json.dumps(chosen, ensure_ascii=False) + "\n")
            out_path = os.path.join(out_dir, f"{lang}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(lines_out)
        print(f"[Done] combined temps for {metric}")
    print(f"[Done] combine_temperature_selections -> {out_subdir}/{dataset}/{model_base}/")


def main():
    parser = argparse.ArgumentParser(description="Combine selected/* across temps 0.3,0.6,0.8,1")
    parser.add_argument("--data_type", type=str, required=True, help="e.g. mgsm_revised")
    parser.add_argument("--model_name", type=str, required=True, help="e.g. distill1.5b, distill7b, qwen4b, qwen8b")
    parser.add_argument("--temperatures", type=str, default="0.3,0.6,0.8,1", help="Comma-separated temps (default: 0.3,0.6,0.8,1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="selected_combined", help="Output subdir")
    parser.add_argument("--stats", action="store_true", help="Run bootstrap significance after combining")
    args = parser.parse_args()
    temps = [float(x.strip()) for x in args.temps.split(",")]
    combine_temperature_selections(args.dataset, args.model_base, temps=temps, seed=args.seed, out_subdir=args.out)
    if args.stats:
        run_bootstrap_analysis(args.dataset, args.model_base, sel_dir=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
