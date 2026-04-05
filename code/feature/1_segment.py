import os
import json
import re
import csv
import glob

# -------------------------------
MODEL_NAMES = ["distill_qwen1.5b", "distill_qwen7b", "qwen4b", "qwen8b"]
DATASET = "mgsm_revised"
MODE = "agnostic"
# -------------------------------

for MODEL_NAME in MODEL_NAMES:
    SRC_DIR = f"../res/{DATASET}_t1/{MODEL_NAME}/"
    OUT_DIR = f"segmentation/{DATASET}/{MODEL_NAME}/"

    if "distill" in MODEL_NAME:
        MODEL = "distill"
    else:
        MODEL = "qwen"

    with open("separators.json", "r", encoding="utf-8") as f:
        SEPARATORS = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    avg_lengths = {}


    if __name__ == "__main__":
        for file_path in glob.glob(os.path.join(SRC_DIR, "*.jsonl")):
            out_path = os.path.join(OUT_DIR, os.path.basename(file_path).replace("_eval", f"_{MODE}"))
            total_steps = 0
            num_entries = 0
            
            if file_path.endswith("_eval.jsonl"):
                with open(file_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
                    language = os.path.basename(file_path)[:2]            
                    
                    for line in f_in:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        LANG_SEPARATORS = SEPARATORS[language][MODEL]

                        reasoning_trace = data.get("reasoning_trace", "").replace("<think>\n", "").replace("\n</think>", "").replace("<think>", "").replace("</think>", "")
                        response = data.get("response", "")

                        if MODE == "specific":
                            if not LANG_SEPARATORS:   # no available separators → return whole trace
                                raise ValueError("No corresponding language-specific separator!")
                            else:
                                pattern = re.compile(
                                    r'(?=(?:' + '|'.join(fr'\b{re.escape(s)}\b\s*' for s in LANG_SEPARATORS) + r'))',
                                    flags=re.IGNORECASE
                                )
                                steps = [
                                    s.strip()
                                    for s in re.split(pattern, reasoning_trace)
                                    if s.strip()
                                ]
                        
                        elif MODE == "agnostic":
                            agnostic_seps = SEPARATORS.get("agnostic", ["\n\n"])
                            pattern = re.compile("|".join(re.escape(s) for s in agnostic_seps))
                            steps = [
                                s.strip()
                                for s in re.split(pattern, reasoning_trace)
                                if s.strip()
                            ]

                        else:
                            raise ValueError("MODE should be either specific or agnostic!")

                        data["reasoning_steps"] = steps
                        data["num_steps"] = len(steps)

                        json.dump(data, f_out, ensure_ascii=False)
                        f_out.write("\n")
                        
                        total_steps += len(steps)
                        num_entries += 1

                avg_length = total_steps / num_entries if num_entries > 0 else 0
                avg_lengths[os.path.basename(file_path).replace("_eval.jsonl", "")] = avg_length
            else:
                pass

        print("\n Summary of average reasoning steps per file:")
        for fname, avg in avg_lengths.items():
            print(f"  • {fname}: {avg:.2f}")
        
        summary_csv = f"{MODE}_{DATASET}_{MODEL_NAME}.csv"
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_name", "avg_steps"])
            for fname, avg in avg_lengths.items():
                writer.writerow([fname, f"{avg:.2f}"])
