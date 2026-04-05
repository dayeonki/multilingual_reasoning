from lingua import Language, LanguageDetectorBuilder
from math_verify import parse, verify
from collections import Counter
import tiktoken
import re
import csv
import numpy as np
import json

# ---------------------------------------------------------
dataset = "mgsm_revised"
model = "distill_qwen7b"
# ---------------------------------------------------------

enc = tiktoken.get_encoding("o200k_base")  # used for GPT-3.5/4


def count_num_tokens(text):
    return len(enc.encode(text))


def parse_time_str(t):
    if not t or not isinstance(t, str):
        return None
    try:
        parts = t.split(":")
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
    except Exception:
        pass
    return None


detector = LanguageDetectorBuilder.from_languages(
    Language.BENGALI, Language.GERMAN, Language.ENGLISH, Language.SPANISH,
    Language.FRENCH, Language.JAPANESE, Language.KOREAN, Language.RUSSIAN,
    Language.SWAHILI, Language.TELUGU, Language.THAI, Language.CHINESE
).build()

LANGUAGE_ISO_MAP = {
    "Language.BENGALI": "bn",
    "Language.GERMAN": "de",
    "Language.ENGLISH": "en",
    "Language.SPANISH": "es",
    "Language.FRENCH": "fr",
    "Language.JAPANESE": "ja",
    "Language.KOREAN": "ko",
    "Language.RUSSIAN": "ru",
    "Language.SWAHILI": "sw",
    "Language.TELUGU": "te",
    "Language.THAI": "th",
    "Language.CHINESE": "zh"
}

def detect_lang(text):
    lang = str(detector.detect_language_of(text))
    return LANGUAGE_ISO_MAP.get(lang, None)


def compute_acc(gold_answer, trace_answer, response_answer):
    # https://github.com/huggingface/Math-Verify
    try:
        gold = parse(gold_answer, raise_on_error=True)
        trace = parse(trace_answer, raise_on_error=True)
        response = parse(response_answer, raise_on_error=True)
        return 1 if verify(gold, response) or verify(gold, trace) else 0
    except:
        return 0


def extract_boxed(text):
    match = re.search(r"\\boxed\{(.+?)\}", text)
    return match.group(1).strip() if match else text.strip()


if __name__ == "__main__":

    tiers = [f"{dataset}_t1", f"{dataset}_t2", f"{dataset}_t3"]
    languages = ["en", "bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
    
    csv_output = f"csv/{dataset}_{model}.csv"
    summary_rows = []
    
    for language in languages:
        tier_tokens = []
        tier_accs = []
        tier_times = []
        combined_lang_counter = Counter()

        print("\n", "="*100)
        print(f"LANGUAGE: {language}")
        print("="*100)

        for tier in tiers:
            input_file = f"../res/{tier}/{model}/{language}.jsonl"
            output_file = f"../res/{tier}/{model}/{language}_eval.jsonl"

            num_tokens_list = []
            acc_list = []
            time_list = []
            lang_counter = Counter()

            with open(input_file, "r", encoding="utf-8") as f_in, \
                 open(output_file, "w", encoding="utf-8") as f_out:

                for line in f_in:
                    if not line.strip():
                        continue

                    item = json.loads(line)

                    gold_answer = item.get("answer", "")
                    reasoning_trace = str(item.get("reasoning_trace", "")).replace("<think>\n", "").replace("\n</think>", "")
                    response = str(item.get("response", ""))

                    num_tokens = count_num_tokens(reasoning_trace)
                    lang_code = detect_lang(reasoning_trace)
                    acc = compute_acc(gold_answer, extract_boxed(reasoning_trace), response)

                    t_str = item.get("time", "")
                    t_sec = parse_time_str(t_str)
                    if t_sec is not None:
                        time_list.append(t_sec)

                    num_tokens_list.append(num_tokens)
                    acc_list.append(acc)
                    lang_counter[lang_code] += 1

                    enriched = {
                        **item,
                        "num_tokens": num_tokens,
                        "lang_code": lang_code,
                        "acc": acc
                    }
                    f_out.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            avg_tokens = np.mean(num_tokens_list) if num_tokens_list else 0
            avg_acc = np.mean(acc_list) if acc_list else 0
            avg_time = np.mean(time_list) if time_list else 0

            print(f"[{tier}] ✔ tokens={avg_tokens:.1f}, ✔ acc={avg_acc:.3f}, ✔ time={avg_time:.1f}, ✔ lang={dict(lang_counter)}")

            tier_tokens.append(avg_tokens)
            tier_accs.append(avg_acc)
            tier_times.append(avg_time)
            combined_lang_counter.update(lang_counter)
        
        sorted_langs = dict(sorted(
            combined_lang_counter.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        summary_rows.append({
            "language": language,
            "avg_tokens_mean": round(np.mean(tier_tokens), 2),
            "avg_tokens_std": round(np.std(tier_tokens), 2),

            "avg_acc_mean": round(np.mean(tier_accs), 4),
            "avg_acc_std": round(np.std(tier_accs), 4),

            "avg_time_mean": round(np.mean(tier_times), 2),
            "avg_time_std": round(np.std(tier_times), 2),

            "lang_code_distribution": json.dumps(sorted_langs, ensure_ascii=False)
        })

    with open(csv_output, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "language",
            "avg_tokens_mean", "avg_tokens_std",
            "avg_acc_mean", "avg_acc_std",
            "avg_time_mean", "avg_time_std",
            "lang_code_distribution"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)
