import json
import os
import argparse
import tiktoken
from openai import OpenAI
from prompt import classification_prompt, classification_prompt_en

client = OpenAI()

# API Pricing
    # https://platform.openai.com/docs/pricing

def count_tokens(messages, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for msg in messages:
        num_tokens += 4  # message framing
        for key, value in msg.items():
            num_tokens += len(enc.encode(value))
    num_tokens += 2
    return num_tokens


PRICE_INPUT = 2.50 / 1_000_000           # non-cached prompt tokens
PRICE_INPUT_CACHED = 1.25 / 1_000_000    # cached prompt tokens
PRICE_OUTPUT = 10.00 / 1_000_000         # output tokens


def estimate_cost(messages, response_text, cached_ratio=0.0):
    prompt_tokens = count_tokens(messages)
    output_tokens = len(tiktoken.encoding_for_model("gpt-4o").encode(response_text))

    cached_tokens = int(prompt_tokens * cached_ratio)
    non_cached_tokens = prompt_tokens - cached_tokens

    cost = (
        non_cached_tokens * PRICE_INPUT +
        cached_tokens * PRICE_INPUT_CACHED +
        output_tokens * PRICE_OUTPUT
    )
    return {
        "prompt_tokens": prompt_tokens,
        # "cached_tokens": cached_tokens,
        # "non_cached_tokens": non_cached_tokens,
        "output_tokens": output_tokens,
        "total_cost_usd": cost
    }


def load_processed_ids(output_file):
    if not os.path.exists(output_file):
        return set()

    processed = set()
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    processed.add(obj["id"])
            except:
                continue
    return processed


def prompt_gpt4(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def eval_per_step_nonen(item, en_item, language):
    en_problem = en_item.get("question", [])
    problem = item.get("question", [])
    reasoning_steps = item.get("reasoning_steps", [])
    if not reasoning_steps:
        return []

    all_steps_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(reasoning_steps))

    labels = []
    embedded_prompt = (
        classification_prompt
        .replace("{{problem}}", problem)
        .replace("{{language}}", language)
        .replace("{{en_problem}}", en_problem)
        .replace("{{reasoning_steps}}", all_steps_text)
    )

    print(f"\n{'='*30} 🧩 Prompt 🧩 {'='*30}\n")
    print(embedded_prompt)

    generation = prompt_gpt4(embedded_prompt)
    print(f"\n{'='*30} ✏️ Generation ✏️ {'='*30}\n")
    print(generation)

    cost_info = estimate_cost(
        messages=[{"role": "user", "content": embedded_prompt}],
        response_text=generation,
        cached_ratio=0.0
    )
    print(cost_info)
    item["token_usage"] = cost_info

    return generation


def eval_per_step_en(item, language="English"):
    problem = item.get("question", [])
    reasoning_steps = item.get("reasoning_steps", [])
    if not reasoning_steps:
        return []

    all_steps_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(reasoning_steps))

    labels = []
    embedded_prompt = (
        classification_prompt_en
        .replace("{{language}}", language)
        .replace("{{problem}}", problem)
        .replace("{{reasoning_steps}}", all_steps_text)
    )

    print(f"\n{'='*100} 🧩 Prompt 🧩 {'='*100}\n")
    print(embedded_prompt)
    
    generation = prompt_gpt4(embedded_prompt)
    print(f"\n{'='*100} ✏️ Generation ✏️ {'='*100}\n")
    print(generation)

    cost_info = estimate_cost(
        messages=[{"role": "user", "content": embedded_prompt}],
        response_text=generation,
        cached_ratio=0.0
    )
    print(cost_info)
    item["token_usage"] = cost_info

    return generation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to model output JSONL file")
    parser.add_argument("--en_input_file", required=True, help="Path to model output English JSONL file")
    parser.add_argument("--language", required=True, help="Language of the input file")
    parser.add_argument("--output_file", required=True, help="Path to save evaluation results")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    processed_ids = load_processed_ids(args.output_file)
    print(processed_ids)
    print(f"🔎 Loaded {len(processed_ids)} processed IDs from output file.")

    if args.input_file == args.en_input_file:
        with open(args.input_file, "r", encoding="utf-8") as f_in, \
            open(args.output_file, "a", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                item = json.loads(line)

                if item.get("id") in processed_ids:
                    print(f"⏩ Skipping ID {item['id']} (already processed)")
                    continue

                generation = eval_per_step_en(item, args.language)
                item["classified_steps"] = generation

                json.dump(item, f_out, ensure_ascii=False)
                f_out.write("\n")
    else:
        with open(args.input_file, "r", encoding="utf-8") as f_in, \
            open(args.en_input_file, "r", encoding="utf-8") as f_en_in, \
            open(args.output_file, "w", encoding="utf-8") as f_out:
            for line, en_line in zip(f_in, f_en_in):
                if not line.strip():
                    continue

                item = json.loads(line)
                en_item = json.loads(en_line)

                if item.get("id") in processed_ids:
                    print(f"⏩ Skipping ID {item['id']} (already processed)")
                    continue

                generation = eval_per_step_nonen(item, en_item, args.language)
                item["classified_steps"] = generation

                json.dump(item, f_out, ensure_ascii=False)
                f_out.write("\n")
