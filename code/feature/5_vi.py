import json
import math
import argparse
import os
from typing import Dict, Tuple, List
from vllm import LLM, SamplingParams


def score_answer_and_entropy_with_vllm(
    llm: LLM,
    tokenizer,
    model_name,
    prompt: str,
    answer: str,
    max_logprobs: int = 20,
    enable_think=False,
) -> Tuple[float, List[float], float]:
    if tokenizer is None:
        tokenizer = llm.get_tokenizer()

    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    logprob_sum = 0.0
    entropies: List[float] = []

    for tok_id in answer_ids:
        # https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,  # recommended for scoring
            top_p=1.0,
            top_k=-1,
            logprobs=max_logprobs,
        )
        text_prompt = prompt

        outputs = llm.generate(
            [text_prompt],
            sampling_params=sampling_params
        )
        out = outputs[0].outputs[0]
        logprob_dict = out.logprobs[0]  # {token_id: Logprob}
        print("Log probs: ", out.logprobs)
        print("\n--- Generated token text:", out.text)
        print("--- Logprobs returned:", {tokenizer.decode([k]): v.logprob for k,v in logprob_dict.items()})

        # Compute entropy over top-k distribution
        lp_values = [lp.logprob for lp in logprob_dict.values()]
        probs = [math.exp(lp) for lp in lp_values]
        Z = sum(probs) + 1e-12
        probs = [p / Z for p in probs]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        entropies.append(entropy)

        if tok_id in logprob_dict:
            lp_correct = logprob_dict[tok_id].logprob
        else:
            lp_correct = math.log(1e-8)
        logprob_sum += lp_correct
        text_prompt += tokenizer.decode([tok_id])

    mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    return logprob_sum, entropies, mean_entropy


# (3) V-Information (entire reasoning trace)
def compute_v_information(
    llm: LLM,
    tokenizer,
    model_name, 
    problem_text: str,
    prompt_lang: str,
    reasoning_text: str,
    answer_text: str,
    max_logprobs: int = 20,
) -> Dict[str, float]:
    if tokenizer is None:
        tokenizer = llm.get_tokenizer()
    
    prompt_with_trace = "Please answer the following question.\n\n" + f"Question: {problem_text}" + "\n\n" + "<think>\n" + reasoning_text + "\n</think>" + "\n\n" + "Answer: "
    prompt_without_trace = "Please answer the following question.\n\n" + f"Question: {problem_text}" + "\n\n" + "Answer: "

    print("="*50, " ✏️ Prompt (with trace) ✏️ ", "="*50)
    print(prompt_with_trace)

    print("="*50, " ✏️ Prompt (without trace) ✏️ ", "="*50)
    print(prompt_without_trace)

    lp_with, ent_with, mean_H_with = score_answer_and_entropy_with_vllm(
        llm, tokenizer, model_name, prompt_with_trace, answer_text, max_logprobs=max_logprobs, enable_think=False,
    )
    lp_without, ent_without, mean_H_without = score_answer_and_entropy_with_vllm(
        llm, tokenizer, model_name, prompt_without_trace, answer_text, max_logprobs=max_logprobs, enable_think=False,
    )

    vi = lp_with - lp_without

    return {
        "VI": vi,
        "logprob_with_trace": lp_with,
        "logprob_without_trace": lp_without,
        "mean_entropy_with_trace": mean_H_with,
        "mean_entropy_without_trace": mean_H_without,
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen-3/Qwen-3-4B")
    parser.add_argument("--num_gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_logprobs", type=int, default=20)
    args = parser.parse_args()

    llm = LLM(
        seed=args.seed,
        model=args.model_name,
        dtype="auto",
        download_dir=os.environ.get("HF_HOME", None),
        tensor_parallel_size=args.num_gpu,
    )
    lm_tokenizer = llm.get_tokenizer()

    languages = ["en", "bn", "de", "es", "fr", "ja", "ko"]
    
    for language in languages:
        input_file = f"{args.input_path}/{language}.jsonl"
        output_file = f"{args.output_path}/{language}_vi.jsonl"

        print(f"Processing language: {language}")
        print(f"  Input:  {input_file}")
        print(f"  Output: {output_file}")

        entries = []
        with open(input_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entries.append(json.loads(line))

        results = []

        for idx, data in enumerate(entries):
            problem = data.get("problem") or data.get("question")
            reasoning_steps = data.get("reasoning_steps", [])
            reasoning_text = "\n".join(reasoning_steps)
            answer = data.get("final_answer") or data.get("answer")

            vi_info = compute_v_information(
                llm,
                lm_tokenizer,
                args.model_name.lower(),
                problem_text=problem,
                prompt_lang="en",
                reasoning_text=reasoning_text,
                answer_text=answer,
                max_logprobs=args.max_logprobs,
            )

            out_entry = {
                "id": data.get("id", idx),
                "problem": problem,
                "answer": answer,
                "VI": vi_info["VI"],
                "logprob_with_trace": vi_info["logprob_with_trace"],
                "logprob_without_trace": vi_info["logprob_without_trace"],
                "mean_entropy_with_trace": vi_info["mean_entropy_with_trace"],
                "mean_entropy_without_trace": vi_info["mean_entropy_without_trace"],
            }
            results.append(out_entry)
            print(out_entry)

        with open(output_file, "w") as out_f:
            for r in results:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
