import json
import argparse
import time
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def prompt_qwen_vllm(
    tokenizer,
    llm,
    input_path,
    output_path,
    prompt_lang,
    max_new_tokens=32768,
    enable_think=True,
):
    # Sampling parameters follow best practice for Qwen
        # https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        max_tokens=max_new_tokens
    )

    # Skip processed IDs
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f_out:
            for line in f_out:
                try:
                    obj = json.loads(line)
                    processed_ids.add(obj.get("id"))
                except:
                    continue

    PROMPT_LANG_MAP = \
        {
            "en": "Please reason step by step, and put your final answer within \\boxed{{}}.\nQuestion: {question}",
            "bn": "অনুগ্রহ করে ধাপে ধাপে যুক্তি দেখান, এবং আপনার চূড়ান্ত উত্তরের চারপাশে \\boxed{{}} ব্যবহার করুন।\nপ্রশ্ন: {question}",
            "de": "Bitte denken Sie Schritt für Schritt nach und schreiben Sie Ihre endgültige Antwort in \\boxed{{}}.\nFrage: {question}",
            "es": "Por favor, razona paso a paso y coloca tu respuesta final dentro de \\boxed{{}}.\nPregunta: {question}",
            "fr": "Veuillez raisonner étape par étape et placer votre réponse finale dans \\boxed{{}}.\nQuestion : {question}",
            "ja": "段階的に考えて、最終的な答えを \\boxed{{}} の中に入れてください。\n質問: {question}",
            "ko": "단계별로 추론하고, 최종 답을 \\boxed{{}} 안에 넣어주세요.\n질문: {question}",
            "ru": "Пожалуйста, рассуждайте шаг за шагом и поместите свой окончательный ответ в \\boxed{{}}.\nВопрос: {question}",
            "sw": "Tafadhali toa hoja hatua kwa hatua, na weka jibu lako la mwisho ndani ya \\boxed{{}}.\nSwali: {question}",
            "te": "దయచేసి దశలవారీగా ఆలోచించండి, మరియు మీ తుది సమాధానాన్ని \\boxed{{}} లో వ్రాయండి.\nప్రశ్న: {question}",
            "th": "กรุณาให้เหตุผลทีละขั้นตอน และใส่คำตอบสุดท้ายของคุณไว้ภายใน \\boxed{{}}\nคำถาม: {question}",
            "zh": "请逐步推理，并将最终答案写在 \\boxed{{}} 中。\n質問: {question}",
        }
    prompt_template = PROMPT_LANG_MAP[prompt_lang]

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "a", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)
            if data["id"] in processed_ids:
                print(f"Skipping already processed ID: {data['id']}")
                continue

            if "mgsm" in input_path or "math500" in input_path:
                question = data["question"]
            elif "aime" in input_path:
                question = data["problem"]
            else:
                raise ValueError("Invalid dataset type.")
            
            prompt = prompt_template.format(question=question)
            print("="*50, " ✏️ Prompt ✏️ ", "="*50)
            print(prompt)

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Run inference with vLLM
            start_time = time.time()
            outputs = llm.generate([text], sampling_params)
            elapsed = time.time() - start_time
            minutes, seconds = divmod(int(elapsed), 60)

            output_text = outputs[0].outputs[0].text

            # Extract thinking and answer segments
            if "</think>" in output_text:
                parts = output_text.split("</think>", 1)
                thinking_content = parts[0].strip()
                answer_content = parts[1].strip()
            else:
                thinking_content, answer_content = "", output_text.strip()

            answer_markers = ["Answer:", "**Answer:**", "Final Answer:", "**Final Answer:**"]
            for marker in answer_markers:
                if marker in answer_content:
                    answer_content = answer_content.split(marker, 1)[1].strip()
                    break

            print("="*50, " 💭 Thinking 💭 ", "="*50)
            print(thinking_content)
            print("="*50, " 🤖 Answer 🤖 ", "="*50)
            print(answer_content)
            print("="*50, " ⏱️ Generation time ⏱️ ", "="*50)
            print(f"{minutes:02d}:{seconds:02d}")

            result = {
                "id": data["id"],
                "question": question,
                "reasoning_trace": thinking_content,
                "response": answer_content,
                "answer": data["answer"],
                "time": f"{minutes:02d}:{seconds:02d}"
            }

            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--save_name", default="distill_qwen_7b")
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    think_token_id = tokenizer.convert_tokens_to_ids("</think>")
    print("#️⃣ Thinking Token ID: ", think_token_id)

    # Initialize vLLM engine
        # max_model_len = prompt length + output token length
        # maximum number of sequences per iteration: max_num_seqs=256 (used to avoid OOM error)
        # tensor_parallel_size: number of GPUs to use
    llm = LLM(
        seed=args.seed,
        model=args.model_name,
        dtype="auto",
        download_dir=os.environ['HF_HOME'],
        tensor_parallel_size=args.num_gpu,   # adjust for multi-GPU
        # gpu_memory_utilization=0.9,
    )

    languages = ["en", "bn", "de", "es", "fr", "ja", "ko", "ru", "sw", "te", "th", "zh"]
    
    for language in languages:
        input_path = f"../data/aime/{language}.jsonl"
        output_path = f"../res/aime_t{str(args.iteration)}/{args.save_name}/{language}.jsonl"

        prompt_qwen_vllm(
            tokenizer=tokenizer,
            llm=llm,
            input_path=input_path,
            output_path=output_path,
            prompt_lang=language,
            max_new_tokens=args.max_new_tokens,
        )
