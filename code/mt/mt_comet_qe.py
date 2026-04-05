import json
from comet import download_model, load_from_checkpoint
import torch


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if isinstance(lines[0], dict):
        return [x.get("problem") or x.get("question") or list(x.values())[0] for x in lines]
    else:
        return lines


def compute_comet_qe(file1, file2, output_path, model_name="Unbabel/wmt22-cometkiwi-da"):
    sents1 = load_jsonl(file1)
    sents2 = load_jsonl(file2)

    assert len(sents1) == len(sents2), "Both files must have the same number of lines"

    print(f"🔹 Loading COMET-QE model: {model_name}")
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    data = [{"src": s1, "mt": s2} for s1, s2 in zip(sents1, sents2)]

    print("🔹 Scoring sentences ...")
    model_output = model.predict(data, batch_size=8, gpus=1)
    print(model_output)
    seg_scores = model_output["scores"]
    sys_score = model_output["system_score"]

    for i, score in enumerate(seg_scores):
        print(score)
        print(f"{i+1:03d}: {score:.4f} | {sents1[i][:40]} → {sents2[i][:40]}")
    
    with open(output_path, "w", encoding="utf-8") as f_out:
        for s1, s2, score in zip(sents1, sents2, seg_scores):
            record = {
                "en_sent": s1,
                "nonen_sent": s2,
                "comet_qe": round(float(score), 4)
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n💡 Average COMET-QE score: {sys_score:.4f}")


if __name__ == "__main__":
    en_file = "../../data/mgsm_revised/en.jsonl"
    languages = ["bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]

    for language in languages:
        nonen_file = f"../../data/mgsm_revised/{language}.jsonl"
        output_path = f"mt_comet_qe/mgsm_revised/mgsm_{language}.jsonl"
        compute_comet_qe(en_file, nonen_file, output_path)
    
    # AIME (MT)
    # en_file = "../data/aime/en.jsonl"
    # languages = ["bn", "de", "es", "fr", "ja", "ko", "ru", "sw", "te", "th", "zh"]

    # for language in languages:
    #     nonen_file = f"../data/aime/{language}.jsonl"
    #     output_path = f"mt_comet_qe/aime_{language}.jsonl"
    #     compute_comet_qe(en_file, nonen_file, output_path)
