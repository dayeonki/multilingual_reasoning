<div align="center">

 # What Makes Good Multilingual Reasoning? <br> Disentangling Reasoning Traces with Measurable Features

<p align="center">
<img width="963" height="220" alt="Screenshot 2026-04-03 at 2 56 09 PM" src="https://github.com/user-attachments/assets/f13fee7d-88b2-4e1c-81b0-d69f553fe910" />
</p>


<a href=https://dayeonki.github.io/>Dayeon Ki</a><sup>1</sup>, <a href=https://www.cs.jhu.edu/~kevinduh/>Kevin Duh</a><sup>2</sup>, <a href=https://www.cs.umd.edu/~marine/>Marine Carpuat<a><sup>1</sup> <br>
<sup>1</sup>University of Maryland, <sup>2</sup>Johns Hopkins University
<br>

This repository contains the code and dataset for our paper <br> **What Makes Good Multilingual Reasoning? Disentangling Reasoning Traces with Measurable Features**.

<p>
  <a href="" target="_blank" style="text-decoration:none">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat&logo=arxiv" alt="arXiv">
  </a>
</p>

</div>

---

## 👾 TL;DR
Large Reasoning Models (LRMs) still exhibit large performance gaps between English and other languages, yet much current work assumes these gaps can be closed simply by making reasoning in every language resemble English reasoning. This work challenges this assumption by asking instead: what actually characterizes effective reasoning in multilingual settings, and to what extent do English‑derived reasoning features genuinely help in other languages?


## 📰 News
- **`2026-04-03`** We release our code and dataset!


## ✏️ Content
- [🗺️ Overview](#overview)
- [🚀 Quick Start](#quick_start)
  - [Data Preparation](#data-preparation)
  - [Measurable Reasoning Features](#measurable-reasoning-features)
  - [Logistic Regression Analysis](#logistic-regression-analysis)
  - [Sparse Autoencoder Analysis](#sparse-autoencoder-analysis)
  - [Test-time Selection](#test-time-selection)
- [🤲 Citation](#citation)
- [📧 Contact](#contact)

---

<a id="overview"></a>
## 🗺️ Overview

We first define a suite of measurable reasoning features spanning multilingual alignment, reasoning step, and reasoning flow aspects of reasoning traces, and use (**1**) **logistic regression** to quantify how each feature associates with final answer accuracy. 
We further train (**2**) **sparse autoencoders** over multilingual traces to automatically discover latent reasoning concepts that instantiate or extend these features.
Finally, we use the features as (**3**) **test-time selection** policies to examine whether they can steer models toward stronger multilingual reasoning.

### Results

<div align="center">
<img width="951" height="514" alt="Screenshot 2026-04-03 at 2 58 42 PM" src="https://github.com/user-attachments/assets/b4bb0443-7892-48c0-9a10-fd1b3d4c164f" />
</div>


<a id="quick_start"></a>
## 🚀 Quick Start

### Data Preparation

- MGSM-Rev2 dataset: `data/mgsm_revised`
   - List of IDs with updated questions could be found in `replaced_questions.json` file
- AIME 2024-25 dataset: `data/aime`


### Measurable Reasoning Features

#### (1) Prompt LRMs to generate reasoning traces

You will first need to prompt Large Reasoning Models (LRMs) with queries in each target language to produce reasoning traces and final answers, which serve as inputs to all subsequent analyses.

For **Distill-Deepseek** series models,
```bash
python -u code/run_distill_vllm_aime/mgsmv2.py \
  --model_name $MODEL_NAME \
  --num_gpu $NUM_GPU \
  --seed $SEED \
  --iteration $ITERATION \
  --save_name $SAVE_NAME
```

Arguments for the code are:
- `model_name`: The HuggingFace identifier name of the LRM
- `num_gpu`: Number of GPUs needed to run the model (We used 2 A5000 GPU for running 7B sized model)
- `seed`: Seed for vLLM setup
- `iteration`: Name of the iteration (e.g., 1, 2, 3 ...)
- `save_name`: Name of the saved file

For **Qwen3** series models,
```bash
python -u code/run_qwen_vllm_aime/mgsmv2.py \
  --model_name $MODEL_NAME \
  --num_gpu $NUM_GPU \
  --seed $SEED \
  --iteration $ITERATION \
  --save_name $SAVE_NAME
```

The arguments are identical to the code above. You can set your save directory in `output_path` in the code.

Then, you can evaluate the generated reasoning traces on the basis of (1) final answer accuracy, (2) language, and (3) length using `python -u code/evaluate.py`.


#### (2) Segment and Classify

We then segment the generated reasoning trace using `\n\n` as the separator and then classify each reasoning step with GPT-4o to each cognitive-behavioral function tags.
- Segmentation: `python -u code/feature/1_segment.py`
- Classification: You will need to set you your `OPENAI_API_KEY`.

```bash
python -u code/feature/2_classify.py \
   --input_file $PATH_TO_INPUT_FILE \
   --en_input_file $PATH_TO_EN_INPUT_FILE \
   --language $LANG \
   --output_file $PATH_TO_OUTPUT_FILE
```

Arguments for running the classification code are as follows:
- `--input_file`: Path to the input jsonl file (saved from segmentation code)
- `--en_input_file`: Path to the English input file
- `--language`: ISO code of the target language
- `--output_file`: Path to the output jsonl file

You can check the distribution of the annotated cognitive-behavioral tag using `python -u code/feature/get_distribution.py` and check the frequency of each tag using `python -u code/feature/tag_frequency.py`. This will give values for **Reasoning Flow** features.

#### (3) Generate feature values

For generating **Multilingual Alignment** features:
- **COMET-QE**: `python -u code/mt/mt_comet_qe.py`
- **Structural similarity**: `python -u code/feature/3_struct_alignment.py`

For generating **Reasoning Step** features:
- **Num. Steps**: You will get a summary when running `python -u code/feature/1_segment.py`
- **Validity**: `python -u code/feature/4_validity_utility.py`
- **Direct/Indirect Utility**: `python -u code/feature/4_validity_utility.py`
- V-Information:
```bash
python -u code/feature/5_vi.py \
  --model_name $MODEL_NAME \
  --num_gpu $NUM_GPU \
  --seed $SEED \
  --input_path $PATH_TO_INPUT_FILE \
  --output_path $PATH_TO_OUTPUT_FILE \
  --max_logprobs $MAX_LOGPROBS
```

Arguments are mostly similar to Step 1, additionally with:
- `--input_path`: Path to the input jsonl file
- `--output_path`: Path to the output jsonl file
- `--max_logprobs`: Flat logprobs of a request into multiple primitive type lists (vLLM feature)


### Logistic Regression Analysis

#### (1) English vs. Non-English
Each code with automatically compute (1) Change in answer accuracy association, (2) Pooled interaction for comparing reasoning traces from English vs. non-English queries, and (3) Visualization used in Figure 1 of the paper.

- Running **univariate** logistic regression model: `python -u code/analysis/univariate.py`
- Running **multivariate** logistic regression model: `python -u code/analysis/multivariate.py`

#### (2) Per language

- Running **univariate** logistic regression model: `python -u code/analysis/univariate_perlang.py`
- Running **multivariate** logistic regression model: `python -u code/analysis/multivariate_perlang.py`


### Sparse Autoencoder Analysis

Please refer to <a href=https://github.com/rmovva/HypotheSAEs>HypotheSAEs</a> Github repo for detailed instruction.

### Test-time Selection

#### (1) Generate candidate reasoning traces

We can generate 8 traces for each temperature (0.3, 0.6, 0.8, 1.0) and then later combine and select using each feature value as the selection policy at test-time.

```bash
python -u code/steer/1_generate.py \
  --model_name $MODEL_NAME \
  --num_gpu $NUM_GPU \
  --seed $SEED \
  --data_type $DATASET_NAME \
  --save_name $SAVE_NAME \
  --n $NUM_TRACES \
  --temperature $TEMPERATURE \
  --languages $LANGUAGES
```

Arguments are mostly similar to [Measurable Reasoning Features](#measurable-reasoning-features) section, additionally with:
- `--data_type`: Name of the dataset (either `aime` or `mgsm_revised`)
- `--n`: Number of candidate reasoning traces to generate (In the paper, we generated 8 traces for each temperature)
- `--temperature`: Temperature value (0.3, 0.6, 0.8, 1.0)
- `--languages`: List of languages to run the code for

#### (2) Re-rank and select based on each feature value

We re-rank the generated candidates using each feature value.

```bash
python -u code/steer/2_selection.py \
  --model_name $MODEL_NAME \
  --data_type $DATASET_NAME \
  --temperature $TEMPERATURE
```

#### (3) Combine traces from each temperature

We combine the re-ranked traces from each temperature and then do the final selection.

```bash
python -u code/steer/3_combine.py \
  --model_name $MODEL_NAME \
  --data_type $DATASET_NAME \
  --temperatures $TEMPERATURE_LIST \
  --seed $SEED \
  --output_path $PATH_TO_OUTPUT_FILE \
  --stats $RUN_STATISTICAL_SIGNIFICANCE_TEST
```
The newly added arguments are:
- `--temperatures`: List of temperatures to consider in the final selection
- `--stats`: Whether to run the statistical significance testing (if passed as argument, then True)


---

<a id="citation"></a>
## 🤲 Citation
If you find our work useful in your research, please consider citing our work:
```
TBD
```

<a id="contact"></a>
## 📧 Contact
For questions, issues, or collaborations, please reach out to [dayeonki@umd.edu](mailto:dayeonki@umd.edu).
