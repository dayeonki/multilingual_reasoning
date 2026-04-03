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
  - [Evaluation](#evaluation)
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



### Measurable Reasoning Features

### Logistic Regression Analysis

### Sparse Autoencoder Analysis

### Test-time Selection

### Evaluation



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
