
# Decision-Making with Large Language Models

This repository contains all the data, transcripts, and analysis scripts supporting the experiments described in:

> **Weber, R., Xu, S., & Gonzales, C. (2025). Decision-Making with Large Language Models.  
> CEUR Workshop Proceedings.**  
> [PDF](Weber_et_al__LLM_Decision_Making_submissions_to_CEUR_Wksp_Proc__CEUR_WS_org.pdf)

---

## Overview

We evaluate how current LLMs perform on a variety of decision-making tasks by:
1. **Prompting** different LLMs across several domains.  
2. **Logging** all prompts & model responses (see `Transcripts/`).  
3. **Computing** n-gram edit distances between model outputs and a reference “ideal” solution (see `Data/`).  
4. **Summarizing** results in tables and figures (see `Scripts/`).

---

## Getting Started

### 1. Clone this repo
```bash
git clone https://github.com/Rosinaweber/decision-making-LLMs.git
cd decision-making-LLMs
````

### 2. Create a Python environment

```bash
python3 -m venv venv
source venv/bin/activate     # macOS/Linux
# .\venv\Scripts\activate    # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

*(If there is no `requirements.txt`, install dependencies manually, e.g.:)*

```bash
pip install numpy pandas matplotlib nltk
```

### 3. Explore the data

* **`Data/`**

  * `reference_solutions.csv`
  * `prompts_responses.csv`
  * `edit_distances.csv`

* **`Transcripts/`**

  * Raw JSON or text logs of every LLM interaction.

### 4. Run the analysis scripts

All scripts assume your current working directory is the repo root.

#### a. Compute n-gram edit distances

```bash
python Scripts/compute_ngram_edit_distance.py \
  --input Transcripts/ \
  --reference Data/reference_solutions.csv \
  --output Data/edit_distances.csv
```

#### b. Generate summary tables

```bash
python Scripts/summarize_edit_distances.py \
  --input Data/edit_distances.csv \
  --output Data/summary_stats.csv
```

#### c. Plot figures from the paper

```bash
python Scripts/plot_results.py \
  --input Data/summary_stats.csv \
  --output figures/
```

---

## Repository Structure

```
├── Data/
│   ├── reference_solutions.csv
│   ├── prompts_responses.csv
│   └── edit_distances.csv
│
├── Scripts/
│   ├── compute_ngram_edit_distance.py
│   ├── summarize_edit_distances.py
│   └── plot_results.py
│
├── Transcripts/
│   └── <LLM_name>_<task>.json
│
└── Weber_et_al__LLM_Decision_Making_submissions_to_CEUR_Wksp_Proc__CEUR_WS_org.pdf
```

---

## Contributing

Feel free to open issues or pull requests for:

* Adding new decision-making benchmarks
* Improving scripts (efficiency, docs, tests)
* Extending to additional LLMs or metrics

---

## Citation

If you use this code in your work, please cite our paper:

> Weber, Rosina O., Christopher B. Rauch, and Savar Amin. “Decision Making in LLMs: First Step.” *Proceedings of the 2nd Workshop on Case-Based Reasoning and Large Language Model Synergies (CBR-LLM) at ICCBR 2025*, Biarritz, France, June 30 2025. CEUR Workshop Proceedings, 2025.&#x20;


```
```
