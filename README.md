
# Decision-Making with Large Language Models

This repository contains all the data, transcripts, and analysis scripts supporting the experiments described in:

> Weber, Rosina O., Christopher B. Rauch, and Savar Amin. “Decision Making in LLMs: First Step.” *Proceedings of the 2nd Workshop on Case-Based Reasoning and Large Language Model Synergies (CBR-LLM) at ICCBR 2025*, Biarritz, France, June 30 2025. CEUR Workshop Proceedings, 2025.&#x20;
> CEUR Workshop Proceedings.**  
> [Read the paper](./CRC%20Weber_Rauch%20Amin%20LLM_Decision_Making.pdf)



---

## Overview
This project investigates whether modern large language models demonstrate an understanding of how problems and solutions are connected, using Simon’s decision-making model as a framework (Intelligence -> Design -> Choice). Specifically, we examine/assess:
- Whether LLMs can distinguish between listing steps to address a problem (Design) and describing how those steps solve the problem - i.e., whether they go beyond instruction to show a causal connection
### Method
We prompted six recent LLMs - Claude 3.7, GPT-4.1, GPT-4o, GPT-o3, Gemini 1.5 Pro, and Gemini 2.0 Flash - across four domains: car repair, human pain, computer faults, and hiring decisions.
For each case:
1. **First prompt**: “What should I do?”
2. **Second prompt**: “How will these steps solve my problem?”
We:
- Logged all responses
- Computed 3-gram edit distances between each model’s first and second responses to evaluate textual overlap
  - Low edit distance indicates repeated content
  - Higher edit distance may indicate more detailed or contextual explanation connecting the steps to the problem
We then:
- Aggregated statistics (average, min, max) across all models and domains
- Manually analyzed sample responses to evaluate whether any connections between problems and solutions were meaningfully made

**Additional Analysis**:
We explored the Choice step through custom hypothetical scenarios involving constraints (e.g., unavailable parts, alien invasions), to observe whether models could adapt and recommend the most context-appropriate solution 

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
└── CRC Weber_Rauch Amin LLM_Decision_Making.pdf
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
