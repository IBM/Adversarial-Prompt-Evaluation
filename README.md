# Adversarial-Prompt-Evaluation

## Overview

This repo contain the source code of "Adversarial Prompt Evaluation: Systematic
Benchmarking of Guardrails Against Prompt Input
Attacks on LLMs" paper for NeurIPS Datasets and Benchmark track. 

## Abstract

As Large Language Models (LLMs) become more integrated into everyday applications, ensuring their robustness and security is increasingly critical.
In particular, LLMs can be manipulated into unsafe behaviour by prompts known as jailbreaks. The variety of jailbreak styles is growing, necessitating the use of external defenses known as guardrails or moderators. While many jailbreak defences have been proposed, not all defences are able to handle new out-of-distribution attacks due to the narrow segment of jailbreaks used to align them.
Moreover, the lack of systematisation around defences has created significant gaps in their practical application.
In this work, we perform a systematic benchmarking across 18 different defences considering a broad swathe of malicious and benign datasets.
We find that there is significant performance variation depending on the style of jailbreak a defence is subject to.
Additionally, we show that based on current datasets available for evaluation, simple baselines can display competitive out-of-distribution performance compared to many state-of-the-art defences.


## Setup Instructions

Following, the instruction to install the correct packages for running the experiments. run:
```bash
#(Python version used during experiments 3.11.7)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Getting Started with APE 🦍
Install (editable) the Adversarial Prompt Evaluation (APE) library:
```bash
pip install -e .[dev]
```

## Project Structure
This project is structured as follows:

```
├── LICENSE
├── README.md
├── requirements.txt
├── scripts
│   ├── README.md
│   ├── configs
│   │   └── neurips_config.json
│   ├── generate_test_set.py
│   ├── main_classification_fine_tuning.py
│   ├── main_evaluate.py
│   ├── main_latency_throughput.py
│   ├── main_n_gram_classifier_train.py
│   ├── run_evaluate.sh
│   └── run_train.sh
├── setup.py
└── src
    └── ape
        ├── __init__.py
        ├── detectors
        │   ├── __init__.py
        │   ├── azureAPI.py
        │   ├── base_refusal.py
        │   ├── bert_classifier.py
        │   ├── detector.py
        │   ├── langkit_detector.py
        │   ├── llm_guard.py
        │   ├── n_gram_classifier.py
        │   ├── openAi_moderation.py
        │   ├── ppl_threshold.py
        │   ├── proactive_detector.py
        │   └── smooth_llm
        │       ├── LICENSE
        │       ├── __init__.py
        │       ├── attacks.py
        │       ├── defenses.py
        │       ├── language_models.py
        │       ├── model_configs.py
        │       └── perturbations.py
        ├── metrics
        │   ├── __init__.py
        │   ├── jailbreak_keyword_asr.py
        │   ├── metric.py
        │   └── metric_computations.py
        └── utils
            ├── __init__.py
            ├── datasets_preprocessing.py
            ├── logging_utils.py
            ├── n_gram.py
            ├── ppl.py
            └── test_data_loader.py

```

The `scripts/` directory include a dedicated README for model reproducibility or `run_train.sh` and `run_evaluate.sh` for full reproduction of the work.

The `dataset/` directory include a dedicated README and python files for reproducing `tap` and `gcg` attacks.
 
## License
