#!/bin/bash

python3 main_n_gram_classifier_train.py

python3 main_classification_fine_tuning.py --model_name bert --model_path_load bert-base-cased

python3 main_classification_fine_tuning.py --model_name deberta --model_path_load microsoft/deberta-v3-base

python3 main_classification_fine_tuning.py --model_name gpt2 --model_path_load gpt2

