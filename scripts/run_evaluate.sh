#!/bin/bash

data_list=("sub_sample_filtered_data.json" "ood_filtered_data.json")
model_list=("n_gram_classifier" "protectAI_v1" "protectAI_v2" "lamaguard" "lamaguard2" "langkit" "proactive"
 "vicuna-7b-v1.5" "vicuna-13b-v1.5" "smooth_llm" "ppl_threshold" "bert" "deberta" "gpt2") #"openAI_moderation"

echo "Add your HF token:"

token_hf = ""

echo "Add your OpenAI token:"

token_openAI = ""

echo "Add your AzureAPI token:"

token_azure = ""

for data in "${data_list[@]}"; do
  for model in "${model_list[@]}"; do
    python3 main_evaluate.py --model_name "$model" --data_location "$data" --token "$token_hf"

  python3 main_evaluate.py --model_name "openAI_moderation" --data_location "$data" --token "$token_openAI"

  python3 main_evaluate.py --model_name "AzureAPI" --data_location "$data" --token "$token_azure"

