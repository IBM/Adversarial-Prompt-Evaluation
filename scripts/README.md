# Experiments launch scripts
This README contain the instruction to reproduce each single code of "Adversarial Prompt Evaluation: Systematic
Benchmarking of Guardrails Against Prompt Input
Attacks on LLMs" paper for NeurIPS Datasets and Benchmark track.

## Training
Following the instruction to reproduce the traing of: 

- ### simple uni-gram
```
python3 main_n_gram_classifier_train.py
```

- ### Bert:
```
python main_classification_fine_tuning.py  --model_name bert --model_name_or_path bert
```

- ### Deberta:
```
python main_classification_fine_tuning.py  --model_name deberta --model_name_or_path deberta
```

- ### GPT2:
```
python main_classification_fine_tuning.py  --model_name gpt2 --model_name_or_path gpt2
```

N.B.: The Transformer-based classifier training generate a finetuned version of the specific model used at `scripts/results/{model_name}/run_{0 or the number of execution}/best_ES_model/`. This path should be used for the evaluation replacing `{model_specific_train_path}`.

## Evaluation
Following, the command to reproduce the evaluation script for each model:

- ### ProtectAI v1 - v2

```
python main_evaluate.py --model_name 'protectAI_v1' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json'
```

```
python main_evaluate.py --model_name 'protectAI_v1' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json'
```

```
python main_evaluate.py --model_name 'protectAI_v2' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json'
```

```
python main_evaluate.py --model_name 'protectAI_v2' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json'
```

- ### LlamaGuard and LlamaGuard2

```
python main_evaluate.py --model_name 'lamaguard' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_HF_TOKEN>
```

```
python main_evaluate.py --model_name 'lamaguard' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_HF_TOKEN>
```

```
python main_evaluate.py --model_name 'lamaguard2' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_HF_TOKEN>
```

```
python main_evaluate.py --model_name 'lamaguard2' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_HF_TOKEN>
```

-  ### Langkit Injection Detection
```
python main_evaluate.py --model_name 'langkit' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

```
python main_evaluate.py --model_name 'langkit' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

- ### Proactive
```
python main_evaluate.py --model_name 'proactive' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

```
python main_evaluate.py --model_name 'proactive' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

- ### Base Refusal

```
python main_evaluate.py --model_name 'vicuna-7b-v1.5' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

```
python main_evaluate.py --model_name 'vicuna-7b-v1.5' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

- ### SmoothLLM
```
python main_evaluate.py --model_name 'smooth_llm' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

```
python main_evaluate.py --model_name 'smooth_llm' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

- ### OpenAI Moderation

```
python main_evaluate.py --model_name 'openAI_modeation' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_OPENAI_TOKEN>
```

```
python main_evaluate.py --model_name 'openAI_modeation' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_OPENAI_TOKEN>
```

- ### PPL_Threshold

```
python main_evaluate.py --model_name 'ppl_threshold' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json'
```

```
python main_evaluate.py --model_name 'ppl_threshold' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json'
```

- ### Bert Roberta GPT2

```
python main_evaluate.py --model_name {bert/roberta/gpt2} --model_load_path {model_specific_train_path} --data_location 'sub_sample_filtered_data.json'
```

```
python main_evaluate.py --model_name {bert/roberta/gpt2} --model_load_path {model_specific_train_path} --data_location 'ood_filtered_data.json'
```

- ### Azure AI Content Safety

```
python main_evaluate.py --model_name 'AzureAPI' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_AZURE_TOKEN> --endpoint <YOUR_AZURE_CONTENT_SAFETY_RESOURCE_ENDPOINT>
```

```
python main_evaluate.py --model_name 'AzureAPI' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_AZURE_TOKEN> --endpoint <YOUR_AZURE_CONTENT_SAFETY_RESOURCE_ENDPOINT>
```
