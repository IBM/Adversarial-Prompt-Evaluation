# Experiments launch scripts

## Training

Bert:
```
python main_classification_fine_tuning.py  --model_name bert --model_name_or_path bert
```
Deberta:
```
python main_classification_fine_tuning.py  --model_name deberta --model_name_or_path deberta
```
GPT2:
```
python main_classification_fine_tuning.py  --model_name gpt2 --model_name_or_path gpt2
```

## Evaluation
Following, the command to reproduce the evaluation script for each model.

## ProtectAI

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

## LlamaGuard

```
python main_evaluate.py --model_name 'lamaguard' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_HF_TOKEN>
```

```
python main_evaluate.py --model_name 'lamaguard' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_HF_TOKEN>
```

## LlamaGuard2 

```
python main_evaluate.py --model_name 'lamaguard2' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_HF_TOKEN>
```

```
python main_evaluate.py --model_name 'lamaguard2' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_HF_TOKEN>
```

## Langkit Injection Detection
```
python main_evaluate.py --model_name 'langkit' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

```
python main_evaluate.py --model_name 'langkit' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

## Proactive
```
python main_evaluate.py --model_name 'proactive' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

```
python main_evaluate.py --model_name 'proactive' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

## Base Refusal

```
python main_evaluate.py --model_name 'vicuna-7b-v1.5' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

```
python main_evaluate.py --model_name 'vicuna-7b-v1.5' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

## SmoothLLM
```
python main_evaluate.py --model_name 'smooth_llm' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

```
python main_evaluate.py --model_name 'smooth_llm' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' 
```

## OpenAI Moderation

```
python main_evaluate.py --model_name 'openAI_modeation' --data_location 'sub_sample_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_OPENAI_TOKEN>
```

```
python main_evaluate.py --model_name 'openAI_modeation' --data_location 'ood_filtered_data.json' --config_location 'configs/neurips_config.json' --token <YOUR_OPENAI_TOKEN>
```


