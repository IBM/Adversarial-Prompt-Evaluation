# Datasets
This folder contains instructions for reproducing the datasets and splits used for the paper evaluation.

### Structure
The final dataset structure should be identical to the following:
```
- datasets
---- SAP200
-------- fraud
------------ generated_cases.json
------------ gpt_output.json
------------ info.log
-------- politics
------------ generated_cases.json
------------ gpt_output.json
------------ info.log
-------- pornography_sexual_minors
------------ generated_cases.json
------------ gpt_output.json
------------ info.log
-------- race
------------ generated_cases.json
------------ gpt_output.json
------------ info.log
-------- religion
------------ generated_cases.json
------------ gpt_output.json
------------ info.log
-------- suicide
------------ generated_cases.json
------------ gpt_output.json
------------ info.log
-------- terrorism
------------ generated_cases.json
------------ gpt_output.json
------------ info.log
-------- violence
------------ generated_cases.json
------------ gpt_output.json
------------ info.log
---- aart-v1-20231117.csv
---- gcg_vicuna_7bv1.5.csv
---- harmful_behaviors.csv
---- jailbreak_prompts.csv
---- MaliciousInstruct.txt
---- super_natural_instructions.csv
---- tap.csv
```

### Obtaining the datasets

- __SAP200__ - https://github.com/Aatrox103/SAP/tree/main/datasets/SAP200
- __aart-v1-20231117__ - https://github.com/google-research-datasets/aart-ai-safety-dataset/blob/main/aart-v1-20231117.csv
- __gcg_vicuna_7bv1.5__<br> This dataset must be generated using the GCG [official repo](https://github.com/llm-attacks/llm-attacks/tree/main). You should format the output of the GCG algorithm to be a csv file where the harmful prompt with adversarial suffix applied is in a column named "prompt".
- __harmful_behaviors__ - https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv
- __jailbreak_prompts__ - https://github.com/verazuo/jailbreak_llms/tree/main/data/prompts
- __MaliciousInstruct__ - https://github.com/Princeton-SysML/Jailbreak_LLM/blob/main/data/MaliciousInstruct.txt
- __super_natural_instructions__<br>This dataset is compiled from the [official repo](https://github.com/allenai/natural-instructions/tree/master/tasks). For every file, only the "Definition" instruction is extracted from the json and added to a csv with column "prompts".
- __tap__<br>
This dataset must be generated using the TAP [official repo](https://github.com/RICommunity/TAP). You should format the output of the TAP algorithm to be a csv file  where the adversarial prompt is in a column named "prompt".