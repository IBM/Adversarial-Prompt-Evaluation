"""
Script for fetching and standardising the datasets used.
"""

import json
import os
import re
from collections import Counter
from itertools import compress
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets


class LLMPromptsDataset(torch.utils.data.Dataset):
    """
    Dataloader for LLM prompts and jailbreaks
    """

    def __init__(self, encodings, labels: np.ndarray, datapoint_index=None):
        """
        :param encodings:
        :param labels:
        :param datapoint_index:
        """
        self.encodings = encodings
        self.labels = labels
        self.datapoint_index = datapoint_index

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.datapoint_index is not None:
            item["datapoint_index"] = torch.tensor(self.datapoint_index[idx])
        return item

    def __len__(self):
        return len(self.labels)


def filter_data(prompts: List[str], info: bool = False) -> List[str]:
    """
    Removes blank entries and duplicate strings

    :param prompts: prompts to filter for blanks and duplicates
    :param info: if to display additional debug info

    :return: dataset with duplicates and blanks removed
    """

    while "" in prompts:
        prompts.remove("")

    while np.nan in prompts:
        prompts.remove(np.nan)

    if info:
        counter = Counter(prompts)
        max_repetition = max(counter, key=counter.get)
        if counter[max_repetition] > 1:
            print(f"Most repeated string is: {max_repetition} with {counter[max_repetition]} duplicates")

    num_samples = len(prompts)
    no_duplicate_prompts = sorted(list(set(prompts)))
    if len(no_duplicate_prompts) < num_samples:
        print(f"{num_samples - len(no_duplicate_prompts)} duplicates removed")

    return no_duplicate_prompts


def filter_combined_data(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, ood: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Filter duplicate data within same subset
    :train: train split of dataset
    :validation: validation split of dataset
    :test: test split of dataset
    :ood: out of distribution data of interest

    :returns: dictionary containing the data with the duplicate datapoints removed
    """
    train["subtype"] = "train"
    validation["subtype"] = "validation"
    test["subtype"] = "test"
    if ood is not None:
        ood["subtype"] = "ood"
        combined_data = pd.concat([train, validation, test, ood])
    else:
        combined_data = pd.concat([train, validation, test])

    num_samples = len(combined_data)
    combined_data = combined_data.drop_duplicates(subset="text", keep="first")
    print(f"Removed {num_samples - len(combined_data)} from cross-dataset duplicates")

    train = combined_data.loc[combined_data["subtype"] == "train"]
    validation = combined_data.loc[combined_data["subtype"] == "validation"]
    test = combined_data.loc[combined_data["subtype"] == "test"]

    outputs: Dict = {}

    for df, subtype in zip([train, validation, test], ["train", "val", "test"]):
        samples = df["text"].astype(str).values.tolist()
        labels = df["label"].values.tolist()
        source = df["source"].values.tolist()
        outputs = outputs | {"x_" + subtype: samples, "y_" + subtype: labels, "source_" + subtype: source}

    if ood is not None:
        df = combined_data.loc[combined_data["subtype"] == "ood"]
        outputs = outputs | {
            "x_ood": df["text"].astype(str).values.tolist(),
            "y_ood": df["label"].values.tolist(),
            "source_ood": df["source"].values.tolist(),
        }

    return outputs


def get_jailbreak_llms_prompts(
    file_name: str = "../datasets/jailbreak_prompts.csv",
) -> Tuple[List[str], np.ndarray, str]:
    """
    Get the jailbreak llm prompts and filter them for
    placeholder values such as [Insert Prompt Here]

    :param file_name: file path of the jailbreak prompts

    """
    df = pd.read_csv(file_name)
    prompts = df["prompt"]
    filtered_prompts = []
    for p in tqdm(prompts):
        if "insert" in p.lower() and ("[" in p.lower() or "{" in p.lower()):
            start_prompt_insertion = None
            end_prompt_insertion = None

            start_filter = p.lower().index("insert")
            end_filter = p.lower().index("insert")

            while start_prompt_insertion is None or end_prompt_insertion is None:
                if p[start_filter] in "[{" and start_prompt_insertion is None:
                    start_prompt_insertion = start_filter
                else:
                    start_filter -= 1
                if p[end_filter] in "]}" and end_prompt_insertion is None:
                    end_prompt_insertion = end_filter
                else:
                    end_filter += 1

                if end_filter == len(p) or start_filter == -1:
                    break

            if start_prompt_insertion is not None and end_prompt_insertion is not None:
                if end_prompt_insertion - start_prompt_insertion < 50:
                    p = p[:start_prompt_insertion] + p[end_prompt_insertion + 1 :]
            filtered_prompts.append(p)
        else:
            filtered_prompts.append(p)

    filtered_prompts = filter_data(filtered_prompts)
    return filtered_prompts, np.ones((len(filtered_prompts),)), "jailbreak_prompts"


def get_harmful_behaviours(file_name: str = "../datasets/harmful_behaviors.csv") -> Tuple[List[str], np.ndarray, str]:
    """
    Get the harmful behaviour prompts.

    :param file_name: file path of the jailbreak prompts
    """
    df = pd.read_csv(file_name)
    prompts = df["goal"].values.tolist()
    prompts = filter_data(prompts)
    return prompts, np.ones((len(prompts),)), "harmful_behaviours"


def get_puffin_dataset() -> Tuple[List[str], np.ndarray, str]:
    """
    Gets the Puffin dataset which was used in https://openreview.net/pdf?id=lNLVvdHyAw
    https://huggingface.co/datasets/LDJnr/Puffin
    Apache Licence
    """

    regular_dataset = load_dataset("LDJnr/Puffin")["train"]  # Only has the train tag.
    prompts = []
    for conversation in regular_dataset["conversations"]:
        for sample in conversation:
            if sample["from"] == "human":
                prompts.append(sample["value"])
    prompts = filter_data(prompts)
    return prompts, np.zeros((len(prompts),)), "puffin"


def get_ignore_instructions_dataset() -> Tuple[List[str], np.ndarray, str]:
    """
    Gets the Gandalf ignore instructions dataset

    https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions
    MIT Licence.
    """
    dataset = load_dataset("Lakera/gandalf_ignore_instructions")
    train_samples = dataset["train"]["text"]
    valid_samples = dataset["validation"]["text"]
    test_samples = dataset["test"]["text"]

    # For convenience group everything together now
    all_samples = train_samples + valid_samples + test_samples
    prompts = filter_data(all_samples)
    return prompts, np.ones((len(prompts),)), "gandalf_ignore_instructions"


def get_hackaprompt_dataset() -> Tuple[List[str], np.ndarray, str]:
    """
    https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset
    MIT Licence.
    """
    dataset = load_dataset("hackaprompt/hackaprompt-dataset")["train"]  # Only has the train tag.
    prompts = []
    for sample in dataset:
        # level 1 was a practice round, and filter for only the successful samples
        if sample["level"] > 1 and sample["completion"] == sample["expected_completion"]:
            prompts.append(sample["user_input"])

    prompts = filter_data(prompts)
    return prompts, np.ones((len(prompts),)), "hackaprompt"


def get_awesome_chatgpt_prompts_dataset() -> Tuple[List[str], np.ndarray, str]:
    """
    https://huggingface.co/datasets/fka/awesome-chatgpt-prompts
    cc0 Licence
    """
    dataset = load_dataset("fka/awesome-chatgpt-prompts")["train"]  # Only has the train tag.
    prompts = filter_data(dataset["prompt"])
    return prompts, np.zeros((len(prompts),)), "awesome_chatgpt_prompts"


def get_no_robots_dataset() -> Tuple[List[str], np.ndarray, str]:
    """
    https://huggingface.co/datasets/HuggingFaceH4/no_robots
    cc by nc Licence
    """
    dataset = load_dataset("HuggingFaceH4/no_robots")
    train_samples = dataset["train"]["prompt"]
    test_samples = dataset["test"]["prompt"]
    # For convenience group everything together now
    all_samples = train_samples + test_samples
    prompts = filter_data(all_samples)
    return prompts, np.zeros((len(prompts),)), "no_robots"


def get_ultrachat_dataset() -> Tuple[List[str], np.ndarray, str]:
    """
    https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
    MIT Licence.
    """
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
    prompts = dataset["train_gen"]["prompt"]  # Take a subset for now
    prompts = filter_data(prompts)
    return prompts, np.zeros(len(prompts)), "ultrachat"


def get_aart_dataset(file_name: str = "../datasets/aart-v1-20231117.csv") -> Tuple[List[str], np.ndarray, str]:
    """
    https://github.com/google-research-datasets/aart-ai-safety-dataset/blob/main/aart-v1-20231117.csv
    CC by 4.0
    """
    df = pd.read_csv(file_name)
    prompts = df["prompt"].astype(str).values.tolist()
    prompts = filter_data(prompts)
    return prompts, np.ones(len(prompts)), "aart"


def get_do_not_answer_dataset() -> Tuple[List[str], np.ndarray, str]:
    """
    https://huggingface.co/datasets/LibrAI/do-not-answer
    MIT License
    """
    dataset = load_dataset("LibrAI/do-not-answer")
    prompts = dataset["train"]["question"]
    prompts = filter_data(prompts)
    return prompts, np.ones(len(prompts)), "donotanswer"


def get_gcg_vicuna_dataset(file_name: str = "../datasets/gcg_vicuna_7bv1.5.csv") -> Tuple[List[str], np.ndarray, str]:
    """
    Get GCG adversarial attack prompts for AdvBench using vicuna-7b-v1.5

    :param file_name: file path of the GCG attack prompts
    """
    df = pd.read_csv(file_name)
    prompts = df["adv_prompt"].values.tolist()
    prompts = filter_data(prompts)
    return prompts, np.ones((len(prompts),)), "gcg_attacks_vicuna_v1.5"


def get_tap_dataset(file_name: str = "../datasets/tap.csv") -> Tuple[List[str], np.ndarray, str]:
    """
    Get TAP adversarial attack prompts for AdvBench

    :param file_name: file path of the TAP attack prompts
    """
    df = pd.read_csv(file_name)
    prompts = df["prompt"].values.tolist()
    prompts = filter_data(prompts)
    return prompts, np.ones((len(prompts),)), "tap"

def get_sap_dataset(path_name: str = "../datasets/SAP200") -> Tuple[List[str], np.ndarray, str]:
    """
    Get SAP adversarial attack prompts dataset from link
    https://github.com/Aatrox103/SAP/tree/main/datasets
    Apache License Version 2.0

    :param path_name: path of the SAP attack prompts dataset directory
    """

    categories = [
        "fraud",
        "politics",
        "pornography_sexual_minors",
        "race",
        "religion",
        "suicide",
        "terrorism",
        "violence",
    ]
    file_name = "generated_cases.json"
    prompts = []
    for p_direct in categories:
        complete_path = os.path.join(path_name, p_direct, file_name)
        assert os.path.exists(complete_path), f"File does not exist: {complete_path}"
        with open(complete_path, mode="r", encoding="utf-8") as file:
            data = json.load(file)
            for row in data:
                match = re.search(r"###(.*?)###", row, re.DOTALL)
                if match is not None:
                    prompts.append(match.group(1).strip())
    prompts = filter_data(prompts)
    return prompts, np.ones((len(prompts),)), "sap"


def get_red_team_attempts_dataset(merge_human_responses=True) -> Tuple[List[str], np.ndarray, str]:
    """
    MIT license

    https://arxiv.org/pdf/2209.07858.pdf
    https://github.com/anthropics/hh-rlhf/tree/master?tab=readme-ov-file

    :param merge_human_responses: If to merge all the text in a human-AI interaction event into a single sample.
                                  Several individual messages between the human and the AI are only harmful
                                  given the whole content of the message. If we would like to use individual
                                  messages as datapoints we need to implement a filtering list to remove the
                                  messages which stand-alone do not have harmful behaviour
                                  (E.g. "Okay", "yes", "I saw a video about it on YouTube.", etc.)
    """
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")

    dataset = dataset["train"]["transcript"]  # Only train tag
    prefix_size = len("Human: ")
    assistant_prefix_size = len("Assistant: ")

    prompts = []
    for sample in tqdm(dataset):
        sub_prompts = []
        start = 0
        while start >= 0 and len(sample) > 0:
            start = sample.find("Human: ")
            end = sample.find("Assistant: ")
            if start >= 0:
                if merge_human_responses:
                    sub_prompts.append(sample[start + prefix_size : end].strip())
                else:
                    prompts.append(sample[start + prefix_size : end].strip())

                sample = sample[end + assistant_prefix_size :]

        if merge_human_responses:
            human_responses = " ".join(sub_prompts)
            prompts.append(human_responses)

    if not merge_human_responses:
        # We can look through the red_team_raw.csv to develop a filter if needed.
        df = pd.DataFrame(data={"col1": prompts})
        df.to_csv("red_team_raw.csv", sep=",", index=False)
    prompts = filter_data(prompts)
    return prompts, np.ones((len(prompts),)), "red_team_attempts"


def get_boolq_dataset() -> Tuple[List[str], np.ndarray, str]:
    """
    https://huggingface.co/datasets/google/boolq
    License: cc-by-sa-3.0
    """
    dataset = load_dataset("google/boolq")
    train_samples = dataset["train"]["question"]
    test_samples = dataset["validation"]["question"]
    all_samples = train_samples + test_samples
    prompts = filter_data(all_samples)
    return prompts, np.zeros((len(prompts),)), "boolq"


def get_alpaca_dataset() -> Tuple[List[str], np.ndarray, str]:
    """
    https://huggingface.co/datasets/tatsu-lab/alpaca
    License: cc-by-nc-4.0
    """
    dataset = load_dataset("tatsu-lab/alpaca")
    train_samples = dataset["train"]
    df = pd.DataFrame(train_samples)
    full_prompts = (df["instruction"] + " " + df["input"]).values.tolist()
    prompts = filter_data(full_prompts)
    return prompts, np.zeros((len(prompts),)), "alpaca"


def get_super_natural_instructions_dataset(
    file_name: str = "../datasets/super_natural_instructions.csv",
) -> Tuple[List[str], np.ndarray, str]:
    """
    Get super natural instructions dataset.
    Paper: https://arxiv.org/abs/2204.07705
    License: Apache 2.0

    :param file_name: file path of the prompts
    """
    df = pd.read_csv(file_name)
    prompts = df["prompts"].values.tolist()
    prompts = filter_data(prompts)
    return prompts, np.zeros((len(prompts),)), "super_natural_instructions"


def get_attaq_dataset() -> Tuple[List[str], np.ndarray, str]:
    """
    MIT Licence
    https://huggingface.co/datasets/ibm/AttaQ
    """
    dataset = load_dataset("ibm/AttaQ")
    all_prompts = dataset["train"]["input"]
    additional_info = dataset["train"]["additional_info"]
    filter_indices = ["anthropics/hh-rlhf/red-team-attempts" in e for e in additional_info]
    prompts = list(compress(all_prompts, filter_indices))
    prompts = filter_data(prompts)
    return prompts, np.ones((len(prompts),)), "attaq"


def get_xstest_dataset(conservative_labels: bool = True) -> Tuple[List[str], np.ndarray, str]:
    """
    Fetches the XSTest Dataset for Testing Exaggerated Safety: https://arxiv.org/abs/2308.01263

    Dataset contains a mix of benign and adversarial prompts.

    CC-by-4.0 Licence
    https://huggingface.co/datasets/natolambert/xstest-v2-copy

    :param conservative_labels: Conservatively label the dataset in favour of maliciousness.
                                Original paper had 250 safe prompts and 200 malicious.
                                However, some safe prompts can be considered malicious. Furthermore, these edge
                                cases can have an overall deterioration in performance if not accompanied by a rich set
                                of data: i.e. prompts such as "Why should cucumbers be barred from practicing law?"
                                can lead a detector to classify prompts like:
                                "Why should <group> be barred from practicing law?" as safe which
                                is not a desirable property.
    """
    dataset = load_dataset("natolambert/xstest-v2-copy")
    types = dataset["prompts"]["type"]
    prompts = dataset["prompts"]["prompt"]
    data_types = {key: item for key, item in zip(prompts, types)}
    prompts = filter_data(prompts)
    types = [data_types[p] for p in prompts]
    labels = np.ones(
        len(prompts),
    )
    for i, label in enumerate(types):
        if "contrast" in label:
            labels[i] = 1.0
        elif label in ["real_group_nons_discr", "nons_group_real_discr", "privacy_fictional"] and conservative_labels:
            labels[i] = 1.0
        else:
            labels[i] = 0.0

    if conservative_labels:
        assert np.sum(labels) == 275
    else:
        assert np.sum(labels) == 200
    assert len(labels) == 450

    return prompts, labels, "xstest"


def get_toxicchat_dataset() -> Tuple[List[str], np.ndarray, str]:
    """
    CC-by-NC 4.0 Licence
    https://huggingface.co/datasets/lmsys/toxic-chat
    """
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    prompts = []
    for d in concatenate_datasets([dataset["test"], dataset["train"]]):
        if d["jailbreaking"] == 1:
            prompts.append(d["user_input"])
    prompts = filter_data(prompts)
    return prompts, np.ones((len(prompts),)), "toxicchat"


def get_malicious_instruct_dataset(
    file_name: str = "../datasets/MaliciousInstruct.txt",
) -> Tuple[List[str], np.ndarray, str]:
    """
    Get Malicious Instruct dataset
    https://github.com/Princeton-SysML/Jailbreak_LLM/blob/main/data/MaliciousInstruct.txt
    :param file_name: file path of the MaliciousInstruct.txt file
    """
    df = pd.read_csv(file_name, names=["prompt"])
    prompts = df["prompt"].values.tolist()
    prompts = filter_data(prompts)
    return prompts, np.ones((len(prompts),)), "malicious_instruct"


SUPPORTED_DATASETS: Dict = {
    "jailbreak_prompts": get_jailbreak_llms_prompts,
    "puffin": get_puffin_dataset,
    "gandalf_ignore_instructions": get_ignore_instructions_dataset,
    "awesome_chatgpt_prompts": get_awesome_chatgpt_prompts_dataset,
    "harmful_behaviours": get_harmful_behaviours,
    "no_robots": get_no_robots_dataset,
    "hackaprompt": get_hackaprompt_dataset,
    "ultrachat": get_ultrachat_dataset,
    "gcg_vicuna_7bv1.5": get_gcg_vicuna_dataset,
    "do_not_answer": get_do_not_answer_dataset,
    "aart": get_aart_dataset,
    "sap": get_sap_dataset,
    "red_team_attempts": get_red_team_attempts_dataset,
    "tap": get_tap_dataset,
    "boolq": get_boolq_dataset,
    "alpaca": get_alpaca_dataset,
    "super_natural_instructions": get_super_natural_instructions_dataset,
    "attaq": get_attaq_dataset,
    "xstest": get_xstest_dataset,
}


OOD_SUPPORTED_DATASETS: Dict = {
    "toxicchat": get_toxicchat_dataset,
    "malicious_instruct": get_malicious_instruct_dataset,
}

def data_processing(
    datasets: Optional[List[str]] = None,
    val_split: float = 0.2,
    test_split: float = 0.2,
    test_data_filter: Optional[List[str]] = None,
    random_seed: int = 77,
    include_ood: bool = False,
) -> Dict:
    """
    Prepare train, validation and test datasets

    Parameters
    --------
    :param datasets: List[str] of supported datasets to include train, validation and test
    :param val_split: float indicating the validation dataset split
    :param test_split: float indicating the test dataset split
    :param test_data_filter: List[str] of (dataset_name) filtering datasets that are permitted in test set
    :param random_seed: seed for reproducible datasets
    :param include_ood: bool indicating to include out-of-distribution datasets

    Returns
    --------
    Dict containing prepared data samples and metadata
    """

    def compose_dataframe(df: pd.DataFrame, sub_data, sub_labels, name: str) -> pd.DataFrame:
        subset = [sub_data, sub_labels, [name] * len(sub_data)]
        subset_df = pd.DataFrame(map(list, zip(*subset)), columns=["text", "label", "source"])
        return pd.concat([df, subset_df])

    train: pd.DataFrame = pd.DataFrame()
    validation: pd.DataFrame = pd.DataFrame()
    test: pd.DataFrame = pd.DataFrame()

    if datasets is None:
        datasets = list(SUPPORTED_DATASETS.keys())
        data_to_fetch = list(SUPPORTED_DATASETS.values())
    else:
        data_to_fetch = [SUPPORTED_DATASETS[name] for name in datasets]

    for dataset_name, data_fetcher in zip(datasets, data_to_fetch):
        print(f"Loading {dataset_name}")
        subset_data, subset_labels, name = data_fetcher()

        subset_train, subset_test, subset_train_labels, subset_test_labels = train_test_split(
            subset_data, subset_labels, test_size=test_split, random_state=random_seed
        )
        subset_train, subset_validation, subset_train_labels, subset_validation_labels = train_test_split(
            subset_train, subset_train_labels, test_size=val_split, random_state=random_seed
        )

        train = compose_dataframe(df=train, sub_data=subset_train, sub_labels=subset_train_labels, name=dataset_name)
        validation = compose_dataframe(
            df=validation, sub_data=subset_validation, sub_labels=subset_validation_labels, name=dataset_name
        )
        test = compose_dataframe(df=test, sub_data=subset_test, sub_labels=subset_test_labels, name=dataset_name)

    if test_data_filter is None:
        test = test[
            test.source.isin(
                [
                    "gcg_vicuna_7bv1.5",
                    "harmful_behaviours",
                    "hackaprompt",
                    "gandalf_ignore_instructions",
                    "red_team_attempts",
                    "jailbreak_prompts",
                    "puffin",
                    "awesome_chatgpt_prompts",
                    "no_robots",
                    "do_not_answer",
                    "aart",
                    "ultrachat",
                    "tap",
                    "sap",
                    "attaq",
                    "xstest",
                    "boolq",
                    "alpaca",
                    "super_natural_instructions",
                    "tap_mixtral",
                ]
            )
        ]
    else:
        test = test[test.source.isin(test_data_filter)]

    num_samples = len(train) + len(validation) + len(test)
    harmful = train.label.sum() + validation.label.sum() + test.label.sum()

    print("----------------------------")
    print("     Dataset Balance        ")
    print(f"Total Samples: {num_samples}")
    print(f"Jailbreaks/Harmful: {harmful}")
    print(f"Regular: {num_samples - harmful}")
    print("----------------------------")

    # Assert that samples do not overlap in case of duplicates existing across datasets
    # Note: this will result in small sample difference if loading all the data simultaneously
    # vs loading individual datasets which will not filter for cross-dataset duplicates.
    datasets.sort()
    all_supported_datasets = list(SUPPORTED_DATASETS.keys())
    all_supported_datasets.sort()

    if include_ood and datasets != all_supported_datasets:
        all_data_loaded = False
        print(
            "\033[1;31mNot all regular data was loaded: OOD data may have overlaps with non-loaded data. "
            "Not returning OOD samples.\033[0;0m"
        )
    else:
        all_data_loaded = True

    if include_ood and all_data_loaded:
        # Out of distribution datasets for testing
        ood_df: pd.DataFrame = pd.DataFrame()

        for dataset_name, data_fetcher in OOD_SUPPORTED_DATASETS.items():
            print(f"Loading {dataset_name}")
            subset_data, subset_labels, name = data_fetcher()
            ood_df = compose_dataframe(df=ood_df, sub_data=subset_data, sub_labels=subset_labels, name=name)
        outputs = filter_combined_data(train, validation, test, ood_df)
    else:
        outputs = filter_combined_data(train, validation, test)

    # Note: currently not returning OOD datasets as pat of the dataset list
    return outputs | {"dataset_names": datasets}
