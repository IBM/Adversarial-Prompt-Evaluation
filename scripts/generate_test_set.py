"""
Script to generate a consistently sampled test set for comparing different methods and approaches.
"""

import sys
sys.path.append("../src/")
import os
import pandas as pd
import json

from ape.utils.datasets_preprocessing import data_processing
from ape.utils import TestLoader


def benchmark_test_set_generation(save_paths: dict, random_state: int = 1234, samples_to_fetch: int = 200) -> None:
    """
    Simple function which loads the data, samples N prompts from each sub-dataset, and saves them.

    This is intended to provide a final analysis suite to compare different

    NB: this excludes the hackaprompt and red_team_attempts data, either due to poor quality (hackaprompt) or
    requiring more manual cleaning (red_team_attempts).

    :param save_paths: Save locations for the files
    :param random_state: Random seed for pandas sampler
    :param samples_to_fetch: Number of prompts to sample from the test data

    :return: None
    """
    
    config_dic = json.load(open('./configs/neurips_config.json'))
    
    # List of datasets excluding hackaprompt and red_team_attempts
    datasets_to_sample_from = config_dic["test_datasets"]

    assert "hackaprompt" not in datasets_to_sample_from
    assert "red_team_attempts" not in datasets_to_sample_from

    if random_state != 1234:
        print("\033[1;31mUsing different to default random seed. Sampled data will differ.\033[0;0m")

    # Use all the datasets and then filter: ensures consistency between cross dataset duplicates.
    data = data_processing(include_ood=True)
    df = pd.DataFrame(
        data={
            "prompt": data["x_train"] + data["x_val"] + data["x_test"],
            "label": data["y_train"] + data["y_val"] + data["y_test"],
            "source": data["source_train"] + data["source_val"] + data["source_test"],
        },
        columns=["prompt", "label", "source"],
    )

    # Ensures consistency in come corner cases
    df.sort_values(by=["prompt", "source"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = df[df.source.isin(datasets_to_sample_from)]

    save_and_check_hash(df, data_type="all", save_paths=save_paths)

    # Create Sub-Selection
    df = pd.DataFrame(
        data={"prompt": data["x_test"], "label": data["y_test"], "source": data["source_test"]},
        columns=["prompt", "label", "source"],
    )
    df = df[df.source.isin(datasets_to_sample_from)]
    save_and_check_hash(df, data_type="test", save_paths=save_paths)
    df_filtered = df[df.apply(lambda x: (len(x["prompt"]) <= 1000), axis=1)]

    for sample_type, data_ in [("sub_sample", df), ("sub_sample_filtered", df_filtered)]:
        sub_sampled = None
        for _, sub_df in data_.groupby("source"):
            if len(sub_df) > samples_to_fetch:
                sub_df = sub_df.sample(n=samples_to_fetch, random_state=random_state)

            if sub_sampled is None:
                sub_sampled = sub_df
            else:
                sub_sampled = pd.concat([sub_sampled, sub_df])
        print(f"Total size of {sample_type}_data is {len(sub_sampled)}")

        save_and_check_hash(sub_sampled, data_type=sample_type, save_paths=save_paths)

    # Create OOD
    ood_df = pd.DataFrame(
        data={"prompt": data["x_ood"], "label": data["y_ood"], "source": data["source_ood"]},
        columns=["prompt", "label", "source"],
    )
    save_and_check_hash(ood_df, data_type="ood", save_paths=save_paths)
    ood_df_filtered = ood_df[ood_df.apply(lambda x: (len(x["prompt"]) <= 1000), axis=1)]
    save_and_check_hash(ood_df_filtered, data_type="ood_filtered", save_paths=save_paths)


def save_and_check_hash(df: pd.DataFrame, data_type: str, save_paths: dict) -> None:
    """
    Saves the data and checks the file hash

    :param df: Dataframe to save
    :param data_type: The type of data to save (one of "all", "sub_sample", "ood", "risk_eval").
    :param save_paths: Save file location.

    :return: None
    """
    if data_type not in ["all", "test", "sub_sample", "sub_sample_filtered", "ood", "ood_filtered", "risk_eval"]:
        raise ValueError(f"{data_type} data_type invalid for hash selection")

    df.to_json(os.path.join(save_paths[data_type], data_type + "_data.json"), orient="records", indent=4, index=False)

    h = TestLoader.hash_file(os.path.join(save_paths[data_type], data_type + "_data.json"))

    print(f"Computed hash for {data_type}: {h}")
    if h != TestLoader.expected_hashes["expected_hash_" + data_type]:
        print(
            f"\033[1;31mHash for {data_type} json does not match! If this is not expected, "
            "check the datasets being loaded and samples fetched\033[0;0m"
        )


if __name__ == "__main__":
    benchmark_test_set_generation(
        save_paths={"all": "./", "test": "./", "sub_sample": "./", "sub_sample_filtered": "./", "ood": "./", "ood_filtered": "./"},
        random_state=1234,
        samples_to_fetch=200,
    )
