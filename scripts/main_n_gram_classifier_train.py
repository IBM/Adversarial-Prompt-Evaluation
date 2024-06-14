"""
Main script for fine-tuning the BERT style classifier.
"""
import sys
sys.path.append("../src/")
import os
import pickle
import argparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import json

from ape.utils.datasets_preprocessing import data_processing
from main_classification_fine_tuning import set_seed

def main(config_dic: dict) -> None:
    """
    Main entrypoint for the training routines.
    :param config_dic: Dictionary containing the relevant configuration for the training.
    """
    pipeline = Pipeline(
        steps=[
            ("n_gram", CountVectorizer(ngram_range=(1, 1), analyzer="word")),
            ("normalizer", StandardScaler(with_mean=False)),
            ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    data = data_processing(datasets=config_dic["datasets"])
    config_dic["datasets"] = data["dataset_names"]
    pipeline.fit(data["x_train"], data["y_train"])
    if not os.path.exists(f"../models/neurips/"):
        os.makedirs(f"../models/neurips/", exist_ok=True)
    with open(f"../models/neurips/{config_dic['model_name']}.pickle", "wb") as f:
        pickle.dump(pipeline, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/neurips_config.json")
    args = parser.parse_args()

    config_dic = json.load(open(args.config_path))

    config_dic["model_name"] = "n_gram_classifier"
    config_dic["save_path"] = os.path.join("results", config_dic["model_name"])

    set_seed()
    main(config_dic)
