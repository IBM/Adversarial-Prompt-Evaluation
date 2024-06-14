"""
Main script for evaluating classifiers.
"""
import sys
sys.path.append("../src/")

import argparse
import json
import random
import time
from typing import List, Callable

import torch
import numpy as np
from tqdm import tqdm
from transformers import set_seed
from ape.utils import TestLoader

from main_evaluate import get_model


SUPPORTED_MODEL = [
    "n_gram_classifier",
    "lamaguard",
    "lamaguard2",
    "protectAI_v1",
    "protectAI_v2",
    "bert",
    "deberta",
    "gpt2",
    "langkit",
    "ppl_threshold",
    "vicunaguard",
    "vicuna_base",
    "smooth_llm",
    "vicuna_7b",
    "vicuna_13b"
]

def smooth_llm_forward(defense, prompt):    
    batch_size = 1
    if len(prompt) < 2000:
        batch_size = 5
    return defense(prompt, batch_size=batch_size)

def benchmark_test(
    model_name: str,
    model: Callable,
    x_test: List[str],
):
    """
    Obtain results for supplied model.

    :param model_name: Name of model for saving/loading results.
    :param model: Model to run data through
    :param x_test: List of prompts to obtain predictions for

    :return: None
    """
    torch.cuda.empty_cache()
    print(f"Test device: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
    print(f"Testing model: {model_name}\n")
    overall_latency = []
    overall_throughput = []
    with torch.inference_mode():
        for _ in range(10):
            case_test_latency = []
            case_test_throughput = []
            for sample in tqdm(x_test):
                time_ = time.time()

                if model_name == "smooth_llm":
                    smooth_llm_forward(model, sample)
                else:
                    model.predict([sample])
                case_test_latency.append(time.time() - time_)
                case_test_throughput.append(1.0/case_test_latency[-1])
                torch.cuda.empty_cache()

            overall_latency.append(np.mean(case_test_latency))
            overall_throughput.append(np.mean(case_test_throughput))
            
            results = {"overall_latency": overall_latency,
                       "mean_latency": np.mean(overall_latency),
                       "std_latency": np.std(overall_latency),
                       "throughput": overall_throughput,
                       "mean_throughput": np.mean(overall_throughput),
                       "std_throughput": np.std(overall_throughput)}
            with open(f"{model_name}_time_results.json", "w", encoding="utf-8") as results_file:
                json.dump(results, results_file, sort_keys=True, indent=4)


def benchmark_model(args: argparse.Namespace):
    """
    Main evaluation loop.

    :param args: Command line args specifying the evaluation.
    """
    data = TestLoader.load_test_set(filename = args.data_file, data_type="all")
    x_test = random.choices(data["prompt"], k=100)
    model = get_model(args.model_name, args.model_load_path, args.token)
    benchmark_test(args.model_name, model, x_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, choices=SUPPORTED_MODEL)
    parser.add_argument("--model_load_path", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument(
        "--data_location",
        type=str,
        default="all_data.json",
        help="Load the data from a test_set json rather than through the dataloaders",
    )
    parser.add_argument(
        "--config_location",
        type=str,
        default="configs/neurips_config.json",
        help="Load the datasets specified by a training configuration json",
    )
    # None to check all models in the SUPPORTED_MODEL list
    args = parser.parse_args()

    set_seed(42)
    benchmark_model(args=args)
