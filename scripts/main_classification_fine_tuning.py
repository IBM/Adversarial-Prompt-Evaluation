"""
Main script for fine-tuning the transformer style classifier.
"""
import sys
sys.path.append("../src/")
import os
import random
import argparse
from typing import Optional, Tuple, Union

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    GPT2Config,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
)
import json

from ape.utils.datasets_preprocessing import data_processing, LLMPromptsDataset
from ape.metrics import MetricComputations
from ape.utils import Logger

if sys.platform == "darwin":
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(_hashed_seed: int = 42):
    """
    set random seed

    Parameters
    ----------
    _hashed_seed: int
        Seed to be set for reproducibility
    """
    random.seed(_hashed_seed)
    np.random.seed(_hashed_seed)
    torch.manual_seed(_hashed_seed)
    torch.cuda.manual_seed(_hashed_seed)
    torch.cuda.manual_seed_all(_hashed_seed)


def get_model_and_tokenizer(model_name_or_path="bert", n_labels: int = 2, precision: str = "half"):
    """
    Fetches the model and tokenizer from huggingface.

    :param model_name_or_path: Model to fetch.
    :param n_labels: Number of classes for the fine-tuning
    :param precision: Precision of the weight if float32 or float16
    """
    torch_dtype = torch.float32
    if precision == "half":
        torch_dtype = torch.float16

    if "deberta" in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base", torch_dtype=torch_dtype, num_labels=n_labels
        )
        tokenizer.model_max_length = model.config.max_position_embeddings - 1
    elif "bert" in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-cased", torch_dtype=torch_dtype, num_labels=n_labels
        )
    elif "gpt2" in model_name_or_path:
        model_config = GPT2Config.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            torch_dtype=torch_dtype,
            num_labels=n_labels,
        )
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        model = GPT2ForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            config=model_config,
            torch_dtype=torch_dtype,
        )

        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


def train(
    model: transformers.models,
    train_loader: torch.utils.data.dataloader.DataLoader,
    config: dict,
    optim: torch.optim.Optimizer,
    lossfn: torch.nn.modules.loss._Loss,
    eval_loader: Optional[torch.utils.data.dataloader.DataLoader] = None,
    tokenizer=None,
) -> transformers.models:
    """
    Train the supplied model on the data with specified training parameters

    :param model: Huggingface model to train
    :param train_loader: Data loader
    :param optim: Optimizer to use
    :param lossfn: Loss function
    :param eval_loader: If to run evaluation at the end of every epoch, provide the relevant data loader here.

    :returns: Fine-tuned model.
    """
    logger = Logger(config)
    num_epochs = config["epochs"]
    best_f1 = 0.0
    count = 0

    for epoch in range(num_epochs):
        metrics = MetricComputations()
        pbar = tqdm(train_loader)
        model.train()

        for batch_num, batch in enumerate(pbar):
            optim.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = lossfn(outputs.logits, labels)

            loss.backward()
            optim.step()

            acc, f1, epoch_loss = metrics.compute(labels, outputs, loss)

            pbar.set_description(
                f"Epoch {epoch + 1}/{num_epochs}: Loss {epoch_loss:.3f} " f"Acc {acc:.3f} " f"f1 {f1:.3f}"
            )

            if batch_num % 1000 == 0 and batch_num >= 0:
                # check Early Stopping every 1000 batch per epoch
                logger.log_results(epoch, metrics, file_name="train_results.csv")
                if config["patience"]:
                    acc, f1, _ = evaluate(
                        model,
                        test_loader=eval_loader,
                        lossfn=lossfn,
                        logger=logger,
                        epoch=epoch,
                    )
                    if f1 > best_f1:
                        best_f1 = f1
                        count = 0
                        logger.save_models(model=model, tokenizer=tokenizer, opt=optim, fname="best_ES_model")
                    else:
                        count += 1
                        if count == config["patience"]:
                            break
                model.train()

        logger.log_results(epoch, metrics, file_name="train_results.csv")
        if config["patience"]:
            if count == config["patience"]:
                break
        evaluate(model, test_loader=eval_loader, lossfn=lossfn, logger=logger, epoch=epoch)
        logger.save_models(model=model, opt=optim, fname="final_model")

    return model


def evaluate(
    model: transformers.models,
    test_loader: torch.utils.data.dataloader.DataLoader,
    lossfn: torch.nn.modules.loss._Loss,
    logger: Optional[Logger] = None,
    epoch: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, float]]:
    """
    Evaluate the suppled model

    :param model: Model to evaluate
    :param test_loader: Data to use for evaluation
    :param lossfn: The loss function
    :param logger: Provide logger to record results
    :param epoch: Provide current epoch to record results
    """

    pbar = tqdm(test_loader)
    model.eval()
    metrics_val = MetricComputations()

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)

            loss = lossfn(outputs.logits, labels)
            acc, f1, test_loss = metrics_val.compute(labels, outputs, loss)

            pbar.set_description(f"Eval: Loss {test_loss:.3f} " f"Acc {acc:.3f} " f"f1 {f1:.3f}")

    if logger is not None and epoch is not None:
        logger.log_results(epoch, metrics_val, file_name="valid_results.csv")

    return acc, f1, test_loss


def main(config_dic: dict) -> None:
    """
    Main entrypoint for the training routines.
    :param config_dic: Dictionary containing the relevant configuration for the training.
    """
    model, tokenizer = get_model_and_tokenizer(
        model_name_or_path=config_dic["model_name_or_path"],
        precision=config_dic["precision"],
    )

    data = data_processing(datasets=config_dic["datasets"])
    config_dic["datasets"] = data["dataset_names"]

    train_encodings = tokenizer(data["x_train"], truncation=True, padding=True)
    val_encodings = tokenizer(data["x_val"], truncation=True, padding=True)

    train_dataset = LLMPromptsDataset(train_encodings, data["y_train"])
    val_dataset = LLMPromptsDataset(val_encodings, data["y_val"])

    train_loader = DataLoader(train_dataset, batch_size=config_dic["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config_dic["batch_size"], shuffle=True)

    model = model.to(device)

    train(
        model,
        train_loader,
        config=config_dic,
        optim=torch.optim.AdamW(
            model.parameters(),
            lr=config_dic["lr"],
            betas=config_dic["betas"],
            eps=config_dic["eps"],
            weight_decay=config_dic["weight_decay"],
        ),
        lossfn=torch.nn.CrossEntropyLoss(),
        eval_loader=val_loader,
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default="./configs/neurips_config.json")
    args = parser.parse_args()

    config_dic = json.load(open(args.config_path))

    config_dic["model_name"] = args.model_name
    config_dic["model_name_or_path"] = args.model_name_or_path
    config_dic["save_path"] = os.path.join("results", args.model_name)

    set_seed()
    main(config_dic)
