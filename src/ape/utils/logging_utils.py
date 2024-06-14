"""
Utility for Logging
"""
import os
from datetime import datetime
import yaml

from typing import TYPE_CHECKING
import json
import torch
import numpy as np
import matplotlib
from sklearn.metrics import ConfusionMatrixDisplay

if TYPE_CHECKING:
    from ape.metrics.metric_utils import MetricComputations


class Logger:
    """
    Utility class to help with logging experimental results and loading/saving models
    """

    def __init__(self, config_dic: dict):
        self.savepath = config_dic["save_path"]

        if not os.path.isdir(self.savepath):
            os.makedirs(self.savepath)

        index = 0
        save_folder = f"run_{str(index)}"

        while os.path.isdir(os.path.join(self.savepath, save_folder)):
            index += 1
            save_folder = "run_" + str(index)

        self.savepath = os.path.join(self.savepath, save_folder)
        os.makedirs(self.savepath)
        print("Experiments will be saved to: ", self.savepath, flush=True)
        os.makedirs(os.path.join(self.savepath, "batch_nums"))

        with open(os.path.join(self.savepath, "configuration.json"), "w", encoding="utf-8") as config_file:
            json.dump(config_dic, config_file, sort_keys=True, indent=4)

        if not os.path.isdir(os.path.join(self.savepath, "models")):
            os.makedirs(os.path.join(self.savepath, "models"))
        self.pytorch_model_savepath = os.path.join(self.savepath, "models")

        for results_fname in ["train_results.csv", "valid_results.csv", "test_results.csv"]:
            with open(os.path.join(self.savepath, results_fname), "a", encoding="utf-8") as f_open:
                f_open.write(
                    ",".join(list(map(str, ["epoch", "loss", "acc", "f1", "TP", "TN", "FP", "FN"])))
                    + "\n"
                )

    def log_results(self, epoch: int, metrics: "MetricComputations", file_name: str = "results.csv") -> None:
        """
        Logs the training/test results
        :param epoch: Current epoch
        :param metrics: MetricComputations instance with the metrics to save
        :param file_name: name of the file to write to
        :return: None
        """

        info = list(
            map(
                str,
                [
                    epoch,
                    np.mean(metrics.epoch_loss),
                    np.mean(metrics.epoch_acc),
                    np.mean(metrics.epoch_f1),
                    metrics.tp,
                    metrics.tn,
                    metrics.fp,
                    metrics.fn,
                ],
            )
        )

        confusion_matrix = np.asarray([[metrics.tn, metrics.fp], [metrics.fn, metrics.tp]])
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)

        if 'train' in file_name:
            cm_fname = 'train_confusion_matrix.png'
        elif 'valid' in file_name:
            cm_fname = 'valid_confusion_matrix.png'
        elif 'test' in file_name:
            cm_fname = 'test_confusion_matrix.png'
        else:
            raise ValueError("Could not determing confusion matrix filename")

        disp.plot().figure_.savefig(os.path.join(self.savepath, cm_fname))
        matplotlib.pyplot.close()

        info_to_log = ",".join(info) + "\n"
        with open(os.path.join(self.savepath, file_name), "a", encoding="utf-8") as f_open:
            f_open.write(info_to_log)

    def save_models(self, model: torch.nn.Module, opt: torch.optim.Optimizer, fname: str = "checkpoint_dict", **kwargs) -> None:
        """
        Save a pytorch model and optimizer

        :param model: Pytorch model to save
        :param opt: Pytorch optimizer to save
        :return: None
        """
        torch.save(
            {"model": model.state_dict(), "opt": opt.state_dict()}, os.path.join(self.savepath, fname)+'.pt'
        )
        if not os.path.isdir(os.path.join(self.savepath, fname)+'/'):
            os.makedirs(os.path.join(self.savepath, fname)+'/')
        print("saved model to {}".format(os.path.join(self.savepath, fname)+'/'))
        model.save_pretrained(os.path.join(self.savepath, fname)+'/')

        if 'tokenizer' in kwargs:
            kwargs['tokenizer'].save_pretrained(os.path.join(self.savepath, fname)+'/')
