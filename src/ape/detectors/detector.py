"""
APE abstract Detector
"""
import abc
import sys
from typing import List

import torch


class Detector(abc.ABC):
    """
    Abstract detector
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        pass

    @staticmethod
    def choose_device(device) -> str:
        """
        Parameters
        ----------
        device: utilised by the detector
        """
        if device:
            return device
        if sys.platform == 'darwin':
            return 'mps'
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'

    @abc.abstractmethod
    def predict(self, prompts: List[str]) -> List[dict]:
        """
        Parameters
        ----------
        prompts: list of prompts to compute detection
        
        Returns
        -------
        List of dicts containing predictions
        """
        pass
