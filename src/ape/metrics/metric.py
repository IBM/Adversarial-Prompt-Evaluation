"""
This module implements the abstract base classes for all metrics.
"""

import abc
from typing import List, Tuple, Union

import torch
import numpy as np


class Metric(abc.ABC):
    """
    Metrics base class
    """

    def __init__(self) -> None:
        super().__init__()
        pass

    @abc.abstractmethod
    def compute(self, outputs: Union[np.ndarray, torch.Tensor], **kwargs) -> Tuple[List[bool], float]:
        """
        Compute the metric
        """
        pass
