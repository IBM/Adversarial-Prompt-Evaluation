from typing import List, Union

import numpy as np

from ape.detectors.detector import Detector


class LangkitDetector(Detector):
    """
    Helper class to convert pre-trained checkpointed models into WatsonNLP format and follow pattern of
    existing detectors.
    """

    def __init__(self, threshold: float = 0.5):
        """
        :param thresh: Detection threshold to classify as malicious
        """
        # Import here so module can be accessed without requiring langkit to be installed
        from langkit import injections  # pylint: disable=C0415

        super().__init__()
        self.threshold = threshold
        self.schema = injections.init()

    def predict_proba(self, prompts: Union[List[str], str]) -> List[np.ndarray]:
        """
        Predict probabilities of prompts via parsing the watson_nlp output.
        This should be used only for inference and not training as gradients are not recorded.

        :param prompts: Set of prompts to obtain predictions for.
        :return: Probabilities for the two classes.
        """
        # Import here so module can be accessed without requiring langkit to be installed
        from langkit import extract  # pylint: disable=C0415

        if isinstance(prompts, str):
            prompts = [prompts]

        outputs = []

        for prompt in prompts:
            result = extract({"prompt": prompt}, schema=self.schema)
            outputs.append(result['prompt.injection'])
        return outputs

    def predict(self, prompts: Union[List[str], str]) -> List[dict]:
        """
        Performs prediction over supplied prompts

        :param prompts: Set of prompts to obtain predictions for

        :return: Predictions of safe/unsafe with confidence score
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        prompts_pred_probs = self.predict_proba(prompts)
        prompts_preds = [
            (
                {"label": "safe", "confidence": float(1 - p)}
                if p < self.threshold
                else {"label": "unsafe", "confidence": float(p)}
            )
            for p in prompts_pred_probs
        ]

        return prompts_preds
