"""
Module containing the BERT and watBERT detector wrappers
"""

from typing import List, Optional
import torch

from transformers import (AutoTokenizer, AutoModelForSequenceClassification, pipeline,
                          GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer)
from ape.detectors.detector import Detector


class BERTclassifier(Detector):
    """
    Class for a BERTclassifier acting as a wrapper around a BERT model.
    """

    def __init__(
        self,
        path: str,
        n_labels: int = 2,
        precision: str = "full",
        stride: int = 10,
        device: Optional[str] = None,
    ):
        """

        Initialise a BERTclassifier instance.

        :param path: path to the huggingface model
        :param model_weight_path: path to the torch dictionary with the model fine-tuned weights
        :param n_labels: number of labels/model outputs. Usually 2, corresponding to 0 - benign and 1 - malicious
        :param device: device for the model (gpu/cpu)
        :param precision: if to run the model in regular (full) precision or float16 (half) precision for memory constraints.
        """

        super().__init__()
        self.path = path
        self.n_labels = n_labels
        self.stride = stride

        if precision == "full":
            torch_dtype = torch.float32
        elif precision == "half":
            torch_dtype = torch.float16
        self.device = self.choose_device(device)
        if "gpt2" in path:
            self.model_config = GPT2Config.from_pretrained(
                pretrained_model_name_or_path=path,
                torch_dtype=torch_dtype,
                num_labels=n_labels,
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=path)

            self.tokenizer.padding_side = "left"

            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = GPT2ForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                device_map=self.device,
            )

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path, num_labels=n_labels, torch_dtype=torch_dtype, device_map=self.device
            )

        self.model.eval()
        self.model.to(self.device)


    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict(self, prompts: List[str]) -> List[dict]:
        """

        Parameters
        ----------
        prompts: prompts to obtain predictions for
        threshold: minumum margin which the malicious class needs to have over the benign class for a prompt to be deemed malicious.

        Returns
        -------
        prompts_preds
        """

        prompts_pred_probs = self.predict_proba(prompts)
        prompts_preds = [
            (
                {"label": "unsafe", "confidence": p[1].numpy().item()}
                if p[1] > p[0]
                else {"label": "safe", "confidence": p[0].numpy().item()}
            )
            for p in prompts_pred_probs
        ]

        return prompts_preds

    def predict_proba(self, prompts: List[str]) -> List[torch.Tensor]:
        """
        Predict probabilities of prompts.
        This should be used only for inference and not training as gradients are not recorded.
        Parameters
        ----------
        prompts
        Returns
        -------
        results
        """

        # Guard condition in case a prompt string is passed
        if isinstance(prompts, str):
            prompts = [prompts]

        with torch.inference_mode():
            results = []
            for prompt in prompts:
                test_encodings = self.tokenizer(prompt, return_tensors="pt", padding=True)
                if len(test_encodings["input_ids"][0]) > self.tokenizer.model_max_length:
                    diff = len(test_encodings["input_ids"][0]) - self.tokenizer.model_max_length
                    input_ids = test_encodings["input_ids"].to(self.device)
                    attention_mask = test_encodings["attention_mask"].to(self.device)
                    outputs = [
                        self.model(
                            input_ids[:, idx : idx + self.tokenizer.model_max_length],
                            attention_mask=attention_mask[:, idx : idx + self.tokenizer.model_max_length],
                        ).logits
                        for idx in range(0, diff, self.stride)
                    ]
                    proba = [torch.softmax(output, dim=1).cpu()[:, 1] for output in outputs]
                    proba_index = torch.argmax(torch.Tensor(proba)).item()
                    results.append(torch.softmax(outputs[proba_index], dim=1).reshape(-1).cpu())
                else:
                    with torch.no_grad():
                        input_ids = test_encodings["input_ids"].to(self.device)
                        attention_mask = test_encodings["attention_mask"].to(self.device)
                        outputs = self.model(input_ids, attention_mask=attention_mask)

                    prompts_pred_probs = torch.softmax(outputs.logits, dim=1).cpu()
                    results.append(prompts_pred_probs.reshape(-1))
            return results
