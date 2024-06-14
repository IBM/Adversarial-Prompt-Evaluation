from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from ape.utils.ppl import get_gpt2_logppl, PerplexityFilter
from typing import List
from ape.detectors.detector import Detector

class PPLThresholdDetector(Detector):

    def __init__(self, threshold:float, stride: int = 512, device: str = 'cpu', paper=False, **kwargs):
        """

        Parameters
        ----------
        threshold
        stride
        device
        """

        super().__init__()
        self.threshold = threshold
        self.stride = stride
        self.device = device
        self.paper = paper
        if self.paper:
            model_id = "gpt2"
            model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
            tokenizer = GPT2TokenizerFast.from_pretrained(model_id, do_sample=True, top_k=1)
            self.ppl = PerplexityFilter(model, tokenizer, threshold, self.stride, self.device)


    def predict(self, prompts: List[str], **kwarg) -> List[dict]:
        """

        Parameters
        ----------
        prompts

        Returns
        -------

        """
        if self.paper:
            prompts_logppl = [self.ppl.get_log_perplexity(prompt[:1024]) for prompt in prompts]
        else:
            prompts_logppl = get_gpt2_logppl(prompts, stride=self.stride, device=self.device)
        prompts_pred = [{'label':'unsafe'} if logppl > self.threshold else {'label':'safe'} for logppl in prompts_logppl]

        return prompts_pred
