import torch
from typing import List, Union

from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ape.detectors.smooth_llm.defenses import Defense

class BaseRefusal(Defense):

    def __init__(self,  model, tokenizer, conv_template):
        super().__init__()

        if isinstance(model, str):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cpu":
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float16

            model = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True,
                device_map="auto",
                torch_dtype=torch_dtype,
            ).to(device).eval()

        self.model = model
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True, use_fast=False)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        self.conv_template = get_conversation_template(conv_template)

    def __call__(self, prompts: Union[List[str], str], verbose_output=False):
        return self.predict_proba(prompts, verbose_output)

    def predict(self, prompts: Union[List[str], str], verbose_output=False):
        prompts_pred_probs = self.predict_proba(prompts, verbose_output)
        if prompts_pred_probs:
            return [{"label": "unsafe", "confidence": None}]
        else:
            return [{"label": "safe", "confidence": None}]

    def predict_proba(self, prompts: Union[List[str], str], verbose_output=False) -> List[float]:
        """
        Obtain prediction for the given prompt

        :param prompts: Set of prompts to obtain predictions for.
        :return: Probabilities for the two classes.
        """

        if isinstance(prompts, str):
            prompts = [prompts]

        generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)

        for prompt in prompts:
            max_new_tokens = max(len(self.tokenizer(prompt).input_ids) + 2, 100)
            # Check if the model refuses to respond
            self.conv_template.append_message(self.conv_template.roles[0], prompt)
            self.conv_template.append_message(self.conv_template.roles[1], "")
            full_prompt = self.conv_template.get_prompt()
            self.conv_template.messages = []
            output = generator(full_prompt, max_new_tokens=max_new_tokens, num_return_sequences=1, do_sample=False)
            output = output[0]["generated_text"][len(full_prompt):]

            if verbose_output:
                return self.is_jailbreak(output), full_prompt, output
            else:
                return self.is_jailbreak(output)