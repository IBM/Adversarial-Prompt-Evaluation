"""
Code adapted from: https://github.com/arobey1/smooth-llm

Original License

MIT License

Copyright (c) 2023 Alex Robey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List

import torch
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLM:
    """Forward pass through a LLM."""

    def __init__(self, model_path: str, tokenizer_path: str, conv_template_name: str, device: torch.device):
        if device.type == "cpu":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16

        # Language model
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True,
                device_map="auto",
                torch_dtype=torch_dtype,
            ).to(device).eval())

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Fastchat conversation template
        self.conv_template = get_conversation_template(conv_template_name)


    def __call__(self, batch: List[str], max_new_tokens=100) -> List[str]:
        # Pass current batch through the tokenizer
        batch_inputs = self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
        batch_input_ids = batch_inputs["input_ids"].to(self.model.device)
        batch_attention_mask = batch_inputs["attention_mask"].to(self.model.device)

        outputs = self.model.generate(
            batch_input_ids, attention_mask=batch_attention_mask, max_new_tokens=max_new_tokens
        )

        # Decode the outputs produced by the LLM
        batch_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        gen_start_idx = [
            len(self.tokenizer.decode(batch_input_ids[i], skip_special_tokens=True))
            for i in range(len(batch_input_ids))
        ]
        return [output[gen_start_idx[i] :] for i, output in enumerate(batch_outputs)]
