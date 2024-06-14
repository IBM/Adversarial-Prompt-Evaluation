"""
Utility functions for prompt features
"""
from typing import List

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def get_prompt_length(prompts: List[str]) -> List[int]:
    """

    Parameters
    ----------
    prompts: list of strings

    Returns
    -------
    a list of integers that represents length of different prompts

    """
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    prompt_lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    return prompt_lengths


# https://huggingface.co/docs/transformers/perplexity
def get_gpt2_logppl(prompts: List[str], stride: int=512, device: str='cpu') -> List[float]:
    """

    Parameters
    ----------
    prompts: list of strings
    stride: stride for the sliding window used for perplexity computation
    device: one of 'cpu', 'gpu', 'mps' to execute torch operators

    Returns
    -------
    a numpy list containing the log perplexity for the prompts

    """
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    logppl_list = []
    for prompt in prompts:
        encodings = tokenizer(prompt, return_tensors="pt")
        max_length = model.config.n_positions
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        logppl = torch.stack(nlls).mean().detach().cpu().numpy()
        logppl_list.append(logppl)

    return logppl_list


class PerplexityFilter:
    """
    Perplexity Filter as per Jain et al.
    Baseline Defenses for Adversarial Attacks Against Aligned Language Models
    https://openreview.net/forum?id=0VZP2Dr9KX
    
    Filter sequences based on perplexity of the sequence.
    
    Parameters
    ----------
    model : transformers.PreTrainedModel
        Language model to use for perplexity calculation.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer to use for encoding sequences.
    threshold : float
        Threshold for -log perplexity. sequences with perplexity below this threshold
        will be considered "good" sequences.
    window_size : int
        Size of window to use for filtering. If window_size is 10, then the
        -log perplexity of the first 10 tokens in the sequence will be compared to
        the threshold. 
    """
    def __init__(self, model, tokenizer, threshold, window_size=10, device="cpu"):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.threshold = threshold
        self.window_threshold = threshold
        self.window_size = window_size
        self.cn_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.device = device
    
    def get_log_perplexity(self, sequence):
        """
        Get the log perplexity of a sequence.

        Parameters
        ----------
        sequence : str
        """
        input_ids = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)
        with torch.no_grad():   
            loss = self.model(input_ids, labels=input_ids).loss
        return loss.item()

    def get_max_log_perplexity_of_goals(self, sequences):
        """
        Get the log perplexity of a sequence.

        Parameters
        ----------
        sequence : str
        """
        all_loss = []
        cal_log_prob = []
        for sequence in sequences:
            input_ids = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)
            with torch.no_grad():   
                output = self.model(input_ids, labels=input_ids)
                loss = output.loss
            all_loss.append(loss.item())
            cal_log_prob.append(self.get_log_prob(sequence).mean().item())
        return max(all_loss)
    
    def get_max_win_log_ppl_of_goals(self, sequences):
        """
        Get the log perplexity of a sequence.

        Parameters
        ----------
        sequence : str
        """
        all_loss = []
        for sequence in sequences:
            input_ids = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)
            with torch.no_grad():   
                loss = self.model(input_ids, labels=input_ids).loss
            all_loss.append(loss.item())
        
        return max(all_loss)
    
    def get_log_prob(self, sequence):
        """
        Get the log probabilities of the token.

        Parameters
        ----------
        sequence : str
        """
        input_ids = self.tokenizer.encode(sequence, return_tensors='pt').to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids, labels=input_ids).logits
        logits = logits[:, :-1, :].contiguous()
        input_ids = input_ids[:, 1:].contiguous()
        log_probs = self.cn_loss(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        return log_probs
    
    def filter(self, sequences):
        """
        Filter sequences based on log perplexity.

        Parameters
        ----------
        sequences : list of str

        Returns
        -------
        filtered_log_ppl : list of float
            List of log perplexity values for each sequence.
        passed_filter : list of bool
            List of booleans indicating whether each sequence passed the filter.
        """
        filtered_log_ppl = []
        passed_filter = []
        for sequence in sequences:
            log_probs = self.get_log_prob(sequence)
            NLL_by_token = log_probs
            if NLL_by_token.mean() <= self.threshold:
                passed_filter.append(True)
                filtered_log_ppl.append(NLL_by_token.mean().item())
            else:
                passed_filter.append(False)
                filtered_log_ppl.append(NLL_by_token.mean().item())
        return filtered_log_ppl, passed_filter
    
    def filter_window(self, sequences, reverse=False):
        """
        Filter sequences based on log perplexity of a window of tokens.
        
        Parameters
        ----------
        sequences : list of str
            List of sequences to filter.
        reverse : bool
            If True, filter sequences based on the last window_size tokens in the sequence.
            If False, filter sequences based on the first window_size tokens in the sequence.

        Returns
        -------
        filtered_log_ppl_by_window : list of list of float
            List of lists of log perplexity values for each sequence.
        passed_filter_by_window : list of list of bool
            List of lists of booleans indicating whether each sequence passed the filter.
        passed : list of bool
            List of booleans indicating whether each sequence passed the filter.
        """
        filtered_log_ppl_by_window = []
        passed_filter_by_window = []
        passed = []
        for sequence in sequences:
            sequence_window_scores = []
            passed_window_filter = []
            log_probs = self.get_log_prob(sequence)
            NLL_by_token = log_probs
            for i in np.arange(0, len(NLL_by_token), self.window_size):
                if not reverse:
                    window = NLL_by_token[i:i+self.window_size]
                else:
                    if i == 0:
                        window = NLL_by_token[-self.window_size:]
                    elif -(-i-self.window_size) > len(NLL_by_token) and i != 0:
                        window = NLL_by_token[:-i]
                    else:
                        window = NLL_by_token[-i-self.window_size:-i]
                if window.mean() <= self.window_threshold:
                    passed_window_filter.append(True)
                    sequence_window_scores.append(window.mean().item())
                else:
                    passed_window_filter.append(False)
                    sequence_window_scores.append(window.mean().item())
            if all(passed_window_filter):
                passed.append(True)
            else:
                passed.append(False)
            passed_filter_by_window.append(passed_window_filter)
            filtered_log_ppl_by_window.append(sequence_window_scores)
        return filtered_log_ppl_by_window, passed_filter_by_window, passed
