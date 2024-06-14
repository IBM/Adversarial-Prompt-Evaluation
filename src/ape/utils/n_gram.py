"""
Utility methods for n-gram computations
"""
from collections import Counter
from typing import Union, Any, Callable, List

import nltk
from transformers import AutoTokenizer


def n_gram(vocabulary: List[str], n: int) -> dict:
    """
    Compute the n-gram

    Parameters
    __________
        Vocabulary: List[str]
            list of tokenized words (not a set)
        n: int
            value of n consecutive words to count

    Returns
    _______
        vocab_freq: dict
            dictionary of n-gram frequencies
    """
    # uni-gram check
    if n == 1:
        vocab_freq = Counter(vocabulary)
        vocab_freq = dict(sorted(vocab_freq.items(), key=lambda item: item[1], reverse=True))
    else:
        tuple_n_grams = [vocabulary[i1:i2] if i2 != 0 else vocabulary[i1:] for i1, i2 in
                         zip(range(0, n), range(-n + 1, 1))]
        n_gram_vocab = ['--'.join(n_g) for n_g in zip(*tuple_n_grams)]
        vocab_freq = Counter(n_gram_vocab)
        vocab_freq = dict(sorted(vocab_freq.items(), key=lambda item: item[1], reverse=True))
    return vocab_freq


def get_n_gram_features(corpus: List[str] | dict, prompt: str | List[str], n_grams: int = 1,
                        tokenizer: str | Callable = 'no punctuation') \
        -> list[list[list[str]] | list[str] | list[list[Any]] | list[Any]]:
    """
    Function fot building n-gram of each words in the prompts, taking the statistics
    of the corpus

    Parameters
    ----------
        corpus : List[str] or dict
            a list of strings or n_gram vocabulary frequency
        prompt : str | list
            a string or list of strings corresponding to a prompt
        n_grams : int
            the number of consecutive word we want to consider
        tokenizer: str | Callable
            type of tokenizer to use for building vocab

    Returns
    -------
        n_gram_features: list
            set of features corresponding to the n-grams for each word in the prompt
    """
    assert n_grams >= 1, ValueError('n_grams must be equal or higher than 1!')
    if isinstance(corpus, dict):
        vocab_freq = corpus
    else:
        vocabulary = get_vocab(corpus, tokenizer=tokenizer)
        print(f'Building n-gram with n=={n_grams}...\n')
        vocab_freq = n_gram(
            vocabulary=vocabulary,
            n=n_grams
        )
    if isinstance(prompt, list):
        features_name = []
        n_gram_features = []
        for p in prompt:
            prompt_vocabulary = get_vocab(p, tokenizer=tokenizer)
            prompt_vocab_freq = n_gram(
                vocabulary=prompt_vocabulary,
                n=n_grams
            )
            features_name.append([f'{n_grams}_grams:{key}' for key in prompt_vocab_freq.keys() if key in vocab_freq])
            n_gram_features.append([vocab_freq[key] for key in prompt_vocab_freq.keys() if key in vocab_freq])
    elif isinstance(prompt, str):
        prompt_vocabulary = get_vocab(prompt, tokenizer=tokenizer, verbose=False)
        prompt_vocab_freq = n_gram(
            vocabulary=prompt_vocabulary,
            n=n_grams
        )
        features_name = [f'{n_grams}_grams:{key}' for key in prompt_vocab_freq.keys() if key in vocab_freq]
        n_gram_features = [vocab_freq[key] for key in prompt_vocab_freq.keys() if key in vocab_freq]
        features_name.append(f'{n_grams}_grams_newTokens')
        n_gram_features.append(len([True for key in prompt_vocab_freq.keys() if key not in vocab_freq]))
    else:
        raise ValueError

    return [features_name, n_gram_features]


def get_vocab(
    corpus: Union[list, str], tokenizer: Union[str, AutoTokenizer] = "no punctuation", verbose: bool = True
) -> List[str]:
    """
    Function for building the Vocabulary
    Parameters
    ----------
        corpus : list | str
            a list of strings or strings with the full corpus
        tokenizer : str | AutoTokenizer
            the strategy to split the corpus to hide or not punctuation (default is 'no punctuation')
        verbose : bool
            whether to print or not
    Returns
    -------
        vocabulary : list
            the set of unique words within the corpus
    """
    assert corpus is not None
    if verbose:
        print("Building Vocabulary...\n")
    if isinstance(corpus, list):
        assert corpus[0] is not None
        if isinstance(corpus[0], str):
            corpus = "".join(corpus)
        elif isinstance(corpus[0][0], str):
            corpus = "".join([line for batch in corpus for line in batch])
        else:
            raise ValueError("Invalid corpus format! Expected list of string, list or string")

    if isinstance(tokenizer, str):
        try:
            _tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            token_ids = _tokenizer(corpus).input_ids
            words = _tokenizer.convert_ids_to_tokens(token_ids)
        except OSError as os_error:
            if tokenizer == "no punctuation":
                words = re.split("\W+", corpus)  # pylint: disable=W1401
            elif tokenizer == "with punctuation":
                words = nltk.tokenize.wordpunct_tokenize(corpus)
            elif tokenizer == "token + punctuation":
                words = corpus.split()
            else:
                raise NotImplementedError from os_error
    elif isinstance(tokenizer, object):
        token_ids = tokenizer(corpus).input_ids
        words = tokenizer.convert_ids_to_tokens(token_ids)
    else:
        raise ValueError

    vocabulary = words
    return vocabulary