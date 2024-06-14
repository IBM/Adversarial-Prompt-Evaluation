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

import random
import string


class Perturbation:
    """Base class for random perturbations."""

    def __init__(self, q: int):
        self.q = q
        self.alphabet = string.printable


class RandomSwapPerturbation(Perturbation):
    """Implementation of random swap perturbations.
    See `RandomSwapPerturbation` in lines 1-5 of Algorithm 2."""

    def __call__(self, s: str) -> str:
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s[i] = random.choice(self.alphabet)
        return "".join(list_s)
