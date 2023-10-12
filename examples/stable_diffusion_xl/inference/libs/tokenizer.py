from typing import Union, List

import numpy as np
from transformers import CLIPTokenizer

from gm.modules.embedders.open_clip.tokenizer import SimpleTokenizer


class SimpleTokenizerWrapper:
    def __init__(self):
        self.tokenizer = SimpleTokenizer()

    def __call__(self, texts: Union[str, List[str]], context_length: int = 77) -> np.ndarray:
        _tokenizer = SimpleTokenizer()
        if isinstance(texts, str):
            texts = [texts]
        sot_token = self.tokenizer.encoder["<start_of_text>"]
        eot_token = self.tokenizer.encoder["<end_of_text>"]
        all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
        result = np.zeros((len(all_tokens), context_length), np.int32)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            result[i, : len(tokens)] = np.array(tokens, np.int32)
        return result


class IdentityTokenizer:

    def __call__(self, x):
        return x


def get_tokenizer(tokenizer_name, version="openai/clip-vit-large-patch14"):
    if tokenizer_name == "CLIPTokenizer":
        tokenizer = CLIPTokenizer.from_pretrained(version)
    elif tokenizer_name == "SimpleTokenizerWrapper":
        tokenizer = SimpleTokenizerWrapper()
    elif tokenizer_name == "IdentityTokenizer":
        tokenizer = IdentityTokenizer()
    else:
        raise NotImplementedError(f"tokenizer {tokenizer_name} not implemented")
    return tokenizer
