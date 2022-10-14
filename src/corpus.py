import json
import os

import pandas as pd
from pathlib import Path

from typing import List


class Loader:

    def __init__(self, data_path, prefix=True):
        self.tokens = None
        self.data_path = data_path
        self.prefix = prefix

    def load_tokens(self) -> pd.DataFrame:
        """
        Loads the folders structure created using the `process_pages` method in order to prevent redundant downloads
        :return: Dataframe with individual tokens
        """
        all_tokens = {'EN_tokens': [], 'CZ_tokens': []}
        for path in Path(self.data_path).glob('[0-9]*-C[0-9]*'):
            all_tokens['EN_tokens'] += self.__load_words(path, 'en')
            all_tokens['CZ_tokens'] += self.__load_words(path, 'cz')
        return pd.DataFrame(all_tokens)

    def save_tokens(self, path):
        tokens = self.load_tokens()
        tokens.to_csv(path)

    def __load_words(self, path, language):
        with open(os.path.join(path, f'{language}_tokens.json')) as f:
            paragraph = json.load(f)
            if self.prefix:
                res = []
                for sentence in paragraph:
                    res += [[f'{language}_{word}' for word in sentence]]
                return res
            return paragraph


class Creator:

    def __init__(self, tokens_path: str = None, tokens: pd.DataFrame = None):
        if tokens is None:
            self.tokens = Loader(tokens_path).load_tokens()
        else:
            self.tokens = tokens

    def save_monolingual(self, cz_path: str, en_path: str):
        cz_tokens = self.tokens['CZ_tokens'].to_frame('tokens')
        cz_tokens.to_csv(cz_path)
        en_tokens = self.tokens['EN_tokens'].to_frame('tokens')
        en_tokens.to_csv(en_path)

    def save_interleaves(self, path):
        tokens = self.interleave_tokens()
        tokens.to_csv(path)

    def interleave_tokens(self) -> pd.DataFrame:
        """
        Takes corpus dataframe and interleaves each pair of sentences using sequential interleaving algorithm
        :return: Dataframe with sequentially interleaved tokens
        """
        return self.tokens.apply(lambda x: self.__interleave_words(x['EN_tokens'], x['CZ_tokens']), axis=1) \
            .explode('data').to_frame(name='tokens')

    @staticmethod
    def __interleave_words(sent1: List[str], sent2: List[str]) -> List[list]:
        """
        Method for sequential interleaving of two sentences
        :param sent1: List of tokens in the first sentence
        :param sent2: List of tokens in the second sentence
        :return: List of interleaved sentences
        """
        if len(sent1) > len(sent2):
            long = sent1
            short = sent2
        else:
            long = sent2
            short = sent1

        res = []
        times = len(long) - len(short)
        if times == 0:
            times = 1
        for start_pos in range(times):
            sent = []
            for i, token in enumerate(long):
                sent.append(token)
                if (i < len(short) + start_pos) and i >= start_pos:
                    sent.append(short[i - start_pos])
            res.append(sent)
        return res
