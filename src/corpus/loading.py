import json
import os
from pathlib import Path

import pandas as pd


class Loader:

    def __init__(self, data_path, prefix=True):
        self.tokens = None
        self.data_path = data_path
        self.prefix = prefix

    def load_tokens(self) -> pd.DataFrame:
        """
        Loads the folders structure created by Downloader in order to prevent redundant downloads
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


