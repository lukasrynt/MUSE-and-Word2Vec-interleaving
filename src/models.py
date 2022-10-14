import math

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from typing import List

from src.utils import series_to_arr


class W2V:

    def __init__(self, path: str = None, **options):
        """
        Initializes embeddings
        """
        if path is not None:
            self.model = Word2Vec.load(path)
        else:
            self.model = Word2Vec(**options)

    def get_vectors(self) -> pd.DataFrame:
        """
        Takes W2V model learned representations and transforms them into dataset
        :return: Learned word embeddings from the given model
        """
        tokens = pd.DataFrame({'token': self.model.wv.index_to_key})
        values = tokens.apply(lambda x: self.model.wv[x['token']], axis=1)
        values = pd.DataFrame(values.to_list(), columns=list(range(1, self.model.vector_size + 1)))
        return pd.concat([tokens, values], axis=1)

    def save_vectors_for_muse(self, path) -> None:
        with open(path, 'w') as f:
            f.write(f'{0} {self.model.vector_size}\n')
        tokens = self.get_vectors()
        tokens.set_index('token').to_csv(path, header=False, sep=' ', mode='a')

    def get_vector(self, word) -> List[int]:
        return self.model.wv.get_vector(word)

    def most_similar(self, vector: List[int] = None, word: str = None, **options):
        return self.model.wv.most_similar([vector or word], **options)

    def fit(self, df: pd.DataFrame) -> None:
        """
        Trains the model on the dataset through series of iterations
        :param df: Dataframe with training data - contains lists of tokens
        """
        split_by_iteration = 100000
        iterations = math.ceil(len(df) / split_by_iteration)
        for i in range(iterations):
            print(f'Iteration {i + 1}/{iterations}')
            self.continual_train_model(df, i)

    def continual_train_model(self, df: pd.DataFrame, iteration: int = 0, split_per_iteration: int = 100000) -> None:
        """
        Takes the whole dataframe and trains the model on multiple iterations. This can be repeated with different parts
        :param df: Dataframe with training data - contains lists of tokens
        :param iteration: Number of iterations to be performed on the training data
        :param split_per_iteration: Number of sentences taken in one iteration
        """
        if iteration == 0:
            update = False
        else:
            update = True
        reduced_df = df.iloc[iteration * split_per_iteration:(iteration + 1) * split_per_iteration]
        self.batch_train_model(reduced_df, update)

    def batch_train_model(self, df: pd.DataFrame, update: bool = False) -> None:
        """
        Trains W2V model in multiple batches
        :param df: Dataframe with training data - contains lists of tokens
        :param update: Set to False if this is the first iteration of training
        :return:
        """
        self.model.build_vocab(series_to_arr(df), update=update)
        batch_size = 1000
        for g, partial_df in df.groupby(np.arange(len(df)) // batch_size):
            self.model.train(series_to_arr(partial_df),
                             total_examples=self.model.corpus_count, epochs=self.model.epochs)

    def save(self, path: str) -> None:
        self.model.save(path)


class MUSE:

    def __init__(self, en_model: W2V, cz_model: W2V):
        self.en_model = en_model
        self.cz_model = cz_model
