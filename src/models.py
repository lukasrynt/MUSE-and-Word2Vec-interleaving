import math
import shutil

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from typing import List, Tuple
import os
from pathlib import Path

from .constants import EMBEDDINGS_PATH, MUSE_EXEC_PATH
from .utils import series_to_arr, form_model_name, rooted_path


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


class MUSE(W2V):

    def __init__(self, en_model: W2V, cz_model: W2V,
                 model_config: dict, epoch_size: int = 1_000_000,
                 epochs: int = 5, root_path: str = './'):
        super().__init__()
        self.root_path = root_path
        self.model_config = model_config
        self.epoch_size = epoch_size
        self.epochs = epochs
        self.en_model = en_model
        self.cz_model = cz_model

    def run_adversarial(self) -> None:
        if not Path(self.__get_aligned_emb_path('cz')).exists() \
                or not Path(self.__get_aligned_emb_path('en')).exists():
            cz_emb_path, en_emb_path = self.__get_monolingual_embeddings()
            path = Path(self.__get_aligned_emb_dir_path())
            if path.exists():
                shutil.rmtree(path)
            os.system(f"""python {self.__muse_exec_path()}\
                            --cuda False --n_refinement 0\
                            --dis_most_frequent 0\
                            --src_emb {cz_emb_path} --src_lang cz\
                            --tgt_emb {en_emb_path} --tgt_lang en\
                            --emb_dim {self.model_config['vector_size']}\
                            --exp_path {self.__export_root_path()}\
                            --exp_name muse\
                            --exp_id {self.__form_model_name()}\
                            --epoch_size {self.epoch_size}\
                            --n_epochs {self.epochs}""")
        cz_aligned, en_aligned = self.__load_aligned_embeddings()
        self.__append_vectors(en_aligned, self.model_config['vector_size'])
        self.__append_vectors(cz_aligned, self.model_config['vector_size'])

    def __get_monolingual_embeddings(self) -> Tuple[str, str]:
        cz_emb_path = self.__get_unaligned_emb_path('cz')
        en_emb_path = self.__get_unaligned_emb_path('en')
        if not Path(cz_emb_path).exists() \
                or not Path(en_emb_path).exists():
            self.cz_model.save_vectors_for_muse(cz_emb_path)
            self.en_model.save_vectors_for_muse(en_emb_path)
        return cz_emb_path, en_emb_path

    def __append_vectors(self, vectors: pd.DataFrame, vector_size: int) -> None:
        self.model.wv.add_vectors(vectors[0].tolist(), vectors[range(1, vector_size + 1)].to_numpy())

    def __load_aligned_embeddings(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cz_aligned = pd.read_csv(self.__get_aligned_emb_path('cz'), sep=' ', skiprows=1, header=None)
        en_aligned = pd.read_csv(self.__get_aligned_emb_path('en'), sep=' ', skiprows=1, header=None)
        return cz_aligned, en_aligned

    def __get_unaligned_emb_path(self, language: str) -> str:
        Path(self.__get_unaligned_emb_dir_path()).mkdir(parents=True, exist_ok=True)
        return os.path.join(self.__get_unaligned_emb_dir_path(), f'{language}.txt')

    def __get_aligned_emb_path(self, language: str) -> str:
        return os.path.join(self.__get_aligned_emb_dir_path(), f'vectors-{language}.txt')

    @rooted_path
    def __get_aligned_emb_dir_path(self) -> str:
        return os.path.join(EMBEDDINGS_PATH, 'muse', self.__form_model_name())

    @rooted_path
    def __get_unaligned_emb_dir_path(self) -> str:
        return os.path.join(EMBEDDINGS_PATH, 'unaligned', self.__form_model_name())

    @rooted_path
    def __muse_exec_path(self) -> str:
        return os.path.join(MUSE_EXEC_PATH)

    @rooted_path
    def __export_root_path(self) -> str:
        return os.path.join(EMBEDDINGS_PATH)

    def __form_model_name(self) -> str:
        return form_model_name(**self.model_config)
