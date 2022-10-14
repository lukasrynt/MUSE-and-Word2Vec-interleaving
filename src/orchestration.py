import os
import pathlib
from typing import List, Dict

import pandas as pd

from .constants import MODELS_PATH, DATA_PATH, EMBEDDINGS_PATH, MUSE_EXEC_PATH
from .evaluation import Evaluator
from .models import W2V, MUSE


class Orchestrator:

    def __init__(self, vector_sizes: List[int], window_sizes: List[int], en_words: List[str], cz_words: List[str],
                 muse_epochs: int = 5, muse_epoch_size: int = 1_000_000):
        self.vector_sizes = vector_sizes
        self.window_sizes = window_sizes
        self.en_words = en_words
        self.cz_words = cz_words
        self.muse_epochs = muse_epochs
        self.muse_epoch_size = muse_epoch_size
        self.model_config = {}

    def run_all(self, progress: Dict[int, Dict[int, List[int]]] = None):
        if progress is None:
            progress = {}
        for vector_size in self.vector_sizes:
            for window_size in self.window_sizes:
                for sg in [0, 1]:
                    if self.__resolve_progress(progress, vector_size, window_size, sg):
                        continue
                    self.model_config = {
                        'window': window_size,
                        'min_count': 3,
                        'negative': 1,
                        'sg': sg,
                        'vector_size': vector_size
                    }
                    self.__log_model_params(model_type='interleaved')
                    self.train_and_evaluate_interleaved()
                    self.model_config['window'] = window_size // 2
                    self.__log_model_params(model_type='MUSE')
                    self.train_and_evaluate_muse()

    def train_and_evaluate_interleaved(self) -> None:
        interleaved_model = self.__build_w2v_model(model_type='interleaved')
        Evaluator(interleaved_model).all_tests(self.en_words, self.cz_words)

    def train_and_evaluate_muse(self) -> None:
        self.__train_monolingual_embeddings('en')
        self.__train_monolingual_embeddings('cz')
        self.__run_muse_adversarial()
        cz_aligned = pd.read_csv(self.__get_aligned_emb_path('cz'), sep=' ', skiprows=1, header=None)
        en_aligned = pd.read_csv(self.__get_aligned_emb_path('en'), sep=' ', skiprows=1, header=None)
        muse_model = MUSE(cz_aligned, en_aligned, self.model_config['vector_size'])
        Evaluator(muse_model).all_tests(self.en_words, self.cz_words)

    @staticmethod
    def __resolve_progress(progress: Dict[int, Dict[int, List[int]]], *args) -> bool:
        return progress.get(args[0], False) and \
               progress[args[0]].get(args[1], False) and \
               args[2] in progress[args[0]][args[1]]

    def __run_muse_adversarial(self):
        if os.path.isfile(self.__get_aligned_emb_path('cz')) and os.path.isfile(self.__get_aligned_emb_path('cz')):
            return
        cz_emb_path = self.__get_emb_path('cz')
        en_emb_path = self.__get_emb_path('en')
        os.system(f"""python {MUSE_EXEC_PATH}\
                        --cuda False --n_refinement 0\
                        --dis_most_frequent 0\
                        --src_emb {cz_emb_path} --src_lang cz\
                        --tgt_emb {en_emb_path} --tgt_lang en\
                        --emb_dim {self.model_config['vector_size']}\
                        --exp_path {EMBEDDINGS_PATH}\
                        --exp_name muse\
                        --exp_id {self.__form_model_name()}\
                        --epoch_size {self.muse_epoch_size}\
                        --n_epochs {self.muse_epochs}""")

    def __train_monolingual_embeddings(self, language: str) -> None:
        emb_path = self.__get_emb_path(language)
        if not os.path.isfile(emb_path):
            model = self.__build_w2v_model(model_type=language)
            model.save_vectors_for_muse(emb_path)

    def __build_w2v_model(self, model_type: str) -> W2V:
        path = self.__get_model_path(model_type)
        if os.path.isfile(path):
            return W2V(path=path)
        else:
            model = W2V(**self.model_config)
            df = pd.read_csv(self.__data_path_for_model(model_type))
            model.fit(df)
            model.save(path)
            return model

    @staticmethod
    def __data_path_for_model(model_type: str) -> str:
        if model_type == 'interleaved':
            file = 'interleaved.csv'
        elif model_type == 'en':
            file = 'en_tokens.csv'
        elif model_type == 'cz':
            file = 'cz_tokens.csv'
        else:
            file = None
        return os.path.join(DATA_PATH, file)

    def __get_emb_path(self, language: str) -> str:
        pathlib.Path(EMBEDDINGS_PATH, 'muse', self.__form_model_name()).mkdir(parents=True, exist_ok=True)
        return os.path.join(EMBEDDINGS_PATH, 'muse',
                            self.__form_model_name(), f'unaligned-{language}.txt')

    def __get_aligned_emb_path(self, language: str) -> str:
        return os.path.join(EMBEDDINGS_PATH, 'muse',
                            self.__form_model_name(), f'vectors-{language}.txt')

    def __get_model_path(self, model_type: str) -> str:
        pathlib.Path(MODELS_PATH, self.__form_model_name()).mkdir(parents=True, exist_ok=True)
        return os.path.join(MODELS_PATH, self.__form_model_name(), model_type)

    def __form_model_name(self) -> str:
        w2v_type = 'sg' if self.model_config['sg'] else 'cbow'
        return f'{w2v_type}_{self.model_config["window"]}x{self.model_config["vector_size"]}'

    def __log_model_params(self, model_type: str):
        w2v_type_name = 'Skip-Gram' if self.model_config['sg'] else 'CBOW'
        print('-' * 30)
        print(model_type.upper())
        print(f'Context window: {self.model_config["window"]}')
        print(f'Vector size: {self.model_config["vector_size"]}')
        print(f'Model type: {w2v_type_name}')
        print('-' * 30)
