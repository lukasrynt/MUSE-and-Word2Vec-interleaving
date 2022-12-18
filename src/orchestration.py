import os
from pathlib import Path
from typing import List, Dict

import pandas as pd

from .constants import MODELS_PATH, DATA_PATH
from .model.evaluation import Evaluator
from .model.word2vec import W2V
from .model.muse import MUSE
from .utils import form_model_name, rooted_path


class Orchestrator:

    def __init__(self, vector_sizes: List[int], window_sizes: List[int], en_words: List[str], cz_words: List[str],
                 muse_epochs: int = 5, muse_epoch_size: int = 1_000_000, root_path: str = './',
                 skip_muse: bool = False):
        self.vector_sizes = vector_sizes
        self.window_sizes = window_sizes
        self.en_words = en_words
        self.cz_words = cz_words
        self.muse_epochs = muse_epochs
        self.muse_epoch_size = muse_epoch_size
        self.root_path = root_path
        self.model_config = {}
        self.skip_muse = skip_muse

    def all_stats_df(self) -> pd.DataFrame:
        all_stats = pd.DataFrame(columns=['Model type', 'W2V type', 'Vector size', 'Context window',
                                          'P@1', 'P@5', 'P@10', 'Relevance'])
        for vector_size in self.vector_sizes:
            for window_size in self.window_sizes:
                for sg in [0, 1]:
                    self.model_config = {
                        'window': window_size,
                        'min_count': 3,
                        'negative': 1,
                        'sg': sg,
                        'vector_size': vector_size
                    }
                    entry = self.__create_stats_interleaved_entry()
                    all_stats = self.__append_entry(all_stats, entry)

                    self.model_config['window'] = window_size // 2
                    entry = self.__create_stats_muse_entry(supervised=True)
                    all_stats = self.__append_entry(all_stats, entry)
                    entry = self.__create_stats_muse_entry(supervised=False)
                    all_stats = self.__append_entry(all_stats, entry)
        return all_stats

    def run_all(self):
        for vector_size in self.vector_sizes:
            for window_size in self.window_sizes:
                for sg in [0, 1]:
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
                    print('Supervised')
                    self.train_and_evaluate_muse(supervised=True)
                    print('Unsupervised')
                    self.train_and_evaluate_muse(supervised=False)

    def train_and_evaluate_interleaved(self) -> None:
        interleaved_model = self.__build_w2v_model(model_type='interleaved')
        Evaluator(interleaved_model).all_tests(self.en_words, self.cz_words)

    def train_and_evaluate_muse(self, supervised) -> None:
        muse_model = self.__build_muse_model(supervised)
        if self.skip_muse:
            print("Skipping adversarial training")
            return
        muse_model.run_adversarial()
        Evaluator(muse_model).all_tests(self.en_words, self.cz_words)

    def __build_muse_model(self, supervised) -> MUSE:
        en_model = self.__build_w2v_model(model_type='en')
        cz_model = self.__build_w2v_model(model_type='cz')
        return MUSE(epoch_size=self.muse_epoch_size,
                    epochs=self.muse_epochs,
                    en_model=en_model, cz_model=cz_model,
                    model_config=self.model_config,
                    root_path=self.root_path,
                    supervised=supervised)

    def __create_stats_interleaved_entry(self) -> pd.DataFrame:
        evaluator = Evaluator(self.__build_w2v_model(model_type='interleaved'))
        return pd.DataFrame([{
            'Model type': 'Interleaved',
            **self.__common_stats_dict(evaluator),
        }])

    def __create_stats_muse_entry(self, supervised) -> pd.DataFrame:
        muse_model = self.__build_muse_model(supervised)
        muse_model.run_adversarial()
        evaluator = Evaluator(muse_model)
        return pd.DataFrame([{
            'Model type': 'MUSE ' + ('supervised' if supervised else 'unsupervised'),
            **self.__common_stats_dict(evaluator),
        }])

    def __common_stats_dict(self, evaluator: Evaluator) -> Dict:
        return {
            'W2V type': 'Skip-Gram' if self.model_config['sg'] else 'CBOW',
            'Vector size': self.model_config['vector_size'],
            'Context window': self.model_config['window'],
            'P@1': evaluator.p_at_k_metric(self.en_words, self.cz_words, k=1),
            'P@5': evaluator.p_at_k_metric(self.en_words, self.cz_words, k=5),
            'P@10': evaluator.p_at_k_metric(self.en_words, self.cz_words, k=10),
            'Relevance': round(evaluator.relevance_metric(self.en_words, self.cz_words), 2)
        }

    @staticmethod
    def __append_entry(stats: pd.DataFrame, entry: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([stats, entry], ignore_index=True)

    def __build_w2v_model(self, model_type: str) -> W2V:
        path = self.__get_model_path(model_type)
        if Path(path).exists():
            return W2V(path=path)
        else:
            model = W2V(**self.model_config)
            df = pd.read_csv(self.__data_path_for_model(model_type))
            model.fit(df)
            model.save(path)
            return model

    @rooted_path
    def __data_path_for_model(self, model_type: str) -> str:
        if model_type == 'interleaved':
            file = 'interleaved.csv'
        elif model_type == 'en':
            file = 'en_tokens.csv'
        elif model_type == 'cz':
            file = 'cz_tokens.csv'
        else:
            file = None
        return os.path.join(DATA_PATH, file)

    @rooted_path
    def __get_model_path(self, model_type: str) -> str:
        Path(self.root_path, MODELS_PATH, self.__form_model_name()).mkdir(parents=True, exist_ok=True)
        return os.path.join(MODELS_PATH, self.__form_model_name(), model_type)

    def __form_model_name(self) -> str:
        return form_model_name(**self.model_config)

    def __log_model_params(self, model_type: str):
        w2v_type_name = 'Skip-Gram' if self.model_config['sg'] else 'CBOW'
        print('-' * 30)
        print('-' * 30)
        print(model_type.upper())
        print(f'Context window: {self.model_config["window"]}')
        print(f'Vector size: {self.model_config["vector_size"]}')
        print(f'Model type: {w2v_type_name}')
        print('-' * 30)
