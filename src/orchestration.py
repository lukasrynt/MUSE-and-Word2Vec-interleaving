import os
from pathlib import Path
from typing import List, Dict

import pandas as pd

from .constants import MODELS_PATH, DATA_PATH
from .evaluation import Evaluator
from .models import W2V, MUSE
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
        en_model = self.__build_w2v_model(model_type='en')
        cz_model = self.__build_w2v_model(model_type='cz')
        muse_model = MUSE(epoch_size=self.muse_epoch_size,
                          epochs=self.muse_epochs,
                          en_model=en_model, cz_model=cz_model,
                          model_config=self.model_config,
                          root_path=self.root_path)
        if self.skip_muse:
            print("Skipping aligning vectors for now")
            return
        muse_model.run_adversarial()
        Evaluator(muse_model).all_tests(self.en_words, self.cz_words)

    @staticmethod
    def __resolve_progress(progress: Dict[int, Dict[int, List[int]]], *args) -> bool:
        return progress.get(args[0], False) and \
               progress[args[0]].get(args[1], False) and \
               args[2] in progress[args[0]][args[1]]

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
        print(model_type.upper())
        print(f'Context window: {self.model_config["window"]}')
        print(f'Vector size: {self.model_config["vector_size"]}')
        print(f'Model type: {w2v_type_name}')
        print('-' * 30)
