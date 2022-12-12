import os
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd
from gensim.models import Word2Vec

from .word2vec import W2V
from ..constants import EMBEDDINGS_PATH, MUSE_EXEC_SUPERVISED_PATH, MUSE_EXEC_UNSUPERVISED_PATH
from ..utils import form_model_name, rooted_path


class MUSE(W2V):

    def __init__(self, en_model: W2V, cz_model: W2V,
                 model_config: dict, epoch_size: int = 1_000_000,
                 epochs: int = 5, root_path: str = './', supervised: bool = False):
        super().__init__(**model_config)
        self.root_path = root_path
        self.model_config = model_config
        self.epoch_size = epoch_size
        self.epochs = epochs
        self.en_model = en_model
        self.cz_model = cz_model
        self.model = Word2Vec(**model_config)
        self.supervised = supervised

    def run_adversarial(self) -> None:
        if not Path(self.__get_aligned_emb_path('cz')).exists() \
                or not Path(self.__get_aligned_emb_path('en')).exists():
            cz_emb_path, en_emb_path = self.__get_monolingual_embeddings()
            path = Path(self.__get_aligned_emb_dir_path())
            if path.exists():
                shutil.rmtree(path)
            if self.supervised:
                self.__run_supervised(cz_emb_path, en_emb_path)
            else:
                self.__run_unsupervised(cz_emb_path, en_emb_path)
        cz_aligned, en_aligned = self.__load_aligned_embeddings()
        self.__append_vectors(en_aligned, self.model_config['vector_size'])
        self.__append_vectors(cz_aligned, self.model_config['vector_size'])
        self.model.wv.fill_norms(force=True)

    def __run_unsupervised(self, cz_emb_path, en_emb_path):
        os.system(f"""python {self.__muse_exec_path()}\
                        --cuda False --n_refinement 5\
                        --dis_most_frequent 7500\
                        --tgt_emb {cz_emb_path} --tgt_lang cz\
                        --src_emb {en_emb_path} --src_lang en\
                        --emb_dim {self.model_config['vector_size']}\
                        --exp_path {self.__export_root_path()}\
                        --exp_name unsupervised\
                        --exp_id {self.__form_model_name()}\
                        --epoch_size {self.epoch_size}\
                        --n_epochs {self.epochs}""")

    def __run_supervised(self, cz_emb_path, en_emb_path):
        os.system(f"""python {self.__muse_exec_path()}\
                        --cuda False --n_refinement 1\
                        --tgt_emb {cz_emb_path} --tgt_lang cz\
                        --src_emb {en_emb_path} --src_lang en\
                        --emb_dim {self.model_config['vector_size']}\
                        --exp_path {self.__export_root_path()}\
                        --exp_name supervised\
                        --exp_id {self.__form_model_name()}""")

    def __get_monolingual_embeddings(self) -> Tuple[str, str]:
        cz_emb_path = self.__get_unaligned_emb_path('cz')
        en_emb_path = self.__get_unaligned_emb_path('en')
        if not Path(cz_emb_path).exists() \
                or not Path(en_emb_path).exists():
            self.cz_model.save_vectors_for_muse(cz_emb_path)
            self.en_model.save_vectors_for_muse(en_emb_path)
        return cz_emb_path, en_emb_path

    def __append_vectors(self, vectors: pd.DataFrame, vector_size: int) -> None:
        self.model.wv.add_vectors(vectors[0].tolist(), vectors[range(1, vector_size + 1)].to_numpy(), replace=True)

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
        path = 'supervised' if self.supervised else 'unsupervised'
        return os.path.join(EMBEDDINGS_PATH, path, self.__form_model_name())

    @rooted_path
    def __get_unaligned_emb_dir_path(self) -> str:
        return os.path.join(EMBEDDINGS_PATH, 'unaligned', self.__form_model_name())

    @rooted_path
    def __muse_exec_path(self) -> str:
        if self.supervised:
            return MUSE_EXEC_SUPERVISED_PATH
        else:
            return MUSE_EXEC_UNSUPERVISED_PATH

    @rooted_path
    def __export_root_path(self) -> str:
        return os.path.join(EMBEDDINGS_PATH)

    def __form_model_name(self) -> str:
        return form_model_name(**self.model_config)
