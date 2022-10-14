import pandas as pd

from src.models import W2V
from typing import List


class Evaluator:

    def __init__(self, model: W2V):
        self.model = model

    def all_tests(self, en_words: List[str], cz_words: List[str]) -> None:
        en_words = self.__prefix_words(en_words, 'en')
        cz_words = self.__prefix_words(cz_words, 'cz')
        success_5 = 0
        success_10 = 0
        for i in range(len(en_words)):
            most_similar = self.__get_most_similar_for(en_words[i], 'cz', top_k=10)
            if cz_words[i] in most_similar['token'].unique():
                success_10 += 1
            if cz_words[i] in most_similar['token'].head(5).unique():
                success_5 += 1

        print("P@5: ", success_5 / len(en_words))
        print("P@10:", success_10 / len(en_words))

    def p_at_k_metric(self, en_words: List[str], cz_words: List[str], k: int = 5) -> float:
        """
        Simplified P@k metric for measuring success of the model. For each word pair checks if the translation is
        within the top k neighbours. If it is - adds this to success counter. The resulting score is then calculated
        by dividing this success counter by the number of word pairs. In contrast to the classical definition,
        this score is binary for each word pair
        :param en_words:
        :param cz_words:
        :param k:
        :return:
        """
        en_words = self.__prefix_words(en_words, 'en')
        cz_words = self.__prefix_words(cz_words, 'cz')
        success_count = 0
        for i in range(len(en_words)):
            most_similar = self.__get_most_similar_for(en_words[i], 'cz', top_k=k)
            if cz_words[i] in most_similar['token'].unique():
                success_count += 1
        return success_count / len(en_words)

    def __get_most_similar_for(self, word: str, language: str, top_k: int) -> pd.DataFrame:
        """
        Gets k most similar words in other language
        :param word: Word in one language to be found
        :param language: Language symbol of the words that we are looking for
        :param top_k: Defines how many words to retrieve
        :return:
        """
        size = 0
        df = pd.DataFrame()
        while size < top_k:
            df = pd.DataFrame(self.model.most_similar(word=word, topn=top_k * (2 + size)), columns=['token', 'dist'])
            df = df[df['token'].str.contains(f'{language}_')]
            size = df.size
        return df.head(top_k)

    @staticmethod
    def __prefix_words(words: List[str], language: str) -> List[str]:
        return [f'{language}_{word}' for word in words]
