import pandas as pd

from .word2vec import W2V
from typing import List


class Evaluator:

    def __init__(self, model: W2V):
        self.model = model

    def all_tests(self, en_words: List[str], cz_words: List[str]) -> None:
        en_words = self.__prefix_words(en_words, 'en')
        cz_words = self.__prefix_words(cz_words, 'cz')
        success_1 = 0
        success_5 = 0
        success_10 = 0
        relevance = 0
        for i in range(len(en_words)):
            most_similar = self.get_most_similar_for(en_words[i], 'cz', top_k=10)
            if cz_words[i] in most_similar['token'].unique():
                success_10 += 1
            if cz_words[i] in most_similar['token'].head(5).unique():
                success_5 += 1
            if cz_words[i] in most_similar['token'].head(1).unique():
                success_1 += 1
            relevance += self.__get_relevance_score(most_similar, cz_words[i])

        print("P@1: {:.2f}%".format(success_1 / len(en_words) * 100))
        print("P@5: {:.2f}%".format(success_5 / len(en_words) * 100))
        print("P@10: {:.2f}%".format(success_10 / len(en_words) * 100))
        print("Relevance: {:.2f}%".format(relevance / len(en_words) * 100))

    def relevance_metric(self, en_words: List[str], cz_words: List[str], max_k: int = 10) -> float:
        """
        Metric that takes max_k words as translations for each word in en_words. Then it takes which of these
        translations were contained the correct translation. If none of them did, it assigns zero, otherwise it
        contributes toward the final metric as 1/l, where l-th word was the correct translation. In the end it
        is normalized by the len of word pairs
        :param en_words:
        :param cz_words:
        :param max_k:
        :return:
        """
        en_words = self.__prefix_words(en_words, 'en')
        cz_words = self.__prefix_words(cz_words, 'cz')
        total_score = 0
        for i in range(len(en_words)):
            most_similar = self.get_most_similar_for(en_words[i], 'cz', top_k=max_k)
            total_score += self.__get_relevance_score(most_similar, cz_words[i])
        return total_score / len(en_words) * 100

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
            most_similar = self.get_most_similar_for(en_words[i], 'cz', top_k=k)
            if cz_words[i] in most_similar['token'].unique():
                success_count += 1
        return success_count / len(en_words) * 100

    def __get_relevance_score(self, most_similar: pd.DataFrame, translation: str) -> float:
        l_val = self.__find_in_list(most_similar['token'].tolist(), translation)
        if l_val == -1:
            return 0
        else:
            return 1 / (l_val + 1)

    @staticmethod
    def __find_in_list(lst: List, seeked):
        for i in range(len(lst)):
            if lst[i] == seeked:
                return i
        return -1

    def get_most_similar_for(self, word: str, language: str, top_k: int) -> pd.DataFrame:
        """
        Gets k most similar words in other language
        :param word: Word in one language to be found
        :param language: Language symbol of the words that we are looking for
        :param top_k: Defines how many words to retrieve
        :return:
        """
        size = 0
        df = pd.DataFrame()
        it = 0
        while size < top_k:
            it += 1
            df = pd.DataFrame(self.model.most_similar(word=word, topn=top_k * (2 + it + size)), columns=['token', 'dist'])
            df = df[df['token'].str.contains(f'{language}_')]
            size = df.size
        return df.head(top_k)

    @staticmethod
    def __prefix_words(words: List[str], language: str) -> List[str]:
        return [f'{language}_{word}' for word in words]
