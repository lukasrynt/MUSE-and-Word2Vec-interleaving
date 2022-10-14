import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
from scipy.spatial import distance

from src.models import W2V
from src.utils import series_to_arr


class Visualizer:
    def __init__(self, model: W2V, tokens_path: str):
        tokens = pd.read_csv(tokens_path)
        self.en_freqs = self.__get_frequencies_list(tokens, 'EN_tokens')
        self.cz_freqs = self.__get_frequencies_list(tokens, 'CZ_tokens')
        self.model = model

    def visualize_most_frequent(self, limit: int = 1000, perplexity: int = 30):
        """
        Visualize the model representation reduced by the most frequent words in the corpus
        :param perplexity: Perplexity of t-SNE dimensionality reduction
        :param limit: Number of tokens take from each language
        """
        reduced_df = self.__most_frequent_representations(limit)
        representations = self.__tsne_reduce(reduced_df, perplexity)
        self.__visualize(representations)

    def visualize_words(self, en_words: list, cz_words: list, perplexity: int = 30):
        """
        Visualize the model representation reduced by the given list of tokens
        :param perplexity: Perplexity of t-SNE dimensionality reduction
        :param en_words: List of english tokens
        :param cz_words: List of czech tokens
        """
        reduced_df = self.__reduced_representations(en_words, cz_words)
        representations = self.__tsne_reduce(reduced_df, perplexity)
        self.__visualize(representations)

    @staticmethod
    def __get_frequencies_list(tokens: pd.DataFrame, col: str) -> list[str]:
        """
        Gets tokens in preferred language ordered by their frequencies
        :return:
        """
        return series_to_arr(tokens, col=col).explode().value_counts().index.to_list()

    def __most_frequent_representations(self, limit: int = 1000):
        """
        Creates representations from the given W2V model reduced by the most frequent tokens in the collection
        :param limit: Number of tokens take from each language
        :return: Reduced learned representations by the most frequent tokens in the collection
        """
        df = self.model.get_vectors()
        cz_words = self.cz_freqs[:limit]
        en_words = self.en_freqs[:limit]
        return self.__reduce_words(df, en_words, cz_words)

    def __reduced_representations(self, en_words: list, cz_words: list) -> pd.DataFrame:
        """
        Creates representations from the given W2V model reduced by provided list of tokens
        :param en_words: List of english words
        :param cz_words: List of czech words
        :return: Reduced dataset containing only the words in the provided lists
        """
        df = self.model.get_vectors()
        return self.__reduce_words(df, en_words, cz_words)

    @staticmethod
    def __visualize(representations: pd.DataFrame):
        """
        Creates a visual representation of the given dataset
        :param representations: Representation to be visualized
        """
        fig = px.scatter(representations, x="t-SNE axe 1", y="t-SNE axe 2", text="token", color="language")
        fig.update_traces(textposition='top center')
        fig.update_layout(
            height=800,
            title_text='Tokens in dataset'
        )
        fig.show()

    @staticmethod
    def __tsne_reduce(df: pd.DataFrame, perplexity: int = 30) -> pd.DataFrame:
        """
        Reduces the dataset into 2 dimensions using t-SNE dimensionality reduction
        :param perplexity: Perplexity of t-SNE dimensionality reduction
        :param df: Dataset of learned representations
        :return: Learned representations reduced to 2 dimensions
        """
        feature_cols = [x for x in df.columns if isinstance(x, int)]
        x = df.loc[:, feature_cols]
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, square_distances=True,
                    init='random', metric=lambda x, y: Visualizer.__cosine_distance(x, y))
        tsne_results = tsne.fit_transform(x)
        cols = [f"t-SNE axe {col + 1}" for col in range(2)]
        tsne_df = pd.DataFrame(data=tsne_results, columns=cols)
        return pd.concat([tsne_df, df[['token', 'language']]], axis=1)

    @staticmethod
    def __cosine_distance(vector1: list[float], vector2: list[float]) -> float:
        """
        Calculates cosine distance between two vectors
        :param vector1: First vector representation
        :param vector2: Second vector representation
        :return: Calculated distance
        """
        return distance.cosine(vector1, vector2)

    @staticmethod
    def __reduce_words(df: pd.DataFrame, en_words: list, cz_words: list) -> pd.DataFrame:
        """
        Reduce tokens present in learned representations taken from W2V model by provided lists of tokens
        :param df: Learned representations of individual tokens taken from W2V model
        :param en_words: List of english words
        :param cz_words: List of czech words
        :return: Reduced dataset containing only the words in the provided lists
        """
        words_df = pd.DataFrame({'word': en_words + cz_words,
                                 'language': ['en'] * len(en_words) + ['cz'] * len(cz_words)})
        words_repr = df.loc[df['token'].isin(words_df['word'])]
        reduced = words_repr.merge(words_df, left_on='token', right_on='word')
        reduced.drop(columns=['word'], inplace=True)
        return reduced
