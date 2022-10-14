import models

def all_tests(model, en_words, cz_words, en_freqs):
    distance_sum = 0
    success_5 = 0
    success_10 = 0
    df = get_vector_representations(model)
    all_en_words = en_freqs['Unnamed: 0'].values.tolist()
    en_representations = df.loc[df['token'].isin(all_en_words)]
    for i in range(len(en_words)):
        cz_repr = get_single_representation(model, cz_words[i])
        en_repr = get_single_representation(model, en_words[i])
        distances = en_representations.apply(lambda x: cosine_distance(x[list(range(1, model.vector_size + 1))].tolist(), cz_repr), axis=1)
        distances.rename('distances', inplace=True)
        largest_dist = distances.max()
        distance_sum += cosine_distance(en_repr, cz_repr) / largest_dist

        merged = pd.concat([distances, en_representations], axis=1)
        neighbors_5 = merged.nsmallest(5, 'distances')['token'].tolist()
        neighbors_10 = merged.nsmallest(10, 'distances')['token'].tolist()
        if en_words[i] in neighbors_5:
            success_5 += 1
        if en_words[i] in neighbors_10:
            success_10 += 1
    print("P@5:", success_5 / len(en_words))
    print("P@10:", success_10 / len(en_words))
    print("distance_sum:", distance_sum)


class Evaluator:

    def __init__(self, model: models.W2V):
        self.model = model

    def p_at_k_metric(self, en_words: list[str], cz_words: list[str], k: int = 5) -> float:
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
        success_count = 0
        res = self.model.most_similar(word=en_words[0], topn=100)
        res

        df = get_vector_representations(self.model)
        all_en_words = en_freqs['Unnamed: 0'].values.tolist()
        en_representations = df.loc[df['token'].isin(all_en_words)]
        for i in range(len(en_words)):
            cz_repr = get_single_representation(model, cz_words[i])
            distances = en_representations \
                .apply(lambda x: cosine_distance(x[list(range(1, model.vector_size + 1))].tolist(), cz_repr), axis=1)
            distances.rename('distances', inplace=True)
            merged = pd.concat([distances, en_representations], axis=1)
            n_neighbors = merged.nsmallest(k, 'distances')['token'].tolist()
            if en_words[i] in n_neighbors:
                success_count += 1
        return success_count / len(en_words)

