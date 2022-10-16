# MUSE vs Word2Vec interleaving for translations

## Corpus
The corpus for these evaluations is created from official reports and legislative acts of European Union from the [EUR-Lex](https://eur-lex.europa.eu/homepage.html) portal. The example of corpus creation process can be seen in `corpus_creation` notebook.

## Training and evaluating models
The training is demonstrated in the `training_and_evaluating` jupyter notebook. Both MUSE and W2V models can be trained and evaluated jointly for multiple configurations. 

### MUSE model training using GPU

For faster MUSE training, Google Colab was used notebooks were used, as GPU cuda can be utilized for adversarial training. I also provide [notebook](https://colab.research.google.com/drive/1aUaIUBeiEd7Y_OpiGxEj4U4I0Imh11-9?usp=sharing) for this purpose.