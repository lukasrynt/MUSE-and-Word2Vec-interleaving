import re

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as ps
from nltk.stem import WordNetLemmatizer as wnl

from .czech_stemmer import cz_stem


# download required data
# nltk.download('punkt')
# nltk.download('omw-1.4')
# nltk.download('stopwords')


# ----------------------------------------------------------------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------------------------------------------------------------
def en_tokenize(sentence: str) -> list:
    """
    Takes english sentence and converts into set of tokens stemmed using Porter stemmer with stopwords removed
    :param sentence: Sentence to be tokenized
    :return: List of tokens
    """
    # tokenization and normalization
    tokens = tokenize(sentence, 'english')

    # stop words removal
    tokens = en_remove_stop_words(tokens)

    # stemming
    return en_stemming(tokens)


def cz_tokenize(sentence: str) -> list:
    """
    Takes czech sentence and converts into list of tokens stemmed using special algorithm with stopwords removed
    :param sentence: Sentence to be tokenized
    :return: List of tokens
    """
    # tokenization and normalization
    tokens = tokenize(sentence, 'czech')

    # stop words removal
    tokens = cz_remove_stop_words(tokens)

    # stemming
    return cz_stemming(tokens)


def tokenize(sentence: str, lang: str) -> list:
    """
    Takes input sentence and converts it into list of tokens. Removes any noise and number inside these tokens. Maps
    all the tokens to lower case.
    :param sentence: Sentence to be tokenized
    :param lang: Language of the sentence -
    either 'czech' or 'english' :return:
    """
    tokens = word_tokenize(sentence, language=lang)
    tokens = [token for token in tokens if
              token.isalnum() and not re.match('^\d*$', token)]  # remove non-alphanumeric tokens
    tokens = [token.lower() for token in tokens]  # convert all tokens to lowercase
    return tokens


# ----------------------------------------------------------------------------------------------------------------------
# Stemming
# ----------------------------------------------------------------------------------------------------------------------
def en_stemming(tokens: list) -> list:
    """
    Stems the tokens using PorterStemmer algorithm
    :param tokens: Tokens to be stemmed
    :return: Stemmed tokens
    """
    stemmer = ps()
    return [stemmer.stem(word=token) for token in tokens]


def cz_stemming(tokens: list) -> list:
    # https://research.variancia.com/czech_stemmer/
    return [cz_stem(token) for token in tokens]


# ----------------------------------------------------------------------------------------------------------------------
# Lemmatization
# ----------------------------------------------------------------------------------------------------------------------
def en_lemmatize(tokens: list) -> list:
    """
    Performs lemmatization on the input list of tokens using WordNetLemmatizer
    :param tokens: Input tokens
    :return: List of lemmas from the process of lemmatization
    """
    nltk.download('wordnet')
    lemmatizer = wnl()
    return [lemmatizer.lemmatize(word=token, pos='v') for token in tokens]


# ----------------------------------------------------------------------------------------------------------------------
# Stop words removal
# ----------------------------------------------------------------------------------------------------------------------
CZ_STOPS = ["ačkoli", "ahoj", "ale", "anebo", "ano", "asi", "aspoň", "během", "bez", "beze", "blízko", "bohužel",
            "brzo", "bude", "budeme", "budeš", "budete", "budou", "budu", "byl", "byla", "byli", "bylo", "byly", "bys",
            "čau", "chce", "chceme", "chceš", "chcete", "chci", "chtějí", "chtít", "chut'", "chuti", "co", "čtrnáct",
            "čtyři", "dál", "dále", "daleko", "děkovat", "děkujeme", "děkuji", "den", "deset", "devatenáct", "devět",
            "do", "dobrý", "docela", "dva", "dvacet", "dvanáct", "dvě", "hodně", "já", "jak", "jde", "je", "jeden",
            "jedenáct", "jedna", "jedno", "jednou", "jedou", "jeho", "její", "jejich", "jemu", "jen", "jenom", "ještě",
            "jestli", "jestliže", "jí", "jich", "jím", "jimi", "jinak", "jsem", "jsi", "jsme", "jsou", "jste", "kam",
            "kde", "kdo", "kdy", "když", "ke", "kolik", "kromě", "která", "které", "kteří", "který", "kvůli", "má",
            "mají", "málo", "mám", "máme", "máš", "máte", "mé", "mě", "mezi", "mí", "mít", "mně", "mnou", "moc", "mohl",
            "mohou", "moje", "moji", "možná", "můj", "musí", "může", "my", "na", "nad", "nade", "nám", "námi",
            "naproti", "nás", "náš", "naše", "naši", "ne", "ně", "nebo", "nebyl", "nebyla", "nebyli", "nebyly", "něco",
            "nedělá", "nedělají", "nedělám", "neděláme", "neděláš", "neděláte", "nějak", "nejsi", "někde", "někdo",
            "nemají", "nemáme", "nemáte", "neměl", "němu", "není", "nestačí", "nevadí", "než", "nic", "nich", "ním",
            "nimi", "nula", "od", "ode", "on", "ona", "oni", "ono", "ony", "osm", "osmnáct", "pak", "patnáct", "pět",
            "po", "pořád", "potom", "pozdě", "před", "přes", "přese", "pro", "proč", "prosím", "prostě", "proti",
            "protože", "rovně", "se", "sedm", "sedmnáct", "šest", "šestnáct", "skoro", "smějí", "smí", "snad", "spolu",
            "sta", "sté", "sto", "ta", "tady", "tak", "takhle", "taky", "tam", "tamhle", "tamhleto", "tamto", "tě",
            "tebe", "tebou", "ted'", "tedy", "ten", "ti", "tisíc", "tisíce", "to", "tobě", "tohle", "toto", "třeba",
            "tři", "třináct", "trošku", "tvá", "tvé", "tvoje", "tvůj", "ty", "určitě", "už", "vám", "vámi", "vás",
            "váš", "vaše", "vaši", "ve", "večer", "vedle", "vlastně", "všechno", "všichni", "vůbec", "vy", "vždy", "za",
            "zač", "zatímco", "ze", "že", "aby", "aj", "ani", "az", "budem", "budes", "by", "byt", "ci", "clanek",
            "clanku", "clanky", "coz", "cz", "dalsi", "design", "dnes", "email", "ho", "jako", "jej", "jeji", "jeste",
            "ji", "jine", "jiz", "jses", "kdyz", "ktera", "ktere", "kteri", "kterou", "ktery", "ma", "mate", "mi",
            "mit", "muj", "muze", "nam", "napiste", "nas", "nasi", "nejsou", "neni", "nez", "nove", "novy", "pod",
            "podle", "pokud", "pouze", "prave", "pred", "pres", "pri", "proc", "proto", "protoze", "prvni", "pta", "re",
            "si", "strana", "sve", "svych", "svym", "svymi", "take", "takze", "tato", "tema", "tento", "teto", "tim",
            "timto", "tipy", "toho", "tohoto", "tom", "tomto", "tomuto", "tu", "tuto", "tyto", "uz", "vam", "vas",
            "vase", "vice", "vsak", "zda", "zde", "zpet", "zpravy", "a", "aniž", "až", "být", "což", "či", "článek",
            "článku", "články", "další", "i", "jenž", "jiné", "již", "jseš", "jšte", "k", "každý", "kteři", "ku", "me",
            "ná", "napište", "nechť", "ní", "nové", "nový", "o", "práve", "první", "přede", "při", "s", "sice", "své",
            "svůj", "svých", "svým", "svými", "také", "takže", "te", "těma", "této", "tím", "tímto", "u", "v", "více",
            "však", "všechen", "z", "zpět", "zprávy"]


def remove_stop_words(tokens: list, stop_words: set) -> list:
    """
    Removes stop words given by a set from input list of tokens
    :param tokens: Input list of tokens
    :param stop_words: Set of stop words
    :return: Tokens reduced by those occurring in stop words set
    """
    return [token for token in tokens if token not in stop_words]


def cz_remove_stop_words(tokens: list) -> list:
    """
    Removes stop words given by a set from input list of czech tokens
    :param tokens: Input list of tokens
    :return: Tokens reduced by those occurring in stop words downloaded from https://countwordsfree.com/stopwords/czech
    """
    stop_words = set(CZ_STOPS)
    return remove_stop_words(tokens, stop_words)


def en_remove_stop_words(tokens: list) -> list:
    """
    Removes stop words given by a set from input list of english tokens
    :param tokens: Input list of tokens
    :return: Tokens reduced by those occurring in stop words downloaded from punkt library
    """
    stop_words = set(stopwords.words('english'))
    return remove_stop_words(tokens, stop_words)
