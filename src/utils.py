from functools import wraps

import pandas as pd
import json
import re
import os


def series_to_arr(df: pd.DataFrame, col: str = 'tokens') -> pd.Series:
    """
    Converts given dataframe with json values to dataframe containing individual tokens
    :param col: Column of the dataframe to be transformed
    :param df: Dataframe with json values
    :return: Converted dataframes where row contains list of tokens
    """
    return df.apply(lambda x: json.loads(re.sub("\'", "\"", x[col])), axis=1)


def form_model_name(**model_config) -> str:
    """
    Takes model parameters and creates name out of them
    :param model_config: parameters of the model
    :return:
    """
    w2v_type = 'sg' if model_config['sg'] else 'cbow'
    return f'{w2v_type}_{model_config["window"]}x{model_config["vector_size"]}'


def rooted_path(function):
    """
    Decorator on objects that have root_path argument.
    Joins the root path to the beginning of the path defined by the function
    :param function: function passed to the decorator
    :return:
    """
    @wraps(function)
    def wrapper(self, *path_args):
        res = function(self, *path_args)
        return os.path.join(self.root_path, res)

    return wrapper
