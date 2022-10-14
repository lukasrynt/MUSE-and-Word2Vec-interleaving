import pandas as pd
import json
import re


def series_to_arr(df: pd.DataFrame, col: str = 'tokens') -> pd.Series:
    """
    Converts given dataframe with json values to dataframe containing individual tokens
    :param col: Column of the dataframe to be transformed
    :param df: Dataframe with json values
    :return: Converted dataframes where row contains list of tokens
    """
    return df.apply(lambda x: json.loads(re.sub("\'", "\"", x[col])), axis=1)
