import pandas as pd


# SINGLE SERIES UTILS
def get_ind_non_empty_elements(series: pd.Series)->list[int]:
    return series[series != ''].index.tolist()

# TWO SERIES UTILS
def build_reaction_series(reactant_series:pd.Series, product_series:pd.Series):
    return reactant_series + ">>" + product_series