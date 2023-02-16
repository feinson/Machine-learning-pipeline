import pandas as pd
import math
import numpy as np

import time

def remove_columns_with_lots_of_nans(df1: pd.DataFrame, prop=0.4):

    #prop is the proportion of elements in the column which must not non-null for that column to survive
    df = df1.copy()
    df.dropna(axis="columns", thresh=math.floor(len(df.index)*prop), inplace= True)
    return df

def remove_rows_with_missing_ratings(df1: pd.DataFrame):

    df = df1.copy()
    rating_columns = [item for item in df.columns if item.endswith("_rating")]
    df = df.dropna(subset=rating_columns)

    return df

def combine_description_strings(df1: pd.DataFrame):

    def string_cleaner(input_string: str):

        try:
            list_of_strings = eval(input_string)
            combination_of_strings = ''.join(list_of_strings)
            output = combination_of_strings.removeprefix("About this space")
            output = output.removeprefix("What this place offers")
        
            return output
        except:
            return np.nan

    def column_string_cleaner(sr1: pd.Series):
        sr = sr1.copy()
        sr = sr.apply(string_cleaner)
        return sr
     
    df = df1.copy()
    df["Description"] = column_string_cleaner(df["Description"])
    df = df.dropna(subset=["Description"])
    df["Amenities"] = column_string_cleaner(df["Amenities"])
    df = df.dropna(subset=["Amenities"])

    return df

def set_default_feature_values(df1: pd.DataFrame):

    df = df1.copy()
    cols_for_defaults = ["guests", "beds", "bathrooms", "bedrooms"]
    df[cols_for_defaults] = df[cols_for_defaults].apply(pd.to_numeric, errors="coerce")
    df[cols_for_defaults] = df[cols_for_defaults].fillna(value=1).astype(int)

    return df

def clean_tabular_data(df: pd.DataFrame):
    df = remove_columns_with_lots_of_nans(df)
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df



if __name__ == "__main__":
    x = time.time()

    df = pd.read_csv('./data/unclean_tabular_data.csv')

    cleaned_tabular_data = clean_tabular_data(df)

    cleaned_tabular_data.to_csv("./data/clean_tabular_data.csv", index=False)

    print("_________")
    print((time.time()-x))