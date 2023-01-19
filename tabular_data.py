import pandas as pd
import math
import numpy as np

import time

def remove_columns_with_lots_of_nans(df: pd.DataFrame, prop=0.4):

    #prop is the proportion of elements in the column which must not non-null for that column to survive
    df2 = df.copy()
    df2.dropna(axis="columns", thresh=math.floor(len(df2.index)*prop), inplace= True)
    return df2

def remove_rows_with_missing_ratings(df: pd.DataFrame):

    df2 = df.copy()
    rating_columns = [item for item in df2.columns if item.endswith("_rating")]
    df2.dropna(subset=rating_columns, inplace=True)

    return df2

def combine_description_strings(df: pd.DataFrame):

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
        sr.dropna(inplace=True)
        sr = sr.apply(string_cleaner)
        sr.dropna(inplace=True)

        return sr
     
    df2 = df.copy()
    df2["Description"] = column_string_cleaner(df2["Description"])

    df2["Amenities"] = column_string_cleaner(df2["Amenities"])  

    return df2

def set_default_feature_values(df: pd.DataFrame):

    df2 = df.copy()
    cols_for_defaults = ["guests", "beds", "bathrooms", "bedrooms"]
    df2[cols_for_defaults] = df2[cols_for_defaults].apply(pd.to_numeric, errors="coerce")
    df2[cols_for_defaults] = df2[cols_for_defaults].fillna(value=1).astype(int)

    return df2

def clean_tabular_data(df: pd.DataFrame):
    df = remove_columns_with_lots_of_nans(df)
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df


def load_airbnb(clean_df: pd.DataFrame, label):
    df = clean_df.copy()
    labels = np.array(df.pop(label))
    features = np.array(df.select_dtypes(['number']))
    return (features, labels)

if __name__ == "__main__":
    x = time.time()

    df = pd.read_csv('./data/listing.csv')

    cleaned_tabular_data = clean_tabular_data(df)

    cleaned_tabular_data.to_csv("./data/clean_tabular_data.csv", index=False)

    print("_________")
    print((time.time()-x))