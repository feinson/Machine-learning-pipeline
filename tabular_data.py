import pandas as pd
import math

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
            return input_string
     
    df2 = df.copy()
    df2.dropna(subset=["Description"], inplace=True)
    df2["Description"] = df2["Description"].apply(string_cleaner)

    df2.dropna(subset=["Amenities"], inplace=True)
    df2["Amenities"] = df2["Amenities"].apply(string_cleaner)
        
    return df2

def set_default_feature_values(df: pd.DataFrame):

    df2 = df.copy()
    df2[["guests", "beds", "bathrooms", "bedrooms"]] = df2[["guests", "beds", "bathrooms", "bedrooms"]].fillna(value=1).astype(int)

    return df2

def clean_tabular_data(df: pd.DataFrame):
    df = remove_columns_with_lots_of_nans(df)
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_values(df)
    return df


if __name__ == "__main__":
    x = time.time()

    df = pd.read_csv('./data/listing.csv')
    df.drop(index = 586, inplace=True)

    cleaned_tabular_data = clean_tabular_data(df)
    print(len(cleaned_tabular_data.columns))
    cleaned_tabular_data.to_csv("./data/clean_tabular_data.csv")

    print("_________")
    print((time.time()-x))