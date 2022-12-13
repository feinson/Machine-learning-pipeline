import os
import pandas as pd
import time
x = time.time()

df = pd.read_csv('listing.csv')
df = df.drop(columns="Unnamed: 19")

def remove_rows_with_missing_ratings(df: pd.DataFrame):

    df2 = df.copy()
    rating_columns = [item for item in df2.columns if item.endswith("_rating")]
    return df2.dropna(subset=rating_columns)


def combine_description_strings(df: pd.DataFrame):

        def string_cleaner(input_string: str):

            try:
                list_of_strings = eval(input_string)
                combination_of_strings = ''.join(list_of_strings)
                return combination_of_strings.removeprefix("About this space")
            except:
                return input_string

        df2 = df.copy()
        df2.loc[:,"Description"] = df["Description"].apply(string_cleaner)
        return df2

def clean_tabular_data(df: pd.DataFrame):
    df = remove_rows_with_missing_ratings(df)
    
    return combine_description_strings(df)


if __name__ == "__main__":
    clean_tabular_data(df).to_csv("clean_tabular_data.csv")

    print("_________")
    print(10*(time.time()-x))