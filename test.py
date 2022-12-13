import pandas as pd
import os

df = pd.read_csv('listing.csv')

df = df.drop(columns="Unnamed: 19")

df = df.dropna(subset=["Description"])
suspect =df.loc[586,"Description"]
print(type(suspect))
print(suspect)

for i, row in df.iterrows():
    list_of_strings=eval(row["Description"])
    print(i)
    print(type(list_of_strings))
    combination_of_strings = ''.join(list_of_strings)
    to_return = combination_of_strings.removeprefix("About this space")
    print(to_return)
    if i == 4:
        break

