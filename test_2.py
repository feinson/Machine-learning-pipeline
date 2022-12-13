import pandas as pd
import numpy as np

np.random.seed(0)
df = pd.DataFrame(np.random.choice(10, (3, 5)), columns=list('ABCDE'))
print(df)

def add_2(numby):
    return numby+2

df2 = df.copy()
df2["A"] = df["A"].apply(add_2)

print(df)
print(df2)
