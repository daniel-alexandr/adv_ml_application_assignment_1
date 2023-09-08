import pandas as pd

def pop_target(df, target_col):

    df_copy = df.copy()
    target = df_copy.pop(target_col)

    return df_copy, target



