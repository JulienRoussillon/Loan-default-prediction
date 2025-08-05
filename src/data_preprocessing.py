import pandas as pd
import numpy as np

def fill_missing_with_median(df, col):
    df[col] = df[col].fillna(df[col].median())
    return df

def log1p_transform(df, cols):
    for col in cols:
        new_col = f"log1p{col}"
        df[new_col] = np.log1p(df[col])
    df.drop(columns=cols, inplace=True)
    return df

def cap_outliers(df, cols, lower_q=0.01, upper_q=0.99):
    for col in cols:
        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        df[col] = df[col].clip(lower, upper)
    return df

