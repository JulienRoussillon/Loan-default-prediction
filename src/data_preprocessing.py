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

def rename(df):
    df.rename(columns={'NumberOfTime30-59DaysPastDueNotWorse': 'NumberOfTime30_59DaysPastDueNotWorse',
                   'NumberOfTime60-89DaysPastDueNotWorse': 'NumberOfTime60_89DaysPastDueNotWorse'}, inplace=True)
    
    return df

def create_new_features(df):

    df['MonthlyIncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['DebtRatioPerDependent'] = df['DebtRatio'] / (df['NumberOfDependents'] + 1)
    df['DebtRatioPerAge'] = df['DebtRatio'] / df['age']
    df['RevolvingUtilizationOfUnsecuredLinesPerDependent'] = df['RevolvingUtilizationOfUnsecuredLines'] / (df['NumberOfDependents'] + 1)
    df['DebtRatioOverIncome'] = df['DebtRatio'] / (df['MonthlyIncome'] + 1)

    df['WeightedLateScore'] = (1 * df['NumberOfTime30_59DaysPastDueNotWorse'] +
                            5 * df['NumberOfTime60_89DaysPastDueNotWorse'] +
                            10 * df['NumberOfTimes90DaysLate'])

    df['WeightedLateScorePerDependent'] = df['WeightedLateScore'] / (df['NumberOfDependents'] + 1)
    df['WeightedLateScorePerAge'] = df['WeightedLateScore'] / df['age']

    #age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #age_labels = ['0-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    #df['AgeGroup'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
    #df = pd.get_dummies(df, columns=['AgeGroup'], drop_first=True)

    df['MonthlyIncomePerAge'] = df['MonthlyIncome'] / df['age']
    df['TotalMonthlyDebtPayment'] = df['MonthlyIncome'] * df['DebtRatio']
    
    return df
