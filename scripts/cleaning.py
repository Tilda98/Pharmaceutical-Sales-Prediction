# Importing the libraries

import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
warnings.filterwarnings('ignore')

# Reading the datasets

store_df = pd.read_csv('../data/store.csv', na_values=['?', None])
test_df = pd.read_csv('../data/test.csv', na_values=['?', None])
train_df = pd.read_csv('../data/train.csv', na_values=['?', None])

# Determining which column(s) has missing values (store_df)
store_df.isna().sum()

# Determining which column(s) has missing values (test_df)
test_df.isna().sum()

# Determining which column(s) has missing values (train_df)
train_df.isna().sum()

# Fixing missing values of store data by 0 and NA

store_df['CompetitionDistance'] = store_df['CompetitionDistance'].fillna(0)
store_df['CompetitionOpenSinceYear'] = store_df['CompetitionOpenSinceYear'].fillna('Not Available')
store_df['CompetitionOpenSinceMonth'] = store_df['CompetitionOpenSinceMonth'].fillna('Not Available')
store_df['PromoInterval'] = store_df['PromoInterval'].fillna('Not Available')
store_df['Promo2SinceYear'] = store_df['Promo2SinceYear'].fillna('Not Available')
store_df['Promo2SinceWeek'] = store_df['Promo2SinceWeek'].fillna('Not Available')

# Confirming missing values (store_df)
store_df.isna().sum()

# Fixing the missing values in the test data by making all the 3 missing values 0

test_df['Open'] = test_df['Open'].fillna(0)

# Confirming missing values (test_df)
test_df.isna().sum()

# Merging the train and store datasets

train_store = pd.merge(train_df, store_df, how = 'left', on = "Store")
train_store.head()

# # Merging the test and store datasets
test_store = pd.merge(test_df, store_df, how = 'left', on = "Store")
test_store.head()

# Plot to fix outliers
def plot_box(df:pd.DataFrame, x_col:str, title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data = df, x=x_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.show()

# Checking outliers in train_store data
plot_box(train_store, "Sales", "Customers")

# Fixing the outliers in the dataset
Q1=train_store['Sales'].quantile(0.25)
Q3=train_store['Sales'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)

train_clean = train_store[train_store['Sales']< Upper_Whisker]

# Checking if outliers have been fixed
plot_box(train_clean, "Sales", "Customers")

# Confirming data is clean
train_clean.head()

# Confirming data is clean
train_clean.isna().sum()

# Creating a csv file of train and test store data

test_store.to_csv('../data/test_store.csv',index=False)
train_clean.to_csv('../data/train_store.csv',index=False)

# Creating a correlation matrix for the main columns to the analysis

correlated_columns = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment','CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
sample_data_for_correlation = train_clean[correlated_columns]
pd.set_option('display.float_format', lambda x: '%.3f' % x)
corr = sample_data_for_correlation.corr()
corr

# Creating a correlation heatmap

plt.figure(figsize=(10,5))
sns.heatmap(data=corr, annot = True)

plt.savefig("plot.png")