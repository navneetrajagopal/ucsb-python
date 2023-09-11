import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from pandas_profiling import ProfileReport

#import original
df = pd.read_csv('/users/navneet/downloads/housing.csv')
missing_values = df.isnull().sum()
print(missing_values)
infinite_values = np.isinf(df).sum()
print(infinite_values)

#cleaning and making changes
df = pd.read_csv('/users/navneet/downloads/housing.csv')
map_dict = {'no': 0, 'yes': 1}
map_dict2 = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
df[['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']] = \
    df[['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']].replace(map_dict)
df['furnishingstatus'] = df['furnishingstatus'].replace(map_dict2)

#export new cleaned data
df.to_csv('/users/navneet/downloads/housing_cleaned.csv', index=False)