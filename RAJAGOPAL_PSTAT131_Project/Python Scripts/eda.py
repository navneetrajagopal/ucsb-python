#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from pandas_profiling import ProfileReport

#cleaned dataset
df = pd.read_csv('/users/navneet/downloads/housing_cleaned.csv')
print(df.head())
df.describe()

#Corr Matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

#Dist
num_vars = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
for var in num_vars:
    sns.histplot(df[var])
    plt.title(var)
    plt.show()

#Relatiobship of all variables and Price
sns.pairplot(df, x_vars=num_vars, y_vars=['price'])
plt.show()

#Categorical Variable Relationship with Price
cat_vars = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
for var in cat_vars:
    sns.boxplot(x=var, y='price', data=df1)
    plt.show()