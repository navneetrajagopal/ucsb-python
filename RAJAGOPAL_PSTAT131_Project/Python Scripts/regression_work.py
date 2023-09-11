#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc, confusion_matrix

#create model
df = pd.read_csv('/users/navneet/downloads/housing_cleaned.csv')
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model1 = LinearRegression()
param_grid1 = {
    'fit_intercept': [True, False],
    'positive': [True, False],
    'copy_X': [True, False],
    'n_jobs': [-1]
}
model1 = LinearRegression()
grid_search1 = GridSearchCV(model1, param_grid=param_grid1, cv=10, scoring='neg_mean_squared_error')
grid_search1.fit(X_train, y_train)
best_model1 = grid_search1.best_estimator_
y_pred1 = best_model1.predict(X_test)
mse = mean_squared_error(y_test, y_pred1)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred1)
print('RMSE:', rmse)
print('R²:', r2)


#visualization
rmse_lr = []
for degree in range(1, 9):
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    y_pred = model.predict(X_poly_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse_lr.append(rmse)
# Plot RMSE values for all four models
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].plot(range(1, 9), rmse_lr, marker='o')
axs[0, 0].set_xlabel('Polynomial Degree')
axs[0, 0].set_ylabel('RMSE')
axs[0, 0].set_title('Linear Regression')
plt.tight_layout()
plt.show()

#fit model to test
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print('RMSE:', rmse)
print('R²:', r2)

#importance
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
coef = pd.Series(lr_model.coef_, index=X_train.columns)
imp_coef = coef.sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=imp_coef.values, y=imp_coef.index)
plt.title("Feature Importance Chart - Linear Regression")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.show()

#predicted vs actual
plt.scatter(y_pred, y_pred, color='blue', label='Predicted')
plt.scatter(y_pred, y_test, color='red', label='Actual')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.legend()
plt.show()
