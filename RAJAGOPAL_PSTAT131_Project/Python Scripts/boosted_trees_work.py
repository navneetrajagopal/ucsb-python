#libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

#model
df = pd.read_csv('/users/navneet/downloads/housing_cleaned.csv')
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model4 = xgb.XGBRegressor(objective='reg:squarederror')
param_grid4 = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.5, 0.7, 0.9],
    'colsample_bytree': [0.5, 0.7, 0.9]
}
grid_search4 = GridSearchCV(model4, param_grid=param_grid4, cv=5, scoring='neg_mean_squared_error')
grid_search4.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)
best_model4 = grid_search4.best_estimator_
y_pred4 = best_model4.predict(X_test)
mse = mean_squared_error(y_test, y_pred4)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred4)
print('RMSE:', rmse)
print('R²:', r2)

#Visualization
rmse_bt = []
for i in range(1, 9):
    model_bt = GradientBoostingRegressor(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=42)
    polynomial_features = PolynomialFeatures(degree=i)
    x_poly_bt = polynomial_features.fit_transform(X_train)
    model_bt.fit(x_poly_bt, y_train)
    x_poly_test_bt = polynomial_features.fit_transform(X_test)
    y_pred_bt = model_bt.predict(x_poly_test_bt)
    rmse_bt.append(mean_squared_error(y_test, y_pred_bt, squared=False))

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 1].set_xlabel('Polynomial Degree')
axs[0, 1].set_ylabel('RMSE')
axs[0, 1].set_title('Boosted Tree')
plt.tight_layout()
plt.show()