#libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

#Dataset
df = pd.read_csv('/users/navneet/downloads/housing_cleaned.csv')
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print('RMSE:', rmse)
print('R²:', r2)

#visualization
rmse_rf = []
for depth in range(1, 9):
    model = RandomForestRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse_rf.append(rmse)

# Plot RMSE values for all four models
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[1, 1].plot(range(1, 9), rmse_rf, marker='o')
axs[1, 1].set_xlabel('Max Depth')
axs[1, 1].set_ylabel('RMSE')
axs[1, 1].set_title('Random Forest')
plt.tight_layout()
plt.show()