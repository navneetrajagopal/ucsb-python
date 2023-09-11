#libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

#model
# Load the dataset
df = pd.read_csv('/users/navneet/downloads/housing_cleaned.csv')
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
param_grid3 = {'n_neighbors': range(1, 21)}
model3 = KNeighborsClassifier()
grid_search3 = GridSearchCV(model3, param_grid=param_grid3, cv=5, scoring='accuracy')
grid_search3.fit(X_train, y_train)
best_model3 = grid_search3.best_estimator_
y_pred3 = best_model3.predict(X_test)
mse = mean_squared_error(y_test, y_pred3)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred3)
print('Best k:', best_model3.get_params()['n_neighbors'])
print('RMSE:', rmse)
print('R²:', r2)

#visualizations
rmse_knn = []
for n_neighbors in range(1, 9):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rmse_knn.append(rmse)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[1, 0].plot(range(1, 9), rmse_knn, marker='o')
axs[1, 0].set_xlabel('Number of Neighbors')
axs[1, 0].set_ylabel('RMSE')
axs[1, 0].set_title('K-Nearest Neighbors')
plt.tight_layout()
plt.show()