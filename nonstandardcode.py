import os
import tarfile
import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedShuffleSplit)
from sklearn.tree import DecisionTreeRegressor


# Function to fetch housing data
def fetch_housing_data(
    housing_url=(
        "https://raw.githubusercontent.com/ageron/"
        "handson-ml/master/datasets/housing/housing.tgz"
    ),
    housing_path="datasets/housing",
):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Function to load housing data
def load_housing_data(housing_path="datasets/housing"):
    csv_path = "housing.csv"
    return pd.read_csv(csv_path)


# Fetch and load housing data
# fetch_housing_data()
housing = load_housing_data()

# Create income categories
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

# Stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Drop income_cat from sets
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Copy training set for exploration
housing = strat_train_set.copy()

# Visualize data
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# Compute correlation matrix
housing_numeric = housing.drop(columns=["ocean_proximity"])
corr_matrix = housing_numeric.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# Feature engineering
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = (
    housing["total_bedrooms"] / housing["total_rooms"]
)

housing["population_per_household"] = (
    housing["population"] / housing["households"]
)

# Prepare data for ML algorithms
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)

housing_cat = housing[["ocean_proximity"]]
housing_prepared = pd.get_dummies(housing_cat, drop_first=True)
housing_prepared = pd.concat([housing_tr, housing_prepared], axis=1)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

# Decision Tree
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

# Random Forest with RandomizedSearchCV
param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_distribs,
    n_iter=10,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
)
rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# Random Forest with GridSearchCV
param_grid = [
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

# Separate numerical features and categorical features
X_test_num = X_test.drop("ocean_proximity", axis=1)
X_test_cat = X_test[["ocean_proximity"]]

# Impute missing values in numerical features
X_test_num_imputed = imputer.transform(X_test_num)
X_test_num_imputed = pd.DataFrame(
    X_test_num_imputed, columns=X_test_num.columns, index=X_test_num.index
)

# Feature engineering for numerical features
X_test_num_imputed["rooms_per_household"] = (
    X_test_num_imputed["total_rooms"] / X_test_num_imputed["households"]
)
X_test_num_imputed["bedrooms_per_room"] = (
    X_test_num_imputed["total_bedrooms"] / X_test_num_imputed["total_rooms"]
)
X_test_num_imputed["population_per_household"] = (
    X_test_num_imputed["population"] / X_test_num_imputed["households"]
)

# One-hot encode categorical features
X_test_cat_encoded = pd.get_dummies(X_test_cat, drop_first=True)

# Combine numerical and categorical features
X_test_prepared = pd.concat([X_test_num_imputed, X_test_cat_encoded], axis=1)

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
