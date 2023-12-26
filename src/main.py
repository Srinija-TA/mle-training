import logging
import logging.config
import os
import pickle

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger(__name__)

def train_linear_regression(housing_prepared, housing_labels, output_folder):
    with mlflow.start_run(run_name="LinearRegressionExperiment", nested=True):
        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)
        linear_filename = os.path.join(
            output_folder, "pickles", "linear_regression.pkl"
        )
        with open(linear_filename, "wb") as linear_filename:
            pickle.dump(lin_reg, linear_filename)
        housing_predictions = lin_reg.predict(housing_prepared)
        lin_mse = mean_squared_error(housing_labels, housing_predictions)
        mlflow.log_metric("mse", lin_mse)
        lin_rmse = np.sqrt(lin_mse)
        mlflow.log_metric("rmse", lin_rmse)
        lin_mae = mean_absolute_error(housing_labels, housing_predictions)
        mlflow.log_metric("mae", lin_mae)
        mlflow.sklearn.log_model(lin_reg, "linear_reg_model")

    return lin_rmse, lin_mse, lin_mae


def train_decision_tree_regression(housing_prepared, housing_labels, output_folder):
    with mlflow.start_run(run_name="DecisionTreeExperiment", nested=True):
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(housing_prepared, housing_labels)
        decisiontree_filename = os.path.join(
            output_folder, "pickles", "DesisionTree_regression.pkl"
        )
        with open(decisiontree_filename, "wb") as decisiontree_filename:
            pickle.dump(tree_reg, decisiontree_filename)
        housing_predictions = tree_reg.predict(housing_prepared)
        tree_mse = mean_squared_error(housing_labels, housing_predictions)
        mlflow.log_metric("mse", tree_mse)
        tree_rmse = np.sqrt(tree_mse)
        mlflow.log_metric("rmse", tree_rmse)
        mlflow.sklearn.log_model(tree_reg, "decision_tree_reg_model")

    return tree_mse, tree_rmse
def train_random_forest_regressor1(housing_prepared, housing_labels, output_folder):
    with mlflow.start_run(run_name="RandomForestRandomizedSearch", nested=True):
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
        randomforest_filename = os.path.join(
            output_folder, "pickles", "randomforest_regression_RandomizedSearch.pkl"
        )
        with open(randomforest_filename, "wb") as randomforest_filename:
            pickle.dump(rnd_search, randomforest_filename)

        mlflow.log_params(rnd_search.best_params_)
        mlflow.log_param("model", "RandomForest")
        best_rf_model = rnd_search.best_estimator_
        mlflow.sklearn.log_model(best_rf_model, "best_rf_model")
        housing_predictions = best_rf_model.predict(housing_prepared)
        rnd_mse = mean_squared_error(housing_labels, housing_predictions)
        mlflow.log_metric("mse", rnd_mse)
        rnd_rmse = mean_absolute_error(housing_labels, housing_predictions)
        mlflow.log_metric("rmse", rnd_rmse)
        cvres = rnd_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            logger.info(" %s, %s", np.sqrt(-mean_score), params)
            print(np.sqrt(-mean_score), params)
    return rnd_mse, rnd_rmse


def train_random_forest_regressor2(housing_prepared,housing_labels,output_folder,strat_test_set,imputer):
    with mlflow.start_run(run_name="RandomForestGridSearch", nested=True):
        param_grid = [
            # try 12 (3×4) combinations of hyperparameters
            {
                "n_estimators": [3, 10, 30],
                "max_features": [2, 4, 6, 8],
            },
            # then try 6 (2×3) combinations with bootstrap set as False
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
        randomforest_filename2 = os.path.join(
            output_folder, "pickles", "randomforest_regression_GridSearch.pkl"
        )
        with open(randomforest_filename2, "wb") as randomforest_filename2:
            pickle.dump(grid_search, randomforest_filename2)
        cvres = grid_search.cv_results_
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model", "RandomForest_with_gridsearchcv")
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            logger.info(" %s, %s", np.sqrt(-mean_score), params)

        feature_importances = grid_search.best_estimator_.feature_importances_
        sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

        final_model = grid_search.best_estimator_
        mlflow.sklearn.log_model(final_model, "best_rf_model")

        X_test = strat_test_set.drop("median_house_value", axis=1)
        y_test = strat_test_set["median_house_value"].copy()

        X_test_num = X_test.drop("ocean_proximity", axis=1)

        X_test_prepared = imputer.transform(X_test_num)
        X_test_prepared = pd.DataFrame(
            X_test_prepared, columns=X_test_num.columns, index=X_test.index
        )
        X_test_prepared["rooms_per_household"] = (
            X_test_prepared["total_rooms"] / X_test_prepared["households"]
        )
        X_test_prepared["bedrooms_per_room"] = (
            X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
        )
        X_test_prepared["population_per_household"] = (
            X_test_prepared["population"] / X_test_prepared["households"]
        )
        X_test_cat = X_test[["ocean_proximity"]]
        X_test_prepared = X_test_prepared.join(
            pd.get_dummies(X_test_cat, drop_first=True)
        )
        test_data = pd.concat([X_test_prepared, y_test], axis=1)
        test_data.to_csv(os.path.join("data/processed/", "test.csv"), index=False)
        final_predictions = final_model.predict(X_test_prepared)
        final_mse = mean_squared_error(y_test, final_predictions)
        mlflow.log_metric("mse", final_mse)
        final_rmse = np.sqrt(final_mse)
        mlflow.log_metric("rmse", final_rmse)

    return final_mse, final_rmse


def score(housing_prepared, housing_labels, strat_test_set, imputer, output_folder):
    """Function to score the modules.

    Parameters
    ----------
    housing_prepared : pd.DataFrame
    housing_labels : pd.Series
    strat_test_set : pd.DataFrame
    imputer : SimpleImputer

    Returns
    -------
    None

    """
    scores_file = open(
        os.path.join(output_folder, "scores.txt"),
        "a",
    )
    exp_name = "housing_price_prediction"
    os.makedirs("../output_folder/pickles",exist_ok=True)
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name="Parent_run"):
        lin_rmse, lin_mse, lin_mae = train_linear_regression(
            housing_prepared, housing_labels, output_folder
        )
        scores_file.write(
            f"Linear Regression Model - MSE: {lin_mse}, RMSE: {lin_rmse}\n"
        )
        tree_mse, tree_rmse = train_decision_tree_regression(
            housing_prepared, housing_labels, output_folder
        )
        scores_file.write(f"Decision Tree Model - MSE: {tree_mse}, RMSE: {tree_rmse}\n")

        rnd_mse, rnd_rmse = train_random_forest_regressor1(
            housing_prepared,
            housing_labels,
            output_folder,
        )
        scores_file.write(f"Random Forest Model1- MSE: {rnd_mse}, RMSE: {rnd_rmse}\n")

        final_mse, final_rmse = train_random_forest_regressor2(
            housing_prepared,
            housing_labels,
            output_folder,
            strat_test_set,
            imputer,
        )
        scores_file.write(
            f"Random Forest Model- MSE: {final_mse}, RMSE: {final_rmse}\n"
        )
        logger.info(" %s", final_rmse)
    return
