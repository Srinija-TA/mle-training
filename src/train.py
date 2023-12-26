import logging
import os
import pickle

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def training_data(housing, output_folder):
    """Function to train the Housing data.

    Parameters
    ----------
    housing : pd.DataFrame
        housing dataframe returned by the ingest_data

    Returns
    -------
    housing_prepared
    housing_labels
    strat_test_set
    imputer

    """
    logger.info("Training the housing data")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    def income_cat_proportions(data):
        """Calculates the frequency of income category in a given dataset.

        Parameters
        -----------
        data : pd.DataFrame
        Pandas DataFrame containing the dataset.

        Returns
        --------
        pd.Series:Pandas Series containing the income category frequency.

        """

        return data["income_cat"].value_counts() / len(data)

    # train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    housing.drop("ocean_proximity", axis=1, inplace=True)
    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    os.makedirs("../output_folder", exist_ok=True)

    imputer_filename = os.path.join(output_folder, "imputer.pkl")
    with open(imputer_filename, "wb") as imputer_file:
        pickle.dump(imputer, imputer_file)
    logger.info(f"Imputer saved to: {imputer_filename}")
    return housing_prepared, housing_labels, strat_test_set, imputer
