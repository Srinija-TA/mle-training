import os
import tarfile
import urllib

import numpy as np
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "/home/srinija2001/mle-training"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def get_data(output_folder):
    """Fetches housing data from a given URL and return it.

    Parameters
    -----------
    None

    Returns
    --------
    housing data

    """
    housing = load_housing_data()

    # train_set, test_set = train_test_split(
    # housing, test_size=0.2,
    # random_state=42
    # )

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    return housing


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Fetches housing data from a given URL and extracts it to path.

    Parameters
    -----------
    housing_url : str
    URL of the housing data file.
    housing_path : str
    Path to save the housing data.

    Returns
    --------
    None

    """

    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """Loads housing data from CSV file.

    Parameters
    -----------

    housing_path : str
    Path to the housing data CSV file.

    Returns
    --------

    pd.DataFrame:Pandas DataFrame containing the loaded housing data.

    """

    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)