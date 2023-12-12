import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib  # pyright: ignore

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("../data", "raw")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

logger = logging.getLogger(__name__)


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Fetches housing data from the given URL and extracts it to a specified path.

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
    logger.info("Fetches housing data from URL and extracting into specified path")
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """Loads housing data from a CSV file.

    Parameters
    -----------
    housing_path : str
    Path to the housing data CSV file.

    Returns
    --------
    pd.DataFrame:Pandas DataFrame containing the loaded housing data.

    """
    logger.info("Loads housing data from a CSV file")
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def get_data(output_folder):
    """Fetches housing data from a given URL and return it.

    Parameters
    -----------
    output_folder : str
    folder path to store the dataset

    Returns
    --------
    pd.DataFrame:Pandas DataFrame containing the loaded housing data.

    """
    logger.info("Fetches housing data from a given URL and returning it")
    fetch_housing_data()
    housing = load_housing_data()
    housing
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    output_dataset_path = os.path.join(output_folder, "housing.csv")
    housing.to_csv(output_dataset_path, index=False)
    return housing