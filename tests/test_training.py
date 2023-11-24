import os

import pandas as pd

from ingest_data import get_data


def test_get_data(tmpdir):
    output_folder = tmpdir.mkdir("output_test_folder")
    preprocessed_data = get_data(output_folder=output_folder)
    assert isinstance(preprocessed_data, pd.DataFrame)
    assert os.path.exists(os.path.join(output_folder, "housing.csv"))
