# tests/unit/test_nonstandardcode_unit.py
import pytest
from mypackage import nonstandardcode

def test_fetch_housing_data():
    nonstandardcode.fetch_housing_data()

def test_load_housing_data():
    data = nonstandardcode.load_housing_data()

def test_data_preprocessing():
    housing = nonstandardcode.load_housing_data()