import pytest
import sklearn
import classifier
import pandas as pd


def test_function():

    df_train = pd.read_json('/home/ramrao0102/project2/yummly.json')

    all_cuisines = classifier.find_cuisines(df_train)

    if 'indian' in all_cuisines:
        assert True
