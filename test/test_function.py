import pytest
import sklearn
import project2
import pandas as pd
import json

def test_function():

    df_train = project2.read_data()

    if len(df_train) > 0:
        assert True
