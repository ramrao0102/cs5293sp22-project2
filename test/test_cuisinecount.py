
import pytest
import sklearn
import project2
import pandas as pd
import json


def test_cuisinescount():

    
    df_train = project2.read_data()


    all_cuisines = project2.find_cuisines(df_train)

    assert len(all_cuisines) == 20
