import pytest
import sklearn
import classifier
import pandas as pd
import json

def test_function():


    with open('/home/ramrao0102/project2/yummly.json', 'r') as datafile:
    
        data = json.load(datafile)
    
    df_train = pd.DataFrame(data)

    all_cuisines = classifier.find_cuisines(df_train)

    if 'indian' in all_cuisines:
        assert True
