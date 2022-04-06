
from classifier import find_cuisines
import pandas as pd

df_train = pd.read_json('/home/ramrao0102/project2/yummly.json')

all_cuisines = find_cuisines(df_train)

print(all_cuisines)
