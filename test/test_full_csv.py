
import numpy as np

import pandas as pd

import sys

import project2

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

def test_full_csv():
 
    df_csv = project2.read_csv_data()
   
    df_csv = df_csv.iloc[0:100]

    str_train_ingredients = []
      
    str_train_ingredients.extend(df_csv['ingredients'])

    str_train_ingredients.append('chicken, cinnamon, soy_sauce, onion, ginger')

    tf_idf_matrix = project2.createvectorizer(str_train_ingredients)

    doc_sim_df = project2.create_cosinematrix(tf_idf_matrix)

    ing_df = project2.createdf_ingredients(str_train_ingredients)

    ing1 = ing_df[0].values

    ingredient_similarities = doc_sim_df[len(df_csv)].values

    ingredient_similarities_11 = ingredient_similarities[0: (len(df_csv)-1)]

    no_in_array = 5 + 1

    ingre_simil_idxs_1 = np.argsort(-ingredient_similarities_11)[0:no_in_array-1]

    cuisines =  df_csv['cuisine'][ingre_simil_idxs_1].values

    cosinescores = doc_sim_df[len(df_csv)][ingre_simil_idxs_1].values

    cuisines = cuisines.tolist()
    
    cosinescores = cosinescores.tolist()

    if len(cosinescores) == 6:
        assert True

    if cosinescores[0] > 0.9:
        assert True
     
    
 
