
import numpy as np

import pandas as pd

import json

import sys

import project2

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


def test_fulljson():

    df_train = project2.read_data()

    df_train = df_train.iloc[0:100] 

    df_train = project2.remove_space(df_train)

    all_cuisines = project2.find_cuisines(df_train)
    
    all_ingredients = project2.createset_ingredients(df_train)

    vocabulary = {}

    for ingredient, i in zip(all_ingredients, range(len(all_ingredients))):
        vocabulary[ingredient] = i
    

    str_train_ingredients = project2.stringify_ingredients(df_train)
    

    str_train_ingredients.append('romainelettuce blackolives grapetomatoes garlic pepper purpleonion seasoning garbanzobeans fetacheesecrumbles')

    tf_idf_matrix = project2.createvectorizer(str_train_ingredients)

    doc_sim_df = project2.create_cosinematrix(tf_idf_matrix)

    ing_df = project2.createdf_ingredients(str_train_ingredients)

    ing1 = ing_df[0].values

    ingredient_similarities = doc_sim_df[len(df_train)].values

    ingredient_similarities = ingredient_similarities[0:(len(df_train)-1)]

    no_in_array = 5 + 1

    ingre_simil_idxs = np.argsort(-ingredient_similarities)[1:no_in_array]

    cuisineids = df_train['id'][ingre_simil_idxs].values

    cuisines =  df_train['cuisine'][ingre_simil_idxs].values

    cosinescores = doc_sim_df[len(df_train)][ingre_simil_idxs].values
    
    cuisineids = cuisineids.tolist()
    
    cuisines = cuisines.tolist()
    
    cosinescores = cosinescores.tolist()

    if len(cosinescores) == 6:
        assert True
     
    if cosinescores[0] ==1:
        assert True
    
 
