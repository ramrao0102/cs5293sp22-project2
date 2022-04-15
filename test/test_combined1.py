
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


def test_combined():
 
    df_train = project2.read_data()

    df_train = df_train.iloc[0:100]

    df_train = project2.remove_space(df_train) 

    df_csv = project2.read_csv_data()

    df_csv = df_csv.iloc[0:100]

    combined_df = pd.concat([df_train, df_csv])

    combined_df.reset_index(drop=True, inplace=True)

    all_cuisines = project2.find_cuisines(df_train)
    
    all_ingredients = project2.createset_ingredients(df_train)

    vocabulary = {}

    for ingredient, i in zip(all_ingredients, range(len(all_ingredients))):
        vocabulary[ingredient] = i
    
    str_train_ingredients = project2.stringify_ingredients(df_train)

    str_train_ingredients.extend(df_csv['ingredients'])

    str_train_ingredients.append('chicken, cinnamon, soy_sauce, onion, ginger')

    tf_idf_matrix = project2.createvectorizer(str_train_ingredients)

    # print(tf_idf_matrix.shape)


    doc_sim_df = project2.create_cosinematrix(tf_idf_matrix)


    # print(doc_sim_df.head())


    ing_df = project2.createdf_ingredients(str_train_ingredients)

    ing1 = ing_df[0].values


    # the below 2 lines find the array index positions in the cosine_similarity matrix for
    # cusines first and then the cuusine ids. cusineids index position only extends to end of 
    # oolumn positions in the json file or the df_train matrix

    ingredient_similarities = doc_sim_df[len(combined_df)].values

    ingredient_similarities11 = ingredient_similarities[0: (len(combined_df)-1)]

    #print(ingredient_similarities)

    ingredient_similarities_1 = ingredient_similarities11[0:(len(df_train))]

    #print(ingredient_similarities_1)

    no_in_array = 5 + 1

    # the below 2 lines provide the index position for the cuisines first and the line below
    # that line shows the index positions for the cuisine ids.

    ingre_simil_idxs0 = np.argsort(-ingredient_similarities)

    ingre_simil_idxs = np.argsort(-ingredient_similarities11)[0:1]

    #print(ingre_simil_idxs)

    #print(ingre_simil_idxs[0])

    #print(type(ingre_simil_idxs[0]))

    ingre_simil_idxs_2 = np.argsort(-ingredient_similarities_1)[0:no_in_array-1]

    ingre_simil_idxs_1 = np.argsort(-ingredient_similarities_1)[1:no_in_array]

    #print(ingre_simil_idxs_1)

    similar_ingredients = ing1[ingre_simil_idxs]

    # print(similar_ingredients)

    if ingre_simil_idxs[0] > (len(df_train)-1):

        cuisineids = df_train['id'][ingre_simil_idxs_2].values
   
    else:

        cuisineids = df_train['id'][ingre_simil_idxs_1].values

    #cuisines = combined_df['cuisine'][11]

    cuisines =  combined_df['cuisine'][ingre_simil_idxs].values

    cosinescores = doc_sim_df[len(combined_df)][ingre_simil_idxs].values
    
    if ingre_simil_idxs[0] > (len(df_train)-1):

        cosinescores1 = doc_sim_df[len(combined_df)][ingre_simil_idxs_2].values
        
    else:        

        cosinescores1 = doc_sim_df[len(combined_df)][ingre_simil_idxs_1].values
    
    cuisineids = cuisineids.tolist()
    
    cuisineids = [0.0] + cuisineids

    cuisines = cuisines.tolist()
    
    cosinescores = cosinescores.tolist()

    cosinescores1 = cosinescores1.tolist()

    cosinescores.extend(cosinescores1)


    assert cosinescores[0] >0.9
    

    assert (len(cosinescores)) ==6
    
     
    
 
