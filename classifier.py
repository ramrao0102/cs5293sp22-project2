
import numpy as np

import pandas as pd

import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


def read_data():

    df_train = pd.read_json('yummly.json')

    df_train = df_train.iloc[:32500]
        
    return df_train

def len_dataframe(df):

    len_df = len(df)

    return len_df


def remove_space(df):

    for i in df.index:
        for j in range(len(df['ingredients'][i])):
            df['ingredients'][i][j] = df['ingredients'][i][j].replace(" ", "")

    return df


def find_cuisines(df):
        
    all_cuisines = set(df['cuisine'])

    return all_cuisines


def createset_ingredients(df):

    all_ingredients = set()

    for ingredients in df['ingredients']:
        all_ingredients = all_ingredients | set(ingredients)

    return all_ingredients


def stringify_ingredients(df):

    return [' '.join(ingredients) for ingredients in df['ingredients']]


def createvectorizer(ingredients):

    tf = TfidfVectorizer()
    tf_idf_matrix = tf.fit_transform(ingredients)

    return tf_idf_matrix
    

def create_cosinematrix(matrix):

    doc_sim = cosine_similarity(matrix)

    doc_sim_df = pd.DataFrame(doc_sim)

    return doc_sim_df


def createdf_ingredients(ingredients):

    ing_df = pd.DataFrame(ingredients)

    return ing_df


if __name__ =='__main__':
 
    df_train = read_data() 

    df_train = remove_space(df_train)

    len_df = len_dataframe(df_train) 

    print(df_train.tail())

    all_cuisines = find_cuisines(df_train)
    
    print(all_cuisines)

    
    all_ingredients = createset_ingredients(df_train)


    print(len(all_ingredients))

    vocabulary = {}

    for ingredient, i in zip(all_ingredients, range(len(all_ingredients))):
        vocabulary[ingredient] = i
    

    str_train_ingredients = stringify_ingredients(df_train)

    str_train_ingredients.append('paprika banana ricekrispies')
 

    tf_idf_matrix = createvectorizer(str_train_ingredients)

    print(tf_idf_matrix.shape)


    doc_sim_df = create_cosinematrix(tf_idf_matrix)


    print(doc_sim_df.head())


    ing_df = createdf_ingredients(str_train_ingredients)

    ing1 = ing_df[0].values


    ingredient_similarities = doc_sim_df[32500].values

    print(ingredient_similarities)

    ingre_simil_idxs = np.argsort(-ingredient_similarities)[1:6]

    print(type(ingre_simil_idxs))

    similar_ingredients = ing1[ingre_simil_idxs]

    print(similar_ingredients)

    print(df_train['id'][ingre_simil_idxs] , df_train['cuisine'][ingre_simil_idxs])

    print(doc_sim_df[32500][ingre_simil_idxs])

