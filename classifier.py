
import numpy as np

import pandas as pd

import json

import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


def read_data():

    with open('yummly.json', 'r') as datafile:
    
        data = json.load(datafile)
    
    df_train = pd.DataFrame(data)

            
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

    #print(df_train.tail())

    all_cuisines = find_cuisines(df_train)
    
    #print(all_cuisines)

    
    all_ingredients = createset_ingredients(df_train)


    #print(len(all_ingredients))

    vocabulary = {}

    for ingredient, i in zip(all_ingredients, range(len(all_ingredients))):
        vocabulary[ingredient] = i
    

    str_train_ingredients = stringify_ingredients(df_train)

    
    arg_ls = sys.argv

    ingr_list = []

    
    for j in range(len(arg_ls)):

        if arg_ls[j] == "--N":

            no_of_closematches = arg_ls[j+1]

        if arg_ls[j] == "--ingredient":
            
            ingr_list.append(arg_ls[j+1])

    

    for i in range(len(ingr_list)):

        ingr_list[i] = ingr_list[i].replace(" ", "")

    

    
    str = ""

    for i in range(len(ingr_list)):

        if i == 0:
            str += ingr_list[i]

        else:
            str += " " + ingr_list[i]

    
    


    str_train_ingredients.append(str)
 

    tf_idf_matrix = createvectorizer(str_train_ingredients)

    #print(tf_idf_matrix.shape)


    doc_sim_df = create_cosinematrix(tf_idf_matrix)


    #print(doc_sim_df.head())


    ing_df = createdf_ingredients(str_train_ingredients)

    ing1 = ing_df[0].values


    ingredient_similarities = doc_sim_df[len(df_train)].values

    no_in_array = int(no_of_closematches) + 2

    ingre_simil_idxs = np.argsort(-ingredient_similarities)[1:no_in_array]

    #print(type(ingre_simil_idxs))

    similar_ingredients = ing1[ingre_simil_idxs]

    #print(similar_ingredients)

    cuisineids = df_train['id'][ingre_simil_idxs].values
    
    cuisines =  df_train['cuisine'][ingre_simil_idxs].values

    cosinescores = doc_sim_df[len(df_train)][ingre_simil_idxs].values
    
    cuisineids = cuisineids.tolist()
    
    cuisines = cuisines.tolist()
    
    cosinescores = cosinescores.tolist()
    
    mylist = []

    for i in range(len(cuisineids)):
        mydict = {'id': None, 'Score': None}
        if i >=1:
            mydict['id'] = cuisineids[i]
            mydict['Score'] = cosinescores[i]
        mylist.append(mydict)
        
    mylist = mylist[1:]

    #print(mylist)
            
    mydictfinal = {}

    mydictfinal.update([('cuisine', cuisines[0])])

    mydictfinal.update([('score', cosinescores[0])])

    mydictfinal.update([('closest', mylist)])

    #print(mydictfinal) 
    
    jsonString = json.dumps(mydictfinal, indent=4)
    print(jsonString)
     
    
 
