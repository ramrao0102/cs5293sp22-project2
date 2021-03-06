
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


def read_csv_data():

    colnames = []
    
    colnames.append('cuisine')

    for i in range(33):

        if i >= 2:

            colnames.append('ingr' + str(i-1))


    df1 = pd.read_csv('srep00196-s3.csv', low_memory = False , skiprows = 4, names = colnames, on_bad_lines='skip')

    df1['ingredients1'] = df1['ingr1'].fillna('') + ' ' +df1['ingr2'].fillna('') + ' ' + df1['ingr3'].fillna('')+ ' ' +df1['ingr4'].fillna('') + ' ' +df1['ingr5'].fillna('') + df1['ingr6'].fillna('') + ' ' +df1['ingr7'].fillna('') + ' ' + df1['ingr8'].fillna('')+ ' ' +df1['ingr9'].fillna('') + ' ' +df1['ingr10'].fillna('')

    df1.drop(columns= ['ingr1', 'ingr2', 'ingr3', 'ingr4', 'ingr5' , 'ingr6', 'ingr7', 'ingr8', 'ingr9', 'ingr10'],  inplace=True)


    df1['ingredients2'] = df1['ingr11'].fillna('') + ' ' +df1['ingr12'].fillna('') + ' ' + df1['ingr13'].fillna('') + ' ' +df1['ingr14'].fillna('') + ' ' +df1['ingr15'].fillna('') + ' ' + df1['ingr16'].fillna('') + ' ' +df1['ingr17'].fillna('') + ' ' + df1['ingr18'].fillna('')+ ' ' +df1['ingr19'].fillna('') + ' ' +df1['ingr20'].fillna('')

    df1.drop(columns= ['ingr11', 'ingr12', 'ingr13', 'ingr14', 'ingr15', 'ingr16', 'ingr17', 'ingr18', 'ingr19', 'ingr20'],  inplace=True)


    df1['ingredients3'] = df1['ingr21'].fillna('') + ' ' +df1['ingr22'].fillna('') + ' ' + df1['ingr23'].fillna('')+ ' ' +df1['ingr24'].fillna('') + ' ' +df1['ingr25'].fillna('') + ' ' +  df1['ingr26'].fillna('') + ' ' +df1['ingr27'].fillna('') + ' ' + df1['ingr28'].fillna('')+ ' ' +df1['ingr29'].fillna('') + ' ' +df1['ingr30'].fillna('')

    df1.drop(columns= ['ingr21', 'ingr22', 'ingr23', 'ingr24', 'ingr25', 'ingr26', 'ingr27', 'ingr28', 'ingr29', 'ingr30'],  inplace=True)


    df1['ingredients4'] = df1['ingr31'].fillna('')

    df1.drop(columns= ['ingr31'],  inplace=True)


    df1['ingredients'] = df1['ingredients1'] + ' ' + df1['ingredients2'] + ' ' +  df1['ingredients3'] + ' ' + df1['ingredients4']


    df1.drop(columns = ['ingredients1', 'ingredients2', 'ingredients3', 'ingredients4'], inplace =True)

    df1 = df1.iloc[0:35000]

    return df1


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

    df_csv = read_csv_data()

    # below code is to concatenante the json input file and the csv file

    combined_df = pd.concat([df_train, df_csv])

    combined_df.reset_index(drop=True, inplace=True)

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

        if "--N" in arg_ls[j]:

            no_of_closematches = arg_ls[j+1]

        if "--ingredient" in arg_ls[j]:
            
            ingr_list.append(arg_ls[j+1])

    

    for i in range(len(ingr_list)):

        ingr_list[i] = ingr_list[i].replace(" ", "")

    

    
    str = ""

    for i in range(len(ingr_list)):

        if i == 0:
            str += ingr_list[i]

        else:
            str += " " + ingr_list[i]

 
    # the below 2 lines help create a list that includes ingredients from both df_train and df_csv

    #str_train_ingredients1 = stringify_ingredients(df_csv)
    
    #print(str_train_ingredients1)

    
    str_train_ingredients.extend(df_csv['ingredients'])

    str_train_ingredients.append(str)

    tf_idf_matrix = createvectorizer(str_train_ingredients)

    # print(tf_idf_matrix.shape)


    doc_sim_df = create_cosinematrix(tf_idf_matrix)


    # print(doc_sim_df.head())


    ing_df = createdf_ingredients(str_train_ingredients)

    ing1 = ing_df[0].values


    # the below 2 lines find the array index positions in the cosine_similarity matrix for
    # cusines first and then the cuusine ids. cusineids index position only extends to end of 
    # oolumn positions in the json file or the df_train matrix

    ingredient_similarities = doc_sim_df[len(combined_df)].values

    ingredient_similarities11 = ingredient_similarities[0: (len(combined_df)-1)]

    #print(ingredient_similarities)

    ingredient_similarities_1 = ingredient_similarities11[0:(len(df_train))]

    #print(ingredient_similarities_1)

    no_in_array = int(no_of_closematches) + 1

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

    # print(cosinescores)

    mylist = []

    for i in range(len(cuisineids)):
        mydict = {'id': None, 'Score': None}
        if i >=1:
            mydict['id'] = cuisineids[i]
            mydict['Score'] = cosinescores[i]
        
        mylist.append(mydict)
        
    mylist = mylist[1:]

    # print(mylist)
            
    mydictfinal = {}

    mydictfinal.update([('cuisine', cuisines[0])])

    mydictfinal.update([('score', cosinescores[0])])

    mydictfinal.update([('closest', mylist)])

    # print(mydictfinal) 
   
    jsonString = json.dumps(mydictfinal, indent=4)
    print(jsonString)
     
    
 
