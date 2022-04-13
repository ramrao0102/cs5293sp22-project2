# This is Ramkishore Rao's Project 2

## Introduction

This project consists of a developing code called project2.py that takes an input file called yummly.json.  The cuisine data and the ingredients needed to make the <br> cuisine are compared to the ingredients that are passed in from the console to identify the cuisine that can be made from the ingredients passed from the console. <br> The yummly file has a json format.

The file has the following format:

{ <br>
    "id": 10259, <br>
    "cuisine": "greek", <br>
    "ingredients": [ <br>
      "romaine lettuce", <br>
      "black olives", <br>
      "grape tomatoes", <br>
      "garlic", <br>
      "pepper", <br>
      "purple onion", <br>
      "seasoning", <br>
      "garbanzo beans", <br>
      "feta cheese crumbles" <br>
    ] <br>
  }, <br>

## Packages

Several packages have been included in the project2.py file.  Note also that the pipfile and piplock have been setup to <br>accomodate use of the noted packages. <br>
1) Numpy <br>
2) Pandas <br>
3) sys <br>
4) json <br>
5) sklearn has several packages imported, but we are using cosine_similarity and TfidfVectorizer

## Functions in Project2.py

The project.py file contains several functions.  The functions and the purpose of each function is outlined below: <br>

1) read_data:  used to read the .json file and load into a pandas dataframe. <br>
2) len_dataframe(df) : reads in the dataframe and estimates the length of the dataframe. <br>
3) remove_space(df): reads in the dataframe and removes any whitespace that is contained within a string such as say rice <br> krispies and combines it into  ricekrispies. <br>
4) find_cuisines(df): reads in the dataframe and returns a set of unique cuisines. <br>
5) createset_ingredients(df): creates a set of ingredients in the dataframe. <br>
6) stringify_ingredients(df): reads in the dataframe and takes the ingrdedients column which has disjointed strings of <br>ingredients and combines them into a single string but does leave white space between each type of ingredient. <br>
7) createvectorizer(ingredients): takes the ingredients from the dataframe that has been processed with the  <br>stringify_ingredients>br> and creates a TfidfVectorizer and returns a transformer tf_idfmatrix of the ingredients.<br>
8) create_cosinematrix(matrix):  takes in the tf_idfmatrix of the ingredients and returns a cosine_similarity matrix that <br>presents the cosine_similarity score between the ingredients. <br>
9) createdf_ingredients(ingredients):  this is used to create a dataframe of 1 column containing the ingredients and this is <br>used to identify an array of ingredients that mostly closely match those that are passed in from the console. <br>

## Console Input of Ingredients

Ingredients are passed in from the console using the following command line input <br>

pipenv run python project2.py --N 5 --ingredient paprika \ <br>
                                    --ingredient banana \ <br>
                                    --ingredient "rice krispies"<br>
Note that 5 is the # of cuisines that most closely match the best cuisine.<br>                                  

## Execution of the Program

1) The passed in ingredients from the console have the spaces removed, for example between rice and krispies and then also <br>
stringified to convert all passed in strings into 1 string with white space between each ingredient. <br>

2) We call the stringify_ingredients to stringify the ingredients in the dataframe and then append the ingredients passed in <br>
   from the console as the final element <br>

3) We call the createvectorizer function to make a Tfidf matrix of the ingredients.<br>

4) We call the create cosine_matrix function on the Tfidf matrix of the ingredients to find the cosine_similarity score <br>
between the ingredients.<br>

5) From there we obtain pairwise ingredient similarities for all ingredients with the passed in list of ingredients <br>
from the console.  Note I am using length of the initial dataframe created from the json file for this as this is the <br>
index position of the ingredients passed in to the ing_df dataframe of the ingredients. <br> 

6) Once we have them depending on the number of closest cuisine matches we need to the best cuisine, we then <br>
  get the index positions of those ingredientss as numpy array using the following command:<br>
   ingre_simil_idxs = np.argsort(-ingredient_similarities)[1:no_in_array]     <br>
where no_in_array is the # after --N passed in from console +2 <br>

7) From there we create dictionary object to present the best cuisine and the 5 closest cuisines and transfer the contents <br>
of the dictionary to a json object to print to the console. The scores for the best cuisine is the cosine similarity score of that cuisine <br>
relative to the ingredients passed in from the console.  The scores of the next 5 cuisines are also cosine similarity scores of those cuisines <br>
relative to the ingredients passed in from the console. <br>

Example json output is provided below. <br>

{<br>
    "cuisine": "spanish", <br>
    "score": 0.1336353701160973,<br>
    "closest": [<br>
        {<br>
            "id": 28972, <br>
            "Score": 0.13084260177062557 <br>
        }, <br>
        { <br>
            "id": 26401, <br>
            "Score": 0.12220620981401238 <br>
        },<br>
        { <br>
            "id": 20283, <br>
            "Score": 0.12144811538221847 <br>
        }, <br>
        { <br>
            "id": 27685, <br>
            "Score": 0.11930782694671059 <br>
        }, <br>
        { <br>
            "id": 19122, <br>
            "Score": 0.11591712687272727 <br>
        } <br>
    ] <br>
} <br>

## Testing of the Program

Several tests were developed to test the following: <br>
    
1) check for creation of the dataframe by checking length of dataframe equal to known length of the index sliced dataframe via an   assert statement. 
2) check that find_cuisines function is running correctly by comparing length of the set of unique cuisines to the known length via an assert statement. <br>
3) check that the create_vectorizer function executes by running a small subset of the dataframe and confirm that the length matches the known length. <br>
4) check that create_cosinematrix(matrix) function executes by running a small subse of the dataframe parsed through the Tfidf function and check that the length of the matrx matches the known length. <br>
    
