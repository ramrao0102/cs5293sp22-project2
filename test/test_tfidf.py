import pytest
import sklearn
import project2
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


def test_tfidf():

    df = project2.read_data()

    df = df[:100]

    ingredients = project2.stringify_ingredients(df)

    tfidf_matrix = project2.createvectorizer(ingredients)

    assert (tfidf_matrix.shape[0]) == 100
