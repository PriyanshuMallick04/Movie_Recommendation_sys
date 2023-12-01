# ----------Movie Recommendation System----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

movies = pd.read_csv(r"D:\PYTHON\Projects\Movie_Recommendation_sys\movies.csv")
credits = pd.read_csv(r"D:\PYTHON\Projects\Movie_Recommendation_sys\credits.csv")

# pd.set_option("display.max_columns", None)
# print(movies.head())
# print(credits.head())


"""Merge the two dataframes into one"""
movies = movies.merge(credits, on='title')


"""Selecting Relevant Columns"""
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
# print(movies.shape)
movies.dropna(inplace=True)    # Deleting the missing values
# print(movies.isnull().sum())
# print(movies.duplicated().sum())
# print(movies.iloc[0]['genres'])


"""Converting some columns to a readable format"""
def convert(text):
    l = []
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l

def convert_cast(text):    # we need the actors name not the character name
    l = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            l.append(i['name'])
        counter += 1
    return l

def fetch_director(text):    # we need only director name from the crew
    l = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)


"""Convert the overview column from string to list"""
movies['overview'] = movies['overview'].apply(lambda x: x.split())


"""Removing spaces between two words"""
def remove_space(word):
    l = []
    for i in word:
        l.append(i.replace(" ", ''))
    return l

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

# print(movies.iloc[0]['keywords'])


"""Merge all the columns to tags"""
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
# print(movies.head())


"""Removing unnecessary Columns"""
new_df = movies[['movie_id', 'title', 'tags']]


"""Converting tags from list to string"""
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


"""Converting to lowerCase"""
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
# print(new_df.head())


"""Stemming"""
ps = PorterStemmer()
def stems(text):
    l = []
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)

new_df['tags'] = new_df['tags'].apply(stems)
# print(new_df.iloc[0]['tags'])


"""Applying count vectorizer"""
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()
# print(vector.shape)


"""Cosine Similarity"""
similarity = cosine_similarity(vector)
# print(similarity)

# print(new_df[new_df['title'] == 'Spider-Man'].index[0])


"""Recommender Function"""
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1: 6]:
        print(new_df.loc[i[0]].title)


# pickle.dump(new_df, open('artifacts/movie_list.pk1', 'wb'))
# pickle.dump(similarity, open('artifacts/similarity.pk1', 'wb'))

# recommend('Spider-Man')
