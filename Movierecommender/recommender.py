import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    return pd.read_csv("movies.csv")

def create_similarity_matrix(df):
    cv = CountVectorizer(tokenizer=lambda x: x.split('|'))
    count_matrix = cv.fit_transform(df['genres'])
    return cosine_similarity(count_matrix)

def recommend(movie_title, df, similarity):
    if movie_title not in df['title'].values:
        return ["Movie not found in database."]
    
    idx = df[df['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return [df.iloc[i[0]].title for i in sorted_scores]
