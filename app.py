import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("mymoviedb.csv")
movies.drop_duplicates(subset='Title', inplace=True)
movies.dropna(subset=['Overview', 'Genre'], inplace=True)

# Create combined feature for content-based filtering
movies['Combined'] = movies['Overview'] + " " + movies['Genre']

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['Combined']).toarray()

# Similarity matrix
similarity = cosine_similarity(vectors)

# Index mapping
movie_indices = pd.Series(movies.index, index=movies['Title'].str.lower())

# Recommend function
def recommend_movies(title, num=5):
    title = title.lower()
    if title not in movie_indices:
        return []
    idx = movie_indices[title]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num+1]
    return [movies.iloc[i[0]] for i in scores]

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="centered")

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(to right, #ddeaff, #ffffff);
        background-size: cover;
    }
    h1, h2, h3, h4, h5, h6, p, div {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)

# User input
movie_name = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    recommendations = recommend_movies(movie_name)
    if recommendations:
        st.success("Here are some movies you might like:")
        for movie in recommendations:
            st.markdown(f"### 🎞️ {movie['Title']}")
            st.write(f"**Release Date:** {movie['Release_Date']}")
            st.write(f"**Original Language:** {movie['Original_Language']}")
            st.image(movie['Poster_Url'], use_column_width=True)
            st.markdown("---")
    else:
        st.error("Movie not found. Please check the title and try again.")
