import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load data
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Helper functions
def convert(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except:
        return []

def get_director(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']
    except:
        return []

def collapse(L):
    return " ".join([i.replace(" ", "") for i in L])

# Data Cleaning
movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])
movies['crew'] = movies['crew'].apply(get_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(collapse)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie = movie.lower()
    if movie not in new_df['title'].str.lower().values:
        return []
    idx = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [new_df.iloc[i[0]].title for i in distances]

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="centered")

# Stylish animated CSS
# Stylish animated CSS
animated_css = """
<style>
body {
    background-color: #f5f5f5;
    color: black;  /* Default text color */
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(to right, #ddeaff, #ffffff);
    background-size: cover;
}
h1.title {
    text-align: center;
    color: #4CAF50;  /* Title color changed to green */
    font-size: 3em;
    animation: slideDown 1s ease-in-out;
}
.stButton>button {
    background-color: #3366cc;
    color: white;  /* Button text color */
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #264d99;
    transform: scale(1.05);
}
.recommendations {
    animation: fadeIn 1s ease-in;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
.stSuccess {
    color: #003366;  /* Dark blue color for success messages */
}
.stError {
    color: #003366;  /* Dark blue color for error messages */
}
</style>
"""
st.markdown(animated_css, unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)

# Input
movie_name = st.text_input("Enter a movie title:")

# Button and Output
if st.button("Recommend"):
    recommendations = recommend(movie_name)
    if recommendations:
        st.markdown("<div class='recommendations'>", unsafe_allow_html=True)
        st.success("Here are 5 movies you might like:")
        for movie in recommendations:
            st.markdown(f"- **{movie}**")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Movie not found. Please try another title.")
