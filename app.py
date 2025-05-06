import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# Page config
st.set_page_config(page_title="Smart Movie Recommender", layout="centered")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("mymoviedb.csv")
    df = df.drop_duplicates(subset='Title')
    df = df.dropna(subset=['Overview', 'Genre', 'Poster_Url', 'Release_Date', 'Original_Language'])
    df['Combined'] = df['Overview'] + " " + df['Genre']
    return df

movies = load_data()

# Load embedding model and compute similarities
@st.cache_resource
def get_embeddings(data):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(data['Combined'].tolist(), convert_to_tensor=False)
    return embeddings, model

embeddings, model = get_embeddings(movies)

similarity = cosine_similarity(embeddings)
movie_indices = pd.Series(movies.index, index=movies['Title'].str.lower())

# Dark mode styling
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #000000;
        color: white;
    }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1c1c1c !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, div {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🎬 Smart Movie Recommender</h1>", unsafe_allow_html=True)

# Input
movie_name = st.text_input("Enter a movie title:")

languages = ["All"] + sorted(movies['Original_Language'].dropna().unique().tolist())
genres = ["All"] + sorted(set(g.strip() for sublist in movies['Genre'].dropna().str.split(",") for g in sublist))

selected_lang = st.selectbox("Filter by language:", languages)
selected_genre = st.selectbox("Filter by genre:", genres)

# Recommendation logic
def recommend_movies(movie_name, selected_lang, selected_genre, top_n=10):
    movie_name = movie_name.lower()
    
    # Franchise/Part Detection using fuzzy matching
    franchise_matches = movies[movies['Title'].str.lower().apply(lambda x: fuzz.partial_ratio(x, movie_name) > 80)]

    # Get index of original movie
    if movie_name not in movie_indices:
        return [], []

    idx = movie_indices[movie_name]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:50]

    similar_movies = [movies.iloc[i[0]] for i in sim_scores]

    # Apply filters
    if selected_lang != "All":
        similar_movies = [m for m in similar_movies if m['Original_Language'].lower() == selected_lang.lower()]
        franchise_matches = franchise_matches[franchise_matches['Original_Language'].str.lower() == selected_lang.lower()]

    if selected_genre != "All":
        similar_movies = [m for m in similar_movies if selected_genre.lower() in m['Genre'].lower()]
        franchise_matches = franchise_matches[franchise_matches['Genre'].str.lower().str.contains(selected_genre.lower())]

    # Sort by relevance
    similar_movies = sorted(similar_movies, key=lambda x: (x['Vote_Average'], x['Popularity']), reverse=True)
    return franchise_matches, similar_movies[:top_n]

# Button click
if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie title.")
    else:
        franchise, recommendations = recommend_movies(movie_name, selected_lang, selected_genre)
        
        if not franchise.empty:
            st.subheader("🔁 Related Movie Parts / Sequels")
            for _, movie in franchise.iterrows():
                st.markdown(f"### 🎞️ {movie['Title']}")
                st.write(f"**Release Date:** {movie['Release_Date']}")
                st.write(f"**Language:** {movie['Original_Language']}")
                st.write(f"**Rating:** {movie['Vote_Average']} ⭐")
                st.image(movie['Poster_Url'], use_column_width=True)
                st.markdown("---")
        else:
            st.info("No direct parts found. Showing similar movies instead.")

        if recommendations:
            st.subheader("🎯 You may also like:")
            for movie in recommendations:
                st.markdown(f"### 🎬 {movie['Title']}")
                st.write(f"**Genre:** {movie['Genre']}")
                st.write(f"**Rating:** {movie['Vote_Average']} ⭐")
                st.write(f"**Popularity:** {round(movie['Popularity'], 2)} 🔥")
                st.image(movie['Poster_Url'], use_column_width=True)
                st.markdown("---")
        elif franchise.empty:
            st.warning("No recommendations found. Try another movie or remove filters.")
