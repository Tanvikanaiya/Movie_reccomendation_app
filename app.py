import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up page
st.set_page_config(page_title="🎬 Smart Movie Recommender", layout="centered")

# Background and UI Styling
st.markdown("""
    <style>
    body {
        color: white;
        background-color: black;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #000000;
    }
    [data-testid="stMarkdownContainer"] h1 {
        color: white;
        text-align: center;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("mymoviedb.csv")
    df = df.drop_duplicates(subset='Title')
    df = df.dropna(subset=['Overview', 'Genre', 'Poster_Url', 'Release_Date', 'Original_Language'])
    df['Combined'] = df['Title'] + " " + df['Overview'] + " " + df['Genre']
    return df

movies = load_data()

# Vectorize with TF-IDF
@st.cache_resource
def compute_similarity_matrix():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['Combined'])
    return cosine_similarity(tfidf_matrix)

similarity_matrix = compute_similarity_matrix()

# Build index
movie_indices = pd.Series(movies.index, index=movies['Title'].str.lower())

# Recommendation logic
def recommend_movies(query, selected_lang, selected_genre, top_n=10):
    query = query.strip().lower()
    if query not in movie_indices:
        return []
    
    idx = movie_indices[query]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:50]

    similar_movies = [movies.iloc[i[0]] for i in sim_scores]

    if selected_lang != "All":
        similar_movies = [m for m in similar_movies if m['Original_Language'].lower() == selected_lang.lower()]
    if selected_genre != "All":
        similar_movies = [m for m in similar_movies if selected_genre.lower() in m['Genre'].lower()]

    return similar_movies[:top_n]

# User UI
st.markdown("<h1>🎥 Smart Movie Recommender</h1>", unsafe_allow_html=True)

movie_name = st.text_input("🎬 Enter a movie title:", placeholder="e.g. Avatar")

languages = ["All"] + sorted(movies['Original_Language'].dropna().unique().tolist())
genres = ["All"] + sorted(set(g.strip() for sublist in movies['Genre'].dropna().str.split(",") for g in sublist))

selected_lang = st.selectbox("🌐 Filter by Language", languages)
selected_genre = st.selectbox("🎭 Filter by Genre", genres)

if st.button("🎯 Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie title.")
    elif movie_name.lower() not in movie_indices:
        st.error("Movie not found in database.")
    else:
        recommendations = recommend_movies(movie_name, selected_lang, selected_genre)
        if recommendations:
    st.success("🔥 Movies You Might Like:")
    for movie in recommendations:
        st.markdown(f"### 🎞️ {movie['Title']}")
        st.write(f"📅 **Release Date:** {movie['Release_Date']}")
        st.write(f"🌐 **Language:** {movie['Original_Language']}")
        st.write(f"🎭 **Genre:** {movie['Genre']}")
        st.write(f"⭐ **Rating:** {movie['Vote_Average']}")
        st.image(movie['Poster_Url'], use_column_width=True)
        st.markdown("---")
        else:
            st.warning("No matching recommendations found. Try different filters.")

