import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Page config MUST be first
st.set_page_config(page_title="Movie Recommender", layout="centered")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("mymoviedb.csv")
    df = df.drop_duplicates(subset='Title')
    df = df.dropna(subset=['Overview', 'Genre', 'Poster_Url', 'Release_Date', 'Original_Language'])
    df['Combined'] = df['Overview'] + " " + df['Genre']
    return df

movies = load_data()

# Vectorization and similarity
@st.cache_resource
def compute_similarity(data):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(data['Combined']).toarray()
    return cosine_similarity(vectors)

similarity = compute_similarity(movies)
movie_indices = pd.Series(movies.index, index=movies['Title'].str.lower())

# Recommendation function with filters
def recommend_movies(title, selected_lang, selected_genre, num=5):
    title = title.lower()
    if title not in movie_indices:
        return []
    idx = movie_indices[title]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:30]

    # Filter and sort recommendations
    candidates = [movies.iloc[i[0]] for i in scores]

    if selected_lang != "All":
        candidates = [m for m in candidates if m['Original_Language'].lower() == selected_lang.lower()]
    if selected_genre != "All":
        candidates = [m for m in candidates if selected_genre.lower() in m['Genre'].lower()]

    sorted_candidates = sorted(
        candidates,
        key=lambda x: (x['Vote_Average'], x['Popularity']),
        reverse=True
    )
    return sorted_candidates[:num]

# Streamlit UI styling
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

# User Input
movie_name = st.text_input("Enter a movie title:")

# Filters
languages = ["All"] + sorted(movies['Original_Language'].dropna().unique().tolist())
genres = ["All"] + sorted(set(g.strip() for sublist in movies['Genre'].dropna().str.split(",") for g in sublist))

selected_lang = st.selectbox("Filter by language:", languages)
selected_genre = st.selectbox("Filter by genre:", genres)

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie title.")
    else:
        recommendations = recommend_movies(movie_name, selected_lang, selected_genre)
        if recommendations:
            st.success("Here are some movies you might like:")
            for movie in recommendations:
                st.markdown(f"### 🎞️ {movie['Title']}")
                st.write(f"**Release Date:** {movie['Release_Date']}")
                st.write(f"**Original Language:** {movie['Original_Language']}")
                st.write(f"**Average Rating:** {movie['Vote_Average']} ⭐")
                st.write(f"**Popularity:** {round(movie['Popularity'], 2)} 🔥")
                st.image(movie['Poster_Url'], use_column_width=True)
                st.markdown("---")
        else:
            st.warning("No recommendations found with the selected filters. Try broadening your search.")
