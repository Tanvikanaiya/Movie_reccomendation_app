import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Page config
st.set_page_config(page_title="Movie Recommender", layout="centered")

# ✅ Load data
@st.cache_data
def load_data():
    df = pd.read_csv("mymoviedb.csv")
    df = df.drop_duplicates(subset='Title')
    df = df.dropna(subset=['Overview', 'Genre', 'Poster_Url', 'Release_Date', 'Original_Language'])
    df['Combined'] = df['Overview'] + " " + df['Genre']
    return df

movies = load_data()

# ✅ Generate embeddings and similarity matrix
@st.cache_resource
def compute_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data['Combined'].tolist(), convert_to_tensor=False)
    return cosine_similarity(embeddings)

similarity = compute_embeddings(movies)
movie_indices = pd.Series(movies.index, index=movies['Title'].str.lower())

# ✅ Recommendation logic
def recommend_movies(title, selected_lang, selected_genre, num=5):
    title = title.lower()
    if title not in movie_indices:
        return []
    idx = movie_indices[title]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:30]
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

# ✅ Dark mode styling
st.markdown("""
    <style>
    body, [data-testid="stAppViewContainer"] {
        background-color: #000000;
        color: #FFFFFF;
    }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #1c1c1c !important;
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

# ✅ UI Title
st.markdown("<h1 style='text-align: center;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)

# ✅ User Inputs
movie_name = st.text_input("Enter a movie title:")

languages = ["All"] + sorted(movies['Original_Language'].dropna().unique().tolist())
genres = ["All"] + sorted(set(g.strip() for sublist in movies['Genre'].dropna().str.split(",") for g in sublist))

selected_lang = st.selectbox("Filter by language:", languages)
selected_genre = st.selectbox("Filter by genre:", genres)

# ✅ Button and results
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
                st.write(f"**Language:** {movie['Original_Language']}")
                st.write(f"**Rating:** {movie['Vote_Average']} ⭐")
                st.write(f"**Popularity:** {round(movie['Popularity'], 2)} 🔥")
                st.image(movie['Poster_Url'], use_column_width=True)
                st.markdown("---")
        else:
            st.warning("No recommendations found. Try changing the filters or title.")
