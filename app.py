import streamlit as st
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle  # To save and load your model

# Load data (assuming 'mymoviedb.csv' is in the same directory)
try:
    movies_data = pd.read_csv('mymoviedb.csv')
except FileNotFoundError:
    st.error("Error: 'mymoviedb.csv' not found. Please make sure the file is in the same directory as the script.")
    #  Important:  Stop execution if the data file is missing.
    st.stop()

# Print the columns of the dataframe to debug.
st.write("Columns in your DataFrame:", movies_data.columns)

# Select relevant features
selected_features = ['Genre', 'keywords', 'tagline', 'cast', 'director'] # Changed 'genres' to 'Genre'

# Check if all selected features are in the dataframe
for feature in selected_features:
    if feature not in movies_data.columns:
        st.error(f"Error: Column '{feature}' not found in the CSV file.  Please check the column names in your 'mymoviedb.csv' file and update the 'selected_features' list if necessary.")
        st.stop()  # Stop if any essential feature is missing.

# Fill NaN values with empty strings
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine selected features
combined_features = movies_data['Genre'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director'] #changed here as well

# ---  Moved model training and data processing outside the function ---
# --- This is more efficient, so it only runs once when the app starts ---

# Convert text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Calculate cosine similarity
similarity = cosine_similarity(feature_vectors)

# Save the trained model, and important data.
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
pickle.dump(similarity, open('similarity_matrix.pkl', 'wb'))
pickle.dump(movies_data['title'].tolist(), open('movie_titles.pkl', 'wb')) #save titles

@st.cache_resource()  # Add this decorator to cache the model loading
def load_model_and_data():
    """Loads the model and data"""
    # Load the vectorizer, similarity matrix, and titles
    tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    similarity_matrix = pickle.load(open('similarity_matrix.pkl', 'rb'))
    movie_titles = pickle.load(open('movie_titles.pkl', 'rb'))
    return tfidf_vectorizer, similarity_matrix, movie_titles
    
tfidf_vectorizer, similarity_matrix, list_of_all_titles = load_model_and_data() # Load

def recommend_movies(movie_name):
    """
    Recommends movies similar to the given movie name.

    Args:
        movie_name (str): The name of the movie to find recommendations for.

    Returns:
        list: A list of similar movie titles, or a message if no matches are found.
    """
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        return ["No close matches found. Please try another movie title."]

    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity_matrix[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, movie in enumerate(sorted_similar_movies):
        if i > 0 and i < 11:  # Exclude the input movie itself, and limit to top 10
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            recommendations.append(title_from_index)
    return recommendations

# --- Streamlit App ---
def main():
    """
    Main function to run the Streamlit app.
    """
    st.title('Movie Recommendation System')

    movie_name = st.text_input('Enter your favorite movie name:', '')

    if st.button('Show Recommendations'):
        if movie_name:
            recommendations = recommend_movies(movie_name)
            if recommendations: #check is list is empty
                st.subheader('Movies suggested for you:')
                for i, recommended_movie in enumerate(recommendations):
                    st.write(f"{i+1}. {recommended_movie}")
            else:
                st.write("No movies found")
        else:
            st.warning('Please enter a movie name.')

if __name__ == '__main__':
    main()
