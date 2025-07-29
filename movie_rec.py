import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Load movies.pkl
movies = pickle.load(open('movies.pkl', 'rb'))

# Limit to Top 1000 Movies FIRST
movies = movies.head(1000)

# Now use pre-computed combined_features column
combined_text = movies['combined_features']

# Vectorization
vectorizer = TfidfVectorizer()
vectorized_form = vectorizer.fit_transform(combined_text)

# Compute Cosine Similarity Matrix
similarity = cosine_similarity(vectorized_form)

# Recommendation Function
def recommend(movie_title):
    list_of_all_titles = movies['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_title, list_of_all_titles, n=1)
    
    if not find_close_match:
        return []
    
    close_match = find_close_match[0]
    index_of_movie = movies[movies.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)[1:6]
    
    recommended_movies = []
    for i in sorted_similar_movies:
        movie_data = movies[movies.index == i[0]].iloc[0]
        title = movie_data['title']
        homepage = movie_data['homepage']
        
        if pd.isna(homepage) or homepage.strip() == '':
            homepage = 'Homepage not available'
        
        recommended_movies.append((title, homepage))
    return recommended_movies

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input('Enter a movie name:')

if st.button('Recommend'):
    if movie_name.strip() == '':
        st.warning('Please enter a movie name.')
    else:
        recommendations = recommend(movie_name)
        if not recommendations:
            st.error('No matching movie found.')
        else:
            st.success('Recommended Movies:')
            for title, homepage in recommendations:
                st.write(f"**{title}**")
                if homepage != 'Homepage not available':
                    st.markdown(f"[Visit Homepage]({homepage})")
                else:
                    st.caption('Homepage not available ')
