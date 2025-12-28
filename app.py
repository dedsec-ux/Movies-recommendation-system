
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load data and models
@st.cache_resource
def load_data():
    movies = pickle.load(open('models/movie_list.pkl', 'rb'))
    similarity = pickle.load(open('models/similarity.pkl', 'rb'))
    knn = pickle.load(open('models/knn_model.pkl', 'rb'))
    kmeans = pickle.load(open('models/kmeans_model.pkl', 'rb'))
    dt_model = pickle.load(open('models/dt_model.pkl', 'rb'))
    vectors = pickle.load(open('models/vectors.pkl', 'rb'))
    return movies, similarity, knn, kmeans, dt_model, vectors

movies, similarity, knn, kmeans, dt_model, vectors = load_data()

# Page Config
st.set_page_config(
    page_title="I KNOW WHAT TO WATCH",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background-color: #0f1116;
        color: #e0e0e0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #6e45e2 0%, #88d3ce 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(110, 69, 226, 0.3);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(110, 69, 226, 0.5);
        color: white;
    }

    .movie-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }

    .movie-card:hover {
        transform: scale(1.05);
        background: rgba(255, 255, 255, 0.1);
        border-color: #6e45e2;
    }

    .recommendation-title {
        color: #ffffff;
        font-weight: 700;
        margin-top: 1rem;
        font-size: 1.1rem;
    }

    .hero-section {
        text-align: center;
        padding: 4rem 0;
        background: linear-gradient(180deg, rgba(110, 69, 226, 0.1) 0%, transparent 100%);
        border-radius: 0 0 50px 50px;
        margin-bottom: 3rem;
    }

    h1 {
        font-size: 4rem !important;
        background: linear-gradient(90deg, #fff 0%, #88d3ce 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-box {
        background: rgba(110, 69, 226, 0.1);
        border-radius: 15px;
        padding: 1rem;
        border-left: 4px solid #6e45e2;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1>I KNOW WHAT TO WATCH</h1>
    <p style="font-size: 1.2rem; opacity: 0.8;">Discover your next favorite movie using AI-powered recommendations</p>
</div>
""", unsafe_allow_html=True)

# Recommendation Logic
def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        recommended_movies = []
        for i in movies_list:
            recommended_movies.append(movies.iloc[i[0]])
        return recommended_movies
    except:
        return []

# Sidebar for additional info
with st.sidebar:
    st.markdown("### 📊 System Insights")
    st.markdown("This system uses **KNN**, **K-Means**, and **Decision Trees** to analyze 5,000+ movies.")
    
    st.divider()
    st.markdown("#### User Preference Profile")
    st.info("The system currently assumes you enjoy **Action** and **Sci-Fi** movies with ratings above **7.5**.")

# Main Interface
selected_movie_name = st.selectbox(
    'Search for a movie...',
    movies['title'].values
)

if st.button('Get Recommendations'):
    recommendations = recommend(selected_movie_name)
    
    if recommendations:
        selected_movie_data = movies[movies['title'] == selected_movie_name].iloc[0]
        
        # Display analysis for the selected movie
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <small>Thematic Cluster</small>
                <h3>Cluster #{selected_movie_data['cluster']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            rating = selected_movie_data['vote_average']
            st.markdown(f"""
            <div class="metric-box">
                <small>Average Rating</small>
                <h3>⭐ {rating}</h3>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            # Predict preference
            # Features: cluster, vote_average, is_action, is_sci_fi
            input_features = [[
                selected_movie_data['cluster'],
                selected_movie_data['vote_average'],
                selected_movie_data['is_action'],
                selected_movie_data['is_sci_fi']
            ]]
            preference = dt_model.predict(input_features)[0]
            status = "Highly Recommended" if preference == 1 else "Might not be your style"
            icon = "✅" if preference == 1 else "⚠️"
            
            st.markdown(f"""
            <div class="metric-box">
                <small>AI Preference Prediction</small>
                <h3>{icon} {status}</h3>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.subheader("Similar Movies You Might Love")
        
        cols = st.columns(5)
        for i, rec in enumerate(recommendations):
            with cols[i]:
                st.markdown(f"""
                <div class="movie-card">
                    <div style="font-size: 2rem;">🎬</div>
                    <div class="recommendation-title">{rec['title']}</div>
                    <p style="font-size: 0.8rem; opacity: 0.7;">Cluster: {rec['cluster']}</p>
                    <p style="font-size: 0.9rem; color: #88d3ce;">⭐ {rec['vote_average']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error("Movie not found in our database!")

st.markdown("""
<div style="text-align: center; margin-top: 5rem; opacity: 0.5; font-size: 0.8rem;">
    Built with ❤️ using Scikit-Learn and Streamlit
</div>
""", unsafe_allow_html=True)
