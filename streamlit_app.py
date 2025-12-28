
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set Page Config
st.set_page_config(
    page_title="I KNOW WHAT TO WATCH",
    page_icon="🎬",
    layout="wide"
)

# Load data and models
@st.cache_resource
def load_data():
    movies = pickle.load(open('models/movie_list.pkl', 'rb'))
    similarity = pickle.load(open('models/similarity.pkl', 'rb'))
    knn = pickle.load(open('models/knn_model.pkl', 'rb'))
    kmeans = pickle.load(open('models/kmeans_model.pkl', 'rb'))
    dt_model = pickle.load(open('models/dt_model.pkl', 'rb'))
    return movies, similarity, knn, kmeans, dt_model

movies, similarity, knn, kmeans, dt_model = load_data()

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background-color: #0f1116;
    }

    .stButton>button {
        background: linear-gradient(135deg, #6e45e2 0%, #88d3ce 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
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
        height: 250px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .movie-card:hover {
        transform: scale(1.05);
        border-color: #6e45e2;
        background: rgba(255, 255, 255, 0.08);
    }

    .algo-badge {
        background: #6e45e2;
        color: white;
        font-size: 0.65rem;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero-section {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(180deg, rgba(110, 69, 226, 0.15) 0%, transparent 100%);
        border-radius: 0 0 50px 50px;
        margin-bottom: 2rem;
    }

    h1 {
        font-size: 4.5rem !important;
        background: linear-gradient(90deg, #fff 0%, #88d3ce 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -2px;
    }

    .metric-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1>I KNOW WHAT TO WATCH</h1>
    <p style="font-size: 1.2rem; opacity: 0.8; color: #88d3ce;">AI-Powered Cinematic Discovery Engine</p>
</div>
""", unsafe_allow_html=True)

# Initialize Session State
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'all_recommendations' not in st.session_state:
    st.session_state.all_recommendations = []
if 'selected_movie' not in st.session_state:
    st.session_state.selected_movie = None

# Sidebar Content
with st.sidebar:
    st.image("https://img.icons8.com/movie-projector", width=100)
    st.title("System Specs")
    st.write("---")
    st.markdown("### 🤖 ML Architecture")
    st.info("**KNN**: Content Filtering\n\n**K-Means**: Thematic Discovery\n\n**Decision Tree**: Preference Predictor")
    st.write("---")
    st.markdown("### 📊 Metrics")
    st.success("DT Accuracy: 100%\n\nRMSE: 1.16\n\nSimilarity: Cosine")

# Search Functionality
selected_movie_name = st.selectbox(
    'Search for a movie in the TMDB 5000 Database:',
    movies['title'].values
)

col1, col2 = st.columns([1, 4])
with col1:
    search_clicked = st.button('🔍 Find Movies')

if search_clicked:
    movie_index = movies[movies['title'] == selected_movie_name].index[0]
    distances = similarity[movie_index]
    
    # Get top 50 recommendations
    rec_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:51]
    
    st.session_state.all_recommendations = [movies.iloc[i[0]] for i in rec_list]
    st.session_state.current_index = 0
    st.session_state.selected_movie = movies.iloc[movie_index]

# Display Analysis if a movie is selected
if st.session_state.selected_movie is not None:
    sel = st.session_state.selected_movie
    
    st.markdown("### 📊 Model Deep Analysis")
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="algo-badge">ALGORITHM: K-MEANS</div>
            <div style="font-size: 0.9rem; opacity: 0.7;">Thematic Cluster</div>
            <h2 style="color: #fff;">Cluster #{sel['cluster']}</h2>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        # Decision Tree Prediction
        feat = [[sel['cluster'], sel['vote_average'], sel['is_action'], sel['is_sci_fi']]]
        pref = dt_model.predict(feat)[0]
        pref_text = "Highly Recommend" if pref == 1 else "Might Not Fit"
        pref_color = "#88d3ce" if pref == 1 else "#ff6b6b"
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="algo-badge">ALGORITHM: DECISION TREE</div>
            <div style="font-size: 0.9rem; opacity: 0.7;">Preference Fit</div>
            <h2 style="color: {pref_color};">{pref_text}</h2>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="algo-badge">DATASET METRIC</div>
            <div style="font-size: 0.9rem; opacity: 0.7;">Average Rating</div>
            <h2 style="color: #fff;">⭐ {sel['vote_average']}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    
    # Recommendation Section
    r_col1, r_col2 = st.columns([4, 1])
    with r_col1:
        st.subheader("Similar Movies You Might Love")
        st.caption("Generated using K-Nearest Neighbors (KNN) Cosine Similarity")
    
    with r_col2:
        if st.button("🔄 Refresh List"):
            st.session_state.current_index = (st.session_state.current_index + 5) % 50

    # Display 5 at a time
    current_recs = st.session_state.all_recommendations[st.session_state.current_index : st.session_state.current_index + 5]
    
    rec_cols = st.columns(5)
    for i, rec in enumerate(current_recs):
        with rec_cols[i]:
            st.markdown(f"""
            <div class="movie-card">
                <div style="font-size: 2.5rem; margin-bottom: 10px;">🎬</div>
                <div style="font-weight: 700; font-size: 1rem; margin-bottom: 5px;">{rec['title']}</div>
                <div style="color: #88d3ce; font-weight: 600;">⭐ {rec['vote_average']}</div>
                <div style="font-size: 0.7rem; opacity: 0.5; margin-top: 10px;">Cluster: {rec['cluster']}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; margin-top: 5rem; opacity: 0.5;">
        <p>Select a movie and hit search to begin your journey.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("---")
st.markdown("<div style='text-align: center; opacity: 0.3;'>Developed for AI Movie Recommendations | TMDB Dataset</div>", unsafe_allow_html=True)
