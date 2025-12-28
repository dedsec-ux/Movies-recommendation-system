
# I KNOW WHAT TO WATCH: Movie Recommendation System Summary

This project is a comprehensive movie recommendation engine that uses Machine Learning to help users find their next favorite film. It utilizes the TMDB 5000 Movie Dataset and implements three distinct algorithms to provide a multi-layered analysis.

## 🚀 What the Code Does
1. **Data Engineering**: It merges movie metadata (genres, keywords, overview) with production credits (cast, crew).
2. **Natural Language Processing (NLP)**: It cleans and processes text data, removing spaces from names (e.g., "Christopher Nolan" to "ChristopherNolan") to create unique tags. It then converts these tags into high-dimensional numerical vectors using **TF-IDF Vectorization**.
3. **Multi-Model Recommendation**: It uses three different AI models to analyze the user's selected movie and provide personalized suggestions.
4. **Interactive Deployment**: It offers two interfaces:
   - A **Streamlit** dashboard for easy, one-page interaction.
   - A **Custom Web Application** with a Flask backend and a premium glassmorphism frontend.

---

## 🛠️ Code Implementation & Algorithm Usage

### 1. Data Preprocessing & Cleaning
```python
# Cleaning: Removing spaces to create unique tags
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Feature Selection: Consolidating text data into 'tags'
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
```

### 2. K-Nearest Neighbors (KNN)
- **Where it's used**: For generating the top-5 similar movie recommendations.
- **Implementation**:
```python
# Vectorization using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()

# KNN Training
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(vectors)
```

### 3. K-Means Clustering
- **Where it's used**: For thematic grouping and unsupervised exploration of movies.
- **Implementation**:
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
new_df['cluster'] = kmeans.fit_predict(vectors)
```

### 4. Decision Tree Classifier
- **Where it's used**: As a "Preference Predictor" to classify if a movie fits a specific user profile.
- **Implementation**:
```python
from sklearn.tree import DecisionTreeClassifier
X_dt = new_df[['cluster', 'vote_average', 'is_action', 'is_sci_fi']]
y_dt = new_df['user_preference']

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
```

---

## 🤖 Why These Algorithms?

| Algorithm | Why it's used | Output |
| :--- | :--- | :--- |
| **KNN** | Excellent for finding similarities in high-dimensional text vectors. | Top 5-50 similar movies. |
| **K-Means** | Uncovers hidden thematic patterns without needing predefined labels. | Cluster ID (Thematic grouping). |
| **Decision Tree** | Interpretable and fast classification for user preference profile. | Recommended / Not Recommended. |

---

## 📊 Evaluation & Performance
- **Decision Tree Accuracy**: 100% (matching Action/Sci-Fi profile).
- **RMSE (Root Mean Squared Error)**: ~1.16 measuring rating variance.
- **Cosine Similarity Score**: Used for KNN to measure the proximity of movie tags.

## 📦 Project Structure
- `train_model.py`: Preprocessing and training pipeline.
- `streamlit_app.py`: Deployment for Streamlit Cloud.
- `server.py` & `index.html`: Custom full-stack web application.
- `models/`: Pre-trained weights (`.pkl`) stored via **Git LFS**.
