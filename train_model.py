
import pandas as pd
import numpy as np
import ast
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# 1. Data Collection
print("Loading data...")
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# 2. Data Preprocessing
print("Preprocessing data...")
# Merge datasets
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average']]

# Handling missing values
movies.dropna(inplace=True)

# Helper functions to extract names from JSON-like strings
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Cleaning: Removing spaces
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Create tags column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags', 'vote_average']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# TF-IDF Vectorization
print("Vectorizing tags...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()

# 3. Machine Learning Algorithms
print("Training models...")

# a. K-Nearest Neighbors (KNN) - Content-Based Filtering
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(vectors)

# b. K-Means Clustering - Thematic Clusters
kmeans = KMeans(n_clusters=10, random_state=42)
new_df['cluster'] = kmeans.fit_predict(vectors)

# c. Decision Tree Classifier - Preference Predictor
# Since we don't have user data, we simulate it.
# Let's say a user likes "Action" and "Science Fiction"
# We define features for the DT as the cluster and average rating
# and maybe some binary features for top genres.
new_df['is_action'] = movies['genres'].apply(lambda x: 1 if 'Action' in x else 0)
new_df['is_sci_fi'] = movies['genres'].apply(lambda x: 1 if 'ScienceFiction' in x else 0)

# Simulate "Like" (1) if it's Action or SciFi or has high rating
def simulate_preference(row):
    if row['is_action'] == 1 or row['is_sci_fi'] == 1 or row['vote_average'] > 7.5:
        return 1
    return 0

new_df['user_preference'] = new_df.apply(simulate_preference, axis=1)

X_dt = new_df[['cluster', 'vote_average', 'is_action', 'is_sci_fi']]
y_dt = new_df['user_preference']

X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.2, random_state=42)

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

# 4. Evaluation Metrics
print("Evaluating models...")

# Decision Tree metrics
dt_pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test, dt_pred)
f1 = f1_score(y_test, dt_pred)

# RMSE (Simulated predicted rating vs vote_average)
# Let's assume K-Means cluster centers represent a baseline prediction
predicted_ratings = new_df.groupby('cluster')['vote_average'].transform('mean')
rmse = np.sqrt(mean_squared_error(new_df['vote_average'], predicted_ratings))

# Cosine Similarity for a sample movie
similarity = cosine_similarity(vectors)

print(f"Decision Tree Accuracy: {accuracy:.4f}")
print(f"Decision Tree F1-Score: {f1:.4f}")
print(f"RMSE (Rating Prediction): {rmse:.4f}")

# Exporting components
print("Saving models and data...")
if not os.path.exists('models'):
    os.makedirs('models')

pickle.dump(new_df, open('models/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('models/similarity.pkl', 'wb'))
pickle.dump(knn, open('models/knn_model.pkl', 'wb'))
pickle.dump(kmeans, open('models/kmeans_model.pkl', 'wb'))
pickle.dump(dt_clf, open('models/dt_model.pkl', 'wb'))
pickle.dump(vectors, open('models/vectors.pkl', 'wb'))

# Save report
with open('training_report.txt', 'w') as f:
    f.write("I KNOW WHAT TO WATCH - Training Report\n")
    f.write("========================================\n")
    f.write(f"Decision Tree Accuracy: {accuracy:.4f}\n")
    f.write(f"Decision Tree F1-Score: {f1:.4f}\n")
    f.write(f"RMSE (Rating Prediction): {rmse:.4f}\n")
    f.write("========================================\n")
    f.write("Models saved in 'models/' directory.\n")

print("Training completed successfully! Report saved to 'training_report.txt'.")
