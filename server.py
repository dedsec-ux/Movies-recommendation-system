
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load data and models
movies = pickle.load(open('models/movie_list.pkl', 'rb'))
similarity = pickle.load(open('models/similarity.pkl', 'rb'))
dt_model = pickle.load(open('models/dt_model.pkl', 'rb'))

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/style.css')
def style():
    return send_from_directory('.', 'style.css')

@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')

@app.route('/api/movies', methods=['GET'])
def get_movies():
    return jsonify(movies['title'].tolist())

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie_title = data.get('movie')
    
    try:
        movie_index = movies[movies['title'] == movie_title].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:51] # Get top 50 instead of 5
        
        selected_movie = movies.iloc[movie_index]
        
        # Predict preference
        input_features = [[
            int(selected_movie['cluster']),
            float(selected_movie['vote_average']),
            int(selected_movie['is_action']),
            int(selected_movie['is_sci_fi'])
        ]]
        preference = int(dt_model.predict(input_features)[0])
        
        recommendations = []
        for i in movies_list:
            rec = movies.iloc[i[0]]
            recommendations.append({
                'title': rec['title'],
                'vote_average': float(rec['vote_average']),
                'cluster': int(rec['cluster'])
            })
            
        return jsonify({
            'success': True,
            'selected_movie': {
                'title': selected_movie['title'],
                'vote_average': float(selected_movie['vote_average']),
                'cluster': int(selected_movie['cluster']),
                'preference': preference
            },
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
