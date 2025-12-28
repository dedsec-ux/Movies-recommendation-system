
# I KNOW WHAT TO WATCH: Movie Recommendation System

An AI-powered movie recommendation system built using TMDB 5000 dataset.

## Features
- **Content-Based Filtering (KNN)**: Recommends movies similar to the one you searched.
- **Thematic Clustering (K-Means)**: Groups movies into 10 distinct thematic clusters.
- **Preference Prediction (Decision Tree)**: Predicts if a movie fits your profile (Action/Sci-Fi fan).
- **Premium UI**: Modern, dark-themed Streamlit interface with glassmorphism effects.

## Project Structure
- `train_model.py`: Script to process data and train the ML models.
- `app.py`: Streamlit application for the user interface.
- `models/`: Directory containing serialized model files (`.pkl`).
- `tmdb_5000_movies.csv` & `tmdb_5000_credits.csv`: Dataset files.

## How to Run
1. **Activate the environment**:
   ```bash
   source venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Train the models** (optional, models are already provided):
   ```bash
   python train_model.py
   ```
4. **Launch the app**:
   ```bash
   streamlit run app.py
   ```

## Evaluation Metrics
See `training_report.txt` for the latest evaluation metrics (Accuracy, F1-Score, RMSE).
