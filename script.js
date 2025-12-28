
const API_BASE = 'http://localhost:5001/api';

const movieInput = document.getElementById('movieInput');
const movieList = document.getElementById('movieList');
const searchBtn = document.getElementById('searchBtn');
const resultsSection = document.getElementById('results');

let allRecommendations = [];
let currentIndex = 0;

// Fetch all movies for autocomplete
async function loadMovies() {
    try {
        const response = await fetch(`${API_BASE}/movies`);
        const movies = await response.json();
        movies.forEach(title => {
            const option = document.createElement('option');
            option.value = title;
            movieList.appendChild(option);
        });
    } catch (err) {
        console.error('Failed to load movies', err);
    }
}

async function getRecommendations() {
    const movie = movieInput.value.trim();
    if (!movie) return;

    searchBtn.disabled = true;
    searchBtn.textContent = 'Analyzing...';

    try {
        const response = await fetch(`${API_BASE}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ movie })
        });

        const data = await response.json();

        if (data.success) {
            allRecommendations = data.recommendations;
            currentIndex = 0;
            displayResults(data);
        } else {
            alert('Movie not found or error occurred!');
        }
    } catch (err) {
        console.error('Search failed', err);
    } finally {
        searchBtn.disabled = false;
        searchBtn.textContent = 'Find Recommendations';
    }
}

function displayResults(data) {
    const { selected_movie } = data;

    // Update stats
    document.getElementById('clusterVal').textContent = `Cluster #${selected_movie.cluster}`;
    document.getElementById('ratingVal').textContent = `⭐ ${selected_movie.vote_average}`;

    const prefEl = document.getElementById('preferenceVal');
    const prefIcon = document.getElementById('preferenceIcon');

    if (selected_movie.preference === 1) {
        prefEl.textContent = 'Highly Recommend';
        prefEl.style.color = '#88d3ce';
        prefIcon.textContent = '✅';
    } else {
        prefEl.textContent = 'Might not fit';
        prefEl.style.color = '#ff6b6b';
        prefIcon.textContent = '⚠️';
    }

    renderCards();

    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function renderCards() {
    const container = document.getElementById('recommendationCards');
    container.innerHTML = '';

    // Get next 5 recommendations
    const slice = allRecommendations.slice(currentIndex, currentIndex + 5);

    slice.forEach(rec => {
        const card = document.createElement('div');
        card.className = 'movie-card';
        card.innerHTML = `
            <div class="emoji">🎬</div>
            <h4>${rec.title}</h4>
            <div class="rating">⭐ ${rec.vote_average}</div>
            <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 0.5rem;">C#${rec.cluster}</div>
        `;
        container.appendChild(card);
    });
}

function cycleRecommendations() {
    currentIndex = (currentIndex + 5) % allRecommendations.length;
    renderCards();
}

searchBtn.addEventListener('click', getRecommendations);
document.getElementById('refreshBtn').addEventListener('click', cycleRecommendations);
movieInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') getRecommendations();
});

loadMovies();
