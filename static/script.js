// static/script.js

// 🚨 PASTE YOUR TMDB API KEY HERE
const API_KEY = "64270cfedbe3933ef55f0be854ba29a2";
const TMDB_BASE_URL = "https://api.themoviedb.org/3";

// A temporary store for search results to keep track of movie IDs
let searchResultsCache = {};

/**
 * Searches for movies using the TMDB API.
 * @param {string} type - Either 'watched' or 'liked' to target the correct search box.
 */
async function searchMovies(type) {
    const queryInput = document.getElementById(type === 'watched' ? 'searchWatched' : 'searchLiked');
    const resultsDiv = document.getElementById(type === 'watched' ? 'searchResultsWatched' : 'searchResultsLiked');
    const query = queryInput.value;

    if (query.length < 3) {
        resultsDiv.innerHTML = '';
        return;
    }

    const url = `${TMDB_BASE_URL}/search/movie?api_key=${API_KEY}&query=${encodeURIComponent(query)}`;
    
    try {
        const response = await fetch(url);
        const data = await response.json();
        resultsDiv.innerHTML = ''; // Clear previous results

        data.results.slice(0, 5).forEach(movie => {
            // Store movie details for later use
            searchResultsCache[movie.id] = { title: movie.title };

            const movieEl = document.createElement('div');
            movieEl.classList.add('search-result-item');
            movieEl.textContent = movie.title;
            movieEl.onclick = () => selectMovie(movie.id, type);
            resultsDiv.appendChild(movieEl);
        });
    } catch (error) {
        console.error('Error fetching movies:', error);
    }
}

/**
 * Handles selecting a movie from the search results.
 * @param {number} movieId - The ID of the selected movie.
 * @param {string} type - Either 'watched' or 'liked'.
 */
function selectMovie(movieId, type) {
    const movie = searchResultsCache[movieId];
    if (!movie) return;

    if (type === 'watched') {
        document.getElementById('watchedMovieTitle').value = movie.title;
        document.getElementById('watchedMovieId').value = movieId;
        document.getElementById('searchResultsWatched').innerHTML = ''; // Hide results
    } else if (type === 'liked') {
        document.getElementById('likedMovieTitle').value = movie.title;
        document.getElementById('likedMovieId').value = movieId;
        document.getElementById('searchResultsLiked').innerHTML = ''; // Hide results
    }
}

/**
 * Adds a movie to the "watched" list by sending data to the Flask backend.
 */
async function addWatchedMovie() {
    const title = document.getElementById('watchedMovieTitle').value;
    const movieId = document.getElementById('watchedMovieId').value;
    const rating = document.getElementById('rating').value;

    if (!movieId || !title || !rating) {
        alert('Please search for a movie and provide a rating.');
        return;
    }

    const response = await fetch('/add_watched_movie', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ movie_id: movieId, title: title, rating: parseInt(rating) })
    });

    const result = await response.json();
    if (result.success) {
        // Add to the list on the page and reload to show the updated list from the server
        location.reload(); 
    } else {
        alert('Failed to add movie: ' + result.message);
    }
}

/**
 * Adds a movie to the "liked" list by sending data to the Flask backend.
 */
async function addLikedMovie() {
    const title = document.getElementById('likedMovieTitle').value;
    const movieId = document.getElementById('likedMovieId').value;

    if (!movieId || !title) {
        alert('Please search for and select a movie.');
        return;
    }

    const response = await fetch('/add_liked_movie', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ movie_id: movieId, title: title })
    });

    const result = await response.json();
    if (result.success) {
        location.reload(); // Reload to see the updated list
    } else {
        alert('Failed to add movie: ' + result.message);
    }
}
