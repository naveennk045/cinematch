from fastapi import FastAPI, HTTPException, Depends, Request, Form, status, Response
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
from mysql.connector import errorcode
import bcrypt
from datetime import datetime, timedelta
import asyncio
import httpx
from contextlib import asynccontextmanager
import secrets


# Pydantic models
class User(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str


class WatchedMovie(BaseModel):
    movie_id: int
    title: str
    rating: int


class LikedMovie(BaseModel):
    movie_id: int
    title: str


class Genre(BaseModel):
    genre: str


class MovieDetails(BaseModel):
    id: int
    title: str
    poster_path: Optional[str] = None
    overview: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


# Configuration - UPDATE THESE WITH YOUR ACTUAL TMDB CREDENTIALS
TMDB_ACCESS_TOKEN = "64270cfedbe3933ef55f0be854ba29a2"  # Get this from TMDB
TMDB_BASE_URL = "https://api.themoviedb.org/3"
SECRET_KEY = secrets.token_hex(32)  # Generate a secure secret key

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Naveennk*@07",
    "database": "cinematch"
}

# Session storage (in production, use a proper session store like Redis)
sessions = {}


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def setup_database():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS watched_movies (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                movie_id BIGINT,
                title TEXT,
                rating INT,
                watched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS liked_movies (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                movie_id BIGINT,
                title TEXT,
                liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS genres (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                genre VARCHAR(255),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')

        # Create demo user with hashed password
        cursor.execute("SELECT * FROM users WHERE username = %s", ('demo',))
        if cursor.fetchone() is None:
            hashed_password = bcrypt.hashpw('password'.encode('utf-8'), bcrypt.gensalt())
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)",
                           ('demo', hashed_password.decode('utf-8')))

        conn.commit()
        conn.close()
        print("✅ MySQL Database initialized successfully.")
    except mysql.connector.Error as err:
        print("❌ Database error:", err)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_database()
    os.makedirs('static/images', exist_ok=True)
    yield
    # Shutdown
    pass


app = FastAPI(title="CineMatch Movie Recommender", version="2.0.0", lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Utility functions
def create_session(user_id: int) -> str:
    """Create a new session for the user"""
    session_id = secrets.token_urlsafe(32)
    sessions[session_id] = {
        "user_id": user_id,
        "created_at": datetime.now()
    }
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    """Get session data by session ID"""
    if session_id in sessions:
        return sessions[session_id]
    return None


def delete_session(session_id: str):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]


def get_current_user(request: Request):
    """Get current user from session"""
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session_data = get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=401, detail="Invalid session")

    # Check if session is expired (30 minutes)
    if datetime.now() - session_data["created_at"] > timedelta(minutes=30):
        delete_session(session_id)
        raise HTTPException(status_code=401, detail="Session expired")

    # Update session creation time to extend session
    session_data["created_at"] = datetime.now()

    # Get user details from database
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, username FROM users WHERE id = %s", (session_data["user_id"],))
    user = cursor.fetchone()
    conn.close()

    if not user:
        delete_session(session_id)
        raise HTTPException(status_code=401, detail="User not found")

    return user


async def get_movie_details(movie_id: int) -> Optional[dict]:
    """Async function to get movie details from TMDB API."""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    headers = {"Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching movie details: {e}")
    return None

async def search_movies(query: str) -> List[dict]:
    """Async function to search movies from TMDB API."""
    url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_ACCESS_TOKEN}&query={query}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])[:10]  # Return top 10 results
        except Exception as e:
            print(f"Error searching movies: {e}")
    return []


async def get_movie_recommendations(movie_id: int) -> List[dict]:
    """Async function to get movie recommendations from TMDB API."""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}/recommendations"
    headers = {"Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
        except Exception as e:
            print(f"Error fetching recommendations: {e}")
    return []


async def get_popular_movies() -> List[dict]:
    """Async function to get popular movies from TMDB API."""
    url = f"{TMDB_BASE_URL}/movie/popular"
    headers = {"Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
        except Exception as e:
            print(f"Error fetching popular movies: {e}")
    return []


# Authentication routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return RedirectResponse(url="/login")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request, response: Response, username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    conn.close()

    if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        session_id = create_session(user['id'])
        response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
        response.set_cookie(key="session_id", value=session_id, httponly=True)
        return response
    else:
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Invalid username or password."
        })


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
async def register(request: Request, response: Response, username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)",
                       (username, hashed_password.decode('utf-8')))
        conn.commit()

        # Get the new user ID
        cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
        user_id = cursor.fetchone()[0]
        conn.close()

        # Create session and redirect to dashboard
        session_id = create_session(user_id)
        response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
        response.set_cookie(key="session_id", value=session_id, httponly=True)
        return response
    except mysql.connector.IntegrityError:
        conn.close()
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Username already exists."
        })


@app.get("/logout")
async def logout(response: Response):
    # Remove session
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="session_id")
    return response


# Dashboard and main functionality
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    try:
        user = get_current_user(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT movie_id, title, rating FROM watched_movies WHERE user_id = %s ORDER BY watched_at DESC",
                   (user['id'],))
    watched_data = cursor.fetchall()

    cursor.execute("SELECT movie_id, title FROM liked_movies WHERE user_id = %s ORDER BY liked_at DESC", (user['id'],))
    liked_data = cursor.fetchall()

    cursor.execute("SELECT genre FROM genres WHERE user_id = %s", (user['id'],))
    genres = [row['genre'] for row in cursor.fetchall()]

    conn.close()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "watched_data": watched_data,
        "liked_data": liked_data,
        "genres": genres
    })


# API endpoints for movie operations
@app.post("/api/watched-movie")
async def add_watched_movie(movie: WatchedMovie, user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO watched_movies (user_id, movie_id, title, rating) VALUES (%s, %s, %s, %s)",
        (user['id'], movie.movie_id, movie.title, movie.rating)
    )
    conn.commit()
    conn.close()
    return {"success": True}


@app.post("/api/liked-movie")
async def add_liked_movie(movie: LikedMovie, user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO liked_movies (user_id, movie_id, title) VALUES (%s, %s, %s)",
        (user['id'], movie.movie_id, movie.title)
    )
    conn.commit()
    conn.close()
    return {"success": True}


@app.post("/api/genre")
async def add_genre(genre: Genre, user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO genres (user_id, genre) VALUES (%s, %s)", (user['id'], genre.genre))
    conn.commit()
    conn.close()
    return {"success": True}


@app.get("/api/search-movies")
async def search_movies_api(q: str):
    movies = await search_movies(q)
    return {"results": movies}


@app.get("/recommendations", response_class=HTMLResponse)
async def recommendations(request: Request):
    try:
        user = get_current_user(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Get IDs of movies the user has already seen
    cursor.execute("SELECT movie_id FROM watched_movies WHERE user_id = %s", (user['id'],))
    watched_movie_ids = {row['movie_id'] for row in cursor.fetchall()}

    cursor.execute("SELECT movie_id FROM liked_movies WHERE user_id = %s", (user['id'],))
    liked_movie_ids = {row['movie_id'] for row in cursor.fetchall()}

    all_seen_movie_ids = watched_movie_ids.union(liked_movie_ids)

    # Find the last liked movie to use as a seed
    cursor.execute("SELECT movie_id FROM liked_movies WHERE user_id = %s ORDER BY liked_at DESC LIMIT 1", (user['id'],))
    last_liked_movie = cursor.fetchone()

    conn.close()

    recommended_movies = []

    try:
        if last_liked_movie:
            # Get recommendations based on the user's last liked movie
            seed_movie_id = last_liked_movie['movie_id']
            movies_data = await get_movie_recommendations(seed_movie_id)
        else:
            # Fallback to popular movies if the user has no liked movies
            movies_data = await get_popular_movies()

        # Filter out movies the user has already seen and ensure they have a poster
        for movie in movies_data:
            if movie.get('id') and movie.get('poster_path'):
                if movie['id'] not in all_seen_movie_ids:
                    recommended_movies.append(movie)
                    if len(recommended_movies) >= 20:  # Limit to 20 recommendations
                        break
    except Exception as e:
        print(f"API request failed: {e}")
        return templates.TemplateResponse('recommendations.html', {
            "request": request,
            "movies": [],
            "error": "Could not fetch recommendations."
        })

    return templates.TemplateResponse('recommendations.html', {
        "request": request,
        "movies": recommended_movies
    })


@app.get("/generate-rating-chart")
async def generate_rating_chart(user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    query = "SELECT title, rating FROM watched_movies WHERE user_id = %s ORDER BY watched_at DESC LIMIT 10"
    df = pd.read_sql(query, conn, params=(user['id'],))
    conn.close()

    if df.empty:
        plt.figure(figsize=(10, 6))
        plt.title('Your Recent Movie Ratings')
        plt.xlabel('Movies')
        plt.ylabel('Rating')
        plt.ylim(0, 10)
        plt.xticks([])
        plt.savefig('static/images/rating_trend.png')
        plt.close()
        return {"message": "Chart generated (empty)"}

    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(df)), df['rating'], color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('Your Recent Movie Ratings', fontsize=16, fontweight='bold')
    plt.xlabel('Movies', fontsize=12)
    plt.ylabel('Rating', fontsize=12)
    plt.ylim(0, 10)

    # Customize x-axis labels
    plt.xticks(range(len(df)), df['title'], rotation=45, ha='right')

    # Add value labels on bars
    for bar, rating in zip(bars, df['rating']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(rating), ha='center', va='bottom', fontweight='bold')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/images/rating_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {"message": "Chart generated successfully"}


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)