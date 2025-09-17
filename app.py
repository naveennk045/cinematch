from fastapi import FastAPI, HTTPException, Depends, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
from mysql.connector import errorcode
import bcrypt
import jwt
from datetime import datetime, timedelta
import asyncio
import httpx
from contextlib import asynccontextmanager


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


class Token(BaseModel):
    access_token: str
    token_type: str


class MovieDetails(BaseModel):
    id: int
    title: str
    poster_path: Optional[str] = None
    overview: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


# Configuration
TMDB_API_KEY = "64270cfedbe3933ef55f0be854ba29a2"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Naveennk*@07",
    "database": "cinematch"
}

security = HTTPBearer()


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
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(user_id: int = Depends(verify_token)):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, username FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


async def get_movie_details(movie_id: int) -> Optional[dict]:
    """Async function to get movie details from TMDB API."""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching movie details: {e}")
    return None


async def search_movies(query: str) -> List[dict]:
    """Async function to search movies from TMDB API."""
    url = f"{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={query}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])[:10]  # Return top 10 results
        except Exception as e:
            print(f"Error searching movies: {e}")
    return []


# Authentication routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    conn.close()

    if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user['id'])}, expires_delta=access_token_expires
        )
        response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
        response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
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
async def register(request: Request, username: str = Form(...), password: str = Form(...)):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)",
                       (username, hashed_password.decode('utf-8')))
        conn.commit()
        conn.close()
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    except mysql.connector.IntegrityError:
        conn.close()
        return templates.TemplateResponse("register.html", {
            "request": request,
            "error": "Username already exists."
        })


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="access_token")
    return response


# Dashboard and main functionality
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    # Extract token from cookie
    token = request.cookies.get("access_token")
    if not token or not token.startswith("Bearer "):
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    try:
        # Remove "Bearer " prefix
        token = token[7:]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT movie_id, title, rating FROM watched_movies WHERE user_id = %s ORDER BY watched_at DESC",
                   (user_id,))
    watched_data = cursor.fetchall()

    cursor.execute("SELECT movie_id, title FROM liked_movies WHERE user_id = %s ORDER BY liked_at DESC", (user_id,))
    liked_data = cursor.fetchall()

    cursor.execute("SELECT genre FROM genres WHERE user_id = %s", (user_id,))
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
async def add_watched_movie(movie: WatchedMovie, user_id: int = Depends(verify_token)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO watched_movies (user_id, movie_id, title, rating) VALUES (%s, %s, %s, %s)",
        (user_id, movie.movie_id, movie.title, movie.rating)
    )
    conn.commit()
    conn.close()
    return {"success": True}


@app.post("/api/liked-movie")
async def add_liked_movie(movie: LikedMovie, user_id: int = Depends(verify_token)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO liked_movies (user_id, movie_id, title) VALUES (%s, %s, %s)",
        (user_id, movie.movie_id, movie.title)
    )
    conn.commit()
    conn.close()
    return {"success": True}


@app.post("/api/genre")
async def add_genre(genre: Genre, user_id: int = Depends(verify_token)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO genres (user_id, genre) VALUES (%s, %s)", (user_id, genre.genre))
    conn.commit()
    conn.close()
    return {"success": True}


@app.get("/api/search-movies")
async def search_movies_api(q: str):
    movies = await search_movies(q)
    return {"results": movies}


@app.get("/recommendations", response_class=HTMLResponse)
async def recommendations(request: Request):
    # Extract token from cookie
    token = request.cookies.get("access_token")
    if not token or not token.startswith("Bearer "):
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    try:
        token = token[7:]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Get IDs of movies the user has already seen
    cursor.execute("SELECT movie_id FROM watched_movies WHERE user_id = %s", (user_id,))
    watched_movie_ids = {row['movie_id'] for row in cursor.fetchall()}

    cursor.execute("SELECT movie_id FROM liked_movies WHERE user_id = %s", (user_id,))
    liked_movie_ids = {row['movie_id'] for row in cursor.fetchall()}

    all_seen_movie_ids = watched_movie_ids.union(liked_movie_ids)

    # Find the last liked movie to use as a seed
    cursor.execute("SELECT movie_id FROM liked_movies WHERE user_id = %s ORDER BY liked_at DESC LIMIT 1", (user_id,))
    last_liked_movie = cursor.fetchone()

    conn.close()

    recommended_movies = []
    api_url = ""

    if last_liked_movie:
        # Get recommendations based on the user's last liked movie
        seed_movie_id = last_liked_movie['movie_id']
        api_url = f"{TMDB_BASE_URL}/movie/{seed_movie_id}/recommendations?api_key={TMDB_API_KEY}"
    else:
        # Fallback to popular movies if the user has no liked movies
        api_url = f"{TMDB_BASE_URL}/movie/popular?api_key={TMDB_API_KEY}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(api_url)
            response.raise_for_status()
            data = response.json()

            # Filter out movies the user has already seen and ensure they have a poster
            for movie in data.get('results', []):
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
async def generate_rating_chart(user_id: int = Depends(verify_token)):
    conn = get_db_connection()
    query = "SELECT title, rating FROM watched_movies WHERE user_id = %s ORDER BY watched_at DESC LIMIT 10"
    df = pd.read_sql(query, conn, params=(user_id,))
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