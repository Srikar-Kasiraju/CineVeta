import os
import json
import time
import sqlite3
from datetime import datetime

import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash

from sentence_transformers import SentenceTransformer
from transformers import pipeline

load_dotenv()

app = Flask(__name__)
app.secret_key = "unmyeong"

DATABASE = "database.db"
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            phone_number TEXT,
            password TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS watch_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            movie_id INTEGER,
            movie_title TEXT,
            watched_at TEXT,
            status TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS mood_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_input TEXT,
            detected_emotion TEXT,
            recommended_movie_ids TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()

with open("movies_cache.json", "r", encoding="utf-8") as f:
    MOVIES_CACHE = json.load(f)

movie_overviews = [m["overview"] for m in MOVIES_CACHE]

print("Loading ML models...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

movie_embeddings = embed_model.encode(
    movie_overviews,
    normalize_embeddings=True,
    show_progress_bar=True
)

EMOTION_TO_MOOD = {
    "joy": "feel good happy comedy romance",
    "sadness": "emotional drama healing",
    "anger": "action revenge thriller",
    "fear": "suspense mystery",
    "surprise": "adventure fantasy",
    "disgust": "dark crime psychological",
    "neutral": "drama slice of life"
}

def fetch_tmdb_movie(movie_id):
    """
    Ultra-robust TMDB fetch.
    Never crashes Flask.
    """
    if not TMDB_API_KEY:
        return None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY}

    try:
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[TMDB ERROR] movie_id={movie_id} → {e}")
        return None

    data = r.json()

    image_path = data.get("poster_path") or data.get("backdrop_path")
    poster_url = (
        f"https://image.tmdb.org/t/p/w500{image_path}"
        if image_path else None
    )

    return {
        "id": data["id"],
        "title": data["title"],
        "overview": data.get("overview", ""),
        "release_date": data.get("release_date"),
        "rating": data.get("vote_average"),
        "poster_url": poster_url
    }

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE username=?",
            (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["username"] = user["username"]
            return redirect(url_for("profile"))

        flash("Invalid credentials")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        pw = generate_password_hash(request.form["password"])
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO users (username, email, phone_number, password)
            VALUES (?, ?, ?, ?)
        """, (
            request.form["username"],
            request.form["email"],
            request.form["phone_number"],
            pw
        ))
        conn.commit()
        conn.close()
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/profile")
def profile():
    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()

    user = conn.execute(
        """
        SELECT id, username, email, phone_number
        FROM users
        WHERE username = ?
        """,
        (session["username"],)
    ).fetchone()


    emotion_rows = conn.execute(
        """
        SELECT detected_emotion
        FROM mood_logs
        WHERE user_id = ?
        """,
        (user["id"],)
    ).fetchall()

    conn.close()

    EMOTION_SCORE = {
        "joy": 5,
        "surprise": 4,
        "neutral": 3,
        "sadness": 2,
        "fear": 2,
        "anger": 1,
        "disgust": 1
    }

    scores = [
        EMOTION_SCORE[row["detected_emotion"]]
        for row in emotion_rows
        if row["detected_emotion"] in EMOTION_SCORE
    ]

    avg_emotion_score = round(sum(scores) / len(scores), 2) if scores else None

    avg_mood_name = None
    if avg_emotion_score is not None:
        if avg_emotion_score >= 4.5:
            avg_mood_name = "Very Positive 😊"
        elif avg_emotion_score >= 3.5:
            avg_mood_name = "Positive 🙂"
        elif avg_emotion_score >= 2.5:
            avg_mood_name = "Neutral 😐"
        elif avg_emotion_score >= 1.5:
            avg_mood_name = "Low 😕"
        else:
            avg_mood_name = "Very Low 😞"

    return render_template(
        "profile.html",
        user=user,
        avg_emotion_score=avg_emotion_score,
        avg_mood_name=avg_mood_name,
        total_entries=len(scores)
    )

@app.route("/recom", methods=["GET", "POST"])
def recom():
    if "username" not in session:
        return redirect(url_for("login"))

    detected_emotion = None
    recommended_movies = []
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("mood_text", "").strip()
        if not user_text:
            flash("Please describe how you feel.")
            return redirect(url_for("recom"))

        detected_emotion = emotion_classifier(user_text)[0][0]["label"]
        mood_query = EMOTION_TO_MOOD.get(detected_emotion, "drama")

        query_embedding = embed_model.encode(
            mood_query,
            normalize_embeddings=True
        )

        scores = np.dot(movie_embeddings, query_embedding)
        top_indices = scores.argsort()[-5:][::-1]

        movie_ids_logged = []

        for idx in top_indices:
            tmdb_id = MOVIES_CACHE[idx]["id"]

            movie = fetch_tmdb_movie(tmdb_id)
            if movie:
                recommended_movies.append(movie)
                movie_ids_logged.append(str(tmdb_id))

            time.sleep(0.15)

        conn = get_db_connection()
        user_id = conn.execute(
            "SELECT id FROM users WHERE username=?",
            (session["username"],)
        ).fetchone()["id"]

        conn.execute("""
            INSERT INTO mood_logs
            (user_id, user_input, detected_emotion, recommended_movie_ids, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            user_text,
            detected_emotion,
            ",".join(movie_ids_logged),
            datetime.utcnow().isoformat()
        ))

        conn.commit()
        conn.close()

    return render_template(
        "recom.html",
        detected_emotion=detected_emotion,
        recommended_movies=recommended_movies,
        user_text=user_text
    )

@app.route("/movie/<int:movie_id>")
def movie_detail(movie_id):
    movie = fetch_tmdb_movie(movie_id)
    if not movie:
        flash("Movie details could not be loaded.")
        return redirect(url_for("recom"))
    return render_template("movie_detail.html", movie=movie)


@app.route("/watch/<int:movie_id>", methods=["POST"])
def watch_movie(movie_id):
    if "username" not in session:
        return redirect(url_for("login"))

    movie = fetch_tmdb_movie(movie_id)
    if not movie:
        return redirect(url_for("recom"))

    conn = get_db_connection()
    user_id = conn.execute(
        "SELECT id FROM users WHERE username=?",
        (session["username"],)
    ).fetchone()["id"]

    conn.execute("""
        INSERT INTO watch_history
        (user_id, movie_id, movie_title, watched_at, status)
        VALUES (?, ?, ?, ?, ?)
    """, (
        user_id,
        movie_id,
        movie["title"],
        datetime.utcnow().isoformat(),
        "watched"
    ))

    conn.commit()
    conn.close()

    return redirect(url_for("movie_detail", movie_id=movie_id))

@app.route("/history")
def history():
    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    user = conn.execute(
        "SELECT id FROM users WHERE username = ?",
        (session["username"],)
    ).fetchone()
    history_rows = conn.execute(
        """
        SELECT user_input, detected_emotion, recommended_movie_ids, created_at
        FROM mood_logs
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user["id"],)
    ).fetchall()

    conn.close()

    history_data = []

    for row in history_rows:
        movie_ids = row["recommended_movie_ids"].split(",")

        movie_titles = [
            m["title"]
            for m in MOVIES_CACHE
            if str(m["id"]) in movie_ids
        ]

        history_data.append({
            "user_input": row["user_input"],
            "emotion": row["detected_emotion"],
            "movies": movie_titles,
            "created_at": row["created_at"]
        })

    return render_template(
        "history.html",
        history_data=history_data
    )

@app.route("/home")
def home():
    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    user = conn.execute(
        "SELECT id FROM users WHERE username = ?",
        (session["username"],)
    ).fetchone()

    popular_movies = []

    if TMDB_API_KEY:
        try:
            url = "https://api.themoviedb.org/3/movie/popular"
            params = {"api_key": TMDB_API_KEY}
            r = requests.get(url, params=params, timeout=5)

            if r.status_code == 200:
                for m in r.json().get("results", [])[:12]:
                    poster = m.get("poster_path") or m.get("backdrop_path")
                    poster_url = (
                        f"https://image.tmdb.org/t/p/w500{poster}"
                        if poster else None
                    )

                    popular_movies.append({
                        "id": m["id"],
                        "title": m["title"],
                        "poster_url": poster_url
                    })
        except Exception as e:
            print("TMDB popular error:", e)

    watched_rows = conn.execute(
        """
        SELECT DISTINCT movie_id
        FROM watch_history
        WHERE user_id = ?
        ORDER BY watched_at DESC
        LIMIT 10
        """,
        (user["id"],)
    ).fetchall()

    conn.close()

    watched_movies = []
    for row in watched_rows:
        movie = fetch_tmdb_movie(row["movie_id"])
        if movie:
            watched_movies.append(movie)
        time.sleep(0.1)  

    return render_template(
        "home.html",
        popular_movies=popular_movies,
        watched_movies=watched_movies
    )

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()

    user = conn.execute(
        "SELECT id FROM users WHERE username=?",
        (session["username"],)
    ).fetchone()
    user_id = user["id"]

    mood_rows = conn.execute(
        """
        SELECT detected_emotion, recommended_movie_ids, created_at
        FROM mood_logs
        WHERE user_id=?
        ORDER BY created_at
        """,
        (user_id,)
    ).fetchall()

    watch_rows = conn.execute(
        """
        SELECT movie_id, movie_title
        FROM watch_history
        WHERE user_id=?
        """,
        (user_id,)
    ).fetchall()

    conn.close()

    EMOTION_SCORE = {
        "joy": 5,
        "surprise": 4,
        "neutral": 3,
        "sadness": 2,
        "fear": 2,
        "anger": 1,
        "disgust": 1
    }

    emotion_count = {}
    timeline_scores = []
    timeline_labels = []

    for i, row in enumerate(mood_rows, start=1):
        emo = row["detected_emotion"]
        emotion_count[emo] = emotion_count.get(emo, 0) + 1

        timeline_labels.append(i)
        timeline_scores.append(EMOTION_SCORE.get(emo, 3))

    dominant_emotion = max(emotion_count, key=emotion_count.get) if emotion_count else None

    emotion_movie_map = {}

    for row in mood_rows:
        emo = row["detected_emotion"]
        movie_ids = row["recommended_movie_ids"].split(",")

        emotion_movie_map.setdefault(emo, {})
        for mid in movie_ids:
            title = next((m["title"] for m in MOVIES_CACHE if str(m["id"]) == mid), None)
            if title:
                emotion_movie_map[emo][title] = emotion_movie_map[emo].get(title, 0) + 1

    total_recommended = sum(len(r["recommended_movie_ids"].split(",")) for r in mood_rows)
    total_watched = len(watch_rows)

    success_rate = round((total_watched / total_recommended) * 100, 2) if total_recommended else 0

    avg_score = round(sum(timeline_scores) / len(timeline_scores), 2) if timeline_scores else 0
    mood_health_score = round((avg_score / 5) * 100, 2)

    return render_template(
        "dashboard.html",

        total_predictions=len(mood_rows),
        total_recommended=total_recommended,
        total_watched=total_watched,
        success_rate=success_rate,
        dominant_emotion=dominant_emotion,
        mood_health_score=mood_health_score,

        emotion_labels=list(emotion_count.keys()),
        emotion_values=list(emotion_count.values()),
        timeline_labels=timeline_labels,
        timeline_scores=timeline_scores,

        emotion_movie_map=emotion_movie_map
    )

@app.route("/playlist")
def playlist():
    if "username" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    user = conn.execute(
        "SELECT id FROM users WHERE username=?",
        (session["username"],)
    ).fetchone()

    watched_rows = conn.execute(
        """
        SELECT DISTINCT movie_id
        FROM watch_history
        WHERE user_id=?
        ORDER BY watched_at DESC
        """,
        (user["id"],)
    ).fetchall()

    conn.close()

    watched_movies = []
    for row in watched_rows:
        movie = fetch_tmdb_movie(row["movie_id"])
        if movie:
            watched_movies.append(movie)
        time.sleep(0.1)

    return render_template(
        "playlist.html",
        watched_movies=watched_movies
    )


if __name__ == "__main__":
    app.run(debug=True)