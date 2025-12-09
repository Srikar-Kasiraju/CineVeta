import os
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline

load_dotenv()

DATA_FILE = "movies_cache.json"

with open(DATA_FILE, "r", encoding="utf-8") as f:
    movies = json.load(f)

movie_texts = [m["overview"] for m in movies]

print("Loading models...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

movie_embeddings = embed_model.encode(
    movie_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

emotion_to_mood = {
    "joy": "feel good happy comedy romance love friendship",
    "sadness": "emotional drama healing hope life",
    "anger": "action revenge power intense thriller",
    "fear": "thriller suspense mystery survival",
    "surprise": "adventure fantasy unexpected journey",
    "disgust": "dark crime thriller psychological",
    "neutral": "drama slice of life meaningful"
}

user_text = input("\n Say something about how you feel:\n> ")
emotion = emotion_classifier(user_text)[0][0]["label"]
print(f"\n Detected emotion: {emotion}")

mood_query = emotion_to_mood.get(emotion, "drama life story")

query_embedding = embed_model.encode(
    mood_query,
    normalize_embeddings=True
)

scores = np.dot(movie_embeddings, query_embedding)
top_indices = scores.argsort()[-5:][::-1]

print("\n Movies for your current mood:")
for rank, idx in enumerate(top_indices, start=1):
    print(f"{rank}. {movies[idx]['title']}")