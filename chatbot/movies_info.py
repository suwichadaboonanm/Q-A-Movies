import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

movies = [
    {"title": "Inception", "genre": "Sci-Fi", "director": "Christopher Nolan", "cast": "Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page", "plot": "A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea into a CEO's mind."},
    {"title": "The Dark Knight", "genre": "Action", "director": "Christopher Nolan", "cast": "Christian Bale, Heath Ledger, Aaron Eckhart", "plot": "When the menace known as the Joker emerges, Batman must confront chaos and anarchy in Gotham City."},
    {"title": "Interstellar", "genre": "Sci-Fi", "director": "Christopher Nolan", "cast": "Matthew McConaughey, Anne Hathaway, Jessica Chastain", "plot": "A team of explorers travel through a wormhole in space to ensure humanity's survival."},
    {"title": "The Matrix", "genre": "Sci-Fi", "director": "Lana Wachowski, Lilly Wachowski", "cast": "Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss", "plot": "A computer hacker learns about the true nature of reality and his role in the war against its controllers."},
    # Add more movies as needed
]

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the movies
movie_texts = [f"{movie['title']} {movie['genre']} {movie['director']} {movie['cast']}" for movie in movies]
movie_embeddings = model.encode(movie_texts)

# Create a FAISS index
dimension = movie_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(movie_embeddings)

def retrieve_movie(query, top_k=1):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    retrieved_movies = [movies[idx] for idx in indices[0]]
    return retrieved_movies

def chatbot():
    print("Welcome to the Movie Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        retrieved_movies = retrieve_movie(user_input)
        if retrieved_movies:
            movie = retrieved_movies[0]
            print(f"Bot: Here's a movie you might like: {movie['title']}")
            print(f"Genre: {movie['genre']}")
            print(f"Director: {movie['director']}")
            print(f"Cast: {movie['cast']}")
            print(f"Plot: {movie['plot']}")
        else:
            print("Bot: Sorry, I couldn't find any movies matching your query.")

# Run the chatbot
chatbot()