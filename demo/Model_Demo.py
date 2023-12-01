import streamlit as st
import os
import pickle
import pandas as pd

# Load the saved model with demographics
model_filename = 'model/best_model_with_demographics.pkl'
with open(model_filename, 'rb') as file:
    loaded_model_with_demographics = pickle.load(file)

loaded_model = loaded_model_with_demographics['model']
loaded_user_embeddings = loaded_model_with_demographics['user_embeddings']
loaded_item_embeddings = loaded_model_with_demographics['item_embeddings']
loaded_user_info_embeddings = loaded_model_with_demographics['user_info_embeddings']

# Load movie details
movie_details = {}
movieId = []
with open(os.path.join('data/ml-100k', 'u.item'), encoding='ISO-8859-1') as f:
    for line in f:
        parts = line.strip().split('|')
        movie_id = int(parts[0])
        movie_title = parts[1]
        movie_genre = parts[5:]
        movie_details[movie_id] = {'title': movie_title, 'genre': movie_genre}
        movieId.append(movie_id)

# Create a Streamlit app
st.title("Movie Recommendation App")

# Sidebar with user selection
selected_user = st.sidebar.selectbox("Select a User", range(1, 11))

# Generate recommendations for the selected user
# Load the test set from the file
trainset_path = 'data/trainset.pkl'
# Load the test set from the file
with open(trainset_path, 'rb') as file:
    trainset = pickle.load(file)

seen_movies = [iid for iid in trainset.ur[trainset.to_inner_uid(str(selected_user))]]

# Get the items that the selected user has already interacted with in the training set
seen_iids = [iid for iid, _ in trainset.ur[trainset.to_inner_uid(str(selected_user))]]
user_predictions = [
    loaded_model.predict(str(selected_user), str(iid), verbose=False) for iid in movieId if iid not in seen_iids
]
user_predictions.sort(key=lambda x: x.est, reverse=True)
top_n_recommendations = [prediction.iid for prediction in user_predictions[:5]]

# Function to decode genre indicators
def decode_genres(genre_indicator):
    genres = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary",
              "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
    decoded_genres = [genre for genre, indicator in zip(genres, genre_indicator) if indicator == '1']
    return ', '.join(decoded_genres)

# Display recommendations
st.header(f"Top Recommendations for User with ID {selected_user}:")
for item_id in top_n_recommendations:
    movie_info = movie_details.get(int(item_id))
    if movie_info:
        st.write(f"Movie ID: {item_id}, Title: {movie_info['title']}, Genre: {decode_genres(movie_info['genre'])}")

# Display top 10 movies with their genres that have been watched by the selected user

# Create a DataFrame to store user interactions
user_interactions_df = pd.DataFrame(seen_movies, columns=['item_id', 'rating'])

# Sort the DataFrame by rating in descending order
top_watched_movies = user_interactions_df.sort_values(by='rating', ascending=False).head(10)

# Get the item IDs of the top 10 watched movies
top_watched_movie_ids = top_watched_movies['item_id'].tolist()

st.header(f"Top 10 Watched Movies for User with ID {selected_user}:")

for item_id in top_watched_movie_ids:
    movie_info = movie_details.get(int(item_id))
    if movie_info:
        st.write(f"Movie ID: {item_id}, Title: {movie_info['title']}, Genre: {decode_genres(movie_info['genre'])}")