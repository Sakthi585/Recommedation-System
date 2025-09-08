# recommender_app.py
import pandas as pd
import streamlit as st
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split


# Load MovieLens dataset
file_path = 'D:/Document/Machine learning Intern/Task 4/ml-100k/u.data'
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

# Train SVD model
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
model = SVD()
model.fit(trainset)

# Load movie titles
movies = pd.read_csv('D:/Document/Machine learning Intern/Task 4/ml-100k/u.item', 
                     sep='|', encoding='latin-1', header=None, usecols=[0,1], names=['movieId','title'])

# Streamlit App
st.title("Movie Recommendation System")
st.write("Get top-5 movie recommendations for a user!")

# User input
user_id = st.number_input("Enter User ID (1-943):", min_value=1, max_value=943, value=1, step=1)

# Generate Top-N recommendations
def get_top_n_for_user(model, user_id, movies, n=5):
    all_movie_ids = movies['movieId'].tolist()
    predictions = [model.predict(str(user_id), str(iid)) for iid in all_movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_n = predictions[:n]
    top_titles = [movies[movies.movieId==int(pred.iid)].title.values[0] for pred in top_n]
    return top_titles

if st.button("Get Recommendations"):
    top_movies = get_top_n_for_user(model, user_id, movies, n=5)
    st.write("Top-5 Recommended Movies:")
    for i, title in enumerate(top_movies, 1):
        st.write(f"{i}. {title}")
