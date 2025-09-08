Movie Recommendation System
This project implements a Movie Recommendation System using Collaborative Filtering with Matrix Factorization (SVD).
The system is trained and tested on the MovieLens 100K dataset and provides personalized recommendations for users.
Additionally, a Streamlit App is included to demonstrate real-time movie recommendations interactively.

Dataset
Source: MovieLens 100K Dataset
Files used:
u.data → user ratings (userId, movieId, rating, timestamp)
u.item → movie information (movieId, title, genres)
Size: 100,000 ratings from 943 users on 1,682 movies

Steps Performed
Load the dataset (u.data and u.item).
Preprocess ratings and map movie IDs to movie titles.
Use Surprise library to:
Define a Reader for parsing ratings.
Train a SVD (Singular Value Decomposition) model.
Evaluate model performance using RMSE.
Generate top-N recommendations for a given user.
Build a Streamlit app to interactively input a user ID and display movie recommendations.

Results
The collaborative filtering model achieved a good RMSE score on test data.
Users can get personalized recommendations by entering their user ID in the Streamlit app.
Movie titles and predicted ratings are displayed in ranked order.

Technologies Used
Python
Scikit-learn
Surprise (Collaborative Filtering library)
Pandas, Numpy
Streamlit (for web app)
