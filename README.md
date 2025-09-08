Movie Recommendation System
This project implements a Movie Recommendation System using Collaborative Filtering with Matrix Factorization (SVD).
The system is trained and tested on the MovieLens 100K dataset and provides personalized recommendations for users.
Additionally, a Streamlit App is included to demonstrate real-time movie recommendations interactively.

Dataset

This project uses the MovieLens 100K Dataset, which contains 100,000 movie ratings from 943 users on 1,682 movies.  
Due to file size limitations, the dataset is not included in this repository.  
You can download it directly from the [MovieLens official website](https://grouplens.org/datasets/movielens/100k/).
After downloading, extract the folder "ml-100k" and place it inside the `Task4_Recommendation_System/` directory, so the structure looks like this:

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
