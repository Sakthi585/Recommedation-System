# recommender_system.py
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load dataset
file_path = 'D:/Document/Machine learning Intern/Task 4/ml-100k/u.data'
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

# Split dataset
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Build SVD model
model = SVD()
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Evaluate model
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Load movie titles
movies = pd.read_csv('D:/Document/Machine learning Intern/Task 4/ml-100k/u.item', 
                     sep='|', encoding='latin-1', header=None, usecols=[0,1], names=['movieId','title'])

# Top-N recommendations for a specific user
def get_top_n(predictions, n=5):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        top_n.setdefault(uid, [])
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n = get_top_n(predictions, n=5)
print("Top-5 recommendations for User 1:", top_n['1'])

# Map movie IDs to titles
user1_recs = top_n['1']
user1_titles = [movies[movies.movieId==int(iid)].title.values[0] for iid, _ in user1_recs]
print("Top-5 movie titles for User 1:", user1_titles)
