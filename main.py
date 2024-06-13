import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load and prepare the data
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

# Split the data into training and test sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use the SVD algorithm to train the model
algo = SVD()
algo.fit(trainset)

# Make predictions on the test set
predictions = algo.test(testset)

# Evaluate the accuracy of the model
print("RMSE:", accuracy.rmse(predictions))


# Function to get top N recommendations for a user
def get_top_n_recommendations(algo, user_id, n=10):
    # Get a list of all item_ids
    all_items = ratings['item_id'].unique()

    # Predict ratings for all items not rated by the user
    user_ratings = [algo.predict(user_id, item_id) for item_id in all_items]

    # Sort predictions by estimated rating
    user_ratings.sort(key=lambda x: x.est, reverse=True)

    # Get top N items
    top_n = user_ratings[:n]
    return [(pred.iid, pred.est) for pred in top_n]


# Get top 10 recommendations for user with ID 1
user_id = 1
top_recommendations = get_top_n_recommendations(algo, user_id, n=10)
print("Top 10 recommendations for user 1:", top_recommendations)
