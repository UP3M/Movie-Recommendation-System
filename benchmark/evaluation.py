##Evaluation

import os
import pickle
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from collections import defaultdict
from surprise.prediction_algorithms.knns import KNNBasic
import numpy as np

# Load the saved model with demographics
model_filename = 'model/best_model_with_demographics.pkl'
with open(model_filename, 'rb') as file:
    loaded_model_with_demographics = pickle.load(file)

loaded_model = loaded_model_with_demographics['model']
loaded_user_embeddings = loaded_model_with_demographics['user_embeddings']
loaded_item_embeddings = loaded_model_with_demographics['item_embeddings']
loaded_user_info_embeddings = loaded_model_with_demographics['user_info_embeddings']

# Load the test set from the file
testset_path = 'benchmark/testset.pkl'
# Load the test set from the file
with open(testset_path, 'rb') as file:
    testset = pickle.load(file)


# Load the MovieLens 100K dataset
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(os.path.join('data/ml-100k', 'u.data'), reader=reader)

# Generate predictions for the test set
predictions = loaded_model.test(testset)

# Evaluate the model using multiple metrics
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# Cross-validate the model for additional metrics
cv_results = cross_validate(loaded_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Display evaluation results

print("Individual Predictions Evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

print("Cross-Validation Results:")
print("Average RMSE:", round(cv_results['test_rmse'].mean(), 4))
print("Average MAE:", round(cv_results['test_mae'].mean(), 4))


def precision_recall_at_k(predictions, k=5, threshold=3.5):
    """
    Compute precision and recall at k.

    Parameters:
    - predictions: List of Prediction objects as returned by the test method of an algorithm.
    - k: The number of recommendations.
    - threshold: The threshold for considering an item relevant.

    Returns:
    - precision: Precision at k.
    - recall: Recall at k.
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    precision = np.mean(list(precisions.values()))
    recall = np.mean(list(recalls.values()))

    return precision, recall

# Precision and Recall evaluation
k_fold = KFold(n_splits=5, random_state=42)
precision_list = []
recall_list = []

for train, test in k_fold.split(data):
    model = KNNBasic(sim_options={'user_based': False})  # Use a simple collaborative filtering model for precision-recall
    model.fit(train)
    test_predictions = model.test(test)
    precision, recall = precision_recall_at_k(test_predictions, k=5, threshold=3.5)
    precision_list.append(precision)
    recall_list.append(recall)

# Average precision and recall over folds
avg_precision = sum(precision_list) / len(precision_list)
avg_recall = sum(recall_list) / len(recall_list)

# Display evaluation results
print("Precision-Recall Metrics:")
print(f"Average Precision at K=5: {avg_precision:.4f}")
print(f"Average Recall at K=5: {avg_recall:.4f}")