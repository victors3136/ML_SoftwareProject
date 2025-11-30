from lib.data import load_data
import pandas as pd
import numpy as np
from supervized_learning import knn_clas

df = load_data(shuffle=True)
"""
print("AVAILABLE KEYS:", list(getattr(df, "_data", {}).keys()))
try:
    first = df.head(1)
    print("HEAD(1):", first)
except Exception as e:
    print("head() failed:", e)

#print("TYPE OF DF:", type(df))
#print("FIRST 5 ROWS OF DF:", df[:5])
#print("TRACK POPULARITY SAMPLE:", df["track_popularity"][:10])
"""
#print(df["track_popularity"][0:])

features = ["danceability", "energy", "tempo", "valence"]

# cleaning the columns mentioned in features by replacing None with column mean
for f in features:
    col = df[f]
    valid = [v for v in col if v is not None]
    mean_v = sum(valid) / len(valid)
    df[f] = [mean_v if v is None else v for v in col]


# Predicting whether a song is popular (>70)
popularity = df["track_popularity"]
df["track_popular"] = [1 if p > 70 else 0 for p in popularity]

y = np.array(df["track_popular"])
X = np.array([[df[f][i] for f in features] for i in range(len(popularity))], dtype=float)
mean_acc, scores = knn_clas.k_fold_cv(knn_clas.KNNClassifier, X, y, k_folds=5, k=7)

#print("Count popular=1:", sum(y)) #  1327
#print("Count not popular=0:", len(y) - sum(y)) # 3504
#popular_mask = (y == 1)
#print("Mean features of popular songs:")
#print(X[popular_mask].mean(axis=0))
#[  0.64826827   0.67180016 121.13051319   0.52517151]

print("Fold accuracies:", scores)
print("Mean accuracy:", mean_acc)
"""
Fold accuracies: [0.6790890269151139, 0.6956521739130435, 0.7080745341614907, 0.6956521739130435, 0.7111801242236024]
Mean accuracy: 0.6979296066252587

accuracies: [0.7018633540372671, 0.7080745341614907, 0.7267080745341615, 0.7028985507246377, 0.6997929606625258]
Mean accuracy: 0.7078674948240166

Fold accuracies: [0.7111801242236024, 0.6842650103519669, 0.7091097308488613, 0.7028985507246377, 0.6853002070393375]
Mean accuracy: 0.6985507246376812
"""