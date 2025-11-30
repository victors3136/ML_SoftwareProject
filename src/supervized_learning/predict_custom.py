from lib.data import load_data
import numpy as np
from supervized_learning.knn_clas import KNNClassifier

df = load_data(shuffle=True)

features = ["danceability", "energy", "tempo", "valence"]

# Prepare labels
popularity = df["track_popularity"]
df["track_popular"] = [1 if p > 70 else 0 for p in popularity]
y = np.array(df["track_popular"])

print("Label counts:", np.unique(y, return_counts=True))

# Extract raw columns as float arrays
raw_columns = {}
for f in features:
    raw = np.array(df[f], dtype=float)
    raw_columns[f] = raw

# IMPUTE missing values
for f in features:
    col = raw_columns[f]
    mean_val = np.nanmean(col)        # compute mean ignoring NaN
    col = np.where(np.isnan(col), mean_val, col)
    raw_columns[f] = col

# Normalize AFTER cleaning
minmax = {}
for f in features:
    col = raw_columns[f]
    mn, mx = col.min(), col.max()
    minmax[f] = (mn, mx)
    df[f] = (col - mn) / (mx - mn)

# Build X matrix
X = np.array([
    [df[f][i] for f in features]
    for i in range(len(popularity))
], dtype=float)

# Train model
model = KNNClassifier(k=7)
model.fit(X, y)

popular = X[y == 1]
print("Popular percentiles:")
for i, f in enumerate(features):
    print(f, np.percentile(popular[:, i], 25),
              np.percentile(popular[:, i], 50),
              np.percentile(popular[:, i], 75))
"""danceability 0.5245082056298229 0.6576459080534726 0.7641560699923922
energy 0.5530157406609354 0.6963313215700973 0.8130884207023867
tempo 0.2733055892004928 0.37148151598910945 0.4543153514084288
valence 0.31585544182160014 0.5174430749947775 0.7221641946939628"""

test_index = 1225 
true_label = y[test_index]
pred_label = model.predict([X[test_index]])[0]
print(X[test_index])
print("True:", true_label, "Pred:", pred_label)
"""
[0.50331486 0.75245491 0.47246809 0.60100272] - pred = 1, true =0
[0.81958483 0.48185905 0.47563589 0.84854815] - same
[0.72720356 0.4157134  0.34117519 0.44432839] - True: 1 Pred: 0
[0.27833931 0.71637546 0.5574345  0.4881972 ] - True: 1 Pred: 1
"""


# Predict a custom song
raw_song = [0.278, 0.71, 0.55, 0.488]

# normalize using SAME min/max
scaled_song = []
for value, f in zip(raw_song, features):
    mn, mx = minmax[f]
    scaled_song.append((value - mn) / (mx - mn))

scaled_song = np.array(scaled_song)

print("Scaled song:", scaled_song)

prediction = model.predict([scaled_song])[0]
print("Predicted popular?", "YES" if prediction > 0.7 else "NO" + str(prediction))
