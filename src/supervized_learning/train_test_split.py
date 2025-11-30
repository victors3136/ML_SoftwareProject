from lib.data import load_data
import numpy as np
from supervized_learning.knn_clas import KNNClassifier

df = load_data(shuffle=True)

popularity = df["track_popularity"]
df["track_popular"] = [1 if p > 70 else 0 for p in popularity]
y = np.array(df["track_popular"])

features = ["danceability", "energy", "tempo", "valence"]
X = np.array([[df[f][i] for f in features] for i in range(len(popularity))], dtype=float)

# 80/20 split
split = int(0.8 * len(X))

X_train = X[:split]
y_train = y[:split]
X_test  = X[split:]
y_test  = y[split:]

# Train KNN
model = KNNClassifier(k=7)
model.fit(X_train, y_train)

# Predict on the 20% test set
preds = model.predict(X_test)

accuracy = (preds == y_test).mean()
print("Test accuracy:", accuracy)
#Test accuracy: 0.7011375387797312
