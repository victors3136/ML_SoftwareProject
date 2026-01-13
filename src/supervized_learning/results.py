import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # won't see lib.data otherwise
import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score)
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
import lime
import lime.lime_tabular

from lib.data import load_data
from supervized_learning.knn_clas import KNNClassifier

print("Loading Data...")
df = load_data(shuffle=True)

features = ["danceability", "energy", "tempo", "valence"]
target_col = "track_popularity"

# Extract raw data to simple lists first
X_raw = []
y_raw = []

rows = df.shape[0]
for i in range(rows):
    # Safe extraction - if a value is missing, put np.nan
    row_feats = [df[f][i] if df[f][i] is not None else np.nan for f in features]
    X_raw.append(row_feats)
    # Target: 1 if > 70, else 0
    pop = df[target_col][i]
    if pop is None:
        y_raw.append(0)
    else:
        y_raw.append(1 if pop > 70 else 0)

# Convert to Numpy Arrays
X = np.array(X_raw, dtype=float)
y = np.array(y_raw, dtype=int)

# Removing rows with ANY NaN values (Sklearn crashes if a NaN exists)
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
y = y[mask]

print(f"Data prepared: {X.shape[0]} valid samples (NaNs removed)")
print(f"Class distribution: {np.unique(y, return_counts=True)}")

# HYPERPARAMETER OPTIMIZATION (Grid Search)
print("\n~~~ Hyperparameter Optimization ~~~")
k_candidates = [3, 5, 7, 9, 11, 15]
best_k = 5
best_val_score = 0

# Spliting for tuning
split = int(0.8 * len(X))
X_train_tune, X_val_tune = X[:split], X[split:]
y_train_tune, y_val_tune = y[:split], y[split:]

for k in k_candidates:
    model = KNNClassifier(k=k)
    model.fit(X_train_tune, y_train_tune)
    preds = model.predict(X_val_tune)
    acc = accuracy_score(y_val_tune, preds)
    print(f"k={k}: Accuracy = {acc:.4f}")
    
    if acc > best_val_score:
        best_val_score = acc
        best_k = k

print(f"-> Best k found: {best_k}")

# STATISTICAL ANALYSIS
print(f"\n~~~ 5-Fold Cross-Validation (k={best_k}) ~~~")
k_folds = 5
indices = np.arange(len(X))
np.random.shuffle(indices)
fold_size = len(X) // k_folds

# metrics for statistics
results = {
    "Accuracy": [], "Precision": [], "Recall": [], 
    "F1-Score": [], "AUC": [], "AUPRC": []
}

for i in range(k_folds):
    # Creating fold indices
    val_idx = indices[i*fold_size : (i+1)*fold_size]
    train_idx = np.setdiff1d(indices, val_idx)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Training
    model = KNNClassifier(k=best_k)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    # Geting Probabilities for AUC/AUPRC
    # entire first column from predict_probAUC (it returns [prob_0, prob_1])
    probs = model.predict_probAUC(X_val)[:, 1] 
    
    results["Accuracy"].append(accuracy_score(y_val, preds))
    results["Precision"].append(precision_score(y_val, preds, zero_division=0))
    results["Recall"].append(recall_score(y_val, preds, zero_division=0))
    results["F1-Score"].append(f1_score(y_val, preds, zero_division=0))
    results["AUC"].append(roc_auc_score(y_val, probs))
    results["AUPRC"].append(average_precision_score(y_val, probs))

# Statistical summary table
print(f"\n{'Metric':<15} | {'Mean':<8} | {'Std Dev':<8} | {'95% CI'}")
print("-" * 55)
for metric, values in results.items():
    mean = np.mean(values)
    std = np.std(values)
    # calculating with 95% confidence interval
    ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=stats.sem(values))
    print(f"{metric:<15} | {mean:.4f}   | {std:.4f}   | ({ci[0]:.3f}, {ci[1]:.3f})")


# LIBRARY COMPARISON
print("\n~~~ Library Comparison (Sklearn) ~~~")
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

sk_model = SklearnKNN(n_neighbors=best_k)
sk_model.fit(X_train_scaled, y_train)
sk_preds = sk_model.predict(X_val_scaled)
sk_acc = accuracy_score(y_val, sk_preds)

print(f"Custom Model Accuracy (Last Fold): {results['Accuracy'][-1]:.4f}")
print(f"Sklearn Model Accuracy (Last Fold): {sk_acc:.4f}")

# EXPLAINABILITY (LIME)
print("\n~~~ Explainability (LIME) ~~~")

# Probabilities for LIME
def predict_fn(data):
    return model.predict_probAUC(data)

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=features,
    class_names=['Not Popular', 'Popular'],
    mode='classification'
)

# Details that explain the first instance
exp = explainer.explain_instance(X_val[0], predict_fn, num_features=len(features))

print("LIME Explanation for one song:")
print(exp.as_list())
# exp.save_to_file('lime_explanation.txt')
