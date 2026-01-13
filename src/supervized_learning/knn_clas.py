import numpy as np

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # FIX SCALING SAFELY SO WE DON'T HAVE 0'S ANYMORE
        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        ranges = maxs - mins
        ranges[ranges == 0] = 1.0    # prevent division by zero

        self.min = mins
        self.range = ranges

        X = (X - self.min) / self.range

        # Lazy learning: simply store the data
        self.X_train = X
        self.y_train = y
        #print("Scaled X[0]:", X[0])
        #print("X_train: ", self.X_train)


    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X_test):
        predictions = []
        #print("scaled song:", X_test[0])
        X_test = (X_test - self.min) / self.range
        for x in X_test:
            distances = np.array([self._euclidean_distance(x, x_train)
                                  for x_train in self.X_train])
            
            k_idx = distances.argsort()[:self.k]
            k_nearest_labels = self.y_train[k_idx]
            
            # Majority vote
            values, counts = np.unique(k_nearest_labels, return_counts=True)
            predictions.append(values[np.argmax(counts)])
            #print("Predictions: ", predictions)
        

        return np.array(predictions)
    
    def predict_probAUC(self, X_test):
        X_test = np.asarray(X_test, dtype=float)
        X_test = (X_test - self.min) / self.range
        
        probas = []
        for x in X_test:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            
            k_idx = distances.argsort()[:self.k]
            k_nearest_labels = self.y_train[k_idx]
            
            # Probability of being popular (== 1)
            prob_popular = np.sum(k_nearest_labels == 1) / self.k
            
            probas.append([1 - prob_popular, prob_popular]) 
            
        return np.array(probas)

def k_fold_cv(model_class, X, y, k_folds=5, **model_params):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        fold_size = len(X) // k_folds
        scores = []

        for i in range(k_folds):
            val_idx = indices[i*fold_size:(i+1)*fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_class(**model_params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            accuracy = np.mean(preds == y_val)
            scores.append(accuracy)

        return np.mean(scores), scores
