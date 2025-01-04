import h5py
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay

# Path to HDF5 file with results from simulations
file_path = f'results.h5'

with h5py.File(file_path, 'r') as f:
    combined_vectors2 = f["combined_vectors"][:]
    gesture_ids_data2 = f["gesture_ids"][:]

print("Gesture IDs data length:", len(gesture_ids_data2))
print("Original Combined vectors shape:", np.array(combined_vectors2).shape)
print("Unique Gesture IDs:", np.unique(gesture_ids_data2))

# Input data
X = combined_vectors2
y = gesture_ids_data2

# Date normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

"""Logistic regression"""

param_grid_lr = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200]
}

grid_search_lr = GridSearchCV(LogisticRegression(), param_grid_lr, refit=True, verbose=3, cv=2)

grid_search_lr.fit(X_train, y_train)

print("Best parameters for logistic regression: ", grid_search_lr.best_params_)

# Predicting on the test set with the best model
y_pred_lr = grid_search_lr.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Accuracy of the best logistic regression model: {accuracy_lr * 100:.2f}%")

# Save trained model to file
# joblib.dump(grid_search_lr.best_estimator_, 'best_lr_model.joblib')

"""kNN"""

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'minkowski']
}

grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, refit=True, verbose=3, cv=2)

grid_search_knn.fit(X_train, y_train)

print("Best parameters for kNN: ", grid_search_knn.best_params_)

y_pred_knn = grid_search_knn.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy of the best kNN model: {accuracy_knn * 100:.2f}%")

# Save trained model to file
# joblib.dump(grid_search_knn.best_estimator_, 'best_knn_model.joblib')

"""Random Forest Classifier"""

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, refit=True, verbose=3, cv=2)

grid_search_rf.fit(X_train, y_train)

print("Best parameters for Random Forest: ", grid_search_rf.best_params_)

y_pred_rf = grid_search_rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy of the best Random Forest model: {accuracy_rf * 100:.2f}%")

# Save trained model to file
# joblib.dump(grid_search_rf.best_estimator_, 'best_rf_model.joblib')

"""MLP"""

param_grid_mlp = {
    'hidden_layer_sizes': [(50,50), (100,100)],
    'activation': ['relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['adaptive']
}

grid_search_mlp = GridSearchCV(MLPClassifier(max_iter=1000), param_grid_mlp, refit=True, verbose=3, cv=2)

grid_search_mlp.fit(X_train, y_train)

print("Best parameters MLPClassifier: ", grid_search_mlp.best_params_)

y_pred_mlp = grid_search_mlp.predict(X_test)

accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"Accuracy of the best MLPClassifier model: {accuracy_mlp * 100:.2f}%")

# Save trained model to file
# joblib.dump(grid_search_mlp.best_estimator_, 'best_mlp_model.joblib')