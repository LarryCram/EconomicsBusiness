import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import archetypes as arch

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Feature engineering: no complex feature engineering as the data is synthetic and simple
# But typically this could include interactions, polynomial features, or transformations.

# Archetypal Analysis Setup
pipe = Pipeline([
    ('aa', arch.AA())
])

# Hyperparameters to tune
param_grid = {
    'aa__n_archetypes': [2, 3, 4, 5],  # different numbers of archetypes to test
}

# Grid Search with cross-validation
grid = GridSearchCV(pipe, param_grid, cv=5, scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# Best model and parameters
print(f"Best parameters: {grid.best_params_}")
best_model = grid.best_estimator_['aa']

# Fitting the model with the best parameters on the full data
best_model.fit(X_scaled)

# Analysis of the results
archetypes = best_model.archetypes_
print(f"Archetypes:\n {archetypes}")

# Predictions and reconstructions
X_reconstructed = best_model.transform(X_scaled)
X_reconstructed = best_model.inverse_transform(X_reconstructed)

# Calculate silhouette score
sil_score = silhouette_score(X_scaled, best_model.labels_)
print(f"Silhouette Score: {sil_score}")

# Plotting original vs reconstructed data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', label='Original')
plt.title('Original Data')
plt.subplot(1, 2, 2)
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], c=best_model.labels_, cmap='viridis', label='Reconstructed')
plt.title('Reconstructed Data')
plt.show()

# Interpretations:
# The plots show how well the archetypes approximate the original data distribution.
# The silhouette score gives an insight into the cohesion and separation of the formed clusters.