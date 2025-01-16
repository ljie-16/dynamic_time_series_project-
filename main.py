from data_preprocessing import load_and_preprocess_data, augment_and_resample
from lstm_autoencoder import build_lstm_autoencoder, train_autoencoder
from ridge_regression import train_ridge_regression, evaluate_ridge_regression
from evaluation import plot_model_results, plot_coefficients
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV

# Load and preprocess data
X, y = load_and_preprocess_data('/path/to/data.csv')
X_resampled, y_resampled = augment_and_resample(X, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)

# Train Ridge Regression with Cross-Validation
ridge = RidgeCV(cv=5)
ridge.fit(X_train, y_train)

# Evaluate and Plot Results
score_train = ridge.score(X_train, y_train)
score_test = ridge.score(X_test, y_test)
print(f"Train Score: {score_train}, Test Score: {score_test}")

# Plot regression results
plot_model_results(ridge, X_train, X_test, y_train, y_test, plot_intervals=True, plot_anomalies=True, scale=1.96)

# Plot coefficients
plot_coefficients(ridge, feature_names=X.columns)
