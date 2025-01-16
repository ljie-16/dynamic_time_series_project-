import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

def plot_model_results(model, X_train, X_test, y_train, y_test, plot_intervals=False, plot_anomalies=False, scale=1.96):
    """
    Plots model predictions vs. actual values, with optional prediction intervals and anomalies.
    """
    # Predictions
    predictions = model.predict(X_test)

    plt.figure(figsize=(15, 5))
    plt.plot(y_test.values, label="Actual", linewidth=2)
    plt.plot(predictions, label="Predicted", linewidth=2, color='g')
    plt.title("Regression Results")
    plt.legend(loc="upper left")

    if plot_intervals:
        # Cross-validation for intervals
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_squared_error")
        mae = np.mean(cv_scores) * (-1)
        deviation = np.sqrt(np.std(cv_scores))

        lower = predictions - scale * deviation
        upper = predictions + scale * deviation

        plt.fill_between(range(len(predictions)), lower, upper, color="gray", alpha=0.2, label="Prediction Interval")

    if plot_anomalies:
        # Detect anomalies outside the interval
        anomalies = np.where((y_test.values < lower) | (y_test.values > upper), y_test.values, np.nan)
        plt.plot(anomalies, "o", color="red", markersize=5, label="Anomalies")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("regression_results_plot.pdf")
    plt.show()

def plot_coefficients(model, feature_names):
    """
    Plots sorted coefficients for linear regression models like Ridge or Lasso.
    """
    coefs = pd.DataFrame(model.coef_, index=feature_names, columns=["Coefficient"])
    coefs["abs"] = coefs["Coefficient"].abs()
    coefs = coefs.sort_values(by="abs", ascending=False)

    plt.figure(figsize=(15, 5))
    coefs["Coefficient"].plot(kind="bar", title="Feature Coefficients")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.ylabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig("feature_coefficients_plot.pdf")
    plt.show()
