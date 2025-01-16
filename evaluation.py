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

def plot_roc_curve(model, X_test, y_test, nb_classes):
    """
    Plots the ROC curve for multi-class classification.
    """
    # Get model predictions
    Y_pred = model.predict(X_test)

    # Binarize the labels and predictions for multi-class ROC curve
    Y_pred_classes = [np.argmax(y) for y in Y_pred]
    Y_test_classes = [np.argmax(y) for y in y_test]

    Y_pred_binarized = label_binarize(Y_pred_classes, classes=range(nb_classes))
    Y_test_binarized = label_binarize(Y_test_classes, classes=range(nb_classes))

    # Compute ROC curve and AUC for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test_binarized[:, i], Y_pred_binarized[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test_binarized.ravel(), Y_pred_binarized.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= nb_classes

    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red"])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot(fpr["micro"], tpr["micro"], linestyle=":", color="deeppink", lw=4,
             label=f"Micro-average (AUC = {roc_auc['micro']:.2f})")
    plt.plot(fpr["macro"], tpr["macro"], linestyle=":", color="navy", lw=4,
             label=f"Macro-average (AUC = {roc_auc['macro']:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("New_ROC_unseen.png", dpi=300)
    plt.show()
