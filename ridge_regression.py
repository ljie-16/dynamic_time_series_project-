from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_ridge_regression(X_train, y_train):
    """
    Trains a Ridge Regression model.
    """
    regressor = Ridge(alpha=1)
    regressor.fit(X_train, y_train)
    return regressor

def evaluate_ridge_regression(regressor, X_test, y_test):
    """
    Evaluates the Ridge Regression model.
    """
    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared: {r2}")
    return mae, mse, r2
