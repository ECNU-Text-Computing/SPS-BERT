from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def cal_all(y_true, y_pred):
    """
    Calculate and return the Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) metrics.

    Parameters:
    y_true (array-like): True values for the target variable.
    y_pred (array-like): Predicted values from the model.

    Returns:
    dict: Dictionary containing the MSE, MAE, and R2 scores.
    """
    mse = mean_squared_error(y_true, y_pred)  # Calculate Mean Squared Error
    mae = mean_absolute_error(y_true, y_pred)  # Calculate Mean Absolute Error
    r2 = r2_score(y_true, y_pred)  # Calculate R-squared Score

    return {'mse': mse, 'mae': mae, 'r2': r2}