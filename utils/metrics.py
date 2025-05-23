from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(model, X_test, y_test):
    preds = model.predict([X_test[:, 0], X_test[:, 1]]).squeeze()
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    print(f"[Evaluation] RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae