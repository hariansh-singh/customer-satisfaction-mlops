import logging
import pandas as pd
import mlflow
from zenml import step
from sklearn.base import RegressorMixin

from typing import Tuple
from typing_extensions import Annotated

from src.evaluation import RMSE, R2, MSE

from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker 


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame,
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
]:  
    """
    Evaluates the model on the test data

    Args: 
        model: Trained model
        X_test: Testing data
        y_test: Testing labels
    """
    try:
        predictions = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, predictions)
        mlflow.log_metric("mse", mse)

        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, predictions)
        mlflow.log_metric("r2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, predictions)
        mlflow.log_metric("rmse", rmse)

        return r2, rmse
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise