import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from typing import Tuple

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessingStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1
            )

            if data is None:
                raise ValueError("Data is None...before fillna")
            # else:
            #     raise ValueError("Everything good...before fillna")

            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data

        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise

class DataSplitStrategy(DataStrategy):
    """
    Stretagy for splitting data into train and test sets
    """
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides data into train and test sets
        """

        review_score = "review_score"
        try:

            X = data.drop(review_score, axis=1)
            y = data[review_score]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test


        except Exception as e:  
            logging.error(f"Error in data splitting: {e}")
            raise

class DataCleaning:
    """
    class for cleaning data which processes the data and divides it into train and test sets
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        # self.data = data
        # self.strategy = strategy

        # if data is None or strategy is None:
        #     raise ValueError("Both 'data' and 'strategy' must be provided.")
        self.data = data
        self.strategy = strategy


    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e: 
            logging.error(f"Error in data handling: {e}")
            raise

