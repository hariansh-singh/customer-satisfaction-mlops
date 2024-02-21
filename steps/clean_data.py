import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataSplitStrategy, DataPreprocessingStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:  
    """
    Cleans the data and divides it into train and test sets

    Args:
        df: Raw data
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    """
    try:
        preprocess_strategy = DataPreprocessingStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        processed_data = data_cleaning.handle_data()
 
        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(processed_data, split_strategy)

        X_train, X_test, y_train, y_test = data_cleaning.handle_data()          

        logging.info("Data cleaning and splitting complete!")
        return X_train, X_test, y_train, y_test
    except Exception as e:  
        logging.error(f"Error in data cleaning: {e}")
        raise