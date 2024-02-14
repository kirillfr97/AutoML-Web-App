import os
import pandas as pd
from ydata_profiling import ProfileReport
from pycaret.classification import setup, compare_models, pull


def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame or None: Loaded DataFrame if successful, else None.
    """
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None


def save_data(df, file_path):
    """
    Save DataFrame to a CSV file.

    Parameters:
    - df (pd.DataFrame): DataFrame to be saved.
    - file_path (str): Path to save the CSV file.
    """
    df.to_csv(file_path, index=False)


def run_data_analysis(df):
    """
    Generate a profiling report for the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame to be analyzed.

    Returns:
    - ProfileReport: Profiling report generated using ydata_profiling.
    """
    return ProfileReport(df, title='Profiling Report')


def train_model(df, target_column):
    """
    Train a machine learning model using PyCaret.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the dataset.
    - target_column (str): Name of the target column for prediction.

    Returns:
    - setup_df (pd.DataFrame): DataFrame containing the ML setup details.
    - compare_df (pd.DataFrame): DataFrame containing model comparison results.
    - best_model: Best trained model.
    """
    setup(df, target=target_column)
    setup_df = pull()

    best_model = compare_models()
    compare_df = pull()

    return setup_df, compare_df, best_model
