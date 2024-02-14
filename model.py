import os
import pandas as pd
from ydata_profiling import ProfileReport
from pycaret.classification import setup, compare_models, pull


def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return None


def save_data(df, file_path):
    df.to_csv(file_path, index=False)


def run_data_analysis(df):
    return ProfileReport(df, title='Profiling Report')


def train_model(df, target_column):
    setup(df, target=target_column)
    setup_df = pull()

    best_model = compare_models()
    compere_df = pull()

    return setup_df, compere_df, best_model
