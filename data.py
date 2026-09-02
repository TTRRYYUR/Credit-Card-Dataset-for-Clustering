import pandas as pd


def load_data(path):
    """Загрузка датасета из CSV файла"""
    data = pd.read_csv(path)
    return data


def clean_data(df):
    """Очистка данных"""
    df = df.copy()
    df = df.drop("CUST_ID", axis=1)
    return df
