import numpy as np
from sklearn.preprocessing import FunctionTransformer


RAW_NUM_COLS =[
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]

RAW_CAT_COLS = []

ENGINEERED_NUM_COLS =["balance_update_ratio",
                      "oneoff_share",
                      "avg_purchase_cost",
                      "limit_usage_ratio",
                      "monthly_purchases"]

ENGINEERED_CAT_COLS = []


def get_raw_feature_names():
    """Возвращает все столбцы до создания новых"""
    return RAW_NUM_COLS, RAW_CAT_COLS


def get_all_feature_names():
    """Возвращает все столбцы"""
    return RAW_NUM_COLS + ENGINEERED_NUM_COLS, RAW_CAT_COLS + ENGINEERED_CAT_COLS


def add_features(df):
    """Feature Engineering"""
    df = df.copy()

    df["balance_update_ratio"] = (df["BALANCE"] / df["BALANCE_FREQUENCY"].replace(0, np.nan)).fillna(0)
    df["oneoff_share"] = (df["ONEOFF_PURCHASES"] / df["PURCHASES"].replace(0, np.nan)).fillna(0)
    df["avg_purchase_cost"] = (df["PURCHASES"] / df["PURCHASES_TRX"].replace(0, np.nan)).fillna(0)
    df["limit_usage_ratio"] = (df["BALANCE"] / df["CREDIT_LIMIT"].replace(0, np.nan)).fillna(0)
    df["monthly_purchases"] = (df["PURCHASES"] / df["TENURE"].replace(0, np.nan)).fillna(0)
    return df


def get_feature_engineering_transformer():
    """Возвращает трансформер для применения генерации признаков в пайплайне"""
    return FunctionTransformer(add_features)
