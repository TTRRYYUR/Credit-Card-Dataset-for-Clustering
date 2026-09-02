import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, RobustScaler

from features import get_all_feature_names, get_feature_engineering_transformer, get_raw_feature_names


NON_LOG_COLS = [
    "BALANCE_FREQUENCY",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "PRC_FULL_PAYMENT"
]


def create_preprocessor():
    """Создание препроцессора с тремя ветками для разных признаков"""
    num_cols, cat_cols = get_all_feature_names()
    log_cols = [col for col in num_cols if col not in NON_LOG_COLS]
    non_log_cols = [col for col in num_cols if col in NON_LOG_COLS]
    return ColumnTransformer([
        ("num_log",Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("log_transform", FunctionTransformer(np.log1p)),
                ("scaler", RobustScaler()),
                ("pca", PCA(n_components=3)),
            ]), log_cols),

        ("num_raw", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]), non_log_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols)
    ], remainder="drop")



def create_model_pipeline(model):
    """Собирает полный пайплайн с генерацией признаков, предобработкой и оценкой модели"""
    feature_engineering_transformer = get_feature_engineering_transformer()
    preprocessor = create_preprocessor()
    return Pipeline(
         [
            ("feature_engineering", feature_engineering_transformer),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
     )
