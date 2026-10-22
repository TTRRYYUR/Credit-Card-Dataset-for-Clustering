import joblib
from functools import lru_cache
import pandas as pd
from pathlib import Path

from features import get_raw_feature_names


@lru_cache(maxsize=1)
def load_model(model_path = None):
    """Загружает сохраненный пайплайн"""
    pipeline = joblib.load(model_path or Path(__file__).parent / "model.pkl")
    if not hasattr(pipeline, "predict"):
        raise ValueError("Модель не поддерживает predict. Переобучите её через main.py.")
    return pipeline


def predict_cluster(features, model_path = None):
    """Предсказывает кластер для новых клиентов"""
    pipeline = load_model(model_path)

    num_cols, cat_cols = get_raw_feature_names()
    order_cols = num_cols + cat_cols

    df = pd.DataFrame([features])[order_cols]
    cluster_labels = pipeline.predict(df)[0]
    return int(cluster_labels)
