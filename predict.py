import joblib
from functools import lru_cache
import pandas as pd

from features import get_raw_feature_names


@lru_cache(maxsize=1)
def load_model(model_path = "model.pkl"):
    """Загружает сохраненный пайплайн"""
    return joblib.load(model_path)


def predict_cluster(features, model_path = "model.pkl"):
    """Предсказывает кластер для новых клиентов"""
    pipeline = load_model(model_path)

    num_cols, cat_cols = get_raw_feature_names()
    order_cols = num_cols + cat_cols

    df = pd.DataFrame([features])[order_cols]
    cluster_labels = pipeline.predict(df)[0]
    return int(cluster_labels)
