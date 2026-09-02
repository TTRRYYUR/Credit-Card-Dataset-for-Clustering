from typing import Any
from sklearn.base import clone
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.mixture import GaussianMixture


def get_model_config() -> dict[str, dict[str, Any]]:
    """Возвращает конфигурацию моделей и пространства их гиперпараметров"""
    return {
        "KMeans": {
            "model": KMeans(random_state=42, n_init=10),
            "params": {
                "model__n_clusters": [3, 4],
            },
        },
        "AgglomerativeClustering": {
            "model": AgglomerativeClustering(),
            "params": {
                "model__n_clusters": [3, 4],
                "model__linkage": ["ward", "complete", "average"],
            },
        },
        "DBSCAN": {
            "model": DBSCAN(n_jobs=-1),
            "params": {
                "model__min_samples": [5, 10, 15],
                "model__eps": [0.4, 0.6, 0.8, 1.0, 1.2],
            },
        },
        "GaussianMixture": {
            "model": GaussianMixture(random_state=42),
            "params": {
                "model__n_components": [3, 4, 5, 6],
                "model__covariance_type": ["full", "tied", "diag"],
            },
        },
    }


def train_model(pipeline, X, param_grid: dict[str, Any]):
    """Обучает модель и вычисление метрик"""
    best_score = -1.0
    best_params = None
    best_pipeline = None
    best_labels = None
    best_metrics = {}

    for params in ParameterGrid(param_grid):
        current_pipe = clone(pipeline)
        current_pipe.set_params(**params)

        X_transformed = current_pipe[:-1].fit_transform(X)
        model = current_pipe.named_steps["model"]
        labels = model.fit_predict(X_transformed)

        unique_labels = set(labels) - {-1}
        if len(unique_labels) < 2:
            continue

        noise_ratio = sum(labels == -1) / len(labels)
        if noise_ratio > 0.3:
            continue

        sil_score = silhouette_score(X_transformed, labels)

        if sil_score > best_score:
            best_score = sil_score
            best_params = params
            best_pipeline = current_pipe
            best_labels = labels
            best_metrics = {
                "silhouette": sil_score,
            }

    return {
        "pipeline": best_pipeline,
        "params": best_params,
        "best_score": best_score,
        "metrics": best_metrics,
        "labels": best_labels,
    }
