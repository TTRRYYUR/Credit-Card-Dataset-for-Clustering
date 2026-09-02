import joblib


from data import clean_data, load_data
from pipelines import create_model_pipeline
from training import get_model_config, train_model
from visualization import plot_cluster_counts, plot_clusters_2d, show_cluster_profiles


def main():
    raw_df = load_data("CC GENERAL.csv")
    df = clean_data(raw_df)

    model_config = get_model_config()

    best_score = -1.0
    best_result = None
    best_model_name = ""

    for model_name, config in model_config.items():
        print(f"Обучение {model_name}")

        pipe = create_model_pipeline(config["model"])

        result = train_model(pipe, df, config["params"])

        score = result["best_score"]
        metrics = result["metrics"]
        current_labels = result["labels"]
        current_pipe = result["pipeline"]

        print(f"Silhouette Score: {metrics['silhouette']:.4f}")

        X_scaled = current_pipe[:-1].transform(df)

        plot_clusters_2d(model_name, X_scaled, current_labels)
        plot_cluster_counts(model_name, current_labels)
        show_cluster_profiles(model_name, df, current_labels)

        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_result = result

    if best_result:
        print(f"\nЛучшая модель: {best_model_name}")
        print(f"Итоговый Silhouette Score: {best_score:.4f}")

        best_pipeline = best_result["pipeline"]
        joblib.dump(best_pipeline, "model.pkl")

if __name__ == "__main__":
    main()