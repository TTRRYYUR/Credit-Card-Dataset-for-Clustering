from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from data import clean_data, load_data
from pipelines import create_model_pipeline


def plot_elbow(X, k_range=range(2, 11)):
    """Inertia for the same feature space used by main.py."""
    ks = list(k_range)
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
                for k in ks]
    plt.figure(figsize=(10, 6))
    plt.plot(ks, inertias, "bo-")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()


def main():
    df = clean_data(load_data(Path(__file__).parent / "CC GENERAL.csv"))
    preprocessing = create_model_pipeline(KMeans())[:-1]
    plot_elbow(preprocessing.fit_transform(df))


if __name__ == "__main__":
    main()
