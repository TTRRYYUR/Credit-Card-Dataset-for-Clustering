import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA


def plot_clusters_2d(model_name, X_scaled, labels):
    """Визуализация кластеров через PCA"""
    pca_data = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=[str(l) for l in labels], palette="tab10")

    plt.title(f"Кластеры {model_name} в 2D (PCA)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_cluster_profiles(model_name, df, labels):
    """Строит график абсолютных медианных значений по кластерам"""
    df_copy = df.copy()
    df_copy['cluster'] = pd.Series(labels).astype(int)

    cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']

    profile = df_copy.groupby('cluster', as_index=False)[cols].median().fillna(0)

    print(f"\n--- АБСОЛЮТНЫЙ ПРОФИЛЬ КЛАСТЕРОВ: {model_name} ---")
    print(profile.round(2))

    plot_df = profile.melt(id_vars='cluster', value_vars=cols,
                           var_name='Признак', value_name='Медиана')

    plt.figure(figsize=(14, 6))
    sns.barplot(data=plot_df, x='cluster', y='Медиана', hue='Признак')

    plt.title(f"Абсолютный профиль кластеров: {model_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Кластер", fontsize=12)
    plt.ylabel("Медианное значение", fontsize=12)
    plt.legend(title="Признак", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cluster_counts(model_name, labels):
    """Распределение клиентов по кластерам"""
    plt.figure(figsize=(6, 4))

    counts = pd.Series(labels).value_counts().sort_index()

    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="tab10")

    plt.title(f"Количество клиентов в кластерах: {model_name}")
    plt.xlabel("Кластер")
    plt.ylabel("Клиентов")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
