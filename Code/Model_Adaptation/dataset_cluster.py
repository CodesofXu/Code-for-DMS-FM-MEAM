
import os
import ast
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import umap
import hdbscan


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_output_directory(config):

    output_root = Path(config['data']['output_root'])
    algorithm = config['clustering']['algorithm']

    algo_dir = output_root / f"{algorithm}_results"
    algo_dir.mkdir(parents=True, exist_ok=True)

    existing_versions = [d.name for d in algo_dir.glob("version_*")]
    version_numbers = [
        int(v.split("_")[1])
        for v in existing_versions
        if v.startswith("version_") and v.split("_")[1].isdigit()
    ]
    next_version = max(version_numbers) + 1 if version_numbers else 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = algo_dir / f"version_{next_version}_{timestamp}"
    version_dir.mkdir()

    with open(version_dir / "config_backup.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

    return version_dir


def load_and_process_data(config):

    csv_files = glob.glob(os.path.join(
        config['data']['csv_dir'],
        config['data']['file_pattern']
    ))

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df['source_file'] = Path(f).name
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    feature_col = config['data']['feature_column']
    full_df[feature_col] = full_df[feature_col].apply(
        lambda x: np.array(ast.literal_eval(x))
    )

    features = np.stack(full_df[feature_col].values)
    return full_df, features


def preprocess_features(features, config):

    if config['preprocessing'].get('standard_scaler', False):
        features = StandardScaler().fit_transform(features)

    pca = None
    if config['preprocessing']['pca'].get('enabled', False):
        pca = PCA(n_components=config['preprocessing']['pca']['n_components'])
        features = pca.fit_transform(features)
        print(f"Cumulative explained variance ratio (PCA): {pca.explained_variance_ratio_.sum():.2f}")

    umap_reducer = None
    if config['preprocessing']['umap'].get('enabled', False):
        umap_reducer = umap.UMAP(
            n_components=config['preprocessing']['umap']['n_components'],
            n_neighbors=config['preprocessing']['umap']['n_neighbors'],
            min_dist=config['preprocessing']['umap']['min_dist'],
            random_state=42
        )
        features = umap_reducer.fit_transform(features)

    return features, pca, umap_reducer


def perform_clustering(features, config):

    algo = config['clustering']['algorithm']

    if algo == 'kmeans':
        params = config['clustering']['kmeans']
        model = KMeans(
            n_clusters=params['n_clusters'],
            max_iter=params['max_iter'],
            n_init=params['n_init'],
            random_state=42
        )
    elif algo == 'hdbscan':
        params = config['clustering']['hdbscan']
        model = hdbscan.HDBSCAN(
            min_cluster_size=params['min_cluster_size'],
            min_samples=params['min_samples'],
            cluster_selection_epsilon=params['cluster_selection_epsilon']
        )
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algo}")

    labels = model.fit_predict(features)
    return labels, model


def evaluate_clustering(labels, features, config):

    metrics = {}
    unique_labels = np.unique(labels)

    if len(unique_labels) > 1:
        if config['evaluation'].get('silhouette', False):
            metrics['silhouette'] = silhouette_score(features, labels)
        if config['evaluation'].get('calinski_harabasz', False):
            metrics['calinski_harabasz'] = calinski_harabasz_score(features, labels)

    return metrics


def optimize_kmeans(features, config):

    init_k = config['clustering']['kmeans']['n_clusters']
    max_k = config['clustering'].get('max_clusters', init_k)
    threshold = config['evaluation'].get('silhouette_threshold', None)

    best_labels = None
    best_model = None
    best_metrics = {}

    for k in range(init_k, max_k + 1):
        config['clustering']['kmeans']['n_clusters'] = k
        labels, model = perform_clustering(features, config)
        metrics = evaluate_clustering(labels, features, config)
        sil = metrics.get('silhouette', None)
        print(f"[K={k}] silhouette = {sil}")

        if threshold is not None and sil is not None and sil >= threshold:
            print(f"Silhouette ≥ {threshold:.2f} achieved at k = {k}")
            return labels, model, metrics

        best_labels, best_model, best_metrics = labels, model, metrics

    print(f"No k in [{init_k}..{max_k}] reached silhouette ≥ {threshold}, using k = {max_k}")
    return best_labels, best_model, best_metrics


def save_results(df, labels, output_dir):

    result_df = df[['source_file', 'img_path']].copy()
    result_df['cluster'] = labels
    out_path = output_dir / 'cluster_results.csv'
    result_df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


def visualize_clusters(features, labels, output_dir, config):

    if config['preprocessing']['umap'].get('enabled', False):
        emb = features
    else:
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(features)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        emb[:, 0], emb[:, 1],
        c=labels, cmap='Spectral',
        s=8, alpha=0.6
    )

    coords = np.column_stack((emb[:, 0], emb[:, 1], labels))
    header = "x_coordinate,y_coordinate,label"
    np.savetxt(output_dir / "cluster_coordinates.csv",
               coords,
               delimiter=",",
               header=header,
               comments="",
               fmt="%f")

    plt.colorbar(scatter, label='Cluster ID')
    plt.title("UMAP Cluster Projection", fontsize=14)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)

    plot_path = output_dir / f"cluster_visualization.{config['visualization']['plot_format']}"
    plt.savefig(plot_path,
                dpi=config['visualization']['plot_dpi'],
                bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {plot_path}")


def generate_summary(labels, df, output_dir):

    summary = pd.DataFrame({
        'cluster': labels,
        'source_file': df['source_file']
    }).groupby('cluster').agg(
        count=('cluster', 'size'),
        source_files=('source_file', lambda x: list(x.unique()))
    ).reset_index()

    summary_path = output_dir / 'cluster_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"Cluster summary saved to {summary_path}")


def main(config_path):

    config = load_config(config_path)
    output_dir = get_output_directory(config)

    full_df, features = load_and_process_data(config)
    processed_features, _, _ = preprocess_features(features, config)

    if (config['clustering']['algorithm'] == 'kmeans'
            and 'silhouette_threshold' in config['evaluation']):
        labels, model, metrics = optimize_kmeans(processed_features, config)
    else:
        labels, model = perform_clustering(processed_features, config)
        metrics = evaluate_clustering(labels, processed_features, config)

    save_results(full_df, labels, output_dir)

    print("\nClustering evaluation results:")
    for name, score in metrics.items():
        print(f"{name:>20}: {score:.4f}")

    if config['visualization'].get('umap_visualization', False):
        visualize_clusters(processed_features, labels, output_dir, config)

    if config['visualization'].get('cluster_summary', False):
        generate_summary(labels, full_df, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run clustering")
    parser.add_argument(
        '--config',
        default='config_cluster.yaml',
        help='Path to clustering configuration YAML'
    )
    args = parser.parse_args()
    main(args.config)
