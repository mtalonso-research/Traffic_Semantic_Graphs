import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
from tqdm import tqdm
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualize_graph_embeddings(
    model_l2d,
    model_nup,
    proj_head,
    l2d_loader,
    nup_loader,
    quantizer,
    metadata,
    node_embed_dim,
    batched_graph_embeddings,
    device="cuda",
    pca_components=2,
    title="Graph Embedding Space (PCA Projection)"
):
    def has_edges(batch):
        return any('edge_index' in store and store['edge_index'].numel() > 0
                   for store in batch.edge_stores)

    # Set models to eval
    model_l2d.eval()
    model_nup.eval()
    proj_head.eval()

    all_embeddings, all_sources, all_paths = [], [], []

    with torch.no_grad():
        # --- Collect L2D embeddings ---
        for batch in tqdm(l2d_loader, desc="Collecting L2D embeddings"):
            if not has_edges(batch):
                continue
            #batch = batch.to(device)
            batch = quantizer.transform_inplace(batch)

            z_dict, feat_logits, edge_logits = model_l2d(batch)
            z = batched_graph_embeddings(z_dict, batch, metadata, embed_dim_per_type=node_embed_dim)
            z_proj = F.normalize(proj_head(z), dim=-1)

            all_embeddings.append(z_proj.cpu())
            all_sources.extend(['l2d'] * z_proj.size(0))
            all_paths.extend(batch['window_meta']['episode_path'])

        # --- Collect NUP embeddings ---
        for batch in tqdm(nup_loader, desc="Collecting NUP embeddings"):
            if not has_edges(batch):
                continue
            #batch = batch.to(device)
            batch = quantizer.transform_inplace(batch)

            z_dict, feat_logits, edge_logits = model_nup(batch)
            z = batched_graph_embeddings(z_dict, batch, metadata, embed_dim_per_type=node_embed_dim)
            z_proj = F.normalize(proj_head(z), dim=-1)

            all_embeddings.append(z_proj.cpu())
            all_sources.extend(['nup'] * z_proj.size(0))
            all_paths.extend(batch['window_meta']['episode_path'])

    # --- Concatenate embeddings ---
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # --- PCA dimensionality reduction ---
    pca = PCA(n_components=pca_components, random_state=42)
    emb_2d = pca.fit_transform(all_embeddings.numpy())

    # --- Prepare DataFrame for Plotly ---
    df = pd.DataFrame({
        'PCA1': emb_2d[:, 0],
        'PCA2': emb_2d[:, 1],
        'Source': all_sources,
        'EpisodePath': all_paths
    })

    # --- Interactive scatter plot ---
    fig = px.scatter(
        df,
        x='PCA1',
        y='PCA2',
        color='Source',
        hover_data=['EpisodePath'],
        opacity=0.6,
        title=title,
        color_discrete_map={'l2d': 'royalblue', 'nup': 'darkorange'}
    )
    fig.update_traces(marker=dict(line=dict(width=0.2, color='black')))
    fig.update_layout(
        xaxis_title="PCA-1",
        yaxis_title="PCA-2",
        legend_title="Source",
        template='simple_white'
    )

    fig.show()
    return fig, all_embeddings, all_sources

def find_optimal_kmeans_k(
    embeddings,
    k_min=2,
    k_max=20,
    pca_components=None,
    random_state=42,
    plot_title="Elbow Plot for KMeans Clustering"
):
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # Optional PCA dimensionality reduction
    if pca_components is not None and pca_components < embeddings.shape[1]:
        pca = PCA(n_components=pca_components, random_state=random_state)
        embeddings = pca.fit_transform(embeddings)

    k_values = range(k_min, k_max + 1)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    # Find elbow using KneeLocator
    kl = KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
    best_k = kl.knee
    print(f"Suggested number of clusters (elbow method): {best_k}")

    return best_k, inertias

def cluster_and_visualize_embeddings(
    embeddings,
    sources,
    n_clusters=7,
    random_state=42,
    use_pca_for_clustering=True,
    pca_cluster_dim=10,
    title_prefix="KMeans Clusters"
):
    # Convert embeddings to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # --- Optional PCA for clustering ---
    if use_pca_for_clustering and pca_cluster_dim < embeddings.shape[1]:
        pca_10d = PCA(n_components=pca_cluster_dim, random_state=random_state)
        embeddings_10d = pca_10d.fit_transform(embeddings)
    else:
        embeddings_10d = embeddings

    # --- Run KMeans ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings_10d)

    # --- PCA 2D projection for visualization ---
    pca_2d = PCA(n_components=2, random_state=random_state)
    proj_2d = pca_2d.fit_transform(embeddings)

    # --- Scatter plot (2D PCA projection) ---
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_clusters)
    colors = [palette[label % len(palette)] for label in cluster_labels]
    plt.scatter(proj_2d[:, 0], proj_2d[:, 1], c=colors, s=10, alpha=0.8, linewidth=0)
    plt.title(f"{title_prefix} (on PCA-{pca_cluster_dim}D) in PCA-2D Space")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.grid(True)

    handles = [
        plt.Line2D([], [], marker='o', color=palette[i % len(palette)],
                   linestyle='', label=f'Cluster {i}')
        for i in range(n_clusters)
    ]
    plt.legend(handles=handles, title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # --- Source counts per cluster ---
    df = pd.DataFrame({"cluster": cluster_labels, "source": sources})
    counts = df.groupby(["cluster", "source"]).size().reset_index(name="count")
    counts["cluster"] = counts["cluster"].astype(str)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=counts, x="cluster", y="count", hue="source", dodge=True)
    plt.title(f"Source Counts per Cluster ({title_prefix})")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.legend(title="Source")
    plt.tight_layout()
    plt.show()

    return cluster_labels, counts
