"""
plotting.py - Visualization utilities for MGS

Functions:
    plot_gene_distribution   : histogram + GMM overlay for individual genes
    plot_score_distribution  : score waterfall / ranked plot
    plot_umap_comparison     : side-by-side UMAP using HVGs vs MGS genes
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse import issparse
from sklearn.mixture import GaussianMixture


def plot_gene_distribution(
    adata,
    genes: list[str],
    layer: str | None = None,
    ncols: int = 3,
    figsize_per_panel: tuple = (4, 3),
    show: bool = True,
):
    """
    Plot expression histogram + GMM fit for each gene.

    Parameters
    ----------
    adata : AnnData
    genes : list of gene names to plot
    layer : which layer to use (None = adata.X)
    ncols : number of columns in the panel grid
    figsize_per_panel : (width, height) per panel
    show : if True, call plt.show()

    Returns
    -------
    fig, axes
    """
    n = len(genes)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
    )
    axes = np.array(axes).flatten()

    X = adata.layers[layer] if layer else adata.X
    gene_idx = {g: i for i, g in enumerate(adata.var_names)}

    for ax, gene in zip(axes, genes):
        if gene not in gene_idx:
            ax.set_visible(False)
            continue

        col = X[:, gene_idx[gene]]
        if issparse(col):
            col = col.toarray().flatten()
        else:
            col = np.asarray(col).flatten()

        nonzero = col[col > 0]
        if len(nonzero) < 10:
            ax.set_title(f"{gene}\n(insufficient data)")
            continue

        # Histogram
        ax.hist(nonzero, bins=40, density=True, alpha=0.55,
                color="#4C72B0", edgecolor="none", label="data")

        # Fit GMM k=2
        try:
            X_fit = nonzero.reshape(-1, 1)
            gmm1 = GaussianMixture(n_components=1, random_state=42).fit(X_fit)
            gmm2 = GaussianMixture(n_components=2, random_state=42).fit(X_fit)

            x_range = np.linspace(nonzero.min(), nonzero.max(), 300).reshape(-1, 1)

            # k=1
            pdf1 = np.exp(gmm1.score_samples(x_range))
            ax.plot(x_range, pdf1, color="#999999", lw=1.5,
                    linestyle="--", label="GMM k=1")

            # k=2
            pdf2 = np.exp(gmm2.score_samples(x_range))
            ax.plot(x_range, pdf2, color="#DD4444", lw=2,
                    label="GMM k=2")

            # Component means
            for mean in gmm2.means_.flatten():
                ax.axvline(mean, color="#DD4444", lw=1, linestyle=":",
                           alpha=0.8)

            delta_bic = gmm1.bic(X_fit) - gmm2.bic(X_fit)
            title_suffix = f"ΔBIC={delta_bic:.1f}"
        except Exception:
            title_suffix = "GMM failed"

        frac = len(nonzero) / len(col)
        ax.set_title(f"{gene}\n(expr={frac:.1%}, {title_suffix})", fontsize=9)
        ax.set_xlabel("log-normalized expression")
        ax.set_ylabel("density")
        ax.legend(fontsize=7)

    # Hide unused panels
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("MGS — Gene Expression Distributions", fontsize=13, y=1.02)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_score_distribution(selector, top_n: int = 50, show: bool = True):
    """
    Plot the ranked multimodal score for all passed genes.

    Parameters
    ----------
    selector : fitted MGS instance
    top_n : how many top genes to label
    show : call plt.show()

    Returns
    -------
    fig, ax
    """
    if selector.results_ is None:
        raise RuntimeError("Call .fit() first.")

    passed = selector.results_[selector.results_["filter_reason"] == "passed"].copy()
    passed = passed.sort_values("multimodal_score", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    colors = ["#DD4444" if g in selector.selected_genes_ else "#4C72B0"
              for g in passed["gene"]]

    ax.bar(range(len(passed)), passed["multimodal_score"], color=colors,
           width=1.0, edgecolor="none")

    # Label top genes
    for i, row in passed.head(top_n).iterrows():
        ax.text(i, row["multimodal_score"] + 0.5, row["gene"],
                rotation=90, fontsize=6, ha="center", va="bottom")

    # Threshold line (min_bic_delta reference)
    ax.axhline(0, color="black", lw=0.5)

    ax.set_xlabel("Genes (ranked by multimodal score)")
    ax.set_ylabel("Multimodal Score")
    ax.set_title("MGS — Ranked Gene Scores\n(red = selected, blue = not selected)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#DD4444", label=f"Selected ({len(selector.selected_genes_)})"),
        Patch(facecolor="#4C72B0", label="Not selected"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_umap_comparison(adata, hvg_key: str = "highly_variable", show: bool = True):
    """
    Side-by-side UMAP: HVGs vs MGS-selected genes.

    Requires:
        - adata.obsm['X_umap_hvg'] : UMAP computed from HVGs
        - adata.obsm['X_umap_mgs'] : UMAP computed from MGS genes
        - adata.obs['leiden'] or similar cluster label

    Parameters
    ----------
    adata : AnnData with both UMAPs computed
    hvg_key : obs key for cluster labels
    show : call plt.show()

    Returns
    -------
    fig, axes
    """
    import scanpy as sc

    if "X_umap_hvg" not in adata.obsm or "X_umap_mgs" not in adata.obsm:
        raise ValueError(
            "Run compute_both_umaps(adata, selector) first to populate "
            "adata.obsm['X_umap_hvg'] and adata.obsm['X_umap_mgs']."
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cluster_key = None
    for key in ["leiden", "louvain", "celltype", "cell_type"]:
        if key in adata.obs:
            cluster_key = key
            break

    for ax, (umap_key, title) in zip(
        axes,
        [("X_umap_hvg", "HVG-based UMAP"), ("X_umap_mgs", "MGS-based UMAP")]
    ):
        coords = adata.obsm[umap_key]
        if cluster_key:
            clusters = adata.obs[cluster_key].astype("category")
            palette = plt.cm.tab20.colors
            for j, cat in enumerate(clusters.cat.categories):
                mask = clusters == cat
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    s=3, alpha=0.6,
                    color=palette[j % len(palette)],
                    label=str(cat)
                )
            ax.legend(markerscale=3, fontsize=7, bbox_to_anchor=(1, 1))
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.4, color="#4C72B0")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("UMAP Comparison: HVG vs MGS Gene Selection", fontsize=13)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def compute_both_umaps(adata, selector, n_neighbors: int = 15, n_pcs: int = 30):
    """
    Helper: compute UMAPs for both HVG and MGS gene sets and store in adata.

    Parameters
    ----------
    adata : AnnData (log-normalized)
    selector : fitted MGS instance
    n_neighbors : for sc.pp.neighbors
    n_pcs : PCs to use

    Returns
    -------
    adata with obsm['X_umap_hvg'] and obsm['X_umap_mgs'] populated
    """
    import scanpy as sc

    print("Computing HVG-based UMAP...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg)
    sc.tl.pca(adata_hvg, n_comps=n_pcs)
    sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors)
    sc.tl.umap(adata_hvg)
    adata.obsm["X_umap_hvg"] = adata_hvg.obsm["X_umap"]

    print("Computing MGS-based UMAP...")
    mgs_genes = selector.selected_genes_
    adata_mgs = adata[:, mgs_genes].copy()
    sc.pp.scale(adata_mgs)
    sc.tl.pca(adata_mgs, n_comps=min(n_pcs, len(mgs_genes) - 1))
    sc.pp.neighbors(adata_mgs, n_neighbors=n_neighbors)
    sc.tl.umap(adata_mgs)
    adata.obsm["X_umap_mgs"] = adata_mgs.obsm["X_umap"]

    return adata
