"""
Example 01: SACI Analysis on Leishmania PBMC dataset

This script demonstrates the full SACI pipeline:
1. Data preprocessing
2. SACI Hierarchical Clustering with CoB rescue
3. Extracting the optimal subset of genes (Levels 0-2)
4. Running UMAP with the optimal subset
"""

import sys
import scanpy as sc
import matplotlib.pyplot as plt

# If testing locally before pip install
sys.path.append('../')
from saci import SACI

def main():
    print("Loading h5ad file...")
    # Replace with the actual path to your dataset
    adata = sc.read_h5ad('../f7dac462-d4a0-41d1-9e2c-fd513fdcd648.h5ad')

    # Prepare data
    if 'feature_name' in adata.var.columns:
        adata.var['feature_name'] = adata.var['feature_name'].astype(str)
        adata.var.index = adata.var['feature_name']
    adata.var_names_make_unique()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Data prepared!")

    # Run SACI
    print("\nRunning SACI Hierarchical Clustering...")
    rec = SACI(
        min_genes_to_split=5,     # Min 5 bimodal genes to justify split
        min_cells_to_split=30,    # Don't split clusters < 30 cells
        max_depth=4,              # Max depth 4
        split_resolution=0.5,
        cob=True,
        cob_n_target_genes=100,
    )

    adata, all_genes, labels = rec.fit_transform(adata)

    print(f"\nTotal genes found (all levels): {len(all_genes)}")
    print(f"Hierarchical clusters (leaves): {len(set(labels))}")

    # Optimal Subset: Levels 0-2
    top_genes_02 = list(adata.var_names[
        (adata.var['rmgs_selected']) & (adata.var['rmgs_level'] <= 2)
    ])
    print(f"Selected optimal genes (Levels 0-2): {len(top_genes_02)}")

    # Downstream UMAP
    adata_final = adata[:, top_genes_02].copy()
    sc.pp.scale(adata_final)
    sc.tl.pca(adata_final, n_comps=30)
    sc.pp.neighbors(adata_final)
    sc.tl.umap(adata_final)
    sc.tl.leiden(adata_final, resolution=0.2, key_added='leiden_final')

    # Plot
    sc.pl.umap(adata_final, color=['leiden_final'], show=False, title='SACI Optimal UMAP')
    plt.tight_layout()
    plt.savefig('saci_umap.png', dpi=300)
    print("UMAP saved to saci_umap.png")

if __name__ == "__main__":
    main()
