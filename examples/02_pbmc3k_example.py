"""
Example 02: SACI Analysis on the standard PBMC 3k dataset

This script demonstrates SACI on the widely used 10x Genomics PBMC 3k dataset,
which is automatically downloaded via Scanpy.

It performs:
1. Basic QC filtering (min_genes, min_cells)
2. Normalization and log1p transformation
3. SACI Hierarchical Clustering with CoB rescue
4. UMAP generation using the hierarchical bimodal markers
"""

import os
import sys
import scanpy as sc
import matplotlib.pyplot as plt

# Dynamically add the parent directory (SACI root) to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from saci import SACI

def main():
    print("Loading PBMC 3k dataset from scanpy...")
    # This automatically downloads the 10x dataset if not present
    adata = sc.datasets.pbmc3k()
    
    # Basic QC filtering
    print("Filtering cells and genes...")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Data prepared!")
    
    # Run SACI
    print("\nRunning SACI Hierarchical Clustering...")
    rec = SACI(
        min_genes_to_split=3,     # Lowered for this smaller dataset
        min_cells_to_split=20,
        max_depth=3,              # Max depth of 3 is usually enough for PBMC3k
        split_resolution=0.5,
        cob=True,
        cob_n_target_genes=50,    # Target 50 rescued genes per branch
    )
    
    adata, all_genes, labels = rec.fit_transform(adata)
    
    print(f"\nTotal genes found (all levels): {len(all_genes)}")
    print(f"Hierarchical clusters (leaves): {len(set(labels))}")
    
    # For a smaller dataset like PBMC 3k, using all discovered genes is fine
    print(f"Generating UMAP using {len(all_genes)} SACI markers...")
    adata_final = adata[:, all_genes].copy()
    
    sc.pp.scale(adata_final)
    sc.tl.pca(adata_final, n_comps=min(30, len(all_genes)-1))
    sc.pp.neighbors(adata_final)
    sc.tl.umap(adata_final)
    
    # Standard leiden for comparison
    sc.tl.leiden(adata_final, resolution=0.4, key_added='leiden_final')
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sc.pl.umap(adata_final, color='leiden_final', ax=axes[0], show=False, 
               title='Standard Leiden (res=0.4)')
               
    sc.pl.umap(adata_final, color='rmgs_cluster', ax=axes[1], show=False, 
               title='SACI Hierarchical Clusters')
               
    plt.tight_layout()
    plt.savefig('pbmc3k_saci_umap.png', dpi=300)
    print("UMAP saved to pbmc3k_saci_umap.png")
    
    # Print canonical markers found
    markers_to_check = ['CD3D', 'CD4', 'CD8A', 'MS4A1', 'CD14', 'FCGR3A', 'NKG7', 'PPBP']
    print("\nCanonical PBMC markers discovered:")
    for gene in markers_to_check:
        if gene in adata.var_names and adata.var.loc[gene, 'rmgs_selected']:
            level = adata.var.loc[gene, 'rmgs_level']
            print(f"  ✅ {gene}: Found at Level {int(level)}")
        else:
            print(f"  ❌ {gene}: Not selected")

if __name__ == "__main__":
    main()
