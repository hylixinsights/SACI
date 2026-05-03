"""
Example 01: SACI Analysis on Leishmania PBMC dataset

This script demonstrates the full SACI pipeline on a modern, high-quality
single-cell dataset.

=== DATASET CONTEXT & PUBLICATION ===

The dataset used in this example comes from the following publication:
de Vrij et al. (2024) Commun Biol. "Persistent T cell unresponsiveness 
associated with chronic visceral leishmaniasis in HIV-coinfected patients."
Link: https://www.nature.com/articles/s42003-024-06225-2

Biological Background:
A large proportion of HIV-coinfected visceral leishmaniasis (VL) patients 
exhibit a chronic disease course with frequent recurrence of VL. The study 
analyzed circulatory immune cells (blood) of a longitudinal HIV cohort in 
Ethiopia, comparing asymptomatically Leishmania-infected individuals with 
active VL-HIV patients. The authors observed that VL chronicity was 
associated with persistent CD8+ T cell exhaustion and CD4+ T cell anergy, 
characterized by high expression of PD-1 and TIGIT.

Dataset Technical Specs:
- Technology: 10x Genomics 5' v1
- Tissue: Blood (PBMC)
- This dataset is ideal for SACI because the modern sequencing chemistry 
  and sufficient read depth allow biologically meaningful marker genes to 
  form clear, statistically robust bimodal distributions. Unlike sparse 
  legacy datasets (e.g. 10x 3' v2 like PBMC 3k), SACI runs perfectly here 
  with its default, strict parameters.

Pipeline steps:
1. Data preprocessing and dynamic dataset download
2. SACI Hierarchical Clustering with CoB rescue
3. Extracting the optimal subset of genes (Levels 0-2)
4. Running UMAP with the optimal subset
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
    h5ad_file = 'f7dac462-d4a0-41d1-9e2c-fd513fdcd648.h5ad'
    
    # Check if dataset exists in current or parent directory
    if os.path.exists(h5ad_file):
        file_path = h5ad_file
    elif os.path.exists(f'../{h5ad_file}'):
        file_path = f'../{h5ad_file}'
    else:
        print("===============================================================")
        print("Dataset not found!")
        print("Please download the Leishmania PBMC dataset from CZI CELLxGENE:")
        print(f"URL: https://datasets.cellxgene.cziscience.com/{h5ad_file}")
        print("Save it in the same directory as this script and run again.")
        print("===============================================================")
        sys.exit(1)

    print(f"Loading {file_path}...")
    adata = sc.read_h5ad(file_path)

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
