"""
Example 02: SACI Analysis on the standard PBMC 3k dataset

This script demonstrates SACI on the widely used 10x Genomics PBMC 3k dataset,
which is automatically downloaded via Scanpy.

=== IMPORTANT NOTE ON DATA QUALITY AND SACI PERFORMANCE ===

SACI's bimodality engine (SaciBimodal) works best on datasets with sufficient
sequencing depth, where marker genes form clear bimodal distributions (one peak
at zero / low expression, and a second peak at higher expression in the
positive population). This is the regime where SACI truly shines — it
discovers hierarchical, level-specific markers that global HVG methods miss.

The PBMC 3k dataset was generated with an older 10x Genomics 3' v2 chemistry
and has very high dropout rates. As a consequence:

  1. Many canonical markers (e.g. CD4) are expressed at very low levels in
     only a small fraction of cells, making their distributions look like a
     single peak with a faint tail rather than a clear bimodal split. The
     Dip Test correctly identifies these as "not bimodal" at default
     thresholds.

  2. Rare populations like platelets (~14 cells out of 2700, i.e. ~0.5%)
     fall below the default min_cell_frac filter, designed to avoid noise.

  3. Co-expression rescue (CoB) is also less effective in very sparse data,
     because Jaccard similarity between binary expression vectors becomes
     unreliable when most values are zero due to technical dropout.

For these reasons, this example uses RELAXED parameters compared to the
defaults. In contrast, modern datasets (10x v3+, Smart-seq2, or larger
CELLxGENE collections like the Leishmania example) work well with default
SACI settings and produce excellent hierarchical marker discovery.

Pipeline steps:
  1. Basic QC filtering (min_genes, min_cells)
  2. Normalization and log1p transformation
  3. SACI Hierarchical Clustering with CoB rescue (relaxed parameters)
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

    # -------------------------------------------------------------------------
    # Run SACI with RELAXED parameters for this older, sparser dataset.
    #
    # Why these values differ from defaults:
    #   - dip_pval_threshold=0.5 : The default (0.2) is too strict for sparse
    #     data where many real markers barely pass the Dip Test. Raising to 0.5
    #     lets more candidates through to the GMM stage for proper evaluation.
    #   - min_cell_frac=0.001 : Allows detection of populations as small as
    #     ~3 cells (e.g. PPBP+ platelets). Default (0.05) would filter them.
    #   - min_genes_fallback=30 : In sparse data, very few genes pass the
    #     strict BIC threshold at each tree node. Lowering the fallback avoids
    #     injecting hundreds of noise genes per branch.
    #   - cob_min_jaccard=0.05 & cob_jaccard_start=0.3 : Co-expression
    #     similarity is weaker in sparse data, so we relax the Jaccard
    #     thresholds to allow CoB to still rescue binary marker modules.
    #
    # For modern, deeply sequenced datasets (10x v3+, Smart-seq2), use the
    # default SACI() parameters — they work out of the box.
    # -------------------------------------------------------------------------
    print("\nRunning SACI Hierarchical Clustering (relaxed parameters)...")
    rec = SACI(
        min_genes_to_split=3,       # Min bimodal genes to justify a split
        min_cells_to_split=20,      # Don't split clusters < 20 cells
        max_depth=3,                # Depth 3 is usually enough for PBMC 3k
        split_resolution=0.5,
        cob=True,
        cob_n_target_genes=50,      # Target 50 rescued genes per branch
        # --- Relaxed parameters for sparse / older data ---
        dip_pval_threshold=0.5,     # More permissive dip pre-filter
        min_cell_frac=0.001,        # Allow very rare populations (~3 cells)
        min_genes_fallback=30,      # Fewer fallback genes to reduce noise
        cob_min_jaccard=0.05,       # Relax co-expression threshold
        cob_jaccard_start=0.3,      # Lower starting Jaccard for CoB cascade
    )

    adata, all_genes, labels = rec.fit_transform(adata)

    print(f"\nTotal genes found (all levels): {len(all_genes)}")
    print(f"Hierarchical clusters (leaves): {len(set(labels))}")

    # For a smaller dataset like PBMC 3k, using all discovered genes is fine
    print(f"Generating UMAP using {len(all_genes)} SACI markers...")
    adata_final = adata[:, all_genes].copy()

    sc.pp.scale(adata_final)
    sc.tl.pca(adata_final, n_comps=min(30, len(all_genes) - 1))
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
    markers_to_check = [
        'CD3D', 'CD4', 'CD8A', 'MS4A1', 'CD14', 'FCGR3A', 'NKG7', 'PPBP',
    ]
    print("\nCanonical PBMC markers discovered:")
    for gene in markers_to_check:
        if gene in adata.var_names and adata.var.loc[gene, 'rmgs_selected']:
            level = adata.var.loc[gene, 'rmgs_level']
            print(f"  ✅ {gene}: Found at Level {int(level)}")
        else:
            print(f"  ❌ {gene}: Not selected")

    # -------------------------------------------------------------------------
    # Expected results with relaxed parameters:
    #   ✅ CD3D  (Level 0-1) — T-cell marker, usually passes even in sparse data
    #   ❌ CD4   — Often missed due to extreme dropout in 10x v2 chemistry
    #   ✅ CD8A  (Level 0-1) — Cytotoxic T-cell marker
    #   ✅ MS4A1 (Level 0-1) — B-cell marker (CD20)
    #   ✅ CD14  (Level 0)   — Monocyte marker
    #   ✅ FCGR3A(Level 1)   — Non-classical monocyte / NK marker (CD16)
    #   ✅ NKG7  (Level 0)   — NK / cytotoxic T-cell marker
    #   ✅ PPBP  (Level 0-2) — Platelet marker (only ~14 cells)
    #
    # Note: CD4 is notoriously difficult to detect as bimodal in the PBMC 3k
    # dataset because its expression is very low and affects only a subset of
    # T cells. This is a known limitation of all bimodality-based methods on
    # shallow-sequenced data. In deeply sequenced datasets (e.g. the
    # Leishmania example), CD4 is correctly identified.
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    main()
