# SACI: Single-cell Adaptive Clustering and Identification 🌪️

**SACI** (Single-cell Adaptive Clustering and Identification) is a Python package for biologically-grounded gene selection and hierarchical clustering of single-cell RNA-seq data. 

While most methods rely on Highly Variable Genes (HVG) — which conflates biological signal with technical noise — SACI selects genes based on the *shape* of their expression distribution. A marker gene defining a cell subpopulation will inherently create two distinct modes (ON vs OFF) across the cells that express it. SACI uses this statistical signature (Bimodality) to guide both gene selection and cluster subdivision.

The acronym refers to the Saci, a famous figure from Brazilian folklore who is elusive and hard to catch. Just like the Saci, rare cell subpopulations and their specific marker genes are elusive and often missed by global gene selection methods.

## Core Components

1. **SaciBimodal**: Selects genes with bimodal expression distributions via Hartigan's Dip Test and Gaussian Mixture Model comparison (BIC).
2. **SaciCoB**: Co-expression Binary rescue. Uses an adaptive Jaccard similarity cascade to rescue perfect binary ON/OFF marker genes for rare populations that are not large enough to form a bimodal distribution.
3. **SACI (Recursive Tree)**: Hierarchical clustering that uses the number of bimodal genes as a natural stopping criterion. It discovers level-specific markers (e.g., CD4) that are invisible to global gene selection.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/SACI.git
cd SACI

# Install the package and dependencies
pip install -e .

# For full functionality (CoB rescue and Recursive Tree):
pip install ".[scanpy,leiden]"
```

## Quick Start

```python
import scanpy as sc
from saci import SACI

# 1. Load and prepare data (must be log-normalized)
adata = sc.datasets.pbmc3k()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# 2. Run SACI Hierarchical Clustering
model = SACI(
    min_genes_to_split=5,     # Need ≥5 bimodal genes to split a cluster
    min_cells_to_split=30,    # Don't split clusters with <30 cells
    max_depth=4,              # Max hierarchy depth
    cob=True,                 # Enable CoB rescue for rare populations
    cob_n_target_genes=100    # Target 100 rescued genes per tree level
)

# 3. Fit and transform (adds cluster labels and tree metadata to adata)
adata, all_genes, labels = model.fit_transform(adata)

# 4. Filter genes from the top levels (Optimal strategy for UMAP)
top_genes = list(adata.var_names[
    (adata.var['saci_selected']) & (adata.var['saci_level'] <= 2)
])

# 5. Standard downstream analysis
adata_final = adata[:, top_genes].copy()
sc.pp.scale(adata_final)
sc.tl.pca(adata_final)
sc.pp.neighbors(adata_final)
sc.tl.umap(adata_final)
```

## Documentation

- [User Manual](docs/saci_user_manual.md): Detailed API reference and parameter tuning guide.
- [Vignette](docs/saci_vignette.md): Biological rationale, theoretical background, and comparison with other methods.

## Example

Check out the `examples/` directory for ready-to-run scripts. The `01_leishmania_example.py` demonstrates the full pipeline achieving +15.4% ARI compared to HVG on a complex PBMC dataset.
