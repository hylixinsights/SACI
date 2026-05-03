"""
SACI - Single-cell Adaptive Clustering and Identification
=========================================================
Selects biologically informative genes for scRNA-seq dimensionality reduction
by detecting multimodal expression distributions, instead of relying solely
on highly variable genes (HVGs).

Pipeline:
    1. **SaciBimodal arm**: Filter genes expressed in at least `min_cell_frac` of cells,
       Dip Test (Hartigan) as fast pre-filter for multimodality, GMM (k=1 vs k=2)
       with BIC to confirm and score bimodality. Captures genes with two distinct
       expression states among non-zero cells.
    2. **SaciCoB arm** (optional): Co-expression Binary rescue. Detects modules of
       genes sharing ON/OFF patterns in the same rare cell populations. Rescues
       perfect marker genes that the bimodality pipeline misses by design.
    3. **SACI recursive tree** (optional): Hierarchical clustering guided by bimodality.
       Recursively subdivides clusters, using the number of bimodal genes as a
       natural stopping criterion. Discovers level-specific markers invisible
       to global gene selection.

Usage:
    import scanpy as sc
    from saci import SACI, SaciBimodal

    adata = sc.datasets.pbmc3k()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    # Bimodal only
    selector = SaciBimodal()
    genes = selector.fit(adata)

    # Bimodal + CoB rescue
    selector = SaciBimodal(cob=True)
    genes = selector.fit(adata)

    # Recursive hierarchical clustering (SACI)
    rec = SACI(cob=True, min_genes_to_split=5)
    adata, genes, labels = rec.fit_transform(adata)
"""

from .bimodal import SaciBimodal
from .cob import SaciCoB
from .recursive import SACI
from .scoring import dip_test, fit_gmm, multimodal_score
from .plotting import plot_gene_distribution, plot_score_distribution, plot_umap_comparison

__version__ = "0.3.0"
__all__ = [
    "SACI",
    "SaciBimodal",
    "SaciCoB",
    "dip_test",
    "fit_gmm",
    "multimodal_score",
    "plot_gene_distribution",
    "plot_score_distribution",
    "plot_umap_comparison",
]

