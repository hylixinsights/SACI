# SACI — Single-cell Adaptive Clustering and Identification
## User Manual (v0.3.0)

### 1. OVERVIEW
SACI v0.3.0 provides three complementary tools for scRNA-seq gene selection:

A) **SaciBimodal**
Selects genes whose non-zero expression distribution is bimodal, indicating they distinguish two or more cell populations.

B) **SaciCoB (Co-expression Binary Rescue)**
Rescues genes with binary ON/OFF expression patterns that form co-expression modules specific to rare cell populations. Supports an adaptive cascade: starts from the strictest Jaccard threshold and progressively relaxes, locking in top genes first.

C) **SACI (Hierarchical Bimodality-Guided Clustering)**
Recursively subdivides cell clusters using SaciBimodal at each level. The number of bimodal genes found within a cluster serves as a natural stopping criterion: no bimodal genes = pure cluster.

---

### 2. CLASS REFERENCE

#### 2.1 SaciBimodal

```python
from saci import SaciBimodal
selector = SaciBimodal(dip_pval_threshold=0.2, min_bic_delta=10.0)
```

**Parameters:**
- `dip_pval_threshold (float, 0.2)`: Dip test p-value threshold (pre-filter).
- `min_bic_delta (float, 10.0)`: BIC difference for k=2 vs k=1.
- `min_peak_separation (float, 0.5)`: Min Cohen's d between peaks.
- `min_cell_frac (float, 0.05)`: Min fraction of cells expressing.
- `cob (bool, False)`: Enable CoB rescue.
- `cob_n_target_genes (int, None)`: Target for adaptive cascade.

**Annotations added to adata.var:**
- `'mgs_score'`: Composite bimodality score.
- `'mgs_selected'`: Boolean, True if selected by SaciBimodal.
- `'mgs_delta_bic'`: BIC difference (k=2 vs k=1).
- `'selection_source'`: 'mgs', 'cob', or 'both'.

#### 2.2 SaciCoB

```python
from saci import SaciCoB
```

**Parameters:**
- `min_cell_frac (float, 0.01)`: Min fraction for candidates.
- `max_cell_frac (float, 0.5)`: Max fraction (exclude housekeeping).
- `n_target_genes (int, None)`: If set, uses adaptive cascade.
- `jaccard_start (float, 0.7)`: Starting threshold for cascade.
- `jaccard_step (float, 0.05)`: Step size for cascade.

**Adaptive Cascade Algorithm:**
1. Compute ALL pairwise Jaccard similarities ONCE (expensive step).
2. Start at `jaccard_start` (e.g., 0.7).
3. Build gene graph, detect modules, rescue qualifying genes.
4. Lower threshold by `jaccard_step`.
5. Repeat until `n_target_genes` reached.

#### 2.3 SACI (Recursive)

```python
from saci import SACI
```

**Parameters:**
- `min_genes_to_split (int, 5)`: Bimodal genes needed to split.
- `min_cells_to_split (int, 30)`: Min cells to attempt split.
- `max_depth (int, 5)`: Max recursion depth.
- `split_resolution (float, 0.5)`: Leiden resolution per split.
- `cob (bool, True)`: Enable CoB at each level.

**Annotations added to adata:**
- `adata.obs['rmgs_cluster']`: Hierarchical leaf labels (e.g., "0.1.2").
- `adata.obs['rmgs_depth']`: Depth at which cell became a leaf.
- `adata.var['rmgs_selected']`: True for genes at any tree level.
- `adata.var['rmgs_level']`: Tree level where gene first discovered.

**Stopping criteria:**
- `too_few_cells`: Cluster has < `min_cells_to_split` cells.
- `max_depth`: Maximum recursion depth reached.
- `few_bimodal_genes`: Found < `min_genes_to_split` bimodal genes.
- `leiden_no_split`: Leiden couldn't split the cluster.

---

### 3. TROUBLESHOOTING

**Issue:** "Only N genes passed all filters"
- **Fix:** This is normal if your data has few bimodal genes. The pipeline falls back to the top genes by score. Lower `min_bic_delta` or `min_peak_separation` if needed.

**Issue:** RuntimeWarning in matmul (sklearn KMeans)
- **Fix:** Harmless warnings from very sparse genes in KMeans initialization. Does not affect results.

**Issue:** SACI recursion is slow
- **Fix:** Reduce `max_depth` (e.g., 3), increase `min_cells_to_split` (e.g., 50).
