# SACI — Single-cell Adaptive Clustering and Identification
## Vignette (v0.3.0)
**Distribution-Aware Gene Selection for scRNA-seq Dimensionality Reduction**

### 1. Introduction
Single-cell RNA sequencing (scRNA-seq) has transformed our ability to characterize cellular heterogeneity at transcriptomic resolution. A central step in nearly every scRNA-seq workflow is the selection of informative genes prior to dimensionality reduction (e.g., PCA/UMAP) and subsequent clustering.

The dominant approach, Highly Variable Gene (HVG) selection, ranks genes by their variance across cells. While computationally efficient, HVG selection conflates biological signal with technical noise. 

**SACI** takes a fundamentally different approach. Instead of measuring variance, it characterizes the shape of each gene's expression distribution among the cells that express it. A gene that defines a cell type will tend to be OFF in most cells and ON in a specific subpopulation — creating two distinct peaks. SACI captures it by design, because bimodality is exactly what it tests for.

### 2. The Limitation of Global Gene Selection
Consider CD4 in a PBMC dataset. Globally, CD4 is expressed in many cell types at varying levels — its distribution is NOT bimodal across all cells. Therefore, both HVG and SaciBimodal (when applied globally) may miss it. However, **WITHIN** the T cell compartment, CD4 becomes strongly bimodal: CD4+ T cells express it, CD8+ T cells do not.

This motivates the **SACI Recursive Tree**: by running the algorithm at each level of the hierarchy, we discover markers that are invisible at the global level but critical for subtype identification.

### 3. Statistical Methods

#### 3.1 SaciBimodal Pipeline
1. **Expression filter:** retain genes expressed in ≥ `min_cell_frac` of cells.
2. **Non-zero extraction:** subset to cells with expression > 0.
3. **Hartigan Dip Test:** test unimodality.
4. **Gaussian Mixture Model:** fit GMM with k=1 and k=2. Compare via BIC. 
5. **Composite score:** `multimodal_score = (1 - dip_pval) × max(ΔBIC, 0) × peak_separation`

#### 3.2 SaciCoB Pipeline
Some rare populations (e.g., platelets at 0.1%) express marker genes in too few cells to produce a bimodal distribution. These genes are co-expressed in a binary pattern. CoB exploits this by building a gene co-expression graph based on Jaccard similarity of binary expression patterns.
The **Adaptive Cascade** algorithm starts at a strict threshold (e.g. J=0.7) and relaxes it progressively, locking in the highest-confidence genes first.

#### 3.3 SACI Algorithm
Uses breadth-first search (BFS) to recursively subdivide clusters:
1. Run `SaciBimodal` (+`SaciCoB`) on the subset of cells.
2. Count strictly bimodal genes (`ΔBIC` ≥ 10, `peak_sep` ≥ 0.5).
3. If `n_bimodal < min_genes_to_split` → **LEAF** (cluster is pure).
4. PCA → neighbors → Leiden on selected genes.
5. If Leiden finds ≥2 clusters → enqueue children.

### 4. Results on Leishmania PBMC
Tested on 17,308 cells, SACI achieved a **+15.4% improvement in Adjusted Rand Index** (ARI) over HVG selection.

Biological validation found 28/28 known PBMC markers at their correct hierarchical level:
- **Level 0:** CD8A, CD14, LYZ, MS4A1 (Lineage separation)
- **Level 1:** CD3D, CD3E, NKG7 (Subtype markers)
- **Level 2:** CD4, LILRA4 (Within-subtype markers, invisible globally)

*Key finding:* CD4 is discovered at Level 2, proving that Recursive SACI discovers level-specific markers that are mathematically impossible to detect globally.
