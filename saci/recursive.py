"""
recursive.py - Recursive Hierarchical Clustering Guided by Bimodality

Uses SaciBimodal bimodality detection as both a gene selector and a stopping
criterion for recursive cluster subdivision. At each tree level, SaciBimodal
runs on only the cells in that branch:
    - If bimodal genes are found → substructure exists → split
    - If no bimodal genes → cluster is homogeneous → stop (leaf)

This discovers level-specific markers that are invisible to global
gene selection (e.g., CD8A is only bimodal within the T cell subset).
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .bimodal import SaciBimodal


class SACI:
    """
    Hierarchical clustering guided by SaciBimodal bimodality detection.

    Recursively subdivides cell clusters using SaciBimodal+CoB gene selection
    at each level. The number of bimodal genes found within a cluster
    serves as a natural stopping criterion: no bimodal genes = pure cluster.

    Parameters
    ----------
    min_genes_to_split : int, default=5
        Minimum number of strictly bimodal genes SaciBimodal must find within
        a cluster to justify splitting it further.
    min_cells_to_split : int, default=30
        Minimum cells in a cluster to attempt subdivision.
    max_depth : int, default=5
        Maximum recursion depth.
    split_resolution : float, default=0.5
        Leiden resolution used at each split step.
    cob : bool, default=True
        Whether to run CoB rescue at each recursion level.
    cob_n_target_genes : int or None, default=None
        Target genes for CoB adaptive cascade at each level.
    verbose_depth : int, default=2
        Show detailed SaciBimodal output only for levels <= this depth.
    bimodal_kw : dict
        Additional keyword arguments forwarded to the internal SaciBimodal
        constructor (e.g., dip_pval_threshold, min_bic_delta, etc.).

    Attributes
    ----------
    tree_ : dict
        Maps node_id → node metadata (parent, children, depth, genes, etc.).
    all_genes_ : list[str]
        Union of all genes discovered at every level of the tree.
    leaf_labels_ : np.ndarray
        Per-cell hierarchical cluster labels.
    """

    def __init__(
        self,
        min_genes_to_split: int = 5,
        min_cells_to_split: int = 30,
        max_depth: int = 5,
        split_resolution: float = 0.5,
        cob: bool = True,
        cob_n_target_genes: int | None = None,
        verbose_depth: int = 2,
        **bimodal_kw,
    ):
        self.min_genes_to_split = min_genes_to_split
        self.min_cells_to_split = min_cells_to_split
        self.max_depth = max_depth
        self.split_resolution = split_resolution
        self.cob = cob
        self.cob_n_target_genes = cob_n_target_genes
        self.verbose_depth = verbose_depth
        self.bimodal_kw = bimodal_kw

        self.tree_: dict = {}
        self.all_genes_: list[str] = []
        self.leaf_labels_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, adata, verbose: bool = True):
        """
        Run recursive hierarchical clustering.

        Parameters
        ----------
        adata : AnnData
            Log-normalized single-cell data.
        verbose : bool
            Print progress.

        Returns
        -------
        all_genes : list[str]
            Union of genes found at all tree levels.
        leaf_labels : np.ndarray
            Per-cell hierarchical cluster labels (e.g., "0.1.2").
        """
        import scanpy as sc

        n_cells = adata.n_obs
        cell_indices = np.arange(n_cells)

        # BFS queue: (cell_idx_array, node_id, depth)
        queue = [(cell_indices, "root", 0)]

        tree = {}
        all_genes_set = set()
        all_genes_ordered = []
        gene_level = {}  # gene → first level discovered
        cell_labels = np.full(n_cells, "", dtype=object)

        while queue:
            cell_idx, node_id, depth = queue.pop(0)
            n = len(cell_idx)
            indent = "  " * depth

            if verbose:
                print(f"\n{indent}▶ Node '{node_id}': {n} cells, depth {depth}")

            # --- Stopping: too few cells ---
            if n < self.min_cells_to_split:
                self._make_leaf(tree, node_id, depth, cell_idx, [],
                                "too_few_cells", cell_labels)
                if verbose:
                    print(f"{indent}  → LEAF (too few cells: {n})")
                continue

            # --- Stopping: max depth ---
            if depth >= self.max_depth:
                self._make_leaf(tree, node_id, depth, cell_idx, [],
                                "max_depth", cell_labels)
                if verbose:
                    print(f"{indent}  → LEAF (max depth)")
                continue

            # --- Run SaciBimodal (+CoB) on subset ---
            adata_sub = adata[cell_idx].copy()

            selector = SaciBimodal(
                cob=self.cob,
                cob_n_target_genes=self.cob_n_target_genes,
                **self.bimodal_kw,
            )

            sub_verbose = verbose and depth < self.verbose_depth
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    bimodal_genes = selector.fit(adata_sub, verbose=sub_verbose)
                except Exception as e:
                    self._make_leaf(tree, node_id, depth, cell_idx, [],
                                    f"error: {e}", cell_labels)
                    if verbose:
                        print(f"{indent}  → LEAF (error: {e})")
                    continue

            # --- Count strictly bimodal genes ---
            n_bimodal = self._count_strict_bimodal(selector)

            if verbose:
                print(
                    f"{indent}  Found {n_bimodal} bimodal genes "
                    f"({len(bimodal_genes)} total selected)"
                )

            # --- Stopping: not enough bimodal genes ---
            if n_bimodal < self.min_genes_to_split:
                self._make_leaf(tree, node_id, depth, cell_idx, bimodal_genes,
                                "few_bimodal_genes", cell_labels)
                if verbose:
                    print(
                        f"{indent}  → LEAF "
                        f"({n_bimodal} < {self.min_genes_to_split})"
                    )
                continue

            # Record newly discovered genes
            for g in bimodal_genes:
                if g not in all_genes_set:
                    all_genes_set.add(g)
                    all_genes_ordered.append(g)
                    gene_level[g] = depth

            # --- PCA → neighbors → Leiden ---
            if verbose:
                print(f"{indent}  Splitting ({len(bimodal_genes)} genes)...")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adata_sub_g = adata_sub[:, bimodal_genes].copy()
                sc.pp.scale(adata_sub_g)
                n_comps = min(30, len(bimodal_genes) - 1, n - 1)
                if n_comps < 2:
                    self._make_leaf(tree, node_id, depth, cell_idx,
                                    bimodal_genes, "too_few_pcs", cell_labels)
                    if verbose:
                        print(f"{indent}  → LEAF (too few PCs)")
                    continue

                sc.tl.pca(adata_sub_g, n_comps=n_comps)
                sc.pp.neighbors(adata_sub_g)
                sc.tl.leiden(
                    adata_sub_g,
                    resolution=self.split_resolution,
                )

            clusters = adata_sub_g.obs["leiden"]
            n_clusters = clusters.nunique()

            # --- Stopping: Leiden can't split ---
            if n_clusters < 2:
                self._make_leaf(tree, node_id, depth, cell_idx, bimodal_genes,
                                "leiden_no_split", cell_labels)
                if verbose:
                    print(f"{indent}  → LEAF (Leiden couldn't split)")
                continue

            if verbose:
                sizes = clusters.value_counts().sort_index()
                size_str = ", ".join(
                    f"{k}:{v}" for k, v in sizes.items()
                )
                print(
                    f"{indent}  → Split into {n_clusters} subclusters "
                    f"[{size_str}]"
                )

            # --- Enqueue children ---
            children = []
            for i, cid in enumerate(sorted(clusters.unique())):
                child_mask = (clusters == cid).values
                child_cells = cell_idx[child_mask]
                child_id = (
                    str(i) if node_id == "root" else f"{node_id}.{i}"
                )
                children.append(child_id)
                queue.append((child_cells, child_id, depth + 1))

            tree[node_id] = {
                "node_id": node_id,
                "parent": self._get_parent(node_id),
                "children": children,
                "depth": depth,
                "n_cells": n,
                "n_genes_found": len(bimodal_genes),
                "n_bimodal_strict": n_bimodal,
                "genes_top10": bimodal_genes[:10],
                "is_leaf": False,
                "leaf_reason": None,
            }

        # --- Summary ---
        self.tree_ = tree
        self.all_genes_ = all_genes_ordered
        self.leaf_labels_ = cell_labels
        self._gene_level_ = gene_level

        if verbose:
            leaves = [v for v in tree.values() if v["is_leaf"]]
            internals = [v for v in tree.values() if not v["is_leaf"]]
            print(f"\n{'=' * 50}")
            print(f"SACI Summary")
            print(f"  Internal nodes: {len(internals)}")
            print(f"  Leaf clusters:  {len(leaves)}")
            print(f"  Total genes:    {len(all_genes_ordered)}")
            max_d = max((v["depth"] for v in tree.values()), default=0)
            print(f"  Max depth:      {max_d}")

        return all_genes_ordered, cell_labels

    def fit_transform(self, adata, verbose: bool = True):
        """
        Run fit() and annotate adata.

        Adds:
            adata.obs['saci_cluster']  : hierarchical leaf labels
            adata.obs['saci_depth']    : depth at which cell's cluster
                                         became a leaf
            adata.var['saci_selected'] : True for genes at any level
            adata.var['saci_level']    : tree level where gene was
                                         first discovered (-1 if never)
            adata.uns['saci_tree']     : tree dict
            adata.uns['saci_genes']    : ordered gene list

        Returns
        -------
        adata, all_genes, leaf_labels
        """
        all_genes, labels = self.fit(adata, verbose=verbose)

        adata.obs["saci_cluster"] = pd.Categorical(labels)

        # Depth per cell from tree
        depth_map = {}
        for node in self.tree_.values():
            if node["is_leaf"]:
                depth_map[node["node_id"]] = node["depth"]
        adata.obs["saci_depth"] = [
            depth_map.get(lbl, -1) for lbl in labels
        ]

        # Gene annotations
        adata.var["saci_selected"] = adata.var_names.isin(all_genes)
        level_series = pd.Series(
            self._gene_level_
        ).reindex(adata.var_names).fillna(-1).astype(int)
        adata.var["saci_level"] = level_series.values

        adata.uns["saci_tree"] = self.tree_
        adata.uns["saci_genes"] = all_genes

        # Backward-compatible aliases (rmgs_* → saci_*)
        # These will be removed in a future major release.
        adata.obs["rmgs_cluster"] = adata.obs["saci_cluster"]
        adata.obs["rmgs_depth"] = adata.obs["saci_depth"]
        adata.var["rmgs_selected"] = adata.var["saci_selected"]
        adata.var["rmgs_level"] = adata.var["saci_level"]
        adata.uns["rmgs_tree"] = adata.uns["saci_tree"]
        adata.uns["rmgs_genes"] = adata.uns["saci_genes"]
        warnings.warn(
            "The 'rmgs_' prefix in adata keys (e.g. rmgs_cluster) is "
            "deprecated and will be removed in the next major release. "
            "Please update your code to use 'saci_' (e.g. saci_cluster).",
            DeprecationWarning,
            stacklevel=2,
        )

        return adata, all_genes, labels

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_parent(node_id: str) -> str | None:
        if node_id == "root":
            return None
        parts = node_id.split(".")
        if len(parts) == 1:
            return "root"
        return ".".join(parts[:-1])

    @staticmethod
    def _count_strict_bimodal(selector: SaciBimodal) -> int:
        """Count genes passing strict BIC + peak_separation filters."""
        if selector.results_ is None:
            return 0
        df = selector.results_
        strict = df[
            (df["filter_reason"] == "passed")
            & (df["delta_bic"] >= selector.min_bic_delta)
            & (df["peak_separation"] >= selector.min_peak_separation)
        ]
        return len(strict)

    def _make_leaf(
        self, tree, node_id, depth, cell_idx, genes, reason, cell_labels,
    ):
        """Record a leaf node and assign labels to its cells."""
        leaf_label = node_id if node_id != "root" else "0"
        cell_labels[cell_idx] = leaf_label

        tree[node_id] = {
            "node_id": node_id,
            "parent": self._get_parent(node_id),
            "children": [],
            "depth": depth,
            "n_cells": len(cell_idx),
            "n_genes_found": len(genes),
            "n_bimodal_strict": 0,
            "genes_top10": genes[:10],
            "is_leaf": True,
            "leaf_reason": reason,
        }
