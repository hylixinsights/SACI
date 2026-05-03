"""
cob.py - Co-expression Binary Rescue Module

Detects modules of genes that are co-expressed (ON/OFF) in the same
rare cell populations. Rescues marker genes that MGS misses because
their non-zero expression distribution is unimodal.

Supports two modes:
    - Fixed threshold: rescue all qualifying genes at a single Jaccard cutoff.
    - Adaptive cascade: start from the strictest threshold and progressively
      relax, locking in top genes first. Guarantees the highest-confidence
      genes are always included.

Pipeline:
    1. Binarize expression (ON/OFF per cell per gene)
    2. Pre-filter candidate genes by expression fraction
    3. Compute pairwise Jaccard similarity via sparse matrix algebra (ONCE)
    4. Cascade from strict to permissive thresholds:
       a. Build gene graph at current threshold
       b. Detect communities (Leiden)
       c. Score modules, lock in qualifying genes
       d. Stop when n_target_genes is reached
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.sparse import issparse, csc_matrix


class SaciCoB:
    """
    Co-expression Binary Rescue for scRNA-seq gene selection.

    Identifies modules of genes that share binary ON/OFF patterns
    across cells, rescuing rare-population markers that bimodality-based
    methods (like MGS) miss by design.

    Parameters
    ----------
    min_cell_frac : float, default=0.01
        Minimum fraction of cells expressing a gene to be considered.
    max_cell_frac : float, default=0.5
        Maximum fraction of cells expressing a gene. Genes above this
        are likely housekeeping and excluded from rescue.
    min_module_size : int, default=3
        Minimum number of genes in a module to qualify for rescue.
    min_cohesion : float, default=0.2
        Minimum mean pairwise Jaccard within a module.
    resolution : float, default=1.0
        Resolution parameter for Leiden community detection.
    n_target_genes : int or None, default=None
        Target number of rescued genes. If set, uses the adaptive
        cascade algorithm (strict → permissive). If None, uses the
        fixed min_jaccard threshold.
    min_jaccard : float, default=0.3
        Jaccard threshold for fixed mode. In adaptive mode, this is
        the floor (the most permissive threshold allowed).
    jaccard_start : float, default=0.7
        Starting (strictest) Jaccard threshold for adaptive cascade.
    jaccard_step : float, default=0.05
        Step size to decrease Jaccard threshold in each cascade round.
    layer : str or None, default=None
        AnnData layer to use. If None, uses adata.X.

    Attributes
    ----------
    results_ : pd.DataFrame
        Per-gene statistics after fitting.
    module_info_ : pd.DataFrame
        Per-module summary statistics.
    rescued_genes_ : list[str]
        Genes rescued after fitting.
    """

    def __init__(
        self,
        min_cell_frac: float = 0.01,
        max_cell_frac: float = 0.5,
        min_module_size: int = 3,
        min_cohesion: float = 0.2,
        resolution: float = 1.0,
        n_target_genes: int | None = None,
        min_jaccard: float = 0.3,
        jaccard_start: float = 0.7,
        jaccard_step: float = 0.05,
        layer: str | None = None,
    ):
        self.min_cell_frac = min_cell_frac
        self.max_cell_frac = max_cell_frac
        self.min_module_size = min_module_size
        self.min_cohesion = min_cohesion
        self.resolution = resolution
        self.n_target_genes = n_target_genes
        self.min_jaccard = min_jaccard
        self.jaccard_start = jaccard_start
        self.jaccard_step = jaccard_step
        self.layer = layer

        self.results_: pd.DataFrame | None = None
        self.module_info_: pd.DataFrame | None = None
        self.rescued_genes_: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        adata,
        exclude_genes: list[str] | None = None,
        verbose: bool = True,
    ) -> list[str]:
        """
        Run the co-expression binary rescue pipeline.

        Parameters
        ----------
        adata : AnnData
            Single-cell data (log-normalized).
        exclude_genes : list[str] or None
            Genes already selected (e.g., by MGS). Excluded from
            CoB candidates to avoid redundancy.
        verbose : bool
            Show progress and summary.

        Returns
        -------
        list[str]
            Rescued gene names, ordered by confidence tier then score.
        """
        X = self._get_matrix(adata)
        gene_names = np.array(adata.var_names)
        n_cells, n_genes = X.shape

        mode = "adaptive" if self.n_target_genes else "fixed"
        if verbose:
            print(f"\nCoB ({mode} mode): {n_cells} cells × {n_genes} genes")
            if mode == "adaptive":
                print(f"  n_target_genes  = {self.n_target_genes}")
                print(f"  jaccard_start   = {self.jaccard_start}")
                print(f"  jaccard_step    = {self.jaccard_step}")
                print(f"  jaccard_floor   = {self.min_jaccard}")
            else:
                print(f"  min_jaccard     = {self.min_jaccard}")
            print(f"  min_module_size = {self.min_module_size}")
            print(f"  min_cohesion    = {self.min_cohesion}")

        # --- Step 1: Binarize ---
        if verbose:
            print("  Binarizing expression matrix...")
        B = self._binarize(X)

        # --- Step 2: Pre-filter candidates ---
        gene_sums = np.array(B.sum(axis=0)).ravel()
        frac_expressed = gene_sums / n_cells

        mask = (
            (frac_expressed >= self.min_cell_frac)
            & (frac_expressed <= self.max_cell_frac)
        )

        if exclude_genes is not None:
            exclude_set = set(exclude_genes)
            exclude_mask = np.array([g in exclude_set for g in gene_names])
            mask = mask & ~exclude_mask

        candidate_idx = np.where(mask)[0]
        candidate_names = gene_names[candidate_idx]
        n_candidates = len(candidate_idx)

        if verbose:
            print(f"  Candidate genes after filtering: {n_candidates}")

        if n_candidates < self.min_module_size:
            if verbose:
                print("  Too few candidates. No genes rescued.")
            self._store_empty(candidate_names, frac_expressed[candidate_idx])
            return []

        B_cand = B[:, candidate_idx]
        cand_sums = gene_sums[candidate_idx].astype(float)
        cand_fracs = frac_expressed[candidate_idx]

        # --- Step 3: Compute ALL pairwise Jaccard values ONCE ---
        if verbose:
            print("  Computing pairwise Jaccard similarities...")

        B_csc = csc_matrix(B_cand)
        intersection_sparse = (B_csc.T @ B_csc)
        intersection_coo = intersection_sparse.tocoo()

        # Upper triangle only
        upper = intersection_coo.row < intersection_coo.col
        all_rows = intersection_coo.row[upper]
        all_cols = intersection_coo.col[upper]
        all_inter = intersection_coo.data[upper].astype(float)

        all_union = cand_sums[all_rows] + cand_sums[all_cols] - all_inter
        all_jaccard = np.divide(
            all_inter, all_union,
            where=all_union > 0,
            out=np.zeros_like(all_inter),
        )

        # Keep intersection as CSR for cohesion lookups
        intersection_csr = intersection_sparse.tocsr()

        if verbose:
            n_total_edges = np.sum(all_jaccard >= self.min_jaccard)
            print(f"  Total Jaccard pairs computed: {len(all_jaccard)}")

        # --- Branch: adaptive cascade vs fixed threshold ---
        if self.n_target_genes is not None:
            return self._fit_adaptive(
                candidate_names, cand_sums, cand_fracs,
                all_rows, all_cols, all_jaccard,
                intersection_csr, n_candidates, verbose,
            )
        else:
            return self._fit_fixed(
                candidate_names, cand_sums, cand_fracs,
                all_rows, all_cols, all_jaccard,
                intersection_csr, n_candidates, verbose,
            )

    def fit_transform(self, adata, exclude_genes=None, verbose=True):
        """
        Run fit() and annotate adata.var / adata.uns with CoB results.

        Adds:
            adata.var['cob_score']      : rescue score (0 if not candidate)
            adata.var['cob_selected']   : bool
            adata.var['cob_module_id']  : module assignment (-1 if N/A)
            adata.var['cob_tier']       : cascade tier (adaptive mode only)
            adata.uns['cob_genes']      : list of rescued genes
            adata.uns['cob_params']     : dict of parameters used

        Returns
        -------
        adata (modified in place), rescued_genes list
        """
        genes = self.fit(adata, exclude_genes=exclude_genes, verbose=verbose)

        if self.results_ is not None and len(self.results_) > 0:
            df = self.results_.set_index("gene")
            adata.var["cob_score"] = (
                df["cob_score"].reindex(adata.var_names).fillna(0.0).values
            )
            adata.var["cob_selected"] = adata.var_names.isin(genes)
            adata.var["cob_module_id"] = (
                df["cob_module_id"]
                .reindex(adata.var_names)
                .fillna(-1)
                .astype(int)
                .values
            )
            if "cob_tier" in df.columns:
                adata.var["cob_tier"] = (
                    df["cob_tier"]
                    .reindex(adata.var_names)
                    .fillna(-1)
                    .astype(int)
                    .values
                )
        else:
            adata.var["cob_score"] = 0.0
            adata.var["cob_selected"] = False
            adata.var["cob_module_id"] = -1

        adata.uns["cob_genes"] = genes
        adata.uns["cob_params"] = {
            "min_cell_frac": self.min_cell_frac,
            "max_cell_frac": self.max_cell_frac,
            "min_jaccard": self.min_jaccard,
            "min_module_size": self.min_module_size,
            "min_cohesion": self.min_cohesion,
            "resolution": self.resolution,
            "n_target_genes": self.n_target_genes,
            "jaccard_start": self.jaccard_start,
            "jaccard_step": self.jaccard_step,
        }

        return adata, genes

    # ------------------------------------------------------------------
    # Adaptive cascade
    # ------------------------------------------------------------------

    def _fit_adaptive(
        self, candidate_names, cand_sums, cand_fracs,
        all_rows, all_cols, all_jaccard,
        intersection_csr, n_candidates, verbose,
    ):
        """Cascade from strict to permissive Jaccard thresholds."""
        target = self.n_target_genes
        thresholds = np.arange(
            self.jaccard_start,
            self.min_jaccard - 1e-9,
            -self.jaccard_step,
        )

        rescued_ordered = []   # Maintains insertion order (strict first)
        rescued_set = set()
        all_gene_records = []
        all_module_records = []
        tier = 0

        for thresh in thresholds:
            tier += 1
            # Filter edges at current threshold
            edge_mask = all_jaccard >= thresh
            e_rows = all_rows[edge_mask]
            e_cols = all_cols[edge_mask]
            e_weights = all_jaccard[edge_mask]
            n_edges = len(e_rows)

            if verbose:
                print(
                    f"\n  --- Tier {tier}: Jaccard ≥ {thresh:.2f} "
                    f"({n_edges} edges) ---"
                )

            if n_edges == 0:
                if verbose:
                    print(f"  No edges at this threshold, skipping.")
                continue

            # Community detection
            try:
                import igraph as ig
            except ImportError:
                raise ImportError(
                    "CoB requires igraph: pip3 install igraph leidenalg"
                )

            edges = list(zip(e_rows.tolist(), e_cols.tolist()))
            g = ig.Graph(n=n_candidates, edges=edges, directed=False)
            g.es["weight"] = e_weights.tolist()

            partition = g.community_leiden(
                objective_function="modularity",
                weights="weight",
                resolution=self.resolution,
            )
            membership = np.array(partition.membership)

            # Score modules and rescue NEW genes
            new_in_tier = 0
            for mod_id in np.unique(membership):
                mod_local = np.where(membership == mod_id)[0]
                mod_size = len(mod_local)
                mod_names = candidate_names[mod_local]

                if mod_size < self.min_module_size:
                    continue

                # Cohesion
                pair_j = []
                for a in range(len(mod_local)):
                    for b in range(a + 1, len(mod_local)):
                        i, j = mod_local[a], mod_local[b]
                        iv = float(intersection_csr[i, j])
                        uv = cand_sums[i] + cand_sums[j] - iv
                        if uv > 0:
                            pair_j.append(iv / uv)

                cohesion = float(np.mean(pair_j)) if pair_j else 0.0

                if cohesion < self.min_cohesion:
                    continue

                specificity = 1.0 - float(np.median(cand_fracs[mod_local]))
                score = float(np.log2(mod_size)) * cohesion * specificity

                all_module_records.append({
                    "module_id": len(all_module_records),
                    "tier": tier,
                    "jaccard_threshold": float(thresh),
                    "size": mod_size,
                    "cohesion": cohesion,
                    "specificity": specificity,
                    "cob_score": score,
                    "top_genes": ", ".join(mod_names[:5]),
                })

                # Add only NEW genes (not already locked in)
                for idx in mod_local:
                    gene = candidate_names[idx]
                    if gene not in rescued_set:
                        rescued_set.add(gene)
                        rescued_ordered.append(gene)
                        new_in_tier += 1
                        all_gene_records.append(self._gene_record_adaptive(
                            gene, cand_fracs[idx],
                            len(all_module_records) - 1, mod_size,
                            cohesion, specificity, score, tier, thresh,
                        ))
                        # Stop immediately if target reached
                        if len(rescued_ordered) >= target:
                            break
                    # (genes already rescued are simply skipped)

                if len(rescued_ordered) >= target:
                    break

            if verbose:
                print(
                    f"  New genes this tier: {new_in_tier} | "
                    f"Cumulative: {len(rescued_ordered)}"
                )

            if len(rescued_ordered) >= target:
                if verbose:
                    print(f"\n  ✅ Target reached ({target} genes)!")
                break

        # Trim to exact target
        rescued_ordered = rescued_ordered[:target]

        self.results_ = (
            pd.DataFrame(all_gene_records)
            .head(target)
            .reset_index(drop=True)
        )
        self.module_info_ = (
            pd.DataFrame(all_module_records)
            .sort_values("cob_score", ascending=False)
            .reset_index(drop=True)
        )
        self.rescued_genes_ = rescued_ordered

        if verbose:
            n_tiers_used = len(set(r["cob_tier"] for r in all_gene_records[:target])) if all_gene_records else 0
            print(f"\n  === CoB Adaptive Summary ===")
            print(f"  Tiers used:     {n_tiers_used}")
            print(f"  Genes rescued:  {len(rescued_ordered)}")

        return rescued_ordered

    # ------------------------------------------------------------------
    # Fixed threshold (original behavior)
    # ------------------------------------------------------------------

    def _fit_fixed(
        self, candidate_names, cand_sums, cand_fracs,
        all_rows, all_cols, all_jaccard,
        intersection_csr, n_candidates, verbose,
    ):
        """Module detection at a single fixed Jaccard threshold."""
        edge_mask = all_jaccard >= self.min_jaccard
        edge_rows = all_rows[edge_mask]
        edge_cols = all_cols[edge_mask]
        edge_weights = all_jaccard[edge_mask]
        n_edges = len(edge_rows)

        if verbose:
            print(
                f"  Gene graph: {n_candidates} nodes, "
                f"{n_edges} edges (Jaccard ≥ {self.min_jaccard})"
            )

        if n_edges == 0:
            if verbose:
                print("  No edges found. No genes rescued.")
            self._store_empty(candidate_names, cand_fracs)
            return []

        if verbose:
            print("  Detecting gene modules (Leiden)...")

        try:
            import igraph as ig
        except ImportError:
            raise ImportError(
                "CoB requires igraph: pip3 install igraph leidenalg"
            )

        edges = list(zip(edge_rows.tolist(), edge_cols.tolist()))
        g = ig.Graph(n=n_candidates, edges=edges, directed=False)
        g.es["weight"] = edge_weights.tolist()

        partition = g.community_leiden(
            objective_function="modularity",
            weights="weight",
            resolution=self.resolution,
        )
        membership = np.array(partition.membership)

        if verbose:
            print("  Scoring modules...")

        unique_modules = np.unique(membership)
        gene_records = []
        module_records = []
        rescued = []

        for mod_id in unique_modules:
            mod_local = np.where(membership == mod_id)[0]
            mod_size = len(mod_local)
            mod_names = candidate_names[mod_local]

            if mod_size < self.min_module_size:
                for idx in mod_local:
                    gene_records.append(self._gene_record(
                        candidate_names[idx], cand_fracs[idx],
                        int(mod_id), mod_size,
                        np.nan, np.nan, 0.0, False, "small_module",
                    ))
                continue

            pair_jaccards = []
            for a in range(len(mod_local)):
                for b in range(a + 1, len(mod_local)):
                    i, j = mod_local[a], mod_local[b]
                    iv = float(intersection_csr[i, j])
                    uv = cand_sums[i] + cand_sums[j] - iv
                    if uv > 0:
                        pair_jaccards.append(iv / uv)

            cohesion = float(np.mean(pair_jaccards)) if pair_jaccards else 0.0
            specificity = 1.0 - float(np.median(cand_fracs[mod_local]))
            score = float(np.log2(mod_size)) * cohesion * specificity
            selected = cohesion >= self.min_cohesion
            reason = "passed" if selected else "low_cohesion"

            for idx in mod_local:
                gene_records.append(self._gene_record(
                    candidate_names[idx], cand_fracs[idx],
                    int(mod_id), mod_size,
                    cohesion, specificity, score, selected, reason,
                ))
                if selected:
                    rescued.append(candidate_names[idx])

            module_records.append({
                "module_id": int(mod_id),
                "size": mod_size,
                "cohesion": cohesion,
                "specificity": specificity,
                "cob_score": score,
                "rescued": selected,
                "top_genes": ", ".join(mod_names[:5]),
            })

        self.results_ = (
            pd.DataFrame(gene_records)
            .sort_values("cob_score", ascending=False)
            .reset_index(drop=True)
        )
        self.module_info_ = (
            pd.DataFrame(module_records)
            .sort_values("cob_score", ascending=False)
            .reset_index(drop=True)
        )
        self.rescued_genes_ = rescued

        if verbose:
            n_mod_total = len(module_records)
            n_mod_rescued = sum(1 for r in module_records if r["rescued"])
            print(f"\n  Modules found:   {n_mod_total}")
            print(f"  Modules rescued: {n_mod_rescued}")
            print(f"  Genes rescued:   {len(rescued)}")

        return rescued

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_matrix(self, adata):
        if self.layer is not None:
            return adata.layers[self.layer]
        return adata.X

    def _binarize(self, X):
        """Convert expression matrix to binary sparse matrix (CSC)."""
        if issparse(X):
            B = X.copy()
            B.data = np.ones_like(B.data)
            return csc_matrix(B)
        else:
            return csc_matrix((np.asarray(X) > 0).astype(float))

    @staticmethod
    def _gene_record(
        gene, frac, mod_id, mod_size,
        cohesion, specificity, score, selected, reason,
    ):
        return {
            "gene": gene,
            "frac_expressed": frac,
            "cob_module_id": mod_id,
            "cob_module_size": mod_size,
            "cob_cohesion": cohesion,
            "cob_specificity": specificity,
            "cob_score": score,
            "cob_selected": selected,
            "cob_filter_reason": reason,
        }

    @staticmethod
    def _gene_record_adaptive(
        gene, frac, mod_id, mod_size,
        cohesion, specificity, score, tier, threshold,
    ):
        return {
            "gene": gene,
            "frac_expressed": frac,
            "cob_module_id": mod_id,
            "cob_module_size": mod_size,
            "cob_cohesion": cohesion,
            "cob_specificity": specificity,
            "cob_score": score,
            "cob_selected": True,
            "cob_filter_reason": "passed",
            "cob_tier": tier,
            "cob_jaccard_threshold": threshold,
        }

    def _store_empty(self, names, fracs):
        records = [
            self._gene_record(
                names[i], fracs[i], -1, 0,
                np.nan, np.nan, 0.0, False, "no_edges",
            )
            for i in range(len(names))
        ]
        self.results_ = pd.DataFrame(records)
        self.module_info_ = pd.DataFrame()
        self.rescued_genes_ = []
