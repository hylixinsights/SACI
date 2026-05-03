"""
selector.py - Main SaciBimodal class

Orchestrates the full pipeline:
    filter → dip test → GMM → score → select

Optionally integrates CoB (Co-expression Binary rescue) to recover
rare-population ON/OFF markers that the bimodality pipeline misses.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm import tqdm

from .scoring import dip_test, fit_gmm, multimodal_score


class SaciBimodal:
    """
    Multimodal Gene Selector for scRNA-seq data.

    Selects genes with bimodal/multimodal expression distributions
    as candidates for cluster-defining marker genes, providing a
    biologically-grounded alternative to Highly Variable Gene (HVG)
    selection for UMAP/clustering.

    Parameters
    ----------
    dip_pval_threshold : float, default=0.2
        Liberal pre-filter. Genes with dip test p-value above this
        threshold are excluded before the more expensive GMM step.
        Use 0.2 to be conservative (keep more candidates);
        use 0.05 for stricter pre-filtering.

    min_bic_delta : float, default=10.0
        Minimum BIC(k=1) - BIC(k=2) required to consider a gene bimodal.
        BIC difference > 10 is conventionally "very strong" evidence
        for the more complex model. Set to 0 to disable.

    min_peak_separation : float, default=0.5
        Minimum Cohen's d between the two GMM components.
        Prevents selecting genes with two overlapping, barely-distinct peaks.

    min_cell_frac : float, default=0.05
        Minimum fraction of cells that must express the gene (non-zero)
        for it to be considered. Removes genes with near-total dropout.

    n_top_genes : int or None, default=None
        If set, return exactly this many top-scoring genes (fallback
        when few genes pass BIC threshold). If None, return all genes
        that pass min_bic_delta and min_peak_separation filters.

    min_genes_fallback : int, default=200
        If fewer than this many genes pass all filters, automatically
        fall back to returning top `min_genes_fallback` genes by score,
        ignoring min_bic_delta threshold. A warning is issued.

    layer : str or None, default=None
        AnnData layer to use. If None, uses adata.X.

    Attributes
    ----------
    results_ : pd.DataFrame
        Per-gene statistics after fitting. Columns:
        gene, n_expressed, frac_expressed, dip_stat, dip_pval,
        delta_bic, peak_separation, multimodal_score, selected.

    selected_genes_ : list[str]
        Genes selected after fitting.

    Examples
    --------
    >>> import scanpy as sc
    >>> from mgs import MGS
    >>> adata = sc.datasets.pbmc3k()
    >>> sc.pp.normalize_total(adata)
    >>> sc.pp.log1p(adata)
    >>> selector = MGS()
    >>> genes = selector.fit(adata)
    >>> print(f"Selected {len(genes)} genes")
    >>> selector.plot_score_distribution()
    """

    def __init__(
        self,
        dip_pval_threshold: float = 0.2,
        min_bic_delta: float = 10.0,
        min_peak_separation: float = 0.5,
        min_cell_frac: float = 0.05,
        n_top_genes: int | None = None,
        min_genes_fallback: int = 200,
        layer: str | None = None,
        # CoB integration parameters
        cob: bool = False,
        cob_max_cell_frac: float = 0.5,
        cob_min_jaccard: float = 0.3,
        cob_min_module_size: int = 3,
        cob_min_cohesion: float = 0.2,
        cob_resolution: float = 1.0,
        cob_n_target_genes: int | None = None,
        cob_jaccard_start: float = 0.7,
        cob_jaccard_step: float = 0.05,
    ):
        self.dip_pval_threshold = dip_pval_threshold
        self.min_bic_delta = min_bic_delta
        self.min_peak_separation = min_peak_separation
        self.min_cell_frac = min_cell_frac
        self.n_top_genes = n_top_genes
        self.min_genes_fallback = min_genes_fallback
        self.layer = layer

        # CoB parameters
        self.cob = cob
        self.cob_max_cell_frac = cob_max_cell_frac
        self.cob_min_jaccard = cob_min_jaccard
        self.cob_min_module_size = cob_min_module_size
        self.cob_min_cohesion = cob_min_cohesion
        self.cob_resolution = cob_resolution
        self.cob_n_target_genes = cob_n_target_genes
        self.cob_jaccard_start = cob_jaccard_start
        self.cob_jaccard_step = cob_jaccard_step

        self.results_: pd.DataFrame | None = None
        self.selected_genes_: list[str] = []
        self.cob_selector_ = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, adata, verbose: bool = True) -> list[str]:
        """
        Run the full gene selection pipeline on an AnnData object.

        Expects log-normalized data in adata.X (or specified layer).
        Run sc.pp.normalize_total() and sc.pp.log1p() beforehand.

        Parameters
        ----------
        adata : AnnData
            Single-cell RNA-seq data. Genes in .var_names, cells in .obs_names.
        verbose : bool, default=True
            Show progress bar and summary statistics.

        Returns
        -------
        list[str]
            Selected gene names, ordered by multimodal_score descending.
        """
        X = self._get_matrix(adata)
        gene_names = list(adata.var_names)
        n_cells, n_genes = X.shape

        if verbose:
            print(f"MGS: {n_cells} cells × {n_genes} genes")
            print(f"  dip_pval_threshold  = {self.dip_pval_threshold}")
            print(f"  min_bic_delta       = {self.min_bic_delta}")
            print(f"  min_peak_separation = {self.min_peak_separation}")
            print(f"  min_cell_frac       = {self.min_cell_frac}")

        records = []
        skipped_low_expr = 0

        gene_iter = tqdm(range(n_genes), desc="Scoring genes") if verbose else range(n_genes)

        for i in gene_iter:
            col = X[:, i]
            if issparse(col):
                col = col.toarray().flatten()
            else:
                col = np.asarray(col).flatten()

            nonzero = col[col > 0]
            n_expressed = len(nonzero)
            frac_expressed = n_expressed / n_cells

            # --- Step 1: Expression filter ---
            if frac_expressed < self.min_cell_frac:
                skipped_low_expr += 1
                records.append(self._empty_record(
                    gene_names[i], n_expressed, frac_expressed, "low_expression"
                ))
                continue

            # --- Step 2: Dip Test pre-filter ---
            dip_stat, dip_pval = dip_test(nonzero)

            if dip_pval > self.dip_pval_threshold:
                records.append(self._empty_record(
                    gene_names[i], n_expressed, frac_expressed, "dip_filtered",
                    dip_stat=dip_stat, dip_pval=dip_pval
                ))
                continue

            # --- Step 3: GMM k=1 vs k=2 ---
            try:
                gmm_result = fit_gmm(nonzero)
            except Exception:
                records.append(self._empty_record(
                    gene_names[i], n_expressed, frac_expressed, "gmm_error",
                    dip_stat=dip_stat, dip_pval=dip_pval
                ))
                continue

            # --- Step 4: Composite score ---
            score = multimodal_score(
                dip_pval,
                gmm_result["delta_bic"],
                gmm_result["peak_separation"],
            )

            records.append({
                "gene": gene_names[i],
                "n_expressed": n_expressed,
                "frac_expressed": frac_expressed,
                "dip_stat": dip_stat,
                "dip_pval": dip_pval,
                "delta_bic": gmm_result["delta_bic"],
                "peak_separation": gmm_result["peak_separation"],
                "bic_k1": gmm_result["bic_k1"],
                "bic_k2": gmm_result["bic_k2"],
                "best_k": gmm_result["best_k"],
                "gmm_means": str(gmm_result["means"]),
                "gmm_weights": str(gmm_result["weights"]),
                "multimodal_score": score,
                "filter_reason": "passed",
            })

        df = pd.DataFrame(records)
        df = df.sort_values("multimodal_score", ascending=False).reset_index(drop=True)
        self.results_ = df

        # --- Step 5: Apply final filters and select ---
        self.selected_genes_ = self._select(df, verbose=verbose)

        if verbose:
            print(f"\n  Skipped (low expression): {skipped_low_expr}")
            print(f"  Passed dip pre-filter:    {(df['filter_reason'] == 'passed').sum()}")
            print(f"  Selected genes (MGS):     {len(self.selected_genes_)}")

        # --- CoB rescue (if enabled) ---
        if self.cob:
            from .cob import SaciCoB
            
            cob_sel = SaciCoB(
                min_cell_frac=self.min_cell_frac,
                max_cell_frac=self.cob_max_cell_frac,
                min_jaccard=self.cob_min_jaccard,
                min_module_size=self.cob_min_module_size,
                min_cohesion=self.cob_min_cohesion,
                resolution=self.cob_resolution,
                n_target_genes=self.cob_n_target_genes,
                jaccard_start=self.cob_jaccard_start,
                jaccard_step=self.cob_jaccard_step,
                layer=self.layer,
            )
            cob_genes = cob_sel.fit(
                adata, exclude_genes=self.selected_genes_, verbose=verbose,
            )
            self.cob_selector_ = cob_sel

            # Merge: MGS genes first, then CoB rescued genes
            combined = list(dict.fromkeys(
                self.selected_genes_ + cob_genes
            ))

            if verbose:
                print(f"\n  === MGS + CoB Union ===")
                print(f"  MGS genes:     {len(self.selected_genes_)}")
                print(f"  CoB rescued:   {len(cob_genes)}")
                print(f"  Total (union): {len(combined)}")

            self.selected_genes_ = combined

        return self.selected_genes_

    def fit_transform(self, adata, verbose: bool = True):
        """
        Run fit() and add results to adata.var and adata.uns.

        Adds:
            adata.var['mgs_score']       : multimodal score (NaN if not tested)
            adata.var['mgs_selected']    : bool, True if selected
            adata.var['mgs_dip_pval']    : dip test p-value
            adata.var['mgs_delta_bic']   : BIC delta
            adata.uns['mgs_params']      : dict of parameters used
            adata.uns['mgs_genes']       : list of selected genes

        Returns
        -------
        adata (modified in place), selected_genes list
        """
        genes = self.fit(adata, verbose=verbose)

        # Map results back to adata.var
        df = self.results_.set_index("gene")
        for col, var_key in [
            ("multimodal_score", "mgs_score"),
            ("dip_pval", "mgs_dip_pval"),
            ("delta_bic", "mgs_delta_bic"),
            ("peak_separation", "mgs_peak_separation"),
        ]:
            adata.var[var_key] = df[col].reindex(adata.var_names).values

        adata.var["mgs_selected"] = adata.var_names.isin(genes)

        adata.uns["mgs_params"] = {
            "dip_pval_threshold": self.dip_pval_threshold,
            "min_bic_delta": self.min_bic_delta,
            "min_peak_separation": self.min_peak_separation,
            "min_cell_frac": self.min_cell_frac,
            "n_top_genes": self.n_top_genes,
            "cob": self.cob,
        }
        adata.uns["mgs_genes"] = genes

        # Annotate CoB results if enabled
        if self.cob and self.cob_selector_ is not None:
            cob_sel = self.cob_selector_
            cob_genes = cob_sel.rescued_genes_

            if cob_sel.results_ is not None and len(cob_sel.results_) > 0:
                cob_df = cob_sel.results_.set_index("gene")
                adata.var["cob_score"] = (
                    cob_df["cob_score"]
                    .reindex(adata.var_names).fillna(0.0).values
                )
                adata.var["cob_selected"] = adata.var_names.isin(cob_genes)
                adata.var["cob_module_id"] = (
                    cob_df["cob_module_id"]
                    .reindex(adata.var_names).fillna(-1).astype(int).values
                )
            else:
                adata.var["cob_score"] = 0.0
                adata.var["cob_selected"] = False
                adata.var["cob_module_id"] = -1

            # Determine selection source for each gene
            is_mgs = adata.var_names.isin(
                [g for g in genes if g not in cob_genes]
            )
            is_cob = adata.var_names.isin(
                [g for g in cob_genes if g not in self.results_[
                    self.results_["filter_reason"] == "passed"
                ]["gene"].tolist()]
            )
            is_both = adata.var["mgs_selected"] & adata.var["cob_selected"]
            source = pd.Series("none", index=adata.var_names)
            source[adata.var["mgs_selected"] & ~adata.var["cob_selected"]] = "mgs"
            source[~adata.var["mgs_selected"] & adata.var["cob_selected"]] = "cob"
            source[is_both] = "both"
            adata.var["selection_source"] = source.values

            adata.uns["cob_genes"] = cob_genes
            adata.uns["cob_params"] = cob_sel.results_ is not None and {
                "min_jaccard": self.cob_min_jaccard,
                "min_module_size": self.cob_min_module_size,
                "min_cohesion": self.cob_min_cohesion,
                "max_cell_frac": self.cob_max_cell_frac,
                "resolution": self.cob_resolution,
            } or {}

        return adata, genes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_matrix(self, adata):
        if self.layer is not None:
            return adata.layers[self.layer]
        return adata.X

    def _select(self, df: pd.DataFrame, verbose: bool = True) -> list[str]:
        passed = df[df["filter_reason"] == "passed"].copy()

        if self.n_top_genes is not None:
            # User explicitly wants top-N
            return passed.head(self.n_top_genes)["gene"].tolist()

        # Apply BIC and peak separation thresholds
        selected = passed[
            (passed["delta_bic"] >= self.min_bic_delta) &
            (passed["peak_separation"] >= self.min_peak_separation)
        ]

        # Fallback: if too few genes pass, relax BIC threshold
        if len(selected) < self.min_genes_fallback:
            warnings.warn(
                f"Only {len(selected)} genes passed all filters "
                f"(min_bic_delta={self.min_bic_delta}, "
                f"min_peak_separation={self.min_peak_separation}). "
                f"Falling back to top {self.min_genes_fallback} genes by score. "
                f"Consider lowering min_bic_delta or min_peak_separation.",
                UserWarning,
                stacklevel=3,
            )
            selected = passed.head(self.min_genes_fallback)

        return selected["gene"].tolist()

    @staticmethod
    def _empty_record(
        gene: str,
        n_expressed: int,
        frac_expressed: float,
        reason: str,
        dip_stat: float = np.nan,
        dip_pval: float = np.nan,
    ) -> dict:
        return {
            "gene": gene,
            "n_expressed": n_expressed,
            "frac_expressed": frac_expressed,
            "dip_stat": dip_stat,
            "dip_pval": dip_pval,
            "delta_bic": np.nan,
            "peak_separation": np.nan,
            "bic_k1": np.nan,
            "bic_k2": np.nan,
            "best_k": np.nan,
            "gmm_means": np.nan,
            "gmm_weights": np.nan,
            "multimodal_score": 0.0,
            "filter_reason": reason,
        }
