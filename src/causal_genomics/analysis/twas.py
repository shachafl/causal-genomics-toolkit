"""
Transcriptome-Wide Association Study (TWAS) Module

Implements methods to identify genes whose predicted expression is associated
with complex traits, including PrediXcan and S-PrediXcan approaches.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import cross_val_predict, KFold
from typing import Dict, List, Tuple, Optional, Union
import warnings


class TWASAnalysis:
    """
    Transcriptome-Wide Association Study analysis.

    Tests for associations between genetically predicted gene expression
    and complex traits/diseases.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize TWAS analysis.

        Parameters
        ----------
        alpha : float
            Significance threshold (will be Bonferroni corrected)
        """
        self.alpha = alpha
        self.expression_weights = {}
        self.results = None

    def train_expression_model(self,
                                genotypes: np.ndarray,
                                expression: np.ndarray,
                                gene_id: str,
                                method: str = 'elastic_net',
                                alpha: float = 0.5,
                                cv_folds: int = 5) -> Dict:
        """
        Train expression prediction model for a gene.

        Parameters
        ----------
        genotypes : array
            Genotype matrix (n_samples x n_variants)
        expression : array
            Gene expression values (n_samples,)
        gene_id : str
            Gene identifier
        method : str
            'elastic_net', 'ridge', or 'lasso'
        alpha : float
            ElasticNet mixing parameter (1=lasso, 0=ridge)
        cv_folds : int
            Number of cross-validation folds

        Returns
        -------
        dict
            Model weights, performance metrics, and cross-validated predictions
        """
        n_samples, n_variants = genotypes.shape

        # Initialize model
        if method == 'elastic_net':
            model = ElasticNet(alpha=0.1, l1_ratio=alpha, max_iter=10000)
        elif method == 'ridge':
            model = Ridge(alpha=1.0)
        elif method == 'lasso':
            model = ElasticNet(alpha=0.1, l1_ratio=1.0, max_iter=10000)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Cross-validated predictions
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_predictions = cross_val_predict(model, genotypes, expression, cv=kf)

        # Fit final model
        model.fit(genotypes, expression)
        weights = model.coef_

        # Compute R-squared
        ss_res = np.sum((expression - cv_predictions)**2)
        ss_tot = np.sum((expression - np.mean(expression))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Compute heritability estimate (cis-h2)
        r2_cv = max(0, r2)

        # Store weights
        self.expression_weights[gene_id] = {
            'weights': weights,
            'r2': r2_cv,
            'n_variants': n_variants,
            'n_nonzero': np.sum(weights != 0)
        }

        return {
            'gene': gene_id,
            'weights': weights,
            'r2_cv': r2_cv,
            'n_variants': n_variants,
            'n_nonzero_weights': np.sum(weights != 0),
            'cv_predictions': cv_predictions
        }

    def predixcan(self,
                  genotypes: np.ndarray,
                  phenotype: np.ndarray,
                  gene_weights: Dict[str, np.ndarray],
                  covariates: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Individual-level PrediXcan analysis.

        Predicts gene expression from genotypes and tests association
        with phenotype.

        Parameters
        ----------
        genotypes : dict
            Dictionary mapping gene_id to genotype matrix
        phenotype : array
            Phenotype values (n_samples,)
        gene_weights : dict
            Dictionary mapping gene_id to weight vectors
        covariates : array, optional
            Covariate matrix (n_samples x n_covariates)

        Returns
        -------
        pd.DataFrame
            Association results for each gene
        """
        results = []

        for gene_id, weights in gene_weights.items():
            if gene_id not in genotypes:
                continue

            geno = genotypes[gene_id]

            # Predict expression
            predicted_expr = geno @ weights

            # Remove covariate effects if provided
            if covariates is not None:
                # Residualize phenotype and predicted expression
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(covariates, phenotype)
                pheno_resid = phenotype - lr.predict(covariates)

                lr.fit(covariates, predicted_expr)
                expr_resid = predicted_expr - lr.predict(covariates)
            else:
                pheno_resid = phenotype
                expr_resid = predicted_expr

            # Test association
            n = len(pheno_resid)
            correlation = np.corrcoef(expr_resid, pheno_resid)[0, 1]
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2 + 1e-10))
            pval = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-2))

            # Effect size (standardized regression coefficient)
            beta = correlation

            results.append({
                'gene': gene_id,
                'beta': beta,
                't_stat': t_stat,
                'pval': pval,
                'n_samples': n
            })

        results_df = pd.DataFrame(results)

        # Multiple testing correction
        if len(results_df) > 0:
            results_df['pval_bonferroni'] = np.minimum(
                results_df['pval'] * len(results_df), 1.0
            )
            results_df['significant'] = results_df['pval_bonferroni'] < self.alpha

        self.results = results_df.sort_values('pval')
        return self.results

    def s_predixcan(self,
                    gwas_z: np.ndarray,
                    ld_matrix: np.ndarray,
                    weights: np.ndarray,
                    gene_id: str,
                    n_gwas: int,
                    model_r2: float) -> Dict:
        """
        Summary-based PrediXcan (S-PrediXcan).

        Uses GWAS summary statistics and LD reference to impute
        gene-trait associations without individual-level data.

        Parameters
        ----------
        gwas_z : array
            Z-scores from GWAS for variants in cis region
        ld_matrix : array
            LD correlation matrix for the variants
        weights : array
            Expression prediction weights
        gene_id : str
            Gene identifier
        n_gwas : int
            GWAS sample size
        model_r2 : float
            Cross-validated R2 of expression model

        Returns
        -------
        dict
            S-PrediXcan association results
        """
        # S-PrediXcan formula
        # Z_twas = (w' * Z_gwas) / sqrt(w' * Sigma * w)

        numerator = np.dot(weights, gwas_z)
        denominator = np.sqrt(np.dot(weights, np.dot(ld_matrix, weights)) + 1e-10)

        z_twas = numerator / denominator
        pval = 2 * (1 - stats.norm.cdf(np.abs(z_twas)))

        # Effect size on liability scale
        # Approximate beta
        beta = z_twas / np.sqrt(n_gwas)

        return {
            'gene': gene_id,
            'z_twas': z_twas,
            'pval': pval,
            'beta': beta,
            'model_r2': model_r2,
            'n_snps_in_model': np.sum(weights != 0)
        }

    def run_s_predixcan(self,
                        gwas_data: pd.DataFrame,
                        ld_matrices: Dict[str, np.ndarray],
                        n_gwas: int,
                        snp_col: str = 'SNP',
                        z_col: str = 'z') -> pd.DataFrame:
        """
        Run S-PrediXcan for all genes with trained models.

        Parameters
        ----------
        gwas_data : DataFrame
            GWAS summary statistics with SNP and z-score columns
        ld_matrices : dict
            Dictionary mapping gene_id to LD matrices
        n_gwas : int
            GWAS sample size
        snp_col : str
            Column name for SNP identifiers
        z_col : str
            Column name for z-scores

        Returns
        -------
        pd.DataFrame
            S-PrediXcan results for all genes
        """
        results = []

        for gene_id, gene_info in self.expression_weights.items():
            if gene_id not in ld_matrices:
                continue

            weights = gene_info['weights']
            r2 = gene_info['r2']

            # Skip genes with very low prediction quality
            if r2 < 0.01:
                continue

            ld_matrix = ld_matrices[gene_id]

            # Get z-scores for this gene's variants
            # Assume variants are in same order as weights
            n_variants = len(weights)
            gwas_z = gwas_data[z_col].values[:n_variants]

            result = self.s_predixcan(
                gwas_z, ld_matrix, weights, gene_id, n_gwas, r2
            )
            results.append(result)

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            results_df['pval_bonferroni'] = np.minimum(
                results_df['pval'] * len(results_df), 1.0
            )
            results_df['significant'] = results_df['pval_bonferroni'] < self.alpha

        self.results = results_df.sort_values('pval')
        return self.results


class MultiTissueTWAS:
    """
    Multi-tissue TWAS analysis.

    Combines evidence across multiple tissues to improve power
    and identify tissue-specific effects.
    """

    def __init__(self):
        """Initialize multi-tissue TWAS."""
        self.tissue_results = {}
        self.combined_results = None

    def add_tissue_result(self,
                          tissue: str,
                          results: pd.DataFrame):
        """
        Add TWAS results for a tissue.

        Parameters
        ----------
        tissue : str
            Tissue name
        results : DataFrame
            TWAS results with 'gene', 'z_twas', 'pval' columns
        """
        self.tissue_results[tissue] = results

    def combine_tissues_fisher(self) -> pd.DataFrame:
        """
        Combine p-values across tissues using Fisher's method.

        Returns
        -------
        pd.DataFrame
            Combined results
        """
        # Get all genes across tissues
        all_genes = set()
        for results in self.tissue_results.values():
            all_genes.update(results['gene'].tolist())

        combined = []
        for gene in all_genes:
            pvals = []
            z_scores = []
            tissues_sig = []

            for tissue, results in self.tissue_results.items():
                gene_result = results[results['gene'] == gene]
                if len(gene_result) > 0:
                    pvals.append(gene_result['pval'].values[0])
                    z_scores.append(gene_result['z_twas'].values[0])
                    if gene_result['pval'].values[0] < 0.05:
                        tissues_sig.append(tissue)

            if len(pvals) == 0:
                continue

            # Fisher's method
            chi2_stat = -2 * np.sum(np.log(np.array(pvals) + 1e-300))
            df = 2 * len(pvals)
            fisher_pval = 1 - stats.chi2.cdf(chi2_stat, df)

            combined.append({
                'gene': gene,
                'fisher_pval': fisher_pval,
                'chi2_stat': chi2_stat,
                'n_tissues': len(pvals),
                'mean_z': np.mean(z_scores),
                'max_abs_z': np.max(np.abs(z_scores)),
                'tissues_nominal_sig': ','.join(tissues_sig) if tissues_sig else 'none'
            })

        self.combined_results = pd.DataFrame(combined).sort_values('fisher_pval')
        return self.combined_results

    def combine_tissues_stouffer(self, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Combine z-scores across tissues using Stouffer's method.

        Parameters
        ----------
        weights : dict, optional
            Tissue-specific weights (e.g., based on sample size or R2)

        Returns
        -------
        pd.DataFrame
            Combined results
        """
        all_genes = set()
        for results in self.tissue_results.values():
            all_genes.update(results['gene'].tolist())

        combined = []
        for gene in all_genes:
            z_scores = []
            w = []

            for tissue, results in self.tissue_results.items():
                gene_result = results[results['gene'] == gene]
                if len(gene_result) > 0:
                    z_scores.append(gene_result['z_twas'].values[0])
                    if weights and tissue in weights:
                        w.append(weights[tissue])
                    else:
                        w.append(1.0)

            if len(z_scores) == 0:
                continue

            z_scores = np.array(z_scores)
            w = np.array(w)

            # Weighted Stouffer
            combined_z = np.sum(w * z_scores) / np.sqrt(np.sum(w**2))
            combined_pval = 2 * (1 - stats.norm.cdf(np.abs(combined_z)))

            combined.append({
                'gene': gene,
                'stouffer_z': combined_z,
                'stouffer_pval': combined_pval,
                'n_tissues': len(z_scores),
                'mean_z': np.mean(z_scores),
                'concordant': np.all(z_scores > 0) or np.all(z_scores < 0)
            })

        self.combined_results = pd.DataFrame(combined).sort_values('stouffer_pval')
        return self.combined_results

    def get_tissue_specific_genes(self,
                                   pval_threshold: float = 0.05,
                                   tissue_specificity_ratio: float = 10) -> pd.DataFrame:
        """
        Identify tissue-specific TWAS hits.

        Genes that are significant in one tissue but not others.

        Parameters
        ----------
        pval_threshold : float
            Significance threshold
        tissue_specificity_ratio : float
            Minimum ratio of p-values (other/target) for tissue specificity

        Returns
        -------
        pd.DataFrame
            Tissue-specific genes
        """
        all_genes = set()
        for results in self.tissue_results.values():
            sig_genes = results[results['pval'] < pval_threshold]['gene'].tolist()
            all_genes.update(sig_genes)

        specific_genes = []
        for gene in all_genes:
            tissue_pvals = {}

            for tissue, results in self.tissue_results.items():
                gene_result = results[results['gene'] == gene]
                if len(gene_result) > 0:
                    tissue_pvals[tissue] = gene_result['pval'].values[0]

            # Check for tissue specificity
            for target_tissue, target_pval in tissue_pvals.items():
                if target_pval >= pval_threshold:
                    continue

                other_pvals = [p for t, p in tissue_pvals.items() if t != target_tissue]
                if len(other_pvals) == 0:
                    continue

                min_other = min(other_pvals)
                ratio = min_other / (target_pval + 1e-300)

                if ratio >= tissue_specificity_ratio:
                    specific_genes.append({
                        'gene': gene,
                        'specific_tissue': target_tissue,
                        'pval': target_pval,
                        'min_other_pval': min_other,
                        'specificity_ratio': ratio
                    })

        return pd.DataFrame(specific_genes).sort_values('specificity_ratio', ascending=False)


class TWAS_FUSION:
    """
    FUSION-style TWAS analysis.

    Implements the FUSION approach for TWAS with multiple
    expression prediction models.
    """

    def __init__(self):
        """Initialize FUSION TWAS."""
        self.models = {}

    def load_fusion_weights(self,
                            weights_file: str,
                            gene_id: str) -> Dict:
        """
        Load pre-computed FUSION weights.

        Parameters
        ----------
        weights_file : str
            Path to FUSION weights file
        gene_id : str
            Gene identifier

        Returns
        -------
        dict
            Weight information
        """
        # In practice, would load from RData or similar format
        # Here we provide interface structure
        warnings.warn("FUSION weight loading requires external R packages")

        return {
            'gene': gene_id,
            'weights': None,
            'model_type': None,
            'r2': None
        }

    def compute_twas_statistic(self,
                               gwas_z: np.ndarray,
                               weights: np.ndarray,
                               ld_matrix: np.ndarray) -> Dict:
        """
        Compute FUSION TWAS test statistic.

        Parameters
        ----------
        gwas_z : array
            GWAS z-scores
        weights : array
            Expression weights
        ld_matrix : array
            LD matrix

        Returns
        -------
        dict
            TWAS test results
        """
        # TWAS statistic
        z_twas = np.dot(weights, gwas_z) / np.sqrt(
            np.dot(weights, np.dot(ld_matrix, weights)) + 1e-10
        )

        pval = 2 * (1 - stats.norm.cdf(np.abs(z_twas)))

        return {
            'z_twas': z_twas,
            'pval': pval
        }

    def conditional_twas(self,
                         gwas_z: np.ndarray,
                         gene_weights: Dict[str, np.ndarray],
                         ld_matrix: np.ndarray,
                         genes_to_condition: List[str]) -> pd.DataFrame:
        """
        Conditional TWAS analysis.

        Tests gene associations while conditioning on other genes,
        to identify independently associated genes.

        Parameters
        ----------
        gwas_z : array
            GWAS z-scores
        gene_weights : dict
            Weights for each gene
        ld_matrix : array
            LD matrix
        genes_to_condition : list
            Gene IDs to condition on

        Returns
        -------
        pd.DataFrame
            Conditional TWAS results
        """
        results = []

        # Build joint prediction matrix
        conditioning_weights = np.column_stack([
            gene_weights[g] for g in genes_to_condition if g in gene_weights
        ])

        for gene_id, weights in gene_weights.items():
            if gene_id in genes_to_condition:
                continue

            # Condition on other genes
            # Residualize TWAS statistic
            joint_weights = np.column_stack([weights.reshape(-1, 1), conditioning_weights])
            cov_matrix = joint_weights.T @ ld_matrix @ joint_weights

            if cov_matrix.shape[0] > 1:
                # Conditional variance
                cov_11 = cov_matrix[0, 0]
                cov_12 = cov_matrix[0, 1:]
                cov_22 = cov_matrix[1:, 1:]
                cov_22_inv = np.linalg.pinv(cov_22)

                cond_var = cov_11 - cov_12 @ cov_22_inv @ cov_12.T
            else:
                cond_var = cov_matrix[0, 0]

            # Marginal z-score
            z_marginal = np.dot(weights, gwas_z) / np.sqrt(
                np.dot(weights, np.dot(ld_matrix, weights)) + 1e-10
            )

            # Conditional z-score (simplified)
            z_conditional = z_marginal / np.sqrt(max(cond_var, 0.01))
            pval = 2 * (1 - stats.norm.cdf(np.abs(z_conditional)))

            results.append({
                'gene': gene_id,
                'z_marginal': z_marginal,
                'z_conditional': z_conditional,
                'pval_conditional': pval,
                'n_genes_conditioned': len(genes_to_condition)
            })

        return pd.DataFrame(results).sort_values('pval_conditional')


class ColocTWAS:
    """
    Combined colocalization and TWAS analysis.

    Integrates TWAS associations with colocalization evidence
    to prioritize causal genes.
    """

    def __init__(self):
        """Initialize ColocTWAS."""
        self.twas_results = None
        self.coloc_results = None

    def integrate_twas_coloc(self,
                              twas_df: pd.DataFrame,
                              coloc_df: pd.DataFrame,
                              twas_gene_col: str = 'gene',
                              coloc_gene_col: str = 'gene') -> pd.DataFrame:
        """
        Integrate TWAS and colocalization results.

        Parameters
        ----------
        twas_df : DataFrame
            TWAS results with gene and p-value
        coloc_df : DataFrame
            Colocalization results with gene and PP.H4
        twas_gene_col : str
            Gene column name in TWAS results
        coloc_gene_col : str
            Gene column name in colocalization results

        Returns
        -------
        pd.DataFrame
            Combined prioritization scores
        """
        # Merge results
        merged = pd.merge(
            twas_df,
            coloc_df,
            left_on=twas_gene_col,
            right_on=coloc_gene_col,
            how='outer'
        )

        # Compute combined score
        # Higher TWAS significance + higher coloc PP.H4 = higher priority
        merged['twas_score'] = -np.log10(merged['pval'].fillna(1) + 1e-300)

        if 'H4_shared_causal' in merged.columns:
            merged['coloc_score'] = merged['H4_shared_causal'].fillna(0)
        elif 'pp_h4' in merged.columns:
            merged['coloc_score'] = merged['pp_h4'].fillna(0)
        else:
            merged['coloc_score'] = 0

        # Combined score (weighted geometric mean)
        merged['combined_score'] = np.sqrt(
            merged['twas_score'] * merged['coloc_score']
        )

        # Prioritization categories
        merged['priority'] = 'low'
        merged.loc[
            (merged['pval'] < 0.05) & (merged['coloc_score'] > 0.5),
            'priority'
        ] = 'medium'
        merged.loc[
            (merged['pval'] < 0.001) & (merged['coloc_score'] > 0.8),
            'priority'
        ] = 'high'

        return merged.sort_values('combined_score', ascending=False)
