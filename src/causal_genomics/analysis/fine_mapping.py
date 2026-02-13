"""
Fine-Mapping Analysis Module

Statistical fine-mapping to identify causal variants within associated genomic loci.
Implements SuSiE (Sum of Single Effects) and related methods.
"""

import numpy as np
import pandas as pd
from scipy import stats, linalg
from typing import Dict, List, Tuple, Optional, Union
import warnings


class FineMappingAnalysis:
    """
    Statistical fine-mapping for causal variant identification.

    Uses Bayesian approaches to identify the most likely causal variants
    within a GWAS-associated locus, accounting for linkage disequilibrium.
    """

    def __init__(self,
                 max_causal: int = 10,
                 coverage: float = 0.95,
                 min_abs_corr: float = 0.5):
        """
        Initialize fine-mapping analysis.

        Parameters
        ----------
        max_causal : int
            Maximum number of causal variants to consider (L in SuSiE)
        coverage : float
            Target coverage for credible sets (default 95%)
        min_abs_corr : float
            Minimum absolute correlation for credible set purity
        """
        self.max_causal = max_causal
        self.coverage = coverage
        self.min_abs_corr = min_abs_corr

    def susie(self,
              z_scores: np.ndarray,
              ld_matrix: np.ndarray,
              n: int,
              var_y: float = 1.0,
              prior_variance: float = 0.2,
              max_iter: int = 100,
              tol: float = 1e-3) -> Dict:
        """
        Sum of Single Effects (SuSiE) fine-mapping.

        Fits a sparse regression model where each component has exactly one
        non-zero effect, allowing for multiple causal variants.

        Parameters
        ----------
        z_scores : array
            Z-scores from GWAS for each variant
        ld_matrix : array
            Linkage disequilibrium correlation matrix (R)
        n : int
            Sample size
        var_y : float
            Variance of the phenotype (typically 1 for standardized)
        prior_variance : float
            Prior variance on effect sizes
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance

        Returns
        -------
        dict
            Contains:
            - pip: Posterior inclusion probabilities for each variant
            - credible_sets: List of credible sets
            - alpha: L x p matrix of posterior probabilities for each effect
            - mu: L x p matrix of posterior means
            - lbf: Log Bayes factors for each effect
        """
        p = len(z_scores)
        L = self.max_causal

        # Convert z-scores to effect estimates
        # beta_hat = z / sqrt(n), se = 1 / sqrt(n)
        beta_hat = z_scores / np.sqrt(n)
        se = 1.0 / np.sqrt(n)

        # Initialize
        alpha = np.ones((L, p)) / p  # Posterior probabilities
        mu = np.zeros((L, p))  # Posterior means
        mu2 = np.zeros((L, p))  # E[beta^2]

        # Precompute XtX approximation from LD
        XtX = ld_matrix * n
        Xty = z_scores * np.sqrt(n) * np.sqrt(var_y)

        # Iterative Bayesian Stepwise Selection (IBSS)
        prev_elbo = -np.inf

        for iteration in range(max_iter):
            for l in range(L):
                # Remove effect l from residual
                residual_Xty = Xty.copy()
                for k in range(L):
                    if k != l:
                        residual_Xty -= XtX @ (alpha[k] * mu[k])

                # Single effect regression
                alpha[l], mu[l], mu2[l], lbf_l = self._single_effect_regression(
                    XtX, residual_Xty, var_y, n, prior_variance
                )

            # Check convergence via ELBO
            elbo = self._compute_elbo(alpha, mu, mu2, XtX, Xty, var_y, n, prior_variance)

            if abs(elbo - prev_elbo) < tol:
                break
            prev_elbo = elbo

        # Compute PIPs
        pip = 1 - np.prod(1 - alpha, axis=0)

        # Get credible sets
        credible_sets = self._get_credible_sets(alpha, ld_matrix)

        # Log Bayes factors for each effect
        lbf = np.zeros(L)
        for l in range(L):
            lbf[l] = np.log(np.sum(np.exp(self._compute_lbf_vector(
                XtX.diagonal(), Xty, var_y, n, prior_variance
            ))))

        return {
            'pip': pip,
            'credible_sets': credible_sets,
            'alpha': alpha,
            'mu': mu,
            'lbf': lbf,
            'n_iterations': iteration + 1,
            'converged': iteration < max_iter - 1
        }

    def _single_effect_regression(self,
                                   XtX: np.ndarray,
                                   Xty: np.ndarray,
                                   var_y: float,
                                   n: int,
                                   prior_variance: float) -> Tuple:
        """
        Fit single-effect regression for one SuSiE component.

        Returns posterior probabilities, means, and second moments.
        """
        p = len(Xty)

        # Posterior variance for each SNP
        diagXtX = np.diag(XtX)
        sigma2 = var_y  # Residual variance estimate

        post_var = 1.0 / (diagXtX / sigma2 + 1.0 / prior_variance)

        # Posterior mean for each SNP
        post_mean = post_var * Xty / sigma2

        # Log Bayes factor for each SNP
        lbf = self._compute_lbf_vector(diagXtX, Xty, var_y, n, prior_variance)

        # Posterior probabilities (softmax of log BFs)
        max_lbf = np.max(lbf)
        alpha = np.exp(lbf - max_lbf)
        alpha = alpha / np.sum(alpha)

        # Second moment
        mu2 = post_var + post_mean**2

        return alpha, post_mean, mu2, lbf

    def _compute_lbf_vector(self,
                            diagXtX: np.ndarray,
                            Xty: np.ndarray,
                            var_y: float,
                            n: int,
                            prior_variance: float) -> np.ndarray:
        """Compute log Bayes factors for each variant."""
        sigma2 = var_y

        # Standard error
        se2 = sigma2 / diagXtX

        # Log Bayes factor using Wakefield approximation
        z2 = (Xty / np.sqrt(diagXtX * sigma2))**2

        lbf = 0.5 * (np.log(se2 / (se2 + prior_variance)) +
                     z2 * prior_variance / (se2 + prior_variance))

        return lbf

    def _compute_elbo(self,
                      alpha: np.ndarray,
                      mu: np.ndarray,
                      mu2: np.ndarray,
                      XtX: np.ndarray,
                      Xty: np.ndarray,
                      var_y: float,
                      n: int,
                      prior_variance: float) -> float:
        """Compute evidence lower bound for convergence check."""
        L, p = alpha.shape

        # Expected log likelihood
        expected_ll = -n/2 * np.log(2 * np.pi * var_y)

        # KL divergence term (simplified)
        kl = 0
        for l in range(L):
            kl += np.sum(alpha[l] * np.log(alpha[l] + 1e-10) + alpha[l] * np.log(p))

        return expected_ll - kl

    def _get_credible_sets(self,
                           alpha: np.ndarray,
                           ld_matrix: np.ndarray) -> List[Dict]:
        """
        Extract credible sets from SuSiE posterior.

        A credible set is the smallest set of variants that contains
        the causal variant with probability >= coverage.
        """
        L, p = alpha.shape
        credible_sets = []

        for l in range(L):
            # Skip effects with low evidence
            if np.max(alpha[l]) < 0.01:
                continue

            # Sort variants by posterior probability
            sorted_idx = np.argsort(alpha[l])[::-1]
            cumsum = np.cumsum(alpha[l, sorted_idx])

            # Find minimum set for target coverage
            cs_size = np.searchsorted(cumsum, self.coverage) + 1
            cs_indices = sorted_idx[:cs_size].tolist()

            # Check purity (minimum pairwise correlation)
            if len(cs_indices) > 1:
                cs_ld = ld_matrix[np.ix_(cs_indices, cs_indices)]
                min_corr = np.min(np.abs(cs_ld))
            else:
                min_corr = 1.0

            credible_sets.append({
                'effect_idx': l,
                'variants': cs_indices,
                'coverage': cumsum[cs_size - 1] if cs_size <= len(cumsum) else cumsum[-1],
                'size': len(cs_indices),
                'purity': min_corr,
                'is_pure': min_corr >= self.min_abs_corr
            })

        return credible_sets

    def abf_finemap(self,
                    beta: np.ndarray,
                    se: np.ndarray,
                    prior_variance: float = 0.04) -> Dict:
        """
        Approximate Bayes Factor fine-mapping.

        Simpler method using Wakefield's ABF for single causal variant.

        Parameters
        ----------
        beta : array
            Effect size estimates
        se : array
            Standard errors
        prior_variance : float
            Prior variance on true effect size (W in original paper)

        Returns
        -------
        dict
            PIPs and credible set under single causal variant assumption
        """
        # Compute ABF for each variant
        z2 = (beta / se)**2
        v = se**2

        # Log approximate Bayes factor
        lbf = 0.5 * (np.log(v / (v + prior_variance)) +
                     z2 * prior_variance / (v + prior_variance))

        # Convert to posterior probabilities (uniform prior)
        max_lbf = np.max(lbf)
        pip = np.exp(lbf - max_lbf)
        pip = pip / np.sum(pip)

        # Get credible set
        sorted_idx = np.argsort(pip)[::-1]
        cumsum = np.cumsum(pip[sorted_idx])
        cs_size = np.searchsorted(cumsum, self.coverage) + 1
        credible_set = sorted_idx[:cs_size].tolist()

        return {
            'pip': pip,
            'lbf': lbf,
            'credible_set': credible_set,
            'credible_set_size': len(credible_set)
        }

    def conditional_analysis(self,
                             z_scores: np.ndarray,
                             ld_matrix: np.ndarray,
                             pval_threshold: float = 5e-8,
                             max_signals: int = 10) -> Dict:
        """
        Stepwise conditional analysis for multiple independent signals.

        Iteratively conditions on lead variants to identify additional
        independent association signals.

        Parameters
        ----------
        z_scores : array
            Z-scores from GWAS
        ld_matrix : array
            LD correlation matrix
        pval_threshold : float
            Significance threshold for declaring signals
        max_signals : int
            Maximum number of signals to identify

        Returns
        -------
        dict
            Lead variants and conditional z-scores
        """
        p = len(z_scores)
        residual_z = z_scores.copy()
        lead_variants = []
        conditional_results = []

        for signal_idx in range(max_signals):
            # Find lead variant
            abs_z = np.abs(residual_z)
            lead_idx = np.argmax(abs_z)
            lead_z = residual_z[lead_idx]
            lead_pval = 2 * (1 - stats.norm.cdf(np.abs(lead_z)))

            if lead_pval >= pval_threshold:
                break

            lead_variants.append({
                'signal': signal_idx + 1,
                'variant_idx': lead_idx,
                'z_score': lead_z,
                'pval': lead_pval
            })

            # Condition on lead variant
            ld_with_lead = ld_matrix[lead_idx, :]
            residual_z = residual_z - ld_with_lead * lead_z

            # Store conditional z-scores
            conditional_results.append(residual_z.copy())

        return {
            'n_signals': len(lead_variants),
            'lead_variants': lead_variants,
            'conditional_z': conditional_results
        }


class MultiLocusFineMapping:
    """
    Fine-mapping across multiple loci simultaneously.

    Handles genome-wide fine-mapping by iterating over independent loci.
    """

    def __init__(self, coverage: float = 0.95):
        """
        Initialize multi-locus fine-mapping.

        Parameters
        ----------
        coverage : float
            Target coverage for credible sets
        """
        self.coverage = coverage
        self.results = {}

    def add_locus(self,
                  locus_name: str,
                  z_scores: np.ndarray,
                  ld_matrix: np.ndarray,
                  variant_ids: Optional[List[str]] = None):
        """
        Add a locus for fine-mapping.

        Parameters
        ----------
        locus_name : str
            Identifier for the locus (e.g., chr:start-end)
        z_scores : array
            Z-scores for variants in locus
        ld_matrix : array
            LD matrix for the locus
        variant_ids : list, optional
            Identifiers for each variant
        """
        self.results[locus_name] = {
            'z_scores': z_scores,
            'ld_matrix': ld_matrix,
            'variant_ids': variant_ids or [f"var_{i}" for i in range(len(z_scores))],
            'finemapped': False
        }

    def run_finemapping(self,
                        method: str = 'susie',
                        n: Optional[int] = None,
                        **kwargs) -> pd.DataFrame:
        """
        Run fine-mapping across all loci.

        Parameters
        ----------
        method : str
            'susie' or 'abf'
        n : int
            Sample size (required for SuSiE)
        **kwargs
            Additional arguments passed to fine-mapping method

        Returns
        -------
        pd.DataFrame
            Combined results across all loci
        """
        all_results = []
        fm = FineMappingAnalysis(coverage=self.coverage)

        for locus_name, locus_data in self.results.items():
            if method == 'susie':
                if n is None:
                    raise ValueError("Sample size n required for SuSiE")
                result = fm.susie(
                    locus_data['z_scores'],
                    locus_data['ld_matrix'],
                    n=n,
                    **kwargs
                )
            elif method == 'abf':
                # Convert z-scores to beta/se
                beta = locus_data['z_scores'] / np.sqrt(n) if n else locus_data['z_scores']
                se = np.ones_like(beta) / np.sqrt(n) if n else np.ones_like(beta)
                result = fm.abf_finemap(beta, se, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Store results
            locus_data['result'] = result
            locus_data['finemapped'] = True

            # Create result DataFrame
            for i, var_id in enumerate(locus_data['variant_ids']):
                all_results.append({
                    'locus': locus_name,
                    'variant': var_id,
                    'pip': result['pip'][i],
                    'z_score': locus_data['z_scores'][i]
                })

        return pd.DataFrame(all_results)

    def get_high_pip_variants(self, pip_threshold: float = 0.5) -> pd.DataFrame:
        """
        Get variants with high posterior inclusion probability.

        Parameters
        ----------
        pip_threshold : float
            Minimum PIP to include

        Returns
        -------
        pd.DataFrame
            High-confidence causal variants
        """
        results = []

        for locus_name, locus_data in self.results.items():
            if not locus_data['finemapped']:
                continue

            pip = locus_data['result']['pip']
            high_pip_idx = np.where(pip >= pip_threshold)[0]

            for idx in high_pip_idx:
                results.append({
                    'locus': locus_name,
                    'variant': locus_data['variant_ids'][idx],
                    'pip': pip[idx],
                    'z_score': locus_data['z_scores'][idx]
                })

        return pd.DataFrame(results).sort_values('pip', ascending=False)

    def get_credible_sets_summary(self) -> pd.DataFrame:
        """
        Get summary of all credible sets across loci.

        Returns
        -------
        pd.DataFrame
            Credible set information
        """
        results = []

        for locus_name, locus_data in self.results.items():
            if not locus_data['finemapped'] or 'credible_sets' not in locus_data['result']:
                continue

            for cs in locus_data['result']['credible_sets']:
                variant_names = [locus_data['variant_ids'][i] for i in cs['variants']]
                results.append({
                    'locus': locus_name,
                    'cs_id': cs['effect_idx'],
                    'size': cs['size'],
                    'coverage': cs['coverage'],
                    'purity': cs['purity'],
                    'is_pure': cs['is_pure'],
                    'variants': ','.join(variant_names)
                })

        return pd.DataFrame(results)


class AnnotationEnrichedFineMaping:
    """
    Fine-mapping with functional annotation priors.

    Incorporates functional annotations (e.g., coding, regulatory)
    to improve fine-mapping resolution.
    """

    def __init__(self):
        """Initialize annotation-enriched fine-mapping."""
        self.annotation_weights = {}

    def set_annotation_prior(self,
                              annotations: pd.DataFrame,
                              enrichment_scores: Dict[str, float]):
        """
        Set annotation-based prior probabilities.

        Parameters
        ----------
        annotations : DataFrame
            Binary matrix of variant x annotation
        enrichment_scores : dict
            Log-odds enrichment for each annotation category
        """
        self.annotations = annotations
        self.enrichment_scores = enrichment_scores

        # Calculate prior weights
        log_prior = np.zeros(len(annotations))
        for annot, score in enrichment_scores.items():
            if annot in annotations.columns:
                log_prior += annotations[annot].values * score

        # Convert to probabilities
        self.prior_weights = np.exp(log_prior)
        self.prior_weights = self.prior_weights / np.sum(self.prior_weights)

    def finemap_with_priors(self,
                            beta: np.ndarray,
                            se: np.ndarray,
                            prior_variance: float = 0.04) -> Dict:
        """
        ABF fine-mapping with annotation priors.

        Parameters
        ----------
        beta : array
            Effect size estimates
        se : array
            Standard errors
        prior_variance : float
            Prior variance on effect sizes

        Returns
        -------
        dict
            PIPs incorporating annotation information
        """
        # Compute Bayes factors
        z2 = (beta / se)**2
        v = se**2

        lbf = 0.5 * (np.log(v / (v + prior_variance)) +
                     z2 * prior_variance / (v + prior_variance))

        # Incorporate annotation priors
        log_posterior = lbf + np.log(self.prior_weights + 1e-10)

        # Normalize
        max_log = np.max(log_posterior)
        pip = np.exp(log_posterior - max_log)
        pip = pip / np.sum(pip)

        # Compare to uniform prior
        pip_uniform = np.exp(lbf - np.max(lbf))
        pip_uniform = pip_uniform / np.sum(pip_uniform)

        return {
            'pip': pip,
            'pip_uniform': pip_uniform,
            'lbf': lbf,
            'prior_weights': self.prior_weights,
            'enrichment_effect': pip - pip_uniform
        }
