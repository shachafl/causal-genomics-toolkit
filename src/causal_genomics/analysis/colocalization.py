"""
Colocalization Analysis Module

Identifies whether two traits share the same causal variant in a genomic region.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
import itertools


class ColocalizationAnalysis:
    """
    Bayesian colocalization analysis.
    
    Tests whether two association signals in the same region are driven by
    the same causal variant (H4) or distinct variants (H3).
    """
    
    def __init__(self, 
                 p1: float = 1e-4,
                 p2: float = 1e-4, 
                 p12: float = 1e-5):
        """
        Initialize colocalization analysis.
        
        Parameters
        ----------
        p1 : float
            Prior probability that trait 1 has a causal variant in region
        p2 : float
            Prior probability that trait 2 has a causal variant in region
        p12 : float
            Prior probability that both traits share the same causal variant
        """
        self.p1 = p1
        self.p2 = p2
        self.p12 = p12
        
    def coloc_abf(self, 
                  trait1_beta: np.ndarray,
                  trait1_se: np.ndarray,
                  trait2_beta: np.ndarray,
                  trait2_se: np.ndarray,
                  trait1_type: str = 'quant',
                  trait2_type: str = 'quant',
                  trait1_n: Optional[int] = None,
                  trait2_n: Optional[int] = None,
                  trait1_s: Optional[float] = None,
                  trait2_s: Optional[float] = None) -> Dict:
        """
        Approximate Bayes Factor colocalization.
        
        Parameters
        ----------
        trait1_beta, trait2_beta : array
            Effect size estimates
        trait1_se, trait2_se : array
            Standard errors
        trait1_type, trait2_type : str
            'quant' or 'cc' (case-control)
        trait1_n, trait2_n : int
            Sample sizes
        trait1_s, trait2_s : float
            Proportion of cases (for case-control)
        
        Returns
        -------
        dict
            Posterior probabilities for each hypothesis:
            H0: no causal variant for either trait
            H1: causal variant for trait 1 only
            H2: causal variant for trait 2 only
            H3: distinct causal variants for each trait
            H4: shared causal variant
        """
        n_snps = len(trait1_beta)
        
        # Calculate approximate Bayes factors for each SNP
        abf1 = self._calculate_abf(trait1_beta, trait1_se, trait1_type, 
                                    trait1_n, trait1_s)
        abf2 = self._calculate_abf(trait2_beta, trait2_se, trait2_type,
                                    trait2_n, trait2_s)
        
        # Prior probabilities
        prior_h0 = 1 - self.p1 - self.p2 + self.p1 * self.p2
        prior_h1 = self.p1 - self.p12
        prior_h2 = self.p2 - self.p12
        prior_h3 = self.p12
        prior_h4 = self.p12
        
        # Calculate likelihood for each hypothesis
        likelihood_h0 = 1
        likelihood_h1 = np.sum(abf1) / n_snps
        likelihood_h2 = np.sum(abf2) / n_snps
        likelihood_h3 = np.sum(abf1 * abf2) / n_snps
        likelihood_h4 = np.sum(abf1 * abf2) / n_snps
        
        # Calculate posterior probabilities
        numerators = np.array([
            prior_h0 * likelihood_h0,
            prior_h1 * likelihood_h1,
            prior_h2 * likelihood_h2,
            prior_h3 * likelihood_h3,
            prior_h4 * likelihood_h4
        ])
        
        denominator = np.sum(numerators)
        posteriors = numerators / denominator
        
        return {
            'H0_no_association': posteriors[0],
            'H1_trait1_only': posteriors[1],
            'H2_trait2_only': posteriors[2],
            'H3_distinct_causal': posteriors[3],
            'H4_shared_causal': posteriors[4],
            'n_snps': n_snps
        }
    
    def _calculate_abf(self, 
                       beta: np.ndarray, 
                       se: np.ndarray,
                       trait_type: str,
                       n: Optional[int] = None,
                       s: Optional[float] = None) -> np.ndarray:
        """
        Calculate approximate Bayes factors.
        
        Uses Wakefield's approximation.
        """
        if trait_type == 'quant':
            # Prior variance for quantitative trait
            V = 0.15
        else:
            # Prior variance for case-control
            if s is None:
                raise ValueError("Must provide proportion of cases (s) for case-control")
            V = 0.2 / (s * (1 - s))
        
        # Wakefield approximation
        z = beta / se
        r = V / (V + se**2)
        
        abf = np.sqrt(1 - r) * np.exp(r * z**2 / 2)
        
        return abf
    
    def ecaviar(self,
                trait1_zscore: np.ndarray,
                trait2_zscore: np.ndarray,
                ld_matrix: np.ndarray,
                causal_config_threshold: float = 0.01) -> Dict:
        """
        eCAVIAR method for fine-mapping and colocalization.
        
        Identifies causal variants and quantifies colocalization probability.
        
        Parameters
        ----------
        trait1_zscore, trait2_zscore : array
            Z-scores for each variant
        ld_matrix : array
            Linkage disequilibrium correlation matrix
        causal_config_threshold : float
            Threshold for reporting causal configurations
        
        Returns
        -------
        dict
            CLPP (colocalization posterior probability) and causal configurations
        """
        n_snps = len(trait1_zscore)
        
        # Calculate marginal likelihoods for each SNP being causal
        trait1_posterior = self._calculate_single_causal_posterior(
            trait1_zscore, ld_matrix
        )
        trait2_posterior = self._calculate_single_causal_posterior(
            trait2_zscore, ld_matrix
        )
        
        # Calculate CLPP (colocalization posterior probability)
        clpp = np.sum(trait1_posterior * trait2_posterior)
        
        # Identify credible sets
        trait1_credible = self._get_credible_set(trait1_posterior)
        trait2_credible = self._get_credible_set(trait2_posterior)
        
        return {
            'CLPP': clpp,
            'trait1_credible_set': trait1_credible,
            'trait2_credible_set': trait2_credible,
            'shared_variants': list(set(trait1_credible) & set(trait2_credible))
        }
    
    def _calculate_single_causal_posterior(self,
                                          zscore: np.ndarray,
                                          ld_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate posterior probability for each SNP being causal.
        
        Assumes single causal variant model.
        """
        n_snps = len(zscore)
        
        # Prior probability
        prior = 1.0 / n_snps
        
        # Calculate Bayes factors
        bf = np.exp(0.5 * zscore**2)
        
        # Account for LD structure
        # Simplified calculation - full implementation would use matrix inversion
        posterior = bf * prior
        posterior = posterior / np.sum(posterior)
        
        return posterior
    
    def _get_credible_set(self, 
                         posterior: np.ndarray,
                         credible_level: float = 0.95) -> list:
        """
        Get 95% credible set of variants.
        
        Returns indices of variants that comprise the credible set.
        """
        sorted_indices = np.argsort(posterior)[::-1]
        cumsum = np.cumsum(posterior[sorted_indices])
        
        credible_set_size = np.searchsorted(cumsum, credible_level) + 1
        credible_set = sorted_indices[:credible_set_size].tolist()
        
        return credible_set
    
    def smr_heidi(self,
                  eqtl_beta: np.ndarray,
                  eqtl_se: np.ndarray,
                  gwas_beta: np.ndarray,
                  gwas_se: np.ndarray,
                  ld_matrix: np.ndarray,
                  top_snp_idx: int = 0) -> Dict:
        """
        SMR (Summary-data-based Mendelian Randomization) with HEIDI test.
        
        Tests whether gene expression is causally associated with trait,
        and whether association is driven by a single variant (HEIDI test).
        
        Parameters
        ----------
        eqtl_beta, gwas_beta : array
            Effect sizes for eQTL and GWAS
        eqtl_se, gwas_se : array
            Standard errors
        ld_matrix : array
            LD correlation matrix
        top_snp_idx : int
            Index of top eQTL variant
        
        Returns
        -------
        dict
            SMR test result and HEIDI heterogeneity test
        """
        # SMR test
        b_xy = gwas_beta[top_snp_idx] / eqtl_beta[top_snp_idx]
        se_xy = np.sqrt((gwas_se[top_snp_idx]**2 / eqtl_beta[top_snp_idx]**2) + 
                       (gwas_beta[top_snp_idx]**2 * eqtl_se[top_snp_idx]**2 / 
                        eqtl_beta[top_snp_idx]**4))
        
        z_smr = b_xy / se_xy
        p_smr = 2 * (1 - stats.norm.cdf(abs(z_smr)))
        
        # HEIDI test - tests for heterogeneity
        # Select SNPs in LD with top SNP
        ld_with_top = ld_matrix[top_snp_idx, :]
        heidi_snps = np.where(np.abs(ld_with_top) > 0.05)[0]
        heidi_snps = heidi_snps[heidi_snps != top_snp_idx]
        
        if len(heidi_snps) < 3:
            heidi_result = {
                'heidi_pval': np.nan,
                'n_heidi_snps': len(heidi_snps),
                'note': 'Insufficient SNPs for HEIDI test'
            }
        else:
            # Calculate heterogeneity statistic
            ratios = gwas_beta / eqtl_beta
            expected_ratio = b_xy
            
            deviations = ratios[heidi_snps] - expected_ratio
            weights = 1 / (gwas_se[heidi_snps]**2 / eqtl_beta[heidi_snps]**2 + 
                          gwas_beta[heidi_snps]**2 * eqtl_se[heidi_snps]**2 / 
                          eqtl_beta[heidi_snps]**4)
            
            heidi_stat = np.sum(weights * deviations**2)
            df = len(heidi_snps) - 1
            heidi_pval = 1 - stats.chi2.cdf(heidi_stat, df)
            
            heidi_result = {
                'heidi_pval': heidi_pval,
                'heidi_stat': heidi_stat,
                'n_heidi_snps': len(heidi_snps)
            }
        
        return {
            'smr_beta': b_xy,
            'smr_se': se_xy,
            'smr_pval': p_smr,
            **heidi_result
        }


class MultiTraitColocalization:
    """
    Colocalization analysis for multiple traits.
    
    Identifies sets of traits sharing causal variants.
    """
    
    def __init__(self):
        self.traits = {}
        
    def add_trait(self, name: str, beta: np.ndarray, se: np.ndarray):
        """Add a trait to the analysis."""
        self.traits[name] = {'beta': beta, 'se': se}
    
    def pairwise_coloc(self) -> pd.DataFrame:
        """
        Perform pairwise colocalization for all trait pairs.
        
        Returns
        -------
        pd.DataFrame
            Pairwise colocalization results
        """
        results = []
        coloc = ColocalizationAnalysis()
        
        trait_names = list(self.traits.keys())
        
        for trait1, trait2 in itertools.combinations(trait_names, 2):
            result = coloc.coloc_abf(
                self.traits[trait1]['beta'],
                self.traits[trait1]['se'],
                self.traits[trait2]['beta'],
                self.traits[trait2]['se']
            )
            
            results.append({
                'trait1': trait1,
                'trait2': trait2,
                **result
            })
        
        return pd.DataFrame(results)
    
    def cluster_traits(self, pp4_threshold: float = 0.8) -> Dict:
        """
        Cluster traits based on colocalization evidence.
        
        Groups traits that share causal variants.
        
        Parameters
        ----------
        pp4_threshold : float
            PP.H4 threshold for considering traits colocalized
        
        Returns
        -------
        dict
            Clusters of colocalized traits
        """
        pairwise_results = self.pairwise_coloc()
        
        # Build adjacency matrix
        trait_names = list(self.traits.keys())
        n_traits = len(trait_names)
        adj_matrix = np.zeros((n_traits, n_traits))
        
        for _, row in pairwise_results.iterrows():
            i = trait_names.index(row['trait1'])
            j = trait_names.index(row['trait2'])
            
            if row['H4_shared_causal'] >= pp4_threshold:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        
        # Simple clustering - connected components
        # In practice, would use more sophisticated graph clustering
        clusters = []
        visited = set()
        
        for i in range(n_traits):
            if i not in visited:
                cluster = self._dfs_cluster(i, adj_matrix, visited)
                clusters.append([trait_names[idx] for idx in cluster])
        
        return {'clusters': clusters, 'n_clusters': len(clusters)}
    
    def _dfs_cluster(self, start: int, adj_matrix: np.ndarray, 
                     visited: set) -> list:
        """Depth-first search to find connected component."""
        stack = [start]
        cluster = []
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                cluster.append(node)
                
                # Add neighbors
                neighbors = np.where(adj_matrix[node, :] > 0)[0]
                stack.extend([n for n in neighbors if n not in visited])
        
        return cluster
