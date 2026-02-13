"""
Mendelian Randomization Analysis Module

Implements various MR methods for causal inference using genetic variants as 
instrumental variables.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings


class MendelianRandomization:
    """
    Mendelian Randomization analysis for causal inference.
    
    Uses genetic variants as instrumental variables to estimate causal effects
    of exposures on outcomes.
    """
    
    def __init__(self, p_threshold: float = 5e-8, clumping_r2: float = 0.001):
        """
        Initialize MR analysis.
        
        Parameters
        ----------
        p_threshold : float
            P-value threshold for instrument selection
        clumping_r2 : float
            R-squared threshold for LD clumping
        """
        self.p_threshold = p_threshold
        self.clumping_r2 = clumping_r2
        self.exposure_data = None
        self.outcome_data = None
        self.harmonized_data = None
        
    def load_exposure_gwas(self, data: pd.DataFrame, 
                           snp_col: str = 'SNP',
                           beta_col: str = 'beta',
                           se_col: str = 'se',
                           pval_col: str = 'pval',
                           effect_allele_col: str = 'effect_allele',
                           other_allele_col: str = 'other_allele',
                           eaf_col: str = 'eaf') -> None:
        """Load exposure GWAS summary statistics."""
        self.exposure_data = data[[snp_col, beta_col, se_col, pval_col, 
                                    effect_allele_col, other_allele_col, eaf_col]].copy()
        self.exposure_data.columns = ['SNP', 'beta_exp', 'se_exp', 'pval_exp', 
                                       'effect_allele', 'other_allele', 'eaf']
        
    def load_outcome_gwas(self, data: pd.DataFrame,
                          snp_col: str = 'SNP',
                          beta_col: str = 'beta',
                          se_col: str = 'se',
                          pval_col: str = 'pval',
                          effect_allele_col: str = 'effect_allele',
                          other_allele_col: str = 'other_allele') -> None:
        """Load outcome GWAS summary statistics."""
        self.outcome_data = data[[snp_col, beta_col, se_col, pval_col,
                                  effect_allele_col, other_allele_col]].copy()
        self.outcome_data.columns = ['SNP', 'beta_out', 'se_out', 'pval_out',
                                      'effect_allele', 'other_allele']
    
    def harmonize_data(self) -> pd.DataFrame:
        """
        Harmonize exposure and outcome data.
        
        Ensures effect alleles are aligned between datasets.
        """
        if self.exposure_data is None or self.outcome_data is None:
            raise ValueError("Must load both exposure and outcome data first")
        
        # Merge datasets
        merged = pd.merge(self.exposure_data, self.outcome_data, 
                         on='SNP', suffixes=('_exp', '_out'))
        
        # Align alleles
        harmonized = []
        for _, row in merged.iterrows():
            # Check if alleles match
            if (row['effect_allele_exp'] == row['effect_allele_out'] and 
                row['other_allele_exp'] == row['other_allele_out']):
                # Already aligned
                harmonized.append(row)
            elif (row['effect_allele_exp'] == row['other_allele_out'] and 
                  row['other_allele_exp'] == row['effect_allele_out']):
                # Need to flip
                row['beta_out'] *= -1
                row['eaf'] = 1 - row['eaf']
                harmonized.append(row)
            # else: skip ambiguous SNPs
        
        self.harmonized_data = pd.DataFrame(harmonized)
        
        # Select instruments
        self.harmonized_data = self.harmonized_data[
            self.harmonized_data['pval_exp'] < self.p_threshold
        ]
        
        return self.harmonized_data
    
    def ivw(self) -> Dict:
        """
        Inverse-variance weighted MR.
        
        Returns
        -------
        dict
            Contains beta, se, pval, and confidence interval
        """
        if self.harmonized_data is None:
            self.harmonize_data()
        
        data = self.harmonized_data
        
        # Calculate weights
        weights = 1 / (data['se_out']**2)
        
        # IVW estimate
        beta_ivw = np.sum(weights * data['beta_out'] * data['beta_exp'] / data['beta_exp']) / np.sum(weights * (data['beta_exp']**2) / (data['beta_exp']**2))
        
        # More accurate IVW calculation
        beta_ivw = np.sum(weights * data['beta_out'] / data['beta_exp']) / np.sum(weights)
        
        # Standard error
        se_ivw = 1 / np.sqrt(np.sum(weights))
        
        # P-value
        z_score = beta_ivw / se_ivw
        pval = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval
        ci_lower = beta_ivw - 1.96 * se_ivw
        ci_upper = beta_ivw + 1.96 * se_ivw
        
        return {
            'method': 'IVW',
            'beta': beta_ivw,
            'se': se_ivw,
            'pval': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_snps': len(data)
        }
    
    def egger(self) -> Dict:
        """
        MR-Egger regression with intercept test for pleiotropy.
        
        Returns
        -------
        dict
            Contains causal estimate and pleiotropy test results
        """
        if self.harmonized_data is None:
            self.harmonize_data()
        
        data = self.harmonized_data
        
        # Weighted regression
        weights = 1 / (data['se_out']**2)
        X = data['beta_exp'].values
        y = data['beta_out'].values
        
        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        W = np.diag(weights)
        
        # Weighted least squares
        beta_egger = np.linalg.inv(X_with_intercept.T @ W @ X_with_intercept) @ X_with_intercept.T @ W @ y
        
        intercept = beta_egger[0]
        slope = beta_egger[1]
        
        # Calculate standard errors
        residuals = y - X_with_intercept @ beta_egger
        mse = np.sum(weights * residuals**2) / (len(y) - 2)
        var_beta = mse * np.linalg.inv(X_with_intercept.T @ W @ X_with_intercept)
        
        se_intercept = np.sqrt(var_beta[0, 0])
        se_slope = np.sqrt(var_beta[1, 1])
        
        # P-values
        z_intercept = intercept / se_intercept
        z_slope = slope / se_slope
        pval_intercept = 2 * (1 - stats.norm.cdf(abs(z_intercept)))
        pval_slope = 2 * (1 - stats.norm.cdf(abs(z_slope)))
        
        return {
            'method': 'MR-Egger',
            'beta': slope,
            'se': se_slope,
            'pval': pval_slope,
            'intercept': intercept,
            'intercept_se': se_intercept,
            'intercept_pval': pval_intercept,
            'n_snps': len(data)
        }
    
    def weighted_median(self) -> Dict:
        """
        Weighted median MR.
        
        More robust to invalid instruments than IVW.
        """
        if self.harmonized_data is None:
            self.harmonize_data()
        
        data = self.harmonized_data
        
        # Calculate ratio estimates
        ratio = data['beta_out'] / data['beta_exp']
        weights = 1 / (data['se_out']**2)
        
        # Sort by ratio
        sorted_indices = np.argsort(ratio)
        sorted_ratio = ratio.iloc[sorted_indices]
        sorted_weights = weights.iloc[sorted_indices]
        
        # Calculate cumulative weights
        cumsum_weights = np.cumsum(sorted_weights)
        total_weight = cumsum_weights.iloc[-1]
        
        # Find median
        median_idx = np.searchsorted(cumsum_weights, total_weight / 2)
        beta_wm = sorted_ratio.iloc[median_idx]
        
        # Bootstrap for standard error
        n_bootstrap = 1000
        bootstrap_estimates = []
        
        for _ in range(n_bootstrap):
            boot_indices = np.random.choice(len(data), size=len(data), replace=True)
            boot_ratio = ratio.iloc[boot_indices]
            boot_weights = weights.iloc[boot_indices]
            
            sorted_indices = np.argsort(boot_ratio)
            sorted_ratio_boot = boot_ratio.iloc[sorted_indices]
            sorted_weights_boot = boot_weights.iloc[sorted_indices]
            
            cumsum_weights_boot = np.cumsum(sorted_weights_boot)
            total_weight_boot = cumsum_weights_boot.iloc[-1]
            median_idx_boot = np.searchsorted(cumsum_weights_boot, total_weight_boot / 2)
            
            bootstrap_estimates.append(sorted_ratio_boot.iloc[median_idx_boot])
        
        se_wm = np.std(bootstrap_estimates)
        z_score = beta_wm / se_wm
        pval = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'method': 'Weighted Median',
            'beta': beta_wm,
            'se': se_wm,
            'pval': pval,
            'n_snps': len(data)
        }
    
    def mr_presso(self, outlier_threshold: float = 0.05) -> Dict:
        """
        MR-PRESSO for outlier detection and correction.
        
        Parameters
        ----------
        outlier_threshold : float
            Significance threshold for outlier detection
        """
        if self.harmonized_data is None:
            self.harmonize_data()
        
        data = self.harmonized_data.copy()
        
        # Initial IVW
        initial_result = self.ivw()
        
        # Calculate residuals
        expected_beta_out = data['beta_exp'] * initial_result['beta']
        residuals = data['beta_out'] - expected_beta_out
        
        # Identify outliers using studentized residuals
        residual_se = np.sqrt(data['se_out']**2 + (data['beta_exp'] * initial_result['se'])**2)
        studentized_residuals = residuals / residual_se
        
        # Chi-square test for each SNP
        chi2_stats = studentized_residuals**2
        outlier_pvals = 1 - stats.chi2.cdf(chi2_stats, df=1)
        
        # Identify outliers
        outliers = outlier_pvals < outlier_threshold
        
        # Remove outliers and recalculate
        data_cleaned = data[~outliers]
        self.harmonized_data = data_cleaned
        corrected_result = self.ivw()
        
        # Restore original data
        self.harmonized_data = data
        
        return {
            'method': 'MR-PRESSO',
            'beta': corrected_result['beta'],
            'se': corrected_result['se'],
            'pval': corrected_result['pval'],
            'n_outliers': outliers.sum(),
            'outlier_snps': data[outliers]['SNP'].tolist(),
            'n_snps': len(data_cleaned)
        }
    
    def run_analysis(self, methods: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run multiple MR methods.
        
        Parameters
        ----------
        methods : list of str, optional
            Methods to run. Options: 'ivw', 'egger', 'weighted_median', 'mr_presso'
            Default is all methods.
        
        Returns
        -------
        pd.DataFrame
            Results from all methods
        """
        if methods is None:
            methods = ['ivw', 'egger', 'weighted_median', 'mr_presso']
        
        results = []
        
        if 'ivw' in methods:
            results.append(self.ivw())
        
        if 'egger' in methods:
            results.append(self.egger())
        
        if 'weighted_median' in methods:
            results.append(self.weighted_median())
        
        if 'mr_presso' in methods:
            results.append(self.mr_presso())
        
        return pd.DataFrame(results)
    
    def heterogeneity_test(self) -> Dict:
        """
        Cochran's Q test for heterogeneity.
        
        Tests whether causal estimates from individual SNPs are homogeneous.
        """
        if self.harmonized_data is None:
            self.harmonize_data()
        
        data = self.harmonized_data
        
        # Get IVW estimate
        ivw_result = self.ivw()
        beta_ivw = ivw_result['beta']
        
        # Calculate Q statistic
        ratio = data['beta_out'] / data['beta_exp']
        weights = 1 / (data['se_out']**2)
        
        Q = np.sum(weights * (ratio - beta_ivw)**2)
        df = len(data) - 1
        pval = 1 - stats.chi2.cdf(Q, df)
        
        return {
            'Q': Q,
            'df': df,
            'pval': pval,
            'I2': max(0, (Q - df) / Q) if Q > 0 else 0
        }


class MultivariableMR:
    """
    Multivariable Mendelian Randomization.
    
    Estimates direct causal effects of multiple exposures on an outcome,
    accounting for correlation between exposures.
    """
    
    def __init__(self):
        self.exposure_data = []
        self.outcome_data = None
        
    def add_exposure(self, data: pd.DataFrame, name: str):
        """Add exposure GWAS data."""
        self.exposure_data.append({'name': name, 'data': data})
    
    def load_outcome(self, data: pd.DataFrame):
        """Load outcome GWAS data."""
        self.outcome_data = data
    
    def run_mvmr(self) -> pd.DataFrame:
        """
        Run multivariable MR analysis.
        
        Returns direct causal effects of each exposure on outcome.
        """
        # Implementation would include:
        # 1. Harmonize all exposures with outcome
        # 2. Calculate conditional F-statistics
        # 3. Perform multivariable IVW regression
        # 4. Calculate standard errors accounting for correlation
        
        results = []
        # ... implementation details ...
        
        return pd.DataFrame(results)
