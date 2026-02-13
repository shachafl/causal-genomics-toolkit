"""
Unit tests for Mendelian Randomization module
"""

import pytest
import numpy as np
import pandas as pd
from causal_genomics.analysis.mendelian_randomization import (
    MendelianRandomization,
    MultivariableMR
)


class TestMendelianRandomization:
    """Test MendelianRandomization class"""

    @pytest.fixture
    def sample_exposure_data(self):
        """Create sample exposure GWAS data with matching alleles for outcome"""
        np.random.seed(42)
        n_snps = 100

        # Use consistent allele pairs
        effect_alleles = ['A'] * n_snps
        other_alleles = ['G'] * n_snps

        data = pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(n_snps)],
            'beta': np.random.normal(0, 0.1, n_snps),
            'se': np.random.uniform(0.01, 0.05, n_snps),
            'pval': np.random.uniform(0, 1, n_snps),
            'effect_allele': effect_alleles,
            'other_allele': other_alleles,
            'eaf': np.random.uniform(0.1, 0.9, n_snps)
        })

        # Make some SNPs highly significant (enough for MR analysis)
        data.loc[:19, 'pval'] = np.random.uniform(1e-10, 1e-8, 20)

        return data

    @pytest.fixture
    def sample_outcome_data(self):
        """Create sample outcome GWAS data with matching alleles for exposure"""
        np.random.seed(43)
        n_snps = 100

        # Use same consistent allele pairs as exposure
        effect_alleles = ['A'] * n_snps
        other_alleles = ['G'] * n_snps

        data = pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(n_snps)],
            'beta': np.random.normal(0, 0.1, n_snps),
            'se': np.random.uniform(0.01, 0.05, n_snps),
            'pval': np.random.uniform(0, 1, n_snps),
            'effect_allele': effect_alleles,
            'other_allele': other_alleles
        })

        return data
    
    def test_initialization(self):
        """Test MR initialization"""
        mr = MendelianRandomization(p_threshold=5e-8, clumping_r2=0.001)
        
        assert mr.p_threshold == 5e-8
        assert mr.clumping_r2 == 0.001
        assert mr.exposure_data is None
        assert mr.outcome_data is None
    
    def test_load_exposure(self, sample_exposure_data):
        """Test loading exposure data"""
        mr = MendelianRandomization()
        mr.load_exposure_gwas(sample_exposure_data)
        
        assert mr.exposure_data is not None
        assert 'beta_exp' in mr.exposure_data.columns
        assert len(mr.exposure_data) == len(sample_exposure_data)
    
    def test_load_outcome(self, sample_outcome_data):
        """Test loading outcome data"""
        mr = MendelianRandomization()
        mr.load_outcome_gwas(sample_outcome_data)
        
        assert mr.outcome_data is not None
        assert 'beta_out' in mr.outcome_data.columns
        assert len(mr.outcome_data) == len(sample_outcome_data)
    
    def test_harmonize_data(self, sample_exposure_data, sample_outcome_data):
        """Test data harmonization"""
        mr = MendelianRandomization()
        mr.load_exposure_gwas(sample_exposure_data)
        mr.load_outcome_gwas(sample_outcome_data)
        
        harmonized = mr.harmonize_data()
        
        assert harmonized is not None
        assert len(harmonized) > 0
        assert all(harmonized['pval_exp'] < mr.p_threshold)
    
    def test_ivw(self, sample_exposure_data, sample_outcome_data):
        """Test IVW method"""
        mr = MendelianRandomization()
        mr.load_exposure_gwas(sample_exposure_data)
        mr.load_outcome_gwas(sample_outcome_data)
        
        result = mr.ivw()
        
        assert 'beta' in result
        assert 'se' in result
        assert 'pval' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert result['method'] == 'IVW'
    
    def test_egger(self, sample_exposure_data, sample_outcome_data):
        """Test MR-Egger method"""
        mr = MendelianRandomization()
        mr.load_exposure_gwas(sample_exposure_data)
        mr.load_outcome_gwas(sample_outcome_data)
        
        result = mr.egger()
        
        assert 'beta' in result
        assert 'intercept' in result
        assert 'intercept_pval' in result
        assert result['method'] == 'MR-Egger'
    
    def test_weighted_median(self, sample_exposure_data, sample_outcome_data):
        """Test weighted median method"""
        mr = MendelianRandomization()
        mr.load_exposure_gwas(sample_exposure_data)
        mr.load_outcome_gwas(sample_outcome_data)
        
        result = mr.weighted_median()
        
        assert 'beta' in result
        assert 'se' in result
        assert 'pval' in result
        assert result['method'] == 'Weighted Median'
    
    def test_run_analysis(self, sample_exposure_data, sample_outcome_data):
        """Test running multiple methods"""
        mr = MendelianRandomization()
        mr.load_exposure_gwas(sample_exposure_data)
        mr.load_outcome_gwas(sample_outcome_data)
        
        results = mr.run_analysis(methods=['ivw', 'egger'])
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 2
        assert 'method' in results.columns
        assert 'beta' in results.columns
        assert 'pval' in results.columns
    
    def test_heterogeneity_test(self, sample_exposure_data, sample_outcome_data):
        """Test heterogeneity test"""
        mr = MendelianRandomization()
        mr.load_exposure_gwas(sample_exposure_data)
        mr.load_outcome_gwas(sample_outcome_data)
        
        result = mr.heterogeneity_test()
        
        assert 'Q' in result
        assert 'df' in result
        assert 'pval' in result
        assert 'I2' in result
        assert 0 <= result['I2'] <= 1


class TestMultivariableMR:
    """Test MultivariableMR class"""
    
    def test_initialization(self):
        """Test MVMR initialization"""
        mvmr = MultivariableMR()
        
        assert mvmr.exposure_data == []
        assert mvmr.outcome_data is None
    
    def test_add_exposure(self):
        """Test adding exposure"""
        mvmr = MultivariableMR()
        
        exposure = pd.DataFrame({
            'SNP': ['rs1', 'rs2'],
            'beta': [0.1, 0.2],
            'se': [0.01, 0.02]
        })
        
        mvmr.add_exposure(exposure, 'Exposure1')
        
        assert len(mvmr.exposure_data) == 1
        assert mvmr.exposure_data[0]['name'] == 'Exposure1'


def test_causal_estimate_validity():
    """Test that causal estimates are reasonable"""
    np.random.seed(42)
    
    # Create data with known causal effect
    true_effect = 0.5
    n_snps = 50
    
    # Exposure
    beta_exp = np.random.normal(0, 0.1, n_snps)
    se_exp = np.random.uniform(0.01, 0.05, n_snps)
    
    # Outcome (with causal effect)
    beta_out = beta_exp * true_effect + np.random.normal(0, 0.05, n_snps)
    se_out = np.random.uniform(0.01, 0.05, n_snps)
    
    exposure_data = pd.DataFrame({
        'SNP': [f'rs{i}' for i in range(n_snps)],
        'beta': beta_exp,
        'se': se_exp,
        'pval': np.random.uniform(1e-10, 1e-8, n_snps),
        'effect_allele': ['A'] * n_snps,
        'other_allele': ['G'] * n_snps,
        'eaf': [0.5] * n_snps
    })
    
    outcome_data = pd.DataFrame({
        'SNP': [f'rs{i}' for i in range(n_snps)],
        'beta': beta_out,
        'se': se_out,
        'pval': np.random.uniform(0, 1, n_snps),
        'effect_allele': ['A'] * n_snps,
        'other_allele': ['G'] * n_snps
    })
    
    # Run MR
    mr = MendelianRandomization(p_threshold=1)  # Include all SNPs
    mr.load_exposure_gwas(exposure_data)
    mr.load_outcome_gwas(outcome_data)
    
    result = mr.ivw()
    
    # Check if estimated effect is close to true effect
    assert abs(result['beta'] - true_effect) < 0.2  # Allow some error


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
