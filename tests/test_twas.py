"""
Unit tests for TWAS module
"""

import pytest
import numpy as np
import pandas as pd
from causal_genomics.analysis.twas import (
    TWASAnalysis,
    MultiTissueTWAS,
    TWAS_FUSION,
    ColocTWAS
)


class TestTWASAnalysis:
    """Test TWASAnalysis class"""

    @pytest.fixture
    def sample_genotypes(self):
        """Create sample genotype data"""
        np.random.seed(42)
        n_samples = 200
        n_variants = 50
        # Genotypes: 0, 1, 2 encoding
        return np.random.binomial(2, 0.3, (n_samples, n_variants)).astype(float)

    @pytest.fixture
    def sample_expression(self):
        """Create sample expression data"""
        np.random.seed(42)
        n_samples = 200
        return np.random.normal(0, 1, n_samples)

    @pytest.fixture
    def sample_phenotype(self):
        """Create sample phenotype data"""
        np.random.seed(43)
        n_samples = 200
        return np.random.normal(0, 1, n_samples)

    def test_initialization(self):
        """Test TWASAnalysis initialization"""
        twas = TWASAnalysis(alpha=0.01)
        assert twas.alpha == 0.01
        assert twas.expression_weights == {}
        assert twas.results is None

    def test_train_expression_model(self, sample_genotypes, sample_expression):
        """Test training expression prediction model"""
        twas = TWASAnalysis()

        result = twas.train_expression_model(
            genotypes=sample_genotypes,
            expression=sample_expression,
            gene_id='GENE1',
            method='elastic_net'
        )

        assert result['gene'] == 'GENE1'
        assert 'weights' in result
        assert 'r2_cv' in result
        assert len(result['weights']) == sample_genotypes.shape[1]
        assert 'GENE1' in twas.expression_weights

    def test_train_expression_model_methods(self, sample_genotypes, sample_expression):
        """Test different expression model methods"""
        twas = TWASAnalysis()

        for method in ['elastic_net', 'ridge', 'lasso']:
            result = twas.train_expression_model(
                genotypes=sample_genotypes,
                expression=sample_expression,
                gene_id=f'GENE_{method}',
                method=method
            )
            assert result['gene'] == f'GENE_{method}'

    def test_predixcan(self, sample_genotypes, sample_expression, sample_phenotype):
        """Test individual-level PrediXcan"""
        twas = TWASAnalysis()

        # Train expression model
        twas.train_expression_model(
            sample_genotypes, sample_expression, 'GENE1'
        )

        # Create gene_weights dict
        gene_weights = {
            'GENE1': twas.expression_weights['GENE1']['weights']
        }

        # Create genotypes dict
        genotypes = {'GENE1': sample_genotypes}

        results = twas.predixcan(
            genotypes=genotypes,
            phenotype=sample_phenotype,
            gene_weights=gene_weights
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        assert 'gene' in results.columns
        assert 'beta' in results.columns
        assert 'pval' in results.columns
        assert 'pval_bonferroni' in results.columns

    def test_s_predixcan(self):
        """Test summary-based S-PrediXcan"""
        np.random.seed(42)
        twas = TWASAnalysis()

        n_variants = 30
        gwas_z = np.random.normal(0, 1, n_variants)
        gwas_z[5] = 4.0  # Strong signal

        ld_matrix = np.eye(n_variants)
        weights = np.random.normal(0, 0.1, n_variants)
        weights[5] = 0.5  # Weight on strong signal

        result = twas.s_predixcan(
            gwas_z=gwas_z,
            ld_matrix=ld_matrix,
            weights=weights,
            gene_id='GENE1',
            n_gwas=50000,
            model_r2=0.1
        )

        assert result['gene'] == 'GENE1'
        assert 'z_twas' in result
        assert 'pval' in result
        assert 'beta' in result
        # Should detect the association
        assert abs(result['z_twas']) > 1.0

    def test_s_predixcan_correlation(self):
        """Test that S-PrediXcan z-score accounts for LD"""
        np.random.seed(42)
        twas = TWASAnalysis()

        n_variants = 20

        # GWAS signal
        gwas_z = np.zeros(n_variants)
        gwas_z[5] = 5.0

        # Weights
        weights = np.zeros(n_variants)
        weights[5] = 1.0

        # LD affects the denominator
        ld_eye = np.eye(n_variants)
        ld_corr = np.eye(n_variants)
        ld_corr[4, 5] = ld_corr[5, 4] = 0.8
        ld_corr[5, 6] = ld_corr[6, 5] = 0.8

        result_eye = twas.s_predixcan(gwas_z, ld_eye, weights, 'G1', 10000, 0.1)
        result_corr = twas.s_predixcan(gwas_z, ld_corr, weights, 'G2', 10000, 0.1)

        # Both should give same result since weights only on index 5
        assert abs(result_eye['z_twas'] - result_corr['z_twas']) < 0.1


class TestMultiTissueTWAS:
    """Test MultiTissueTWAS class"""

    @pytest.fixture
    def sample_tissue_results(self):
        """Create sample TWAS results for multiple tissues"""
        np.random.seed(42)
        tissues = {}

        genes = ['GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5']

        for tissue in ['Brain', 'Liver', 'Adipose']:
            z_scores = np.random.normal(0, 1, len(genes))
            # Make GENE1 significant in Brain
            if tissue == 'Brain':
                z_scores[0] = 4.0

            pvals = 2 * (1 - np.abs(np.random.normal(0, 1, len(genes))))
            pvals = np.clip(pvals, 1e-10, 1)
            pvals[z_scores > 3] = 1e-5

            tissues[tissue] = pd.DataFrame({
                'gene': genes,
                'z_twas': z_scores,
                'pval': pvals
            })

        return tissues

    def test_initialization(self):
        """Test MultiTissueTWAS initialization"""
        mt = MultiTissueTWAS()
        assert mt.tissue_results == {}
        assert mt.combined_results is None

    def test_add_tissue_result(self, sample_tissue_results):
        """Test adding tissue results"""
        mt = MultiTissueTWAS()

        for tissue, results in sample_tissue_results.items():
            mt.add_tissue_result(tissue, results)

        assert len(mt.tissue_results) == 3
        assert 'Brain' in mt.tissue_results

    def test_combine_tissues_fisher(self, sample_tissue_results):
        """Test Fisher's method for combining tissues"""
        mt = MultiTissueTWAS()

        for tissue, results in sample_tissue_results.items():
            mt.add_tissue_result(tissue, results)

        combined = mt.combine_tissues_fisher()

        assert isinstance(combined, pd.DataFrame)
        assert 'gene' in combined.columns
        assert 'fisher_pval' in combined.columns
        assert 'n_tissues' in combined.columns
        assert len(combined) == 5  # 5 genes

    def test_combine_tissues_stouffer(self, sample_tissue_results):
        """Test Stouffer's method for combining tissues"""
        mt = MultiTissueTWAS()

        for tissue, results in sample_tissue_results.items():
            mt.add_tissue_result(tissue, results)

        combined = mt.combine_tissues_stouffer()

        assert isinstance(combined, pd.DataFrame)
        assert 'stouffer_z' in combined.columns
        assert 'stouffer_pval' in combined.columns
        assert 'concordant' in combined.columns

    def test_weighted_stouffer(self, sample_tissue_results):
        """Test weighted Stouffer combination"""
        mt = MultiTissueTWAS()

        for tissue, results in sample_tissue_results.items():
            mt.add_tissue_result(tissue, results)

        weights = {'Brain': 2.0, 'Liver': 1.0, 'Adipose': 0.5}
        combined = mt.combine_tissues_stouffer(weights=weights)

        assert len(combined) == 5

    def test_tissue_specific_genes(self, sample_tissue_results):
        """Test identification of tissue-specific genes"""
        mt = MultiTissueTWAS()

        for tissue, results in sample_tissue_results.items():
            mt.add_tissue_result(tissue, results)

        specific = mt.get_tissue_specific_genes(
            pval_threshold=0.01,
            tissue_specificity_ratio=5
        )

        assert isinstance(specific, pd.DataFrame)
        if len(specific) > 0:
            assert 'specific_tissue' in specific.columns
            assert 'specificity_ratio' in specific.columns


class TestTWAS_FUSION:
    """Test TWAS_FUSION class"""

    def test_initialization(self):
        """Test TWAS_FUSION initialization"""
        fusion = TWAS_FUSION()
        assert fusion.models == {}

    def test_compute_twas_statistic(self):
        """Test TWAS statistic computation"""
        np.random.seed(42)
        fusion = TWAS_FUSION()

        n_variants = 25
        gwas_z = np.random.normal(0, 1, n_variants)
        gwas_z[10] = 4.0

        weights = np.zeros(n_variants)
        weights[10] = 1.0

        ld_matrix = np.eye(n_variants)

        result = fusion.compute_twas_statistic(gwas_z, weights, ld_matrix)

        assert 'z_twas' in result
        assert 'pval' in result
        assert abs(result['z_twas'] - 4.0) < 0.1

    def test_conditional_twas(self):
        """Test conditional TWAS analysis"""
        np.random.seed(42)
        fusion = TWAS_FUSION()

        n_variants = 30
        gwas_z = np.random.normal(0, 1, n_variants)
        gwas_z[5] = 4.0
        gwas_z[15] = 3.5

        # Gene weights
        gene_weights = {
            'GENE_A': np.zeros(n_variants),
            'GENE_B': np.zeros(n_variants),
            'GENE_C': np.zeros(n_variants)
        }
        gene_weights['GENE_A'][5] = 1.0
        gene_weights['GENE_B'][15] = 1.0
        gene_weights['GENE_C'][20] = 0.5

        ld_matrix = np.eye(n_variants)

        results = fusion.conditional_twas(
            gwas_z=gwas_z,
            gene_weights=gene_weights,
            ld_matrix=ld_matrix,
            genes_to_condition=['GENE_A']
        )

        assert isinstance(results, pd.DataFrame)
        assert 'z_marginal' in results.columns
        assert 'z_conditional' in results.columns
        # GENE_A should not be in results (being conditioned on)
        assert 'GENE_A' not in results['gene'].values


class TestColocTWAS:
    """Test ColocTWAS class"""

    @pytest.fixture
    def sample_twas_df(self):
        """Create sample TWAS results"""
        return pd.DataFrame({
            'gene': ['GENE1', 'GENE2', 'GENE3', 'GENE4'],
            'z_twas': [4.0, 2.5, 1.0, 3.2],
            'pval': [1e-5, 0.01, 0.3, 1e-3]
        })

    @pytest.fixture
    def sample_coloc_df(self):
        """Create sample colocalization results"""
        return pd.DataFrame({
            'gene': ['GENE1', 'GENE2', 'GENE3', 'GENE5'],
            'H4_shared_causal': [0.9, 0.3, 0.1, 0.85]
        })

    def test_initialization(self):
        """Test ColocTWAS initialization"""
        ct = ColocTWAS()
        assert ct.twas_results is None
        assert ct.coloc_results is None

    def test_integrate_twas_coloc(self, sample_twas_df, sample_coloc_df):
        """Test integrating TWAS and colocalization"""
        ct = ColocTWAS()

        combined = ct.integrate_twas_coloc(sample_twas_df, sample_coloc_df)

        assert isinstance(combined, pd.DataFrame)
        assert 'twas_score' in combined.columns
        assert 'coloc_score' in combined.columns
        assert 'combined_score' in combined.columns
        assert 'priority' in combined.columns

    def test_priority_assignment(self, sample_twas_df, sample_coloc_df):
        """Test priority assignment logic"""
        ct = ColocTWAS()

        combined = ct.integrate_twas_coloc(sample_twas_df, sample_coloc_df)

        # GENE1 should be high priority (low pval, high coloc)
        gene1 = combined[combined['gene'] == 'GENE1']
        if len(gene1) > 0:
            assert gene1['priority'].values[0] in ['medium', 'high']


def test_twas_with_causal_effect():
    """Test TWAS detects true causal gene"""
    np.random.seed(42)

    n_samples = 500
    n_variants = 30
    causal_gene = 'CAUSAL_GENE'

    # Generate genotypes
    genotypes = np.random.binomial(2, 0.3, (n_samples, n_variants)).astype(float)

    # Generate expression with genetic component
    true_weights = np.random.normal(0, 0.1, n_variants)
    true_weights[5] = 0.5  # Strong eQTL
    expression = genotypes @ true_weights + np.random.normal(0, 0.5, n_samples)

    # Generate phenotype with expression effect
    causal_effect = 0.3
    phenotype = expression * causal_effect + np.random.normal(0, 1, n_samples)

    # Run TWAS
    twas = TWASAnalysis()
    twas.train_expression_model(genotypes, expression, causal_gene)

    gene_weights = {causal_gene: twas.expression_weights[causal_gene]['weights']}
    geno_dict = {causal_gene: genotypes}

    results = twas.predixcan(geno_dict, phenotype, gene_weights)

    # Should detect association
    assert results['pval'].values[0] < 0.05


def test_multi_tissue_power():
    """Test that combining tissues increases power"""
    np.random.seed(42)

    genes = ['GENE1', 'GENE2', 'GENE3']

    # Create marginal results for multiple tissues
    mt = MultiTissueTWAS()

    # Each tissue has weak signal
    for tissue in ['T1', 'T2', 'T3', 'T4']:
        z_scores = np.array([1.8, 0.5, -0.2])  # GENE1 marginal in each
        pvals = 2 * (1 - np.abs(z_scores) / 5)

        mt.add_tissue_result(tissue, pd.DataFrame({
            'gene': genes,
            'z_twas': z_scores,
            'pval': pvals
        }))

    combined = mt.combine_tissues_stouffer()

    # Combined z for GENE1 should be higher than individual
    gene1_combined = combined[combined['gene'] == 'GENE1']['stouffer_z'].values[0]
    # Stouffer: z_combined = sum(z) / sqrt(n) = 1.8 * 4 / 2 = 3.6
    assert gene1_combined > 3.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
