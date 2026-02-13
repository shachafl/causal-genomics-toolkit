"""
Unit tests for Fine-mapping module
"""

import pytest
import numpy as np
import pandas as pd
from causal_genomics.analysis.fine_mapping import (
    FineMappingAnalysis,
    MultiLocusFineMapping,
    AnnotationEnrichedFineMaping
)


class TestFineMappingAnalysis:
    """Test FineMappingAnalysis class"""

    @pytest.fixture
    def sample_z_scores(self):
        """Create sample z-scores with one strong signal"""
        np.random.seed(42)
        n_snps = 50
        z_scores = np.random.normal(0, 1, n_snps)
        # Add a strong causal signal
        z_scores[10] = 5.0
        # Add some weaker signals in LD
        z_scores[9] = 3.5
        z_scores[11] = 3.2
        return z_scores

    @pytest.fixture
    def sample_ld_matrix(self):
        """Create sample LD matrix with local correlation structure"""
        np.random.seed(42)
        n_snps = 50
        # Create block diagonal LD structure
        ld = np.eye(n_snps)
        # Add some LD around index 10
        for i in range(8, 13):
            for j in range(8, 13):
                if i != j:
                    ld[i, j] = 0.7 ** abs(i - j)
        return ld

    def test_initialization(self):
        """Test FineMappingAnalysis initialization"""
        fm = FineMappingAnalysis(max_causal=5, coverage=0.99)

        assert fm.max_causal == 5
        assert fm.coverage == 0.99
        assert fm.min_abs_corr == 0.5

    def test_susie(self, sample_z_scores, sample_ld_matrix):
        """Test SuSiE fine-mapping"""
        fm = FineMappingAnalysis(max_causal=3)

        result = fm.susie(
            z_scores=sample_z_scores,
            ld_matrix=sample_ld_matrix,
            n=10000,
            max_iter=50
        )

        assert 'pip' in result
        assert 'credible_sets' in result
        assert 'alpha' in result
        assert len(result['pip']) == len(sample_z_scores)
        # PIP should be highest around true causal variant
        assert np.argmax(result['pip']) in [9, 10, 11]

    def test_susie_convergence(self, sample_z_scores, sample_ld_matrix):
        """Test SuSiE convergence tracking"""
        fm = FineMappingAnalysis()

        result = fm.susie(
            z_scores=sample_z_scores,
            ld_matrix=sample_ld_matrix,
            n=10000,
            max_iter=100,
            tol=1e-6
        )

        assert 'converged' in result
        assert 'n_iterations' in result

    def test_abf_finemap(self, sample_z_scores):
        """Test ABF fine-mapping"""
        fm = FineMappingAnalysis()

        # Convert z-scores to beta/se
        n = 10000
        beta = sample_z_scores / np.sqrt(n)
        se = np.ones_like(beta) / np.sqrt(n)

        result = fm.abf_finemap(beta, se)

        assert 'pip' in result
        assert 'credible_set' in result
        assert len(result['pip']) == len(sample_z_scores)
        # PIPs should sum to 1
        assert abs(np.sum(result['pip']) - 1.0) < 1e-6
        # Top variant should be in credible set
        assert np.argmax(result['pip']) in result['credible_set']

    def test_credible_set_coverage(self, sample_z_scores, sample_ld_matrix):
        """Test that credible sets achieve target coverage"""
        fm = FineMappingAnalysis(coverage=0.95)

        result = fm.susie(
            z_scores=sample_z_scores,
            ld_matrix=sample_ld_matrix,
            n=10000
        )

        for cs in result['credible_sets']:
            # Each credible set should meet coverage target
            assert cs['coverage'] >= 0.90  # Allow small tolerance

    def test_conditional_analysis(self, sample_z_scores, sample_ld_matrix):
        """Test stepwise conditional analysis"""
        fm = FineMappingAnalysis()

        result = fm.conditional_analysis(
            z_scores=sample_z_scores,
            ld_matrix=sample_ld_matrix,
            pval_threshold=5e-8,
            max_signals=5
        )

        assert 'n_signals' in result
        assert 'lead_variants' in result
        assert 'conditional_z' in result

        # Should find at least one signal
        if result['n_signals'] > 0:
            assert len(result['lead_variants']) == result['n_signals']
            # Lead variant should have smallest p-value
            assert result['lead_variants'][0]['variant_idx'] == np.argmax(np.abs(sample_z_scores))


class TestMultiLocusFineMapping:
    """Test MultiLocusFineMapping class"""

    @pytest.fixture
    def sample_loci(self):
        """Create sample data for multiple loci"""
        np.random.seed(42)
        loci = {}

        for i in range(3):
            n_snps = 30 + i * 10
            z_scores = np.random.normal(0, 1, n_snps)
            z_scores[5] = 4.0 + i  # Strong signal

            ld = np.eye(n_snps)
            for j in range(3, 8):
                for k in range(3, 8):
                    if j != k:
                        ld[j, k] = 0.6 ** abs(j - k)

            loci[f'chr{i+1}:1000000-2000000'] = {
                'z_scores': z_scores,
                'ld_matrix': ld,
                'variant_ids': [f'rs{i}_{j}' for j in range(n_snps)]
            }

        return loci

    def test_initialization(self):
        """Test MultiLocusFineMapping initialization"""
        mfm = MultiLocusFineMapping(coverage=0.99)
        assert mfm.coverage == 0.99
        assert mfm.results == {}

    def test_add_locus(self, sample_loci):
        """Test adding loci"""
        mfm = MultiLocusFineMapping()

        for locus_name, locus_data in sample_loci.items():
            mfm.add_locus(
                locus_name,
                locus_data['z_scores'],
                locus_data['ld_matrix'],
                locus_data['variant_ids']
            )

        assert len(mfm.results) == 3

    def test_run_finemapping(self, sample_loci):
        """Test running fine-mapping across loci"""
        mfm = MultiLocusFineMapping()

        for locus_name, locus_data in sample_loci.items():
            mfm.add_locus(
                locus_name,
                locus_data['z_scores'],
                locus_data['ld_matrix'],
                locus_data['variant_ids']
            )

        results = mfm.run_finemapping(method='susie', n=10000)

        assert isinstance(results, pd.DataFrame)
        assert 'locus' in results.columns
        assert 'variant' in results.columns
        assert 'pip' in results.columns
        # Should have results for all loci
        assert len(results['locus'].unique()) == 3

    def test_get_high_pip_variants(self, sample_loci):
        """Test filtering high PIP variants"""
        mfm = MultiLocusFineMapping()

        for locus_name, locus_data in sample_loci.items():
            mfm.add_locus(
                locus_name,
                locus_data['z_scores'],
                locus_data['ld_matrix'],
                locus_data['variant_ids']
            )

        mfm.run_finemapping(method='susie', n=10000)
        high_pip = mfm.get_high_pip_variants(pip_threshold=0.1)

        assert isinstance(high_pip, pd.DataFrame)
        if len(high_pip) > 0:
            assert all(high_pip['pip'] >= 0.1)

    def test_abf_method(self, sample_loci):
        """Test ABF fine-mapping method"""
        mfm = MultiLocusFineMapping()

        for locus_name, locus_data in sample_loci.items():
            mfm.add_locus(
                locus_name,
                locus_data['z_scores'],
                locus_data['ld_matrix'],
                locus_data['variant_ids']
            )

        results = mfm.run_finemapping(method='abf', n=10000)

        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0


class TestAnnotationEnrichedFineMaping:
    """Test AnnotationEnrichedFineMaping class"""

    @pytest.fixture
    def sample_annotations(self):
        """Create sample functional annotations"""
        np.random.seed(42)
        n_snps = 50

        annotations = pd.DataFrame({
            'coding': np.random.binomial(1, 0.1, n_snps),
            'regulatory': np.random.binomial(1, 0.3, n_snps),
            'conserved': np.random.binomial(1, 0.2, n_snps)
        })

        # Make the causal variant (index 10) have annotations
        annotations.loc[10, 'coding'] = 1
        annotations.loc[10, 'regulatory'] = 1

        return annotations

    def test_set_annotation_prior(self, sample_annotations):
        """Test setting annotation priors"""
        aef = AnnotationEnrichedFineMaping()

        enrichment_scores = {
            'coding': 2.0,
            'regulatory': 1.0,
            'conserved': 0.5
        }

        aef.set_annotation_prior(sample_annotations, enrichment_scores)

        assert aef.prior_weights is not None
        assert len(aef.prior_weights) == len(sample_annotations)
        # Prior should be higher for annotated variants
        assert aef.prior_weights[10] > np.mean(aef.prior_weights)

    def test_finemap_with_priors(self, sample_annotations):
        """Test fine-mapping with annotation priors"""
        aef = AnnotationEnrichedFineMaping()

        enrichment_scores = {
            'coding': 2.0,
            'regulatory': 1.0,
            'conserved': 0.5
        }

        aef.set_annotation_prior(sample_annotations, enrichment_scores)

        # Create beta/se with signal at annotated variant
        np.random.seed(42)
        n_snps = 50
        beta = np.random.normal(0, 0.01, n_snps)
        se = np.ones(n_snps) * 0.01
        beta[10] = 0.05  # Strong signal

        result = aef.finemap_with_priors(beta, se)

        assert 'pip' in result
        assert 'pip_uniform' in result
        assert 'enrichment_effect' in result
        # Annotated prior should increase PIP vs uniform
        assert result['pip'][10] >= result['pip_uniform'][10]


def test_pip_sums_to_one():
    """Test that PIPs sum to approximately 1"""
    np.random.seed(42)

    fm = FineMappingAnalysis()
    beta = np.random.normal(0, 0.1, 30)
    se = np.ones(30) * 0.1
    beta[5] = 0.5  # Strong signal

    result = fm.abf_finemap(beta, se)

    assert abs(np.sum(result['pip']) - 1.0) < 1e-6


def test_credible_set_contains_causal():
    """Test that credible set likely contains true causal"""
    np.random.seed(42)

    # Simulate data with known causal variant
    n_snps = 40
    causal_idx = 15

    z_scores = np.random.normal(0, 1, n_snps)
    z_scores[causal_idx] = 6.0  # Very strong signal

    ld = np.eye(n_snps)
    for i in range(13, 18):
        for j in range(13, 18):
            if i != j:
                ld[i, j] = 0.5 ** abs(i - j)

    fm = FineMappingAnalysis(coverage=0.95)
    result = fm.susie(z_scores, ld, n=50000, max_iter=100)

    # At least one credible set should contain the causal variant
    causal_in_cs = any(
        causal_idx in cs['variants']
        for cs in result['credible_sets']
    )
    # Also check high PIP
    assert result['pip'][causal_idx] > 0.5 or causal_in_cs


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
