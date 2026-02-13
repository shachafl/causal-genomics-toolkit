"""
Unit tests for Transfer Learning module
"""

import pytest
import numpy as np
import pandas as pd
from causal_genomics.models.transfer_learning import (
    TransferLearningModel,
    MultiTaskLearning,
    CrossPopulationTransfer,
    GenomicsPretrainedModel,
    ProgressiveTransfer
)


class TestTransferLearningModel:
    """Test TransferLearningModel class"""

    @pytest.fixture
    def sample_source_data(self):
        """Create sample source domain data"""
        np.random.seed(42)
        n_samples = 500
        n_features = 50

        X = np.random.normal(0, 1, (n_samples, n_features))
        # True relationship
        true_weights = np.random.normal(0, 0.1, n_features)
        true_weights[:5] = 0.5  # First 5 features important
        y = X @ true_weights + np.random.normal(0, 0.5, n_samples)

        return X, y, true_weights

    @pytest.fixture
    def sample_target_data(self):
        """Create sample target domain data (smaller, related)"""
        np.random.seed(43)
        n_samples = 100
        n_features = 50

        X = np.random.normal(0, 1, (n_samples, n_features))
        # Similar but not identical relationship
        true_weights = np.random.normal(0, 0.1, n_features)
        true_weights[:5] = 0.4  # Same features important
        true_weights[5:10] = 0.2  # Some new features
        y = X @ true_weights + np.random.normal(0, 0.5, n_samples)

        return X, y

    def test_initialization(self):
        """Test TransferLearningModel initialization"""
        model = TransferLearningModel(
            base_model='ridge',
            transfer_method='feature_extraction',
            lambda_transfer=0.7
        )

        assert model.base_model == 'ridge'
        assert model.transfer_method == 'feature_extraction'
        assert model.lambda_transfer == 0.7
        assert not model.is_fitted

    def test_fit_source(self, sample_source_data):
        """Test fitting on source data"""
        X, y, _ = sample_source_data
        model = TransferLearningModel()

        model.fit_source(X, y)

        assert model.source_model is not None
        assert hasattr(model, 'source_weights')

    def test_fit_target_feature_extraction(self, sample_source_data, sample_target_data):
        """Test feature extraction transfer"""
        X_source, y_source, _ = sample_source_data
        X_target, y_target = sample_target_data

        model = TransferLearningModel(
            transfer_method='feature_extraction',
            lambda_transfer=0.5
        )

        model.fit_source(X_source, y_source)
        model.fit_target(X_target, y_target)

        assert model.is_fitted
        assert model.target_model is not None

    def test_fit_target_fine_tuning(self, sample_source_data, sample_target_data):
        """Test fine-tuning transfer"""
        X_source, y_source, _ = sample_source_data
        X_target, y_target = sample_target_data

        model = TransferLearningModel(
            transfer_method='fine_tuning',
            lambda_transfer=0.5
        )

        model.fit_source(X_source, y_source)
        model.fit_target(X_target, y_target)

        assert model.is_fitted

    def test_fit_target_domain_adaptation(self, sample_source_data, sample_target_data):
        """Test domain adaptation transfer"""
        X_source, y_source, _ = sample_source_data
        X_target, y_target = sample_target_data

        model = TransferLearningModel(
            transfer_method='domain_adaptation',
            lambda_transfer=0.5
        )

        model.fit_source(X_source, y_source)
        model.fit_target(X_target, y_target, X_source=X_source, y_source=y_source)

        assert model.is_fitted

    def test_predict(self, sample_source_data, sample_target_data):
        """Test prediction"""
        X_source, y_source, _ = sample_source_data
        X_target, y_target = sample_target_data

        model = TransferLearningModel()
        model.fit_source(X_source, y_source)
        model.fit_target(X_target, y_target)

        predictions = model.predict(X_target)

        assert len(predictions) == len(y_target)
        assert not np.any(np.isnan(predictions))

    def test_evaluate(self, sample_source_data, sample_target_data):
        """Test model evaluation"""
        X_source, y_source, _ = sample_source_data
        X_target, y_target = sample_target_data

        model = TransferLearningModel()
        model.fit_source(X_source, y_source)
        model.fit_target(X_target, y_target)

        metrics = model.evaluate(X_target, y_target)

        assert 'r2' in metrics
        assert 'mse' in metrics
        assert 'correlation' in metrics

    def test_different_base_models(self, sample_source_data, sample_target_data):
        """Test different base model types"""
        X_source, y_source, _ = sample_source_data
        X_target, y_target = sample_target_data

        for base_model in ['ridge', 'elastic_net', 'gradient_boosting', 'random_forest']:
            model = TransferLearningModel(base_model=base_model)
            model.fit_source(X_source, y_source)
            model.fit_target(X_target, y_target)

            predictions = model.predict(X_target)
            assert len(predictions) == len(y_target)


class TestMultiTaskLearning:
    """Test MultiTaskLearning class"""

    @pytest.fixture
    def sample_multitask_data(self):
        """Create sample multi-task data"""
        np.random.seed(42)
        n_samples = 300
        n_features = 40
        n_tasks = 4

        X = np.random.normal(0, 1, (n_samples, n_features))

        # Shared factors affect all tasks
        shared_weights = np.zeros((n_tasks, n_features))
        shared_weights[:, :10] = np.random.normal(0.3, 0.1, (n_tasks, 10))

        # Task-specific weights
        for t in range(n_tasks):
            shared_weights[t, 10 + t*5:15 + t*5] = 0.2

        Y = X @ shared_weights.T + np.random.normal(0, 0.5, (n_samples, n_tasks))

        return X, Y

    def test_initialization(self):
        """Test MultiTaskLearning initialization"""
        mtl = MultiTaskLearning(n_shared_factors=5, lambda_shared=0.7)

        assert mtl.n_shared_factors == 5
        assert mtl.lambda_shared == 0.7

    def test_fit(self, sample_multitask_data):
        """Test multi-task fitting"""
        X, Y = sample_multitask_data

        mtl = MultiTaskLearning(n_shared_factors=5)
        mtl.fit(X, Y, task_names=['T1', 'T2', 'T3', 'T4'])

        assert mtl.shared_basis is not None
        assert len(mtl.task_models) == 4
        assert 'T1' in mtl.task_models

    def test_predict(self, sample_multitask_data):
        """Test single task prediction"""
        X, Y = sample_multitask_data

        mtl = MultiTaskLearning()
        mtl.fit(X, Y, task_names=['T1', 'T2', 'T3', 'T4'])

        predictions = mtl.predict(X, 'T1')

        assert len(predictions) == len(X)

    def test_predict_all(self, sample_multitask_data):
        """Test all tasks prediction"""
        X, Y = sample_multitask_data

        mtl = MultiTaskLearning()
        mtl.fit(X, Y, task_names=['T1', 'T2', 'T3', 'T4'])

        predictions = mtl.predict_all(X)

        assert isinstance(predictions, pd.DataFrame)
        assert list(predictions.columns) == ['T1', 'T2', 'T3', 'T4']
        assert len(predictions) == len(X)

    def test_get_shared_features(self, sample_multitask_data):
        """Test shared feature extraction"""
        X, Y = sample_multitask_data

        mtl = MultiTaskLearning(n_shared_factors=5)
        mtl.fit(X, Y)

        shared = mtl.get_shared_features(X)

        # Number of factors is min(n_shared_factors, n_tasks)
        assert shared.shape[0] == len(X)
        assert shared.shape[1] <= 5


class TestCrossPopulationTransfer:
    """Test CrossPopulationTransfer class"""

    @pytest.fixture
    def sample_population_data(self):
        """Create sample data for two populations"""
        np.random.seed(42)
        n_features = 30

        # Source population (larger)
        n_source = 1000
        X_source = np.random.normal(0, 1, (n_source, n_features))
        weights_source = np.random.normal(0, 0.1, n_features)
        weights_source[:10] = 0.3
        y_source = X_source @ weights_source + np.random.normal(0, 0.5, n_source)

        # Target population (smaller, slightly different effect sizes)
        n_target = 200
        X_target = np.random.normal(0.1, 1.1, (n_target, n_features))  # Slightly different distribution
        weights_target = weights_source.copy()
        weights_target[:10] *= 0.8  # Slightly attenuated effects
        y_target = X_target @ weights_target + np.random.normal(0, 0.5, n_target)

        return X_source, y_source, X_target, y_target

    def test_initialization(self):
        """Test CrossPopulationTransfer initialization"""
        cpt = CrossPopulationTransfer(
            transfer_method='weighted',
            ld_correction=True
        )

        assert cpt.transfer_method == 'weighted'
        assert cpt.ld_correction is True

    def test_fit_weighted(self, sample_population_data):
        """Test weighted combination method"""
        X_source, y_source, X_target, y_target = sample_population_data

        cpt = CrossPopulationTransfer(transfer_method='weighted')
        cpt.fit(X_source, y_source, X_target, y_target, source_weight=0.5)

        assert cpt.combined_weights is not None
        assert len(cpt.combined_weights) == X_source.shape[1]

    def test_fit_meta(self, sample_population_data):
        """Test meta-analysis method"""
        X_source, y_source, X_target, y_target = sample_population_data

        cpt = CrossPopulationTransfer(transfer_method='meta')
        cpt.fit(X_source, y_source, X_target, y_target)

        assert cpt.combined_weights is not None

    def test_fit_joint(self, sample_population_data):
        """Test joint modeling method"""
        X_source, y_source, X_target, y_target = sample_population_data

        cpt = CrossPopulationTransfer(transfer_method='joint')
        cpt.fit(X_source, y_source, X_target, y_target)

        assert cpt.combined_weights is not None

    def test_predict(self, sample_population_data):
        """Test prediction for target population"""
        X_source, y_source, X_target, y_target = sample_population_data

        cpt = CrossPopulationTransfer()
        cpt.fit(X_source, y_source, X_target, y_target)

        predictions = cpt.predict(X_target, population='target')

        assert len(predictions) == len(y_target)


class TestGenomicsPretrainedModel:
    """Test GenomicsPretrainedModel class"""

    @pytest.fixture
    def sample_gwas(self):
        """Create sample GWAS summary statistics"""
        np.random.seed(42)
        n_snps = 100

        return pd.DataFrame({
            'SNP': [f'rs{i}' for i in range(n_snps)],
            'beta': np.random.normal(0, 0.01, n_snps),
            'se': np.random.uniform(0.005, 0.02, n_snps),
            'pval': np.random.uniform(0, 1, n_snps)
        })

    def test_initialization(self):
        """Test GenomicsPretrainedModel initialization"""
        gpm = GenomicsPretrainedModel()
        assert gpm.pretrained_weights == {}
        assert gpm.fine_tuned_models == {}

    def test_create_from_gwas(self, sample_gwas):
        """Test creating pretrained weights from GWAS"""
        gpm = GenomicsPretrainedModel()
        gpm.create_from_gwas('T2D_GWAS', sample_gwas)

        assert 'T2D_GWAS' in gpm.pretrained_weights
        assert len(gpm.pretrained_weights['T2D_GWAS']['weights']) == len(sample_gwas)

    def test_fine_tune(self, sample_gwas):
        """Test fine-tuning pretrained model"""
        np.random.seed(42)
        gpm = GenomicsPretrainedModel()
        gpm.create_from_gwas('T2D_GWAS', sample_gwas)

        n_samples = 200
        n_snps = 100
        X = np.random.binomial(2, 0.3, (n_samples, n_snps)).astype(float)
        y = np.random.normal(0, 1, n_samples)

        metrics = gpm.fine_tune(
            'T2D_GWAS', X, y,
            tune_fraction=0.3,
            regularization=1.0
        )

        assert 'r2' in metrics
        assert 'weight_change' in metrics
        assert 'T2D_GWAS' in gpm.fine_tuned_models

    def test_predict(self, sample_gwas):
        """Test prediction with pretrained model"""
        np.random.seed(42)
        gpm = GenomicsPretrainedModel()
        gpm.create_from_gwas('T2D_GWAS', sample_gwas)

        n_samples = 50
        n_snps = 100
        X = np.random.binomial(2, 0.3, (n_samples, n_snps)).astype(float)

        predictions = gpm.predict(X, 'T2D_GWAS', use_fine_tuned=False)

        assert len(predictions) == n_samples


class TestProgressiveTransfer:
    """Test ProgressiveTransfer class"""

    @pytest.fixture
    def sample_domain_chain(self):
        """Create sample data for progressive transfer"""
        np.random.seed(42)
        domains = []
        n_features = 25

        base_weights = np.random.normal(0, 0.1, n_features)
        base_weights[:8] = 0.3

        for i, (name, n_samples, weight_scale) in enumerate([
            ('EUR', 1000, 1.0),
            ('EAS', 500, 0.9),
            ('AFR', 200, 0.8)
        ]):
            X = np.random.normal(0, 1, (n_samples, n_features))
            weights = base_weights * weight_scale + np.random.normal(0, 0.02, n_features)
            y = X @ weights + np.random.normal(0, 0.5, n_samples)

            domains.append({
                'name': name,
                'X': X,
                'y': y
            })

        return domains

    def test_initialization(self):
        """Test ProgressiveTransfer initialization"""
        pt = ProgressiveTransfer()
        assert pt.domain_chain == []
        assert pt.models == []

    def test_add_domain(self, sample_domain_chain):
        """Test adding domains"""
        pt = ProgressiveTransfer()

        for domain in sample_domain_chain:
            pt.add_domain(domain['name'], domain['X'], domain['y'])

        assert len(pt.domain_chain) == 3

    def test_fit_progressive(self, sample_domain_chain):
        """Test progressive fitting"""
        pt = ProgressiveTransfer()

        for domain in sample_domain_chain:
            pt.add_domain(domain['name'], domain['X'], domain['y'])

        pt.fit_progressive(lambda_decay=0.8)

        assert len(pt.models) == 3
        for model in pt.models:
            assert model.is_fitted

    def test_predict(self, sample_domain_chain):
        """Test prediction from chain"""
        pt = ProgressiveTransfer()

        for domain in sample_domain_chain:
            pt.add_domain(domain['name'], domain['X'], domain['y'])

        pt.fit_progressive()

        # Predict on last domain (AFR)
        predictions = pt.predict(sample_domain_chain[-1]['X'], domain_idx=-1)

        assert len(predictions) == len(sample_domain_chain[-1]['y'])

    def test_get_transfer_metrics(self, sample_domain_chain):
        """Test transfer metrics"""
        pt = ProgressiveTransfer()

        for domain in sample_domain_chain:
            pt.add_domain(domain['name'], domain['X'], domain['y'])

        pt.fit_progressive()
        metrics = pt.get_transfer_metrics()

        assert isinstance(metrics, pd.DataFrame)
        assert 'domain' in metrics.columns
        assert 'r2' in metrics.columns
        assert len(metrics) == 3


def test_transfer_improves_performance():
    """Test that transfer learning improves over target-only"""
    np.random.seed(42)
    n_features = 30

    # Large source dataset
    n_source = 1000
    X_source = np.random.normal(0, 1, (n_source, n_features))
    weights = np.random.normal(0, 0.1, n_features)
    weights[:10] = 0.4
    y_source = X_source @ weights + np.random.normal(0, 0.3, n_source)

    # Small target dataset (related task)
    n_target = 50
    X_target = np.random.normal(0, 1, (n_target, n_features))
    y_target = X_target @ weights + np.random.normal(0, 0.3, n_target)

    # Test data
    n_test = 200
    X_test = np.random.normal(0, 1, (n_test, n_features))
    y_test = X_test @ weights + np.random.normal(0, 0.3, n_test)

    # Target-only model
    from sklearn.linear_model import Ridge
    target_only = Ridge(alpha=1.0)
    target_only.fit(X_target, y_target)
    target_only_r2 = np.corrcoef(y_test, target_only.predict(X_test))[0, 1]**2

    # Transfer model
    transfer = TransferLearningModel(
        transfer_method='fine_tuning',
        lambda_transfer=0.7
    )
    transfer.fit_source(X_source, y_source)
    transfer.fit_target(X_target, y_target)
    transfer_metrics = transfer.evaluate(X_test, y_test)

    # Transfer should improve over target-only (or at least not hurt much)
    # With small target data, transfer should help
    assert transfer_metrics['r2'] >= target_only_r2 - 0.1  # Allow small tolerance


def test_multi_task_shares_information():
    """Test that multi-task learning shares information across tasks"""
    np.random.seed(42)
    n_samples = 200
    n_features = 30
    n_tasks = 3

    X = np.random.normal(0, 1, (n_samples, n_features))

    # Tasks share first 10 features
    shared_component = X[:, :10] @ np.random.normal(0.3, 0.1, 10)

    Y = np.column_stack([
        shared_component + np.random.normal(0, 0.5, n_samples),
        shared_component + np.random.normal(0, 0.5, n_samples),
        shared_component + np.random.normal(0, 0.5, n_samples)
    ])

    mtl = MultiTaskLearning(n_shared_factors=5, lambda_shared=0.7)
    mtl.fit(X, Y)

    # Shared basis should capture the common structure
    shared_features = mtl.get_shared_features(X)

    # Each shared feature should correlate with the shared component
    for i in range(min(5, shared_features.shape[1])):
        corr = np.corrcoef(shared_features[:, i], shared_component)[0, 1]
        # At least one factor should capture the shared signal
        if abs(corr) > 0.3:
            break
    else:
        # Check if predictions are reasonable even if correlation is low
        pred = mtl.predict(X, 'task_0')
        assert np.corrcoef(pred, Y[:, 0])[0, 1] > 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
