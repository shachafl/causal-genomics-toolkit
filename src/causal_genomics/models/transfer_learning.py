"""
Transfer Learning Module for Genomics

Implements domain adaptation and transfer learning methods for
cross-trait, cross-tissue, and cross-population prediction.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings


class TransferLearningModel(BaseEstimator, RegressorMixin):
    """
    Base transfer learning model for genomics applications.

    Enables knowledge transfer from source domain (e.g., large GWAS)
    to target domain (e.g., smaller cohort or related trait).
    """

    def __init__(self,
                 base_model: str = 'ridge',
                 transfer_method: str = 'feature_extraction',
                 lambda_transfer: float = 0.5):
        """
        Initialize transfer learning model.

        Parameters
        ----------
        base_model : str
            'ridge', 'elastic_net', 'gradient_boosting', or 'random_forest'
        transfer_method : str
            'feature_extraction', 'fine_tuning', or 'domain_adaptation'
        lambda_transfer : float
            Regularization weight for transfer (0=no transfer, 1=full transfer)
        """
        self.base_model = base_model
        self.transfer_method = transfer_method
        self.lambda_transfer = lambda_transfer
        self.source_model = None
        self.target_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _create_base_model(self):
        """Create instance of base model."""
        if self.base_model == 'ridge':
            return Ridge(alpha=1.0)
        elif self.base_model == 'elastic_net':
            return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)
        elif self.base_model == 'gradient_boosting':
            return GradientBoostingRegressor(n_estimators=100, max_depth=4)
        elif self.base_model == 'random_forest':
            return RandomForestRegressor(n_estimators=100, max_depth=10)
        else:
            raise ValueError(f"Unknown base model: {self.base_model}")

    def fit_source(self, X_source: np.ndarray, y_source: np.ndarray) -> 'TransferLearningModel':
        """
        Fit model on source domain data.

        Parameters
        ----------
        X_source : array
            Source domain features (n_samples x n_features)
        y_source : array
            Source domain targets

        Returns
        -------
        self
        """
        self.source_model = self._create_base_model()
        X_scaled = self.scaler.fit_transform(X_source)
        self.source_model.fit(X_scaled, y_source)

        # Store source feature importance for feature extraction
        if hasattr(self.source_model, 'coef_'):
            self.source_weights = self.source_model.coef_
        elif hasattr(self.source_model, 'feature_importances_'):
            self.source_weights = self.source_model.feature_importances_
        else:
            self.source_weights = np.ones(X_source.shape[1])

        return self

    def fit_target(self,
                   X_target: np.ndarray,
                   y_target: np.ndarray,
                   X_source: Optional[np.ndarray] = None,
                   y_source: Optional[np.ndarray] = None) -> 'TransferLearningModel':
        """
        Fit model on target domain with transfer from source.

        Parameters
        ----------
        X_target : array
            Target domain features
        y_target : array
            Target domain targets
        X_source : array, optional
            Source domain features (for some transfer methods)
        y_source : array, optional
            Source domain targets

        Returns
        -------
        self
        """
        X_target_scaled = self.scaler.transform(X_target)

        if self.transfer_method == 'feature_extraction':
            self._fit_feature_extraction(X_target_scaled, y_target)
        elif self.transfer_method == 'fine_tuning':
            self._fit_fine_tuning(X_target_scaled, y_target)
        elif self.transfer_method == 'domain_adaptation':
            if X_source is None:
                raise ValueError("Source data required for domain adaptation")
            X_source_scaled = self.scaler.transform(X_source)
            self._fit_domain_adaptation(X_source_scaled, y_source,
                                       X_target_scaled, y_target)
        else:
            raise ValueError(f"Unknown transfer method: {self.transfer_method}")

        self.is_fitted = True
        return self

    def _fit_feature_extraction(self, X_target: np.ndarray, y_target: np.ndarray):
        """Transfer via feature weighting from source model."""
        # Weight features by source importance
        feature_weights = np.abs(self.source_weights)
        feature_weights = feature_weights / (np.max(feature_weights) + 1e-10)

        # Apply feature weighting
        X_weighted = X_target * (1 - self.lambda_transfer + self.lambda_transfer * feature_weights)

        # Fit target model on weighted features
        self.target_model = self._create_base_model()
        self.target_model.fit(X_weighted, y_target)

        self._feature_weights = feature_weights

    def _fit_fine_tuning(self, X_target: np.ndarray, y_target: np.ndarray):
        """Transfer via fine-tuning source model."""
        if self.source_model is None:
            raise ValueError("Must fit source model first")

        # Initialize from source model weights
        self.target_model = clone(self.source_model)

        # For linear models, initialize with source weights
        if hasattr(self.target_model, 'coef_'):
            # Regularize towards source weights
            self.target_model = Ridge(alpha=1.0)
            self.target_model.fit(X_target, y_target)

            # Blend source and target weights
            target_weights = self.target_model.coef_
            blended_weights = (self.lambda_transfer * self.source_weights +
                             (1 - self.lambda_transfer) * target_weights)
            self.target_model.coef_ = blended_weights
        else:
            # For tree-based models, just re-fit
            self.target_model.fit(X_target, y_target)

    def _fit_domain_adaptation(self,
                               X_source: np.ndarray, y_source: np.ndarray,
                               X_target: np.ndarray, y_target: np.ndarray):
        """Transfer via domain adaptation (TrAdaBoost-style)."""
        n_source = len(y_source)
        n_target = len(y_target)

        # Initialize weights
        source_weights = np.ones(n_source) / n_source
        target_weights = np.ones(n_target) / n_target

        # Combined dataset
        X_combined = np.vstack([X_source, X_target])
        y_combined = np.concatenate([y_source, y_target])

        # Iterative reweighting
        n_iterations = 10
        for iteration in range(n_iterations):
            # Combine weights
            all_weights = np.concatenate([
                source_weights * self.lambda_transfer,
                target_weights * (1 - self.lambda_transfer)
            ])
            all_weights = all_weights / np.sum(all_weights)

            # Fit weighted model
            model = self._create_base_model()

            # For models that support sample weights
            if hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                model.fit(X_combined, y_combined, sample_weight=all_weights)
            else:
                # Resample according to weights
                indices = np.random.choice(
                    len(y_combined),
                    size=len(y_combined),
                    replace=True,
                    p=all_weights
                )
                model.fit(X_combined[indices], y_combined[indices])

            # Update source weights based on error
            source_pred = model.predict(X_source)
            source_error = np.abs(source_pred - y_source)
            source_error = source_error / (np.max(source_error) + 1e-10)

            # Decrease weights for poorly predicted source samples
            beta = 1.0 / (1 + np.sqrt(2 * np.log(n_source) / (n_iterations + 1)))
            source_weights = source_weights * np.power(beta, source_error)
            source_weights = source_weights / np.sum(source_weights)

        self.target_model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using transfer-learned model.

        Parameters
        ----------
        X : array
            Features for prediction

        Returns
        -------
        array
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_target first.")

        X_scaled = self.scaler.transform(X)

        if self.transfer_method == 'feature_extraction' and hasattr(self, '_feature_weights'):
            X_weighted = X_scaled * (1 - self.lambda_transfer +
                                    self.lambda_transfer * self._feature_weights)
            return self.target_model.predict(X_weighted)
        else:
            return self.target_model.predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance.

        Parameters
        ----------
        X : array
            Test features
        y : array
            True values

        Returns
        -------
        dict
            Performance metrics
        """
        y_pred = self.predict(X)

        return {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'correlation': np.corrcoef(y, y_pred)[0, 1]
        }


class MultiTaskLearning:
    """
    Multi-task learning for related phenotypes.

    Jointly learns models for multiple related traits to improve
    prediction by sharing information across tasks.
    """

    def __init__(self,
                 n_shared_factors: int = 10,
                 lambda_shared: float = 0.5):
        """
        Initialize multi-task learner.

        Parameters
        ----------
        n_shared_factors : int
            Number of shared latent factors
        lambda_shared : float
            Regularization weight for shared structure
        """
        self.n_shared_factors = n_shared_factors
        self.lambda_shared = lambda_shared
        self.task_models = {}
        self.shared_basis = None
        self.scaler = StandardScaler()

    def fit(self,
            X: np.ndarray,
            Y: np.ndarray,
            task_names: Optional[List[str]] = None) -> 'MultiTaskLearning':
        """
        Fit multi-task model.

        Parameters
        ----------
        X : array
            Shared features (n_samples x n_features)
        Y : array
            Task targets (n_samples x n_tasks)
        task_names : list, optional
            Names for each task

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape
        n_tasks = Y.shape[1]

        if task_names is None:
            task_names = [f'task_{i}' for i in range(n_tasks)]

        X_scaled = self.scaler.fit_transform(X)

        # Learn shared basis via SVD on concatenated task weights
        # First, fit individual models
        individual_weights = []
        for t in range(n_tasks):
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, Y[:, t])
            individual_weights.append(model.coef_)

        weight_matrix = np.array(individual_weights)

        # SVD to extract shared factors
        U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
        n_factors = min(self.n_shared_factors, len(S))
        self.shared_basis = Vt[:n_factors, :]

        # Re-fit models with shared regularization
        for t, name in enumerate(task_names):
            # Project features onto shared basis
            shared_features = X_scaled @ self.shared_basis.T
            task_specific = X_scaled - shared_features @ self.shared_basis

            # Combined features
            combined = np.hstack([
                self.lambda_shared * shared_features,
                (1 - self.lambda_shared) * task_specific
            ])

            model = Ridge(alpha=1.0)
            model.fit(combined, Y[:, t])
            self.task_models[name] = {
                'model': model,
                'task_idx': t
            }

        return self

    def predict(self, X: np.ndarray, task: str) -> np.ndarray:
        """
        Predict for a specific task.

        Parameters
        ----------
        X : array
            Features
        task : str
            Task name

        Returns
        -------
        array
            Predictions
        """
        if task not in self.task_models:
            raise ValueError(f"Unknown task: {task}")

        X_scaled = self.scaler.transform(X)

        # Apply same feature transformation
        shared_features = X_scaled @ self.shared_basis.T
        task_specific = X_scaled - shared_features @ self.shared_basis
        combined = np.hstack([
            self.lambda_shared * shared_features,
            (1 - self.lambda_shared) * task_specific
        ])

        return self.task_models[task]['model'].predict(combined)

    def predict_all(self, X: np.ndarray) -> pd.DataFrame:
        """
        Predict for all tasks.

        Parameters
        ----------
        X : array
            Features

        Returns
        -------
        pd.DataFrame
            Predictions for all tasks
        """
        predictions = {}
        for task in self.task_models:
            predictions[task] = self.predict(X, task)

        return pd.DataFrame(predictions)

    def get_shared_features(self, X: np.ndarray) -> np.ndarray:
        """
        Get shared feature representation.

        Parameters
        ----------
        X : array
            Original features

        Returns
        -------
        array
            Shared latent features
        """
        X_scaled = self.scaler.transform(X)
        return X_scaled @ self.shared_basis.T


class CrossPopulationTransfer:
    """
    Transfer learning across genetic populations.

    Adapts models trained on one population (e.g., European)
    to another (e.g., African, Asian).
    """

    def __init__(self,
                 transfer_method: str = 'weighted',
                 ld_correction: bool = True):
        """
        Initialize cross-population transfer.

        Parameters
        ----------
        transfer_method : str
            'weighted', 'meta', or 'joint'
        ld_correction : bool
            Whether to apply LD correction between populations
        """
        self.transfer_method = transfer_method
        self.ld_correction = ld_correction
        self.source_model = None
        self.target_model = None
        self.combined_model = None

    def fit(self,
            X_source: np.ndarray, y_source: np.ndarray,
            X_target: np.ndarray, y_target: np.ndarray,
            source_ld: Optional[np.ndarray] = None,
            target_ld: Optional[np.ndarray] = None,
            source_weight: float = 0.5) -> 'CrossPopulationTransfer':
        """
        Fit cross-population transfer model.

        Parameters
        ----------
        X_source : array
            Source population genotypes
        y_source : array
            Source population phenotypes
        X_target : array
            Target population genotypes
        y_target : array
            Target population phenotypes
        source_ld : array, optional
            LD matrix for source population
        target_ld : array, optional
            LD matrix for target population
        source_weight : float
            Weight for source population (0 to 1)

        Returns
        -------
        self
        """
        # Standardize
        scaler_source = StandardScaler()
        scaler_target = StandardScaler()

        X_source_scaled = scaler_source.fit_transform(X_source)
        X_target_scaled = scaler_target.fit_transform(X_target)

        self.scaler_source = scaler_source
        self.scaler_target = scaler_target

        if self.transfer_method == 'weighted':
            self._fit_weighted(X_source_scaled, y_source,
                             X_target_scaled, y_target,
                             source_weight)
        elif self.transfer_method == 'meta':
            self._fit_meta(X_source_scaled, y_source,
                          X_target_scaled, y_target)
        elif self.transfer_method == 'joint':
            self._fit_joint(X_source_scaled, y_source,
                           X_target_scaled, y_target,
                           source_ld, target_ld)
        else:
            raise ValueError(f"Unknown transfer method: {self.transfer_method}")

        return self

    def _fit_weighted(self,
                      X_source: np.ndarray, y_source: np.ndarray,
                      X_target: np.ndarray, y_target: np.ndarray,
                      source_weight: float):
        """Weighted combination of source and target models."""
        # Fit source model
        self.source_model = Ridge(alpha=1.0)
        self.source_model.fit(X_source, y_source)

        # Fit target model
        self.target_model = Ridge(alpha=1.0)
        self.target_model.fit(X_target, y_target)

        # Combined weights
        source_weights = self.source_model.coef_
        target_weights = self.target_model.coef_

        self.combined_weights = (source_weight * source_weights +
                                (1 - source_weight) * target_weights)

    def _fit_meta(self,
                  X_source: np.ndarray, y_source: np.ndarray,
                  X_target: np.ndarray, y_target: np.ndarray):
        """Meta-analysis approach combining effect estimates."""
        # Fit both models
        self.source_model = Ridge(alpha=1.0)
        self.source_model.fit(X_source, y_source)

        self.target_model = Ridge(alpha=1.0)
        self.target_model.fit(X_target, y_target)

        # Inverse-variance weighted meta-analysis
        source_weights = self.source_model.coef_
        target_weights = self.target_model.coef_

        # Estimate variances (simplified)
        n_source = len(y_source)
        n_target = len(y_target)

        var_source = np.var(y_source - self.source_model.predict(X_source)) / n_source
        var_target = np.var(y_target - self.target_model.predict(X_target)) / n_target

        # IVW meta-analysis
        w_source = 1 / (var_source + 1e-10)
        w_target = 1 / (var_target + 1e-10)

        self.combined_weights = ((w_source * source_weights + w_target * target_weights) /
                                (w_source + w_target))

    def _fit_joint(self,
                   X_source: np.ndarray, y_source: np.ndarray,
                   X_target: np.ndarray, y_target: np.ndarray,
                   source_ld: Optional[np.ndarray],
                   target_ld: Optional[np.ndarray]):
        """Joint modeling with LD correction."""
        # Stack data
        X_combined = np.vstack([X_source, X_target])
        y_combined = np.concatenate([y_source, y_target])

        # Create population indicator
        pop_indicator = np.concatenate([
            np.zeros(len(y_source)),
            np.ones(len(y_target))
        ])

        # Fit model with population as covariate
        X_with_pop = np.column_stack([X_combined, pop_indicator])

        self.combined_model = Ridge(alpha=1.0)
        self.combined_model.fit(X_with_pop, y_combined)

        # Extract combined weights (excluding population effect)
        self.combined_weights = self.combined_model.coef_[:-1]

    def predict(self, X: np.ndarray, population: str = 'target') -> np.ndarray:
        """
        Predict for new samples.

        Parameters
        ----------
        X : array
            Genotypes
        population : str
            'source' or 'target' population

        Returns
        -------
        array
            Predictions
        """
        if population == 'source':
            X_scaled = self.scaler_source.transform(X)
        else:
            X_scaled = self.scaler_target.transform(X)

        return X_scaled @ self.combined_weights


class GenomicsPretrainedModel:
    """
    Pre-trained genomics model for transfer learning.

    Leverages models pre-trained on large-scale genomics data
    (e.g., UK Biobank, GTEx) for new prediction tasks.
    """

    def __init__(self, pretrained_weights: Optional[Dict] = None):
        """
        Initialize with pre-trained weights.

        Parameters
        ----------
        pretrained_weights : dict, optional
            Pre-trained model weights and metadata
        """
        self.pretrained_weights = pretrained_weights or {}
        self.fine_tuned_models = {}

    def load_pretrained(self, model_name: str, weights_path: str) -> 'GenomicsPretrainedModel':
        """
        Load pre-trained weights from file.

        Parameters
        ----------
        model_name : str
            Name for the pretrained model
        weights_path : str
            Path to weights file

        Returns
        -------
        self
        """
        # In practice, would load from various formats
        try:
            weights = np.load(weights_path, allow_pickle=True)
            self.pretrained_weights[model_name] = {
                'weights': weights.get('weights', weights),
                'feature_names': weights.get('feature_names', None),
                'metadata': weights.get('metadata', {})
            }
        except Exception as e:
            warnings.warn(f"Could not load pretrained weights: {e}")

        return self

    def create_from_gwas(self,
                         model_name: str,
                         gwas_data: pd.DataFrame,
                         beta_col: str = 'beta',
                         se_col: str = 'se',
                         snp_col: str = 'SNP') -> 'GenomicsPretrainedModel':
        """
        Create pre-trained weights from GWAS summary statistics.

        Parameters
        ----------
        model_name : str
            Name for this pretrained model
        gwas_data : DataFrame
            GWAS summary statistics
        beta_col : str
            Column name for effect sizes
        se_col : str
            Column name for standard errors
        snp_col : str
            Column name for variant IDs

        Returns
        -------
        self
        """
        # LDpred-style weights
        # Simplified: just use beta values
        weights = gwas_data[beta_col].values
        snps = gwas_data[snp_col].tolist()

        self.pretrained_weights[model_name] = {
            'weights': weights,
            'feature_names': snps,
            'metadata': {
                'n_variants': len(weights),
                'source': 'gwas'
            }
        }

        return self

    def fine_tune(self,
                  model_name: str,
                  X: np.ndarray,
                  y: np.ndarray,
                  tune_fraction: float = 0.1,
                  regularization: float = 1.0) -> Dict:
        """
        Fine-tune pre-trained model on new data.

        Parameters
        ----------
        model_name : str
            Name of pretrained model to fine-tune
        X : array
            Training features
        y : array
            Training targets
        tune_fraction : float
            Fraction of weights to update (0=frozen, 1=full update)
        regularization : float
            L2 regularization strength towards pretrained weights

        Returns
        -------
        dict
            Fine-tuned weights and performance metrics
        """
        if model_name not in self.pretrained_weights:
            raise ValueError(f"No pretrained model named {model_name}")

        pretrained = self.pretrained_weights[model_name]['weights']

        # Ensure dimension match
        if len(pretrained) != X.shape[1]:
            warnings.warn(f"Dimension mismatch: pretrained has {len(pretrained)} features, "
                         f"X has {X.shape[1]}. Padding/truncating.")
            if len(pretrained) < X.shape[1]:
                pretrained = np.pad(pretrained, (0, X.shape[1] - len(pretrained)))
            else:
                pretrained = pretrained[:X.shape[1]]

        # Ridge regression with prior centered at pretrained weights
        # (y - X @ beta)^2 + lambda * (beta - pretrained)^2 * tune_fraction
        #                  + lambda * beta^2 * (1 - tune_fraction)

        # This is equivalent to augmented regression
        n_samples, n_features = X.shape

        # Regularization towards pretrained
        reg_strength = regularization * np.sqrt(n_samples)

        # Augmented system
        X_aug = np.vstack([
            X,
            reg_strength * tune_fraction * np.eye(n_features)
        ])
        y_aug = np.concatenate([
            y,
            reg_strength * tune_fraction * pretrained
        ])

        # Fit
        model = Ridge(alpha=regularization * (1 - tune_fraction), fit_intercept=True)
        model.fit(X_aug, y_aug)

        fine_tuned_weights = model.coef_

        # Store
        self.fine_tuned_models[model_name] = {
            'weights': fine_tuned_weights,
            'intercept': model.intercept_,
            'pretrained_weights': pretrained,
            'tune_fraction': tune_fraction
        }

        # Evaluate
        y_pred = X @ fine_tuned_weights + model.intercept_
        metrics = {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'weight_change': np.mean(np.abs(fine_tuned_weights - pretrained))
        }

        return metrics

    def predict(self, X: np.ndarray, model_name: str, use_fine_tuned: bool = True) -> np.ndarray:
        """
        Predict using pretrained or fine-tuned model.

        Parameters
        ----------
        X : array
            Features
        model_name : str
            Model name
        use_fine_tuned : bool
            Whether to use fine-tuned weights if available

        Returns
        -------
        array
            Predictions
        """
        if use_fine_tuned and model_name in self.fine_tuned_models:
            weights = self.fine_tuned_models[model_name]['weights']
            intercept = self.fine_tuned_models[model_name]['intercept']
            return X @ weights + intercept
        elif model_name in self.pretrained_weights:
            weights = self.pretrained_weights[model_name]['weights']
            return X @ weights
        else:
            raise ValueError(f"No model named {model_name}")


class ProgressiveTransfer:
    """
    Progressive transfer learning through intermediate domains.

    Enables transfer through a chain of related domains when
    direct transfer is difficult (e.g., EUR -> EAS -> AFR).
    """

    def __init__(self):
        """Initialize progressive transfer."""
        self.domain_chain = []
        self.models = []

    def add_domain(self,
                   name: str,
                   X: np.ndarray,
                   y: np.ndarray,
                   ld_matrix: Optional[np.ndarray] = None):
        """
        Add a domain to the transfer chain.

        Parameters
        ----------
        name : str
            Domain name
        X : array
            Features for this domain
        y : array
            Targets for this domain
        ld_matrix : array, optional
            LD matrix for this domain
        """
        self.domain_chain.append({
            'name': name,
            'X': X,
            'y': y,
            'ld_matrix': ld_matrix
        })

    def fit_progressive(self, lambda_decay: float = 0.8) -> 'ProgressiveTransfer':
        """
        Fit models progressively through domain chain.

        Parameters
        ----------
        lambda_decay : float
            Decay factor for transfer weight at each step

        Returns
        -------
        self
        """
        if len(self.domain_chain) < 2:
            raise ValueError("Need at least 2 domains for progressive transfer")

        current_lambda = 1.0

        for i in range(len(self.domain_chain)):
            domain = self.domain_chain[i]

            if i == 0:
                # First domain: standard fit
                model = TransferLearningModel(
                    transfer_method='feature_extraction',
                    lambda_transfer=0.0
                )
                model.fit_source(domain['X'], domain['y'])
                model.target_model = model.source_model
                model.is_fitted = True
            else:
                # Progressive transfer from previous model
                current_lambda *= lambda_decay

                model = TransferLearningModel(
                    transfer_method='fine_tuning',
                    lambda_transfer=current_lambda
                )

                # Use previous model as source
                prev_model = self.models[-1]
                model.source_model = prev_model.target_model
                model.source_weights = prev_model.target_model.coef_ if hasattr(
                    prev_model.target_model, 'coef_'
                ) else np.ones(domain['X'].shape[1])
                model.scaler = prev_model.scaler

                model.fit_target(domain['X'], domain['y'])

            self.models.append(model)

        return self

    def predict(self, X: np.ndarray, domain_idx: int = -1) -> np.ndarray:
        """
        Predict using model at specified domain.

        Parameters
        ----------
        X : array
            Features
        domain_idx : int
            Index of domain model to use (default: last/target)

        Returns
        -------
        array
            Predictions
        """
        return self.models[domain_idx].predict(X)

    def get_transfer_metrics(self) -> pd.DataFrame:
        """
        Get metrics for each transfer step.

        Returns
        -------
        pd.DataFrame
            Transfer performance at each step
        """
        metrics = []

        for i, (domain, model) in enumerate(zip(self.domain_chain, self.models)):
            eval_metrics = model.evaluate(domain['X'], domain['y'])
            metrics.append({
                'domain': domain['name'],
                'step': i,
                **eval_metrics
            })

        return pd.DataFrame(metrics)
