"""
Causal Effect Prediction Module

Machine learning models to predict causal gene-phenotype effects.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Any
import warnings


class CausalPredictor:
    """
    Predict causal effects using machine learning.
    
    Integrates genetic variants, expression data, network features,
    and perturbation screen results to predict gene-phenotype effects.
    """
    
    def __init__(self, 
                 model_type: str = 'xgboost',
                 **model_params):
        """
        Initialize predictor.
        
        Parameters
        ----------
        model_type : str
            'xgboost', 'gradient_boosting', or 'random_forest'
        model_params : dict
            Parameters for the chosen model
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model."""
        if self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            self.model = xgb.XGBRegressor(**params)
            
        elif self.model_type == 'gradient_boosting':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            self.model = GradientBoostingRegressor(**params)
            
        elif self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42
            }
            params = {**default_params, **self.model_params}
            self.model = RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_features(self,
                        genetic_features: Optional[pd.DataFrame] = None,
                        expression_features: Optional[pd.DataFrame] = None,
                        network_features: Optional[pd.DataFrame] = None,
                        annotation_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Combine and prepare features from multiple sources.
        
        Parameters
        ----------
        genetic_features : DataFrame
            Genetic variant features (e.g., number of variants, effect sizes)
        expression_features : DataFrame
            Gene expression features (e.g., tissue-specific expression)
        network_features : DataFrame
            Network topology features (e.g., degree, centrality)
        annotation_features : DataFrame
            Functional annotations (e.g., GO terms, pathways)
        
        Returns
        -------
        pd.DataFrame
            Combined feature matrix
        """
        feature_dfs = []
        
        if genetic_features is not None:
            feature_dfs.append(genetic_features)
        
        if expression_features is not None:
            feature_dfs.append(expression_features)
        
        if network_features is not None:
            feature_dfs.append(network_features)
        
        if annotation_features is not None:
            feature_dfs.append(annotation_features)
        
        if not feature_dfs:
            raise ValueError("Must provide at least one feature type")
        
        # Merge on gene/sample identifier
        combined = feature_dfs[0]
        for df in feature_dfs[1:]:
            combined = combined.merge(df, left_index=True, right_index=True, how='inner')
        
        self.feature_names = combined.columns.tolist()
        
        return combined
    
    def train(self,
             X: pd.DataFrame,
             y: pd.Series,
             validate: bool = True,
             cv_folds: int = 5) -> Dict:
        """
        Train the causal effect prediction model.
        
        Parameters
        ----------
        X : DataFrame
            Feature matrix
        y : Series
            Target causal effects (e.g., MR estimates)
        validate : bool
            Whether to perform cross-validation
        cv_folds : int
            Number of CV folds
        
        Returns
        -------
        dict
            Training metrics
        """
        # Store feature names
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Training performance
        y_pred_train = self.model.predict(X_scaled)
        train_metrics = {
            'train_r2': r2_score(y, y_pred_train),
            'train_mse': mean_squared_error(y, y_pred_train),
            'train_mae': mean_absolute_error(y, y_pred_train)
        }
        
        # Cross-validation
        if validate:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X_scaled, y, 
                                       cv=kf, scoring='r2')
            
            train_metrics['cv_r2_mean'] = cv_scores.mean()
            train_metrics['cv_r2_std'] = cv_scores.std()
        
        return train_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict causal effects for new data.
        
        Parameters
        ----------
        X : DataFrame
            Feature matrix
        
        Returns
        -------
        array
            Predicted causal effects
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top feature importances.
        
        Parameters
        ----------
        top_n : int
            Number of top features to return
        
        Returns
        -------
        pd.DataFrame
            Top features and their importances
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet")
        
        return self.feature_importance.head(top_n)
    
    def explain_prediction(self, X: pd.DataFrame, use_shap: bool = False) -> Dict:
        """
        Explain model predictions.
        
        Parameters
        ----------
        X : DataFrame
            Features for samples to explain
        use_shap : bool
            Whether to use SHAP values (requires shap package)
        
        Returns
        -------
        dict
            Feature contributions to predictions
        """
        if not use_shap:
            # Simple feature importance-based explanation
            predictions = self.predict(X)
            
            return {
                'predictions': predictions,
                'feature_importance': self.feature_importance
            }
        else:
            # SHAP values for detailed explanations
            try:
                import shap
            except ImportError:
                raise ImportError("SHAP package required for detailed explanations")
            
            X_scaled = self.scaler.transform(X)
            
            if self.model_type == 'xgboost':
                explainer = shap.TreeExplainer(self.model)
            else:
                explainer = shap.Explainer(self.model)
            
            shap_values = explainer(X_scaled)
            
            return {
                'predictions': self.predict(X),
                'shap_values': shap_values,
                'base_value': explainer.expected_value
            }


class EnsemblePredictor:
    """
    Ensemble of multiple causal effect predictors.
    
    Combines predictions from multiple models for improved performance.
    """
    
    def __init__(self):
        self.models = []
        self.weights = None
    
    def add_model(self, model: CausalPredictor, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append({'model': model, 'weight': weight})
    
    def train_all(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train all models in the ensemble."""
        results = []
        
        for i, model_dict in enumerate(self.models):
            model = model_dict['model']
            metrics = model.train(X, y)
            results.append({
                'model_idx': i,
                'model_type': model.model_type,
                **metrics
            })
        
        return pd.DataFrame(results)
    
    def predict(self, X: pd.DataFrame, method: str = 'weighted_average') -> np.ndarray:
        """
        Ensemble prediction.
        
        Parameters
        ----------
        X : DataFrame
            Features
        method : str
            'weighted_average', 'median', or 'stacking'
        
        Returns
        -------
        array
            Ensemble predictions
        """
        predictions = []
        weights = []
        
        for model_dict in self.models:
            model = model_dict['model']
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(model_dict['weight'])
        
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        if method == 'weighted_average':
            return np.average(predictions, axis=0, weights=weights)
        elif method == 'median':
            return np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")


class CausalEffectRegressor:
    """
    Specialized regressor for causal effect estimation.
    
    Incorporates uncertainty quantification and handles censored data.
    """
    
    def __init__(self):
        self.point_predictor = None
        self.uncertainty_predictor = None
    
    def train(self,
             X: pd.DataFrame,
             y: pd.Series,
             y_se: Optional[pd.Series] = None):
        """
        Train point estimate and uncertainty models.
        
        Parameters
        ----------
        X : DataFrame
            Features
        y : Series
            Causal effect estimates
        y_se : Series, optional
            Standard errors of effect estimates
        """
        # Point estimate model
        self.point_predictor = CausalPredictor(model_type='xgboost')
        self.point_predictor.train(X, y)
        
        # Uncertainty model (if SEs provided)
        if y_se is not None:
            self.uncertainty_predictor = CausalPredictor(model_type='xgboost')
            self.uncertainty_predictor.train(X, y_se)
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict causal effects with uncertainty estimates.
        
        Returns
        -------
        tuple
            (point_estimates, standard_errors)
        """
        point_estimates = self.point_predictor.predict(X)
        
        if self.uncertainty_predictor is not None:
            standard_errors = self.uncertainty_predictor.predict(X)
        else:
            # Use prediction intervals from base model
            standard_errors = np.std(point_estimates) * np.ones_like(point_estimates)
        
        return point_estimates, standard_errors
    
    def get_confidence_intervals(self,
                                 X: pd.DataFrame,
                                 confidence: float = 0.95) -> pd.DataFrame:
        """
        Get confidence intervals for predictions.
        
        Parameters
        ----------
        X : DataFrame
            Features
        confidence : float
            Confidence level (e.g., 0.95 for 95% CI)
        
        Returns
        -------
        pd.DataFrame
            Predictions with confidence intervals
        """
        point_est, se = self.predict_with_uncertainty(X)
        
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        return pd.DataFrame({
            'prediction': point_est,
            'se': se,
            'ci_lower': point_est - z * se,
            'ci_upper': point_est + z * se
        })


class FeatureEngineering:
    """
    Feature engineering for causal effect prediction.
    
    Creates informative features from raw genetic and omics data.
    """
    
    @staticmethod
    def create_genetic_features(variants_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from genetic variants.
        
        Parameters
        ----------
        variants_df : DataFrame
            Contains columns: effect_allele_freq, beta, pval, etc.
        
        Returns
        -------
        pd.DataFrame
            Engineered genetic features
        """
        features = pd.DataFrame(index=variants_df.index)
        
        # Number of significant variants
        features['n_variants'] = (variants_df['pval'] < 5e-8).sum()
        
        # Average effect size
        features['mean_beta'] = variants_df['beta'].mean()
        features['max_beta'] = variants_df['beta'].abs().max()
        
        # Allele frequency features
        features['mean_eaf'] = variants_df['effect_allele_freq'].mean()
        features['rare_variant_count'] = (variants_df['effect_allele_freq'] < 0.01).sum()
        
        return features
    
    @staticmethod
    def create_expression_features(expression_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from gene expression data.
        
        Parameters
        ----------
        expression_df : DataFrame
            Gene expression matrix (genes x tissues/samples)
        
        Returns
        -------
        pd.DataFrame
            Expression-based features
        """
        features = pd.DataFrame(index=expression_df.index)
        
        # Expression statistics
        features['mean_expression'] = expression_df.mean(axis=1)
        features['std_expression'] = expression_df.std(axis=1)
        features['max_expression'] = expression_df.max(axis=1)
        
        # Tissue specificity (tau)
        exp_norm = expression_df.div(expression_df.max(axis=1), axis=0)
        features['tau'] = (1 - exp_norm).sum(axis=1) / (expression_df.shape[1] - 1)
        
        return features
    
    @staticmethod
    def create_network_features(network_graph: Any) -> pd.DataFrame:
        """
        Create features from gene regulatory network.
        
        Parameters
        ----------
        network_graph : networkx.Graph
            Gene network
        
        Returns
        -------
        pd.DataFrame
            Network topology features
        """
        import networkx as nx
        
        nodes = list(network_graph.nodes())
        features = pd.DataFrame(index=nodes)
        
        # Degree centrality
        degree_cent = nx.degree_centrality(network_graph)
        features['degree_centrality'] = pd.Series(degree_cent)
        
        # Betweenness centrality
        between_cent = nx.betweenness_centrality(network_graph)
        features['betweenness_centrality'] = pd.Series(between_cent)
        
        # Clustering coefficient
        clustering = nx.clustering(network_graph)
        features['clustering_coefficient'] = pd.Series(clustering)
        
        return features
