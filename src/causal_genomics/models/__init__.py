"""
Models module for causal genomics.

Contains machine learning models for:
- Causal effect prediction
- Transfer learning across traits/tissues/populations
- Multi-task learning for related phenotypes
"""

from .causal_predictor import (
    CausalPredictor,
    EnsemblePredictor,
    CausalEffectRegressor,
    FeatureEngineering
)
from .transfer_learning import (
    TransferLearningModel,
    MultiTaskLearning,
    CrossPopulationTransfer,
    GenomicsPretrainedModel,
    ProgressiveTransfer
)

__all__ = [
    # Causal Prediction
    "CausalPredictor",
    "EnsemblePredictor",
    "CausalEffectRegressor",
    "FeatureEngineering",
    # Transfer Learning
    "TransferLearningModel",
    "MultiTaskLearning",
    "CrossPopulationTransfer",
    "GenomicsPretrainedModel",
    "ProgressiveTransfer",
]
