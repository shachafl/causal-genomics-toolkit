"""
Causal Genomics Toolkit

A comprehensive Python framework for discovering and validating causal
gene-phenotype relationships using multi-omics data, GWAS/xQTL summary
statistics, and perturbation experiments.
"""

__version__ = "0.1.0"
__author__ = "Lior Shachaf"

from .analysis.mendelian_randomization import MendelianRandomization
from .analysis.colocalization import ColocalizationAnalysis
from .analysis.fine_mapping import FineMappingAnalysis, MultiLocusFineMapping
from .analysis.twas import TWASAnalysis, MultiTissueTWAS
from .models.causal_predictor import CausalPredictor
from .models.transfer_learning import TransferLearningModel, MultiTaskLearning, CrossPopulationTransfer
from .data.data_loader import GWASDataLoader, QTLDataLoader
from .utils.multi_omics_integrator import MultiOmicsIntegrator

__all__ = [
    # Analysis
    "MendelianRandomization",
    "ColocalizationAnalysis",
    "FineMappingAnalysis",
    "MultiLocusFineMapping",
    "TWASAnalysis",
    "MultiTissueTWAS",
    # Models
    "CausalPredictor",
    "TransferLearningModel",
    "MultiTaskLearning",
    "CrossPopulationTransfer",
    # Data
    "GWASDataLoader",
    "QTLDataLoader",
    # Utils
    "MultiOmicsIntegrator",
]
