"""
Analysis module for causal genomics.

Contains statistical methods for causal inference including:
- Mendelian Randomization
- Colocalization analysis
- Fine-mapping
- TWAS (Transcriptome-Wide Association Study)
"""

from .mendelian_randomization import MendelianRandomization, MultivariableMR
from .colocalization import ColocalizationAnalysis, MultiTraitColocalization
from .fine_mapping import FineMappingAnalysis, MultiLocusFineMapping, AnnotationEnrichedFineMaping
from .twas import TWASAnalysis, MultiTissueTWAS, TWAS_FUSION, ColocTWAS

__all__ = [
    # Mendelian Randomization
    "MendelianRandomization",
    "MultivariableMR",
    # Colocalization
    "ColocalizationAnalysis",
    "MultiTraitColocalization",
    # Fine-mapping
    "FineMappingAnalysis",
    "MultiLocusFineMapping",
    "AnnotationEnrichedFineMaping",
    # TWAS
    "TWASAnalysis",
    "MultiTissueTWAS",
    "TWAS_FUSION",
    "ColocTWAS",
]
