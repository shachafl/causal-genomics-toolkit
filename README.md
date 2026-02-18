# Causal Genomics Toolkit

A comprehensive Python framework for discovering and validating causal gene-phenotype relationships using multi-omics data, GWAS/xQTL summary statistics, and perturbation experiments.

## Features

### 1. Causal Discovery
- **Mendelian Randomization (MR)**: Two-sample and multi-variable MR with multiple sensitivity analyses
- **Colocalization Analysis**: Identify shared causal variants across traits
- **Fine-mapping**: Statistical fine-mapping to identify causal variants
- **Transcriptome-Wide Association Studies (TWAS)**: Gene-level associations using predicted expression

### 2. Multi-Omics Integration
- **Data Harmonization**: Integrate GWAS, eQTL, pQTL, mQTL data
- **Network Analysis**: Build gene regulatory and protein interaction networks
- **Pathway Enrichment**: Identify enriched biological pathways
- **Disease Subtyping**: Clustering and classification of patient subgroups
- **Biomarker Discovery**: Identify candidate biomarkers using causal inference

### 3. Predictive Modeling
- **Causal Effect Prediction**: ML models to predict gene-phenotype effect sizes
- **Transfer Learning**: Leverage perturbation screen data for prediction
- **Feature Engineering**: Integrate genetic variation, expression, network topology
- **Model Interpretation**: SHAP values and feature importance analysis

## Installation

```bash
git clone https://github.com/shachafl/causal-genomics-toolkit.git
cd causal-genomics-toolkit
pip install -e .
```

## Quick Start

```python
from causal_genomics import MendelianRandomization, MultiOmicsIntegrator, CausalPredictor

# Example 1: Mendelian Randomization
mr = MendelianRandomization()
mr.load_exposure_gwas('exposure_sumstats.txt')
mr.load_outcome_gwas('outcome_sumstats.txt')
results = mr.run_analysis(methods=['ivw', 'egger', 'weighted_median'])

# Example 2: Multi-omics integration
integrator = MultiOmicsIntegrator()
integrator.add_layer('gwas', gwas_data)
integrator.add_layer('eqtl', eqtl_data)
integrator.add_layer('pqtl', pqtl_data)
network = integrator.build_causal_network()

# Example 3: Causal effect prediction
predictor = CausalPredictor()
predictor.train(features, labels)
predictions = predictor.predict(new_features)
```

## Project Structure

```
causal-genomics-toolkit/
├── src/causal_genomics/
│   ├── data/              # Data loading and preprocessing
│   ├── analysis/          # Core analysis modules
│   │   ├── mendelian_randomization.py
│   │   ├── colocalization.py
│   │   ├── fine_mapping.py
│   │   └── twas.py
│   ├── models/            # Predictive models
│   │   ├── causal_predictor.py
│   │   └── transfer_learning.py
│   ├── utils/             # Utility functions
│   └── visualization/     # Plotting functions
├── tests/                 # Unit tests
├── examples/             # Example notebooks and scripts
├── data/                 # Sample data
└── docs/                 # Documentation
```

## Key Methods

### Mendelian Randomization
- Inverse-variance weighted (IVW)
- MR-Egger with pleiotropy test
- Weighted median
- MR-PRESSO for outlier detection
- Multivariable MR

### Network Analysis
- Gene regulatory network inference
- Protein-protein interaction networks
- Pathway enrichment (GO, KEGG, Reactome)
- Network-based disease module identification

### Machine Learning
- Gradient boosting for causal effect prediction
- Neural networks for multi-task learning
- SHAP-based model interpretation
- Cross-validation with stratification

## Example Use Cases

1. **Identify causal genes for complex disease**: Integrate GWAS with eQTLs to find genes whose expression causally affects disease risk

2. **Discover therapeutic targets**: Use MR and colocalization to prioritize genes with evidence of causality and druggability

3. **Predict perturbation effects**: Train models on CRISPR screens to predict effects of novel perturbations

4. **Stratify patients**: Identify disease subtypes using multi-omics clustering

## Dependencies

- numpy, pandas, scipy
- scikit-learn, xgboost
- statsmodels
- networkx
- matplotlib, seaborn
- biopython (for biological annotations)

## Citation

If you use this toolkit in your research, please cite:

```
[Lior Shachaf] (2025). Causal Genomics Toolkit: A Python framework for 
causal inference in genomics. GitHub repository.
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
