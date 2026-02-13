# Quick Start Guide

Get started with Causal Genomics Toolkit in 5 minutes!

## Installation

```bash
git clone https://github.com/yourusername/causal-genomics-toolkit.git
cd causal-genomics-toolkit
pip install -e .
```

## Example 1: Simple Mendelian Randomization

```python
import pandas as pd
from causal_genomics import MendelianRandomization

# Load your GWAS summary statistics
exposure_data = pd.read_csv('exposure_gwas.txt', sep='\t')
outcome_data = pd.read_csv('outcome_gwas.txt', sep='\t')

# Initialize MR analysis
mr = MendelianRandomization(p_threshold=5e-8)
mr.load_exposure_gwas(exposure_data)
mr.load_outcome_gwas(outcome_data)

# Run analysis with multiple methods
results = mr.run_analysis(methods=['ivw', 'egger', 'weighted_median'])
print(results)
```

## Example 2: Test Colocalization

```python
from causal_genomics import ColocalizationAnalysis

# Initialize colocalization
coloc = ColocalizationAnalysis()

# Run analysis
result = coloc.coloc_abf(
    trait1_beta=eqtl_beta,
    trait1_se=eqtl_se,
    trait2_beta=gwas_beta,
    trait2_se=gwas_se
)

# Check if traits colocalize
if result['H4_shared_causal'] > 0.8:
    print("Strong evidence for shared causal variant!")
```

## Example 3: Predict Causal Effects

```python
from causal_genomics import CausalPredictor

# Prepare features
features = predictor.prepare_features(
    genetic_features=genetic_data,
    expression_features=expression_data,
    network_features=network_data
)

# Train model
predictor = CausalPredictor(model_type='xgboost')
metrics = predictor.train(features, causal_effects)

# Predict for new genes
predictions = predictor.predict(new_features)
```

## Example 4: Multi-Omics Integration

```python
from causal_genomics import MultiOmicsIntegrator

# Initialize integrator
integrator = MultiOmicsIntegrator()

# Add data layers
integrator.add_layer('gwas', gwas_data)
integrator.add_layer('eqtl', eqtl_data)
integrator.add_layer('pqtl', pqtl_data)

# Build causal network
network = integrator.build_causal_network()

# Identify disease modules
modules = integrator.identify_disease_modules(disease_genes)

# Find enriched pathways
pathways = integrator.pathway_enrichment(causal_genes)
```

## Example 5: Identify Biomarkers

```python
# Load case-control data
cases = pd.read_csv('cases_proteomics.csv', index_col=0)
controls = pd.read_csv('controls_proteomics.csv', index_col=0)

# Identify biomarkers
biomarkers = integrator.identify_biomarkers(
    cases, 
    controls,
    method='differential',
    n_biomarkers=50
)

print("Top biomarkers:")
print(biomarkers.head(10))
```

## Data Format Requirements

### GWAS Summary Statistics
Required columns:
- `SNP`: Variant identifier
- `CHR`: Chromosome
- `POS`: Position
- `A1`: Effect allele
- `A2`: Other allele
- `BETA`: Effect size
- `SE`: Standard error
- `P`: P-value

### eQTL/pQTL Data
Required columns:
- `SNP`: Variant identifier
- `gene`: Gene identifier
- `beta`: Effect size
- `se`: Standard error
- `pval`: P-value

## Next Steps

1. Check out the [examples/](examples/) directory for detailed tutorials
2. Read the [full documentation](docs/)
3. See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Getting Help

- Open an issue on GitHub
- Check existing issues and discussions
- Read the full documentation

## Citation

If you use this toolkit in your research, please cite:

```
[Your Name] (2025). Causal Genomics Toolkit: A Python framework for 
causal inference in genomics. GitHub: https://github.com/yourusername/causal-genomics-toolkit
```
