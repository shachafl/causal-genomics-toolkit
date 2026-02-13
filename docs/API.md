# API Documentation

## Core Modules

### 1. Mendelian Randomization (`causal_genomics.analysis.mendelian_randomization`)

#### `MendelianRandomization`

Main class for performing Mendelian Randomization analyses.

**Methods:**

- `load_exposure_gwas(data, snp_col='SNP', beta_col='beta', ...)`: Load exposure GWAS data
- `load_outcome_gwas(data, ...)`: Load outcome GWAS data  
- `harmonize_data()`: Harmonize exposure and outcome datasets
- `ivw()`: Inverse-variance weighted MR
- `egger()`: MR-Egger with pleiotropy test
- `weighted_median()`: Weighted median MR
- `mr_presso(outlier_threshold=0.05)`: MR-PRESSO outlier detection
- `run_analysis(methods=['ivw', 'egger', 'weighted_median'])`: Run multiple methods
- `heterogeneity_test()`: Cochran's Q test for heterogeneity

**Example:**
```python
mr = MendelianRandomization(p_threshold=5e-8)
mr.load_exposure_gwas(exposure_data)
mr.load_outcome_gwas(outcome_data)
results = mr.run_analysis()
```

---

### 2. Fine-Mapping (`causal_genomics.analysis.fine_mapping`)

#### `FineMappingAnalysis`

Statistical fine-mapping to identify causal variants within associated loci.

**Parameters:**
- `max_causal`: Maximum number of causal variants to consider (default: 10)
- `coverage`: Target coverage for credible sets (default: 0.95)
- `min_abs_corr`: Minimum correlation for credible set purity (default: 0.5)

**Methods:**

- `susie(z_scores, ld_matrix, n, var_y=1.0, prior_variance=0.2, max_iter=100)`: SuSiE fine-mapping
- `abf_finemap(beta, se, prior_variance=0.04)`: Approximate Bayes Factor fine-mapping
- `conditional_analysis(z_scores, ld_matrix, pval_threshold=5e-8, max_signals=10)`: Stepwise conditional analysis

**Returns:** Dictionary with:
- `pip`: Posterior inclusion probabilities for each variant
- `credible_sets`: List of credible sets with coverage and purity
- `alpha`: Posterior probability matrix (L x p)

**Example:**
```python
from causal_genomics import FineMappingAnalysis

fm = FineMappingAnalysis(max_causal=5, coverage=0.95)
result = fm.susie(z_scores, ld_matrix, n=50000)

# Get high-confidence causal variants
for cs in result['credible_sets']:
    if cs['is_pure']:
        print(f"Credible set: {cs['variants']}, coverage: {cs['coverage']:.2f}")
```

#### `MultiLocusFineMapping`

Fine-mapping across multiple independent loci.

**Methods:**
- `add_locus(locus_name, z_scores, ld_matrix, variant_ids)`: Add a locus
- `run_finemapping(method='susie', n=None)`: Run fine-mapping on all loci
- `get_high_pip_variants(pip_threshold=0.5)`: Get likely causal variants
- `get_credible_sets_summary()`: Summary of all credible sets

#### `AnnotationEnrichedFineMaping`

Fine-mapping with functional annotation priors.

**Methods:**
- `set_annotation_prior(annotations, enrichment_scores)`: Set annotation-based priors
- `finemap_with_priors(beta, se)`: Fine-mapping incorporating annotations

---

### 3. Colocalization (`causal_genomics.analysis.colocalization`)

#### `ColocalizationAnalysis`

Bayesian colocalization to test if two traits share causal variants.

**Methods:**

- `coloc_abf(trait1_beta, trait1_se, trait2_beta, trait2_se, ...)`: Approximate Bayes Factor colocalization
- `ecaviar(trait1_zscore, trait2_zscore, ld_matrix, ...)`: eCAVIAR fine-mapping
- `smr_heidi(eqtl_beta, eqtl_se, gwas_beta, gwas_se, ld_matrix, ...)`: SMR with HEIDI test

**Returns:** Dictionary with posterior probabilities for:
- H0: No association
- H1: Trait 1 only
- H2: Trait 2 only
- H3: Distinct causal variants
- H4: Shared causal variant (colocalization)

**Example:**
```python
coloc = ColocalizationAnalysis()
result = coloc.coloc_abf(eqtl_beta, eqtl_se, gwas_beta, gwas_se)
if result['H4_shared_causal'] > 0.8:
    print("Strong colocalization evidence")
```

---

### 4. TWAS (`causal_genomics.analysis.twas`)

#### `TWASAnalysis`

Transcriptome-Wide Association Study for gene-level associations.

**Methods:**

- `train_expression_model(genotypes, expression, gene_id, method='elastic_net')`: Train expression prediction model
- `predixcan(genotypes, phenotype, gene_weights, covariates=None)`: Individual-level PrediXcan
- `s_predixcan(gwas_z, ld_matrix, weights, gene_id, n_gwas, model_r2)`: Summary-based S-PrediXcan
- `run_s_predixcan(gwas_data, ld_matrices, n_gwas)`: Run S-PrediXcan for all genes

**Example:**
```python
from causal_genomics import TWASAnalysis

twas = TWASAnalysis(alpha=0.05)

# Train expression models
twas.train_expression_model(genotypes, expression, 'GENE1')

# Summary-based TWAS
result = twas.s_predixcan(
    gwas_z=gwas_z_scores,
    ld_matrix=ld_matrix,
    weights=expression_weights,
    gene_id='GENE1',
    n_gwas=50000,
    model_r2=0.1
)
print(f"TWAS Z: {result['z_twas']:.2f}, P: {result['pval']:.2e}")
```

#### `MultiTissueTWAS`

Combine TWAS results across multiple tissues.

**Methods:**
- `add_tissue_result(tissue, results)`: Add tissue-specific TWAS results
- `combine_tissues_fisher()`: Fisher's method for p-value combination
- `combine_tissues_stouffer(weights=None)`: Weighted Stouffer's method
- `get_tissue_specific_genes(pval_threshold, tissue_specificity_ratio)`: Identify tissue-specific genes

**Example:**
```python
from causal_genomics.analysis.twas import MultiTissueTWAS

mt_twas = MultiTissueTWAS()
mt_twas.add_tissue_result('Brain', brain_results)
mt_twas.add_tissue_result('Liver', liver_results)

combined = mt_twas.combine_tissues_stouffer(
    weights={'Brain': 2.0, 'Liver': 1.0}
)
```

#### `ColocTWAS`

Combined TWAS and colocalization prioritization.

**Methods:**
- `integrate_twas_coloc(twas_df, coloc_df)`: Combine TWAS and colocalization evidence

---

### 5. Causal Predictor (`causal_genomics.models.causal_predictor`)

#### `CausalPredictor`

Machine learning models to predict causal gene-phenotype effects.

**Parameters:**
- `model_type`: 'xgboost', 'gradient_boosting', or 'random_forest'
- `**model_params`: Model-specific parameters

**Methods:**

- `prepare_features(genetic_features=None, expression_features=None, network_features=None, annotation_features=None)`: Combine features
- `train(X, y, validate=True, cv_folds=5)`: Train the model
- `predict(X)`: Predict causal effects
- `get_feature_importance(top_n=20)`: Get top features
- `explain_prediction(X, use_shap=False)`: Explain predictions

**Example:**
```python
predictor = CausalPredictor(model_type='xgboost')
metrics = predictor.train(features, causal_effects)
predictions = predictor.predict(new_features)
importance = predictor.get_feature_importance()
```

---

### 6. Transfer Learning (`causal_genomics.models.transfer_learning`)

#### `TransferLearningModel`

Transfer learning for cross-trait, cross-tissue, or cross-population prediction.

**Parameters:**
- `base_model`: 'ridge', 'elastic_net', 'gradient_boosting', or 'random_forest'
- `transfer_method`: 'feature_extraction', 'fine_tuning', or 'domain_adaptation'
- `lambda_transfer`: Transfer regularization weight (0=no transfer, 1=full transfer)

**Methods:**
- `fit_source(X_source, y_source)`: Fit on source domain data
- `fit_target(X_target, y_target, X_source=None, y_source=None)`: Transfer to target domain
- `predict(X)`: Predict using transfer-learned model
- `evaluate(X, y)`: Evaluate model performance

**Example:**
```python
from causal_genomics import TransferLearningModel

# Transfer from large GWAS to small cohort
model = TransferLearningModel(
    transfer_method='fine_tuning',
    lambda_transfer=0.7
)

model.fit_source(X_large_gwas, y_large_gwas)
model.fit_target(X_small_cohort, y_small_cohort)

predictions = model.predict(X_new)
metrics = model.evaluate(X_test, y_test)
```

#### `MultiTaskLearning`

Joint learning across related phenotypes.

**Parameters:**
- `n_shared_factors`: Number of shared latent factors (default: 10)
- `lambda_shared`: Weight for shared structure (default: 0.5)

**Methods:**
- `fit(X, Y, task_names=None)`: Fit multi-task model
- `predict(X, task)`: Predict for specific task
- `predict_all(X)`: Predict for all tasks
- `get_shared_features(X)`: Extract shared feature representation

**Example:**
```python
from causal_genomics import MultiTaskLearning

mtl = MultiTaskLearning(n_shared_factors=10)
mtl.fit(X_genotypes, Y_phenotypes, task_names=['BMI', 'T2D', 'HDL'])

# Predict for specific phenotype
bmi_pred = mtl.predict(X_new, 'BMI')

# Get shared genetic factors
shared = mtl.get_shared_features(X_new)
```

#### `CrossPopulationTransfer`

Transfer learning across genetic populations.

**Parameters:**
- `transfer_method`: 'weighted', 'meta', or 'joint'
- `ld_correction`: Whether to apply LD correction

**Methods:**
- `fit(X_source, y_source, X_target, y_target, source_weight=0.5)`: Fit cross-population model
- `predict(X, population='target')`: Predict for target population

**Example:**
```python
from causal_genomics import CrossPopulationTransfer

# Transfer from EUR to AFR population
cpt = CrossPopulationTransfer(transfer_method='meta')
cpt.fit(X_eur, y_eur, X_afr, y_afr)
predictions = cpt.predict(X_afr_new, population='target')
```

#### `GenomicsPretrainedModel`

Use pre-trained models from large-scale genomics studies.

**Methods:**
- `create_from_gwas(model_name, gwas_data)`: Create weights from GWAS summary stats
- `fine_tune(model_name, X, y, tune_fraction=0.1)`: Fine-tune on new data
- `predict(X, model_name, use_fine_tuned=True)`: Predict using model

#### `ProgressiveTransfer`

Chain-of-domains progressive transfer (e.g., EUR → EAS → AFR).

**Methods:**
- `add_domain(name, X, y, ld_matrix=None)`: Add domain to chain
- `fit_progressive(lambda_decay=0.8)`: Fit through domain chain
- `predict(X, domain_idx=-1)`: Predict at specific domain
- `get_transfer_metrics()`: Get R² at each transfer step

---

### 7. Multi-Omics Integrator (`causal_genomics.utils.multi_omics_integrator`)

#### `MultiOmicsIntegrator`

Integrate multiple omics layers for systems-level analysis.

**Methods:**

- `add_layer(name, data, layer_type='association')`: Add omics layer
- `build_causal_network(method='multi_layer', significance_threshold=0.05)`: Build network
- `identify_disease_modules(disease_genes, method='community')`: Find disease modules
- `pathway_enrichment(gene_list, pathway_database='GO')`: Pathway analysis
- `identify_biomarkers(case_data, control_data, method='differential', n_biomarkers=50)`: Find biomarkers
- `cluster_samples(data, n_clusters=3, method='kmeans')`: Cluster for subtypes
- `integrate_perturbation_data(crispr_data, observational_data)`: Integrate screens

**Example:**
```python
integrator = MultiOmicsIntegrator()
integrator.add_layer('gwas', gwas_data)
integrator.add_layer('eqtl', eqtl_data)
network = integrator.build_causal_network()
modules = integrator.identify_disease_modules(disease_genes)
biomarkers = integrator.identify_biomarkers(cases, controls)
```

---

### 8. Data Loaders (`causal_genomics.data.data_loader`)

#### `GWASDataLoader`

Load and preprocess GWAS summary statistics.

**Methods:**
- `load_from_file(filepath, format='standard')`: Load from file
- `load_from_gwas_catalog(study_id)`: Download from GWAS Catalog
- `clump(r2_threshold=0.1, kb_distance=500)`: LD clumping

#### `QTLDataLoader`

Load QTL (eQTL, pQTL, mQTL) data.

**Methods:**
- `load_eqtl(filepath, tissue=None)`: Load eQTL data
- `load_from_gtex(gene, tissue)`: Load from GTEx
- `get_cis_qtls(window_kb=1000)`: Filter cis-QTLs

#### `PerturbationDataLoader`

Load CRISPR/RNAi screen data.

**Methods:**
- `load_crispr_screen(filepath, screen_type='knockout')`: Load screen
- `load_from_depmap(cell_line=None)`: Load from DepMap

---

### 9. Visualization (`causal_genomics.visualization.plots`)

#### `MRVisualizer`

Visualize MR results.

**Methods:**
- `forest_plot(results, figsize=(10,8), save_path=None)`: Forest plot
- `scatter_plot(exposure_beta, outcome_beta, ...)`: MR scatter plot
- `funnel_plot(beta, se, ...)`: Funnel plot for pleiotropy

#### `NetworkVisualizer`

Visualize networks and pathways.

**Methods:**
- `plot_network(G, node_color=None, ...)`: Plot gene network
- `plot_pathway_enrichment(enrichment_results, top_n=15, ...)`: Bar plot of pathways

#### `MultiOmicsVisualizer`

Multi-omics visualizations.

**Methods:**
- `heatmap(data, figsize=(12,10), ...)`: Heatmap
- `manhattan_plot(data, chr_col='CHR', pos_col='POS', pval_col='P', ...)`: Manhattan plot
- `volcano_plot(data, beta_col='beta', pval_col='pval', ...)`: Volcano plot

---

## Data Format Specifications

### GWAS Summary Statistics

Standard format requires these columns:

| Column | Type | Description |
|--------|------|-------------|
| SNP | str | Variant identifier (rsID) |
| CHR | int | Chromosome (1-22, X, Y) |
| POS | int | Base pair position |
| A1 | str | Effect allele |
| A2 | str | Other allele |
| BETA | float | Effect size |
| SE | float | Standard error |
| P | float | P-value |
| EAF | float | Effect allele frequency (optional) |

### eQTL/pQTL Data

| Column | Type | Description |
|--------|------|-------------|
| SNP | str | Variant identifier |
| gene | str | Gene symbol or ID |
| beta | float | Effect on expression/protein |
| se | float | Standard error |
| pval | float | P-value |
| tissue | str | Tissue type (optional) |

### CRISPR Screen Data

| Column | Type | Description |
|--------|------|-------------|
| gene | str | Gene symbol |
| log2fc | float | Log2 fold change |
| pval | float | P-value |
| fdr | float | FDR-adjusted p-value |

---

## Common Workflows

### Workflow 1: Basic MR Analysis

```python
# 1. Load data
mr = MendelianRandomization()
mr.load_exposure_gwas(exposure)
mr.load_outcome_gwas(outcome)

# 2. Run analysis
results = mr.run_analysis()

# 3. Check heterogeneity
het = mr.heterogeneity_test()

# 4. Visualize
from causal_genomics.visualization import MRVisualizer
viz = MRVisualizer()
viz.forest_plot(results, save_path='mr_results.png')
```

### Workflow 2: Colocalization + MR

```python
# 1. MR to test causality
mr_result = mr.ivw()

# 2. Colocalization to validate
coloc = ColocalizationAnalysis()
coloc_result = coloc.coloc_abf(...)

# 3. Combined evidence
if mr_result['pval'] < 0.05 and coloc_result['H4_shared_causal'] > 0.8:
    print("Strong causal evidence with shared variant")
```

### Workflow 3: Multi-Omics Discovery

```python
# 1. Integrate data layers
integrator = MultiOmicsIntegrator()
integrator.add_layer('gwas', gwas)
integrator.add_layer('eqtl', eqtl)
integrator.add_layer('pqtl', pqtl)

# 2. Build network
network = integrator.build_causal_network()

# 3. Find modules
modules = integrator.identify_disease_modules(disease_genes)

# 4. Pathway enrichment
pathways = integrator.pathway_enrichment(module_genes)

# 5. Identify biomarkers
biomarkers = integrator.identify_biomarkers(cases, controls)
```

### Workflow 4: Prediction Pipeline

```python
# 1. Engineer features
from causal_genomics.models import FeatureEngineering
fe = FeatureEngineering()
genetic_feat = fe.create_genetic_features(variants)
expression_feat = fe.create_expression_features(expression)
network_feat = fe.create_network_features(network)

# 2. Train predictor
predictor = CausalPredictor()
features = predictor.prepare_features(
    genetic_features=genetic_feat,
    expression_features=expression_feat,
    network_features=network_feat
)
predictor.train(features, known_effects)

# 3. Predict novel effects
predictions = predictor.predict(new_features)

# 4. Interpret
importance = predictor.get_feature_importance()
```

### Workflow 5: Fine-Mapping a GWAS Locus

```python
from causal_genomics import FineMappingAnalysis, MultiLocusFineMapping

# Single locus fine-mapping
fm = FineMappingAnalysis(max_causal=5, coverage=0.95)
result = fm.susie(
    z_scores=gwas_z_scores,
    ld_matrix=ld_matrix,
    n=50000
)

# Identify high-confidence causal variants
for i, pip in enumerate(result['pip']):
    if pip > 0.5:
        print(f"Variant {i}: PIP = {pip:.3f}")

# Multi-locus analysis
mfm = MultiLocusFineMapping()
for locus_name, locus_data in loci.items():
    mfm.add_locus(locus_name, locus_data['z'], locus_data['ld'], locus_data['snps'])

results_df = mfm.run_finemapping(method='susie', n=50000)
high_pip = mfm.get_high_pip_variants(pip_threshold=0.5)
```

### Workflow 6: TWAS Analysis

```python
from causal_genomics import TWASAnalysis
from causal_genomics.analysis.twas import MultiTissueTWAS

# Summary-based TWAS
twas = TWASAnalysis()

# Run S-PrediXcan for each gene
results = []
for gene, weights in expression_weights.items():
    result = twas.s_predixcan(
        gwas_z=gwas_z,
        ld_matrix=ld_matrices[gene],
        weights=weights,
        gene_id=gene,
        n_gwas=50000,
        model_r2=model_r2[gene]
    )
    results.append(result)

# Combine across tissues
mt = MultiTissueTWAS()
mt.add_tissue_result('Brain', brain_twas)
mt.add_tissue_result('Liver', liver_twas)
combined = mt.combine_tissues_stouffer()

# Identify tissue-specific genes
specific = mt.get_tissue_specific_genes(pval_threshold=0.05)
```

### Workflow 7: Cross-Population Transfer Learning

```python
from causal_genomics import TransferLearningModel, CrossPopulationTransfer

# Transfer from EUR GWAS to AFR prediction
cpt = CrossPopulationTransfer(transfer_method='meta')
cpt.fit(
    X_eur, y_eur,  # Large EUR training data
    X_afr, y_afr   # Small AFR training data
)

# Predict in AFR population
afr_predictions = cpt.predict(X_afr_test)

# Or use progressive transfer: EUR -> EAS -> AFR
from causal_genomics.models.transfer_learning import ProgressiveTransfer

pt = ProgressiveTransfer()
pt.add_domain('EUR', X_eur, y_eur)
pt.add_domain('EAS', X_eas, y_eas)
pt.add_domain('AFR', X_afr, y_afr)

pt.fit_progressive(lambda_decay=0.8)
metrics = pt.get_transfer_metrics()
```

---

## Performance Tips

1. **Large datasets**: Use data loaders' clumping functionality to reduce SNPs
2. **Multiple traits**: Use `MultiTraitColocalization` for efficient pairwise analysis
3. **Feature engineering**: Pre-compute network features for faster prediction
4. **Cross-validation**: Set `cv_folds=5` for validation without excessive compute
5. **Parallel processing**: Most methods are numpy-based and benefit from BLAS parallelization

## Troubleshooting

**Issue**: MR returns NaN values
- **Solution**: Check that exposure and outcome have overlapping SNPs after harmonization

**Issue**: Colocalization PP.H4 always near 0
- **Solution**: Verify you have sufficient SNPs (>10) in the region and they show association with at least one trait

**Issue**: Predictor low accuracy
- **Solution**: Ensure adequate training data (>100 samples), check for feature-target leakage, try different model types

**Issue**: Network visualization too cluttered
- **Solution**: Filter to top N most significant edges, use subgraph extraction for specific genes

**Issue**: SuSiE returns empty credible sets
- **Solution**: Check that z-scores contain genome-wide significant signals; increase `max_causal` parameter

**Issue**: TWAS model R² very low
- **Solution**: Ensure sufficient cis-eQTL signal; try different regularization methods (elastic_net vs ridge)

**Issue**: Transfer learning doesn't improve performance
- **Solution**: Increase `lambda_transfer` for more source influence; ensure source and target domains are related

**Issue**: Cross-population transfer produces poor predictions
- **Solution**: Check for population-specific effects; try 'meta' method for more balanced combination

## Advanced Topics

See the examples/ directory for advanced tutorials on:
- Multivariable MR with correlated exposures
- Fine-mapping with functional annotations (AnnotationEnrichedFineMaping)
- SuSiE fine-mapping for multiple causal variants
- Multi-tissue TWAS with Fisher/Stouffer combination
- Transfer learning across populations (EUR → EAS → AFR)
- Multi-task learning for correlated phenotypes
- Network-based drug repurposing
- Patient stratification pipelines

## New in v0.1.0

- **Fine-Mapping Module**: SuSiE, ABF, conditional analysis, annotation-enriched fine-mapping
- **TWAS Module**: PrediXcan, S-PrediXcan, multi-tissue combination, FUSION-style conditional TWAS
- **Transfer Learning Module**: Cross-trait transfer, multi-task learning, cross-population prediction, progressive transfer chains
