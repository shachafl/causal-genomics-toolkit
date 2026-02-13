# Causal Genomics Toolkit - Complete Example

This notebook demonstrates a complete analysis pipeline for discovering and validating causal gene-phenotype relationships.

## Example: Identifying Causal Genes for Type 2 Diabetes

```python
import pandas as pd
import numpy as np
from causal_genomics import (
    MendelianRandomization,
    ColocalizationAnalysis,
    CausalPredictor,
    MultiOmicsIntegrator
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
```

## 1. Load Data

```python
# Load GWAS summary statistics for Type 2 Diabetes
t2d_gwas = pd.read_csv('data/t2d_gwas_summary.txt', sep='\t')

# Load eQTL data (expression quantitative trait loci)
eqtl_data = pd.read_csv('data/pancreas_eqtl.txt', sep='\t')

# Load protein QTL data
pqtl_data = pd.read_csv('data/blood_pqtl.txt', sep='\t')

# Load CRISPR perturbation screen data
crispr_data = pd.read_csv('data/beta_cell_crispr_screen.txt', sep='\t')

print(f"GWAS variants: {len(t2d_gwas)}")
print(f"eQTL associations: {len(eqtl_data)}")
print(f"pQTL associations: {len(pqtl_data)}")
print(f"CRISPR perturbations: {len(crispr_data)}")
```

## 2. Mendelian Randomization Analysis

Test if gene expression causally affects T2D risk.

```python
# For each gene with significant eQTL
genes_to_test = eqtl_data[eqtl_data['pval'] < 5e-8]['gene'].unique()

mr_results = []

for gene in genes_to_test[:10]:  # Test first 10 genes
    # Get eQTL data for this gene
    gene_eqtl = eqtl_data[eqtl_data['gene'] == gene]
    
    # Initialize MR analysis
    mr = MendelianRandomization(p_threshold=5e-8)
    
    # Load exposure (eQTL) and outcome (GWAS) data
    mr.load_exposure_gwas(gene_eqtl)
    mr.load_outcome_gwas(t2d_gwas)
    
    # Run multiple MR methods
    results = mr.run_analysis(methods=['ivw', 'egger', 'weighted_median'])
    results['gene'] = gene
    
    mr_results.append(results)

# Combine results
mr_results_df = pd.concat(mr_results, ignore_index=True)

# Display significant results
sig_results = mr_results_df[mr_results_df['pval'] < 0.05]
print(f"\nSignificant causal associations: {len(sig_results)}")
print(sig_results[['gene', 'method', 'beta', 'pval']])
```

## 3. Colocalization Analysis

Verify that eQTL and GWAS signals share the same causal variant.

```python
coloc = ColocalizationAnalysis()

coloc_results = []

for gene in sig_results['gene'].unique():
    gene_eqtl = eqtl_data[eqtl_data['gene'] == gene]
    
    # Get overlapping SNPs
    overlap_snps = set(gene_eqtl['SNP']) & set(t2d_gwas['SNP'])
    
    if len(overlap_snps) > 10:
        # Subset to common SNPs
        eqtl_subset = gene_eqtl[gene_eqtl['SNP'].isin(overlap_snps)].sort_values('SNP')
        gwas_subset = t2d_gwas[t2d_gwas['SNP'].isin(overlap_snps)].sort_values('SNP')
        
        # Run colocalization
        result = coloc.coloc_abf(
            eqtl_subset['beta'].values,
            eqtl_subset['se'].values,
            gwas_subset['beta'].values,
            gwas_subset['se'].values
        )
        
        result['gene'] = gene
        coloc_results.append(result)

coloc_df = pd.DataFrame(coloc_results)

# Genes with strong colocalization evidence
colocalized = coloc_df[coloc_df['H4_shared_causal'] > 0.8]
print(f"\nGenes with colocalization evidence: {len(colocalized)}")
print(colocalized[['gene', 'H4_shared_causal']])
```

## 4. Multi-Omics Integration

Integrate GWAS, eQTL, and pQTL data to build causal network.

```python
# Initialize integrator
integrator = MultiOmicsIntegrator()

# Add data layers
integrator.add_layer('gwas', t2d_gwas, layer_type='association')
integrator.add_layer('eqtl', eqtl_data, layer_type='association')
integrator.add_layer('pqtl', pqtl_data, layer_type='association')

# Build multi-layer causal network
causal_network = integrator.build_causal_network(significance_threshold=5e-8)

print(f"\nCausal network: {causal_network.number_of_nodes()} nodes, "
      f"{causal_network.number_of_edges()} edges")

# Identify disease modules
disease_genes = sig_results['gene'].unique().tolist()
modules = integrator.identify_disease_modules(disease_genes, method='community')

print(f"Identified {modules['n_modules']} disease modules")
for i, module in enumerate(modules['modules'][:3]):
    print(f"Module {i+1}: {len(module)} genes - {module[:5]}...")
```

## 5. Pathway Enrichment

What biological pathways are enriched in causal genes?

```python
# Get all significant causal genes
causal_genes = sig_results['gene'].unique().tolist()

# Pathway enrichment
enrichment_results = integrator.pathway_enrichment(causal_genes)

print("\nTop enriched pathways:")
print(enrichment_results[['pathway', 'overlap', 'pval', 'fdr']].head(10))
```

## 6. Biomarker Identification

Identify proteins that could serve as biomarkers.

```python
# Load case-control proteomic data
case_proteomics = pd.read_csv('data/t2d_cases_proteomics.csv', index_col=0)
control_proteomics = pd.read_csv('data/controls_proteomics.csv', index_col=0)

# Identify candidate biomarkers
biomarkers = integrator.identify_biomarkers(
    case_proteomics,
    control_proteomics,
    method='differential',
    n_biomarkers=50
)

print("\nTop biomarker candidates:")
print(biomarkers[['feature', 'log2_fc', 'cohens_d', 'pval', 'fdr']].head(10))

# Filter for causal genes
causal_biomarkers = biomarkers[biomarkers['feature'].isin(causal_genes)]
print(f"\nCausal genes that are also biomarkers: {len(causal_biomarkers)}")
```

## 7. Patient Subtype Discovery

Cluster patients to identify disease subtypes.

```python
# Cluster patients based on multi-omics profiles
clustering_result = integrator.cluster_samples(
    case_proteomics,
    n_clusters=3,
    method='kmeans'
)

print(f"\nIdentified {clustering_result['n_clusters']} patient subtypes")
for cluster_info in clustering_result['cluster_stats']:
    print(f"Subtype {cluster_info['cluster']}: {cluster_info['size']} patients")

# Visualize clusters with PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
coords = pca.fit_transform(case_proteomics)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                     c=clustering_result['labels'],
                     cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('T2D Patient Subtypes')
plt.colorbar(scatter, label='Subtype')
plt.tight_layout()
plt.savefig('patient_subtypes.png', dpi=300)
plt.show()
```

## 8. Integrate Perturbation Data

Compare observational associations with CRISPR screen results.

```python
# Integrate CRISPR perturbation data with MR results
integrated = integrator.integrate_perturbation_data(
    crispr_data,
    mr_results_df[mr_results_df['method'] == 'IVW']
)

# Genes with consistent evidence across methods
high_confidence = integrated[
    (integrated['direction_consistent']) &
    (integrated['pval_obs'] < 0.05) &
    (integrated['pval_perturb'] < 0.01)
]

print(f"\nHigh-confidence causal genes: {len(high_confidence)}")
print(high_confidence[['gene', 'effect_obs', 'effect_perturb', 
                       'evidence_rank']].head(10))
```

## 9. Causal Effect Prediction

Train ML model to predict causal effects for new genes.

```python
from causal_genomics.models import FeatureEngineering

# Engineer features
fe = FeatureEngineering()

genetic_features = fe.create_genetic_features(t2d_gwas)
expression_features = fe.create_expression_features(gene_expression_matrix)
network_features = fe.create_network_features(causal_network)

# Combine features
X = predictor.prepare_features(
    genetic_features=genetic_features,
    expression_features=expression_features,
    network_features=network_features
)

# Target: MR effect estimates
y = mr_results_df[mr_results_df['method'] == 'IVW'].set_index('gene')['beta']

# Align features and labels
X_aligned = X.loc[y.index]

# Train predictor
predictor = CausalPredictor(model_type='xgboost')
metrics = predictor.train(X_aligned, y, validate=True)

print("\nPrediction model performance:")
print(f"R² (CV): {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")
print(f"Training R²: {metrics['train_r2']:.3f}")

# Feature importance
print("\nTop predictive features:")
print(predictor.get_feature_importance(top_n=10))

# Predict for new genes
new_genes_features = X.loc[~X.index.isin(y.index)]
predictions = predictor.predict(new_genes_features)

predicted_effects = pd.DataFrame({
    'gene': new_genes_features.index,
    'predicted_effect': predictions
}).sort_values('predicted_effect', key=abs, ascending=False)

print("\nTop predicted causal effects for unstudied genes:")
print(predicted_effects.head(10))
```

## 10. Visualization and Reporting

```python
# Create summary visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Volcano plot of MR results
ax = axes[0, 0]
mr_plot_data = mr_results_df[mr_results_df['method'] == 'IVW']
ax.scatter(mr_plot_data['beta'], -np.log10(mr_plot_data['pval']), alpha=0.5)
ax.axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
ax.set_xlabel('MR Effect Size (Beta)')
ax.set_ylabel('-log10(P-value)')
ax.set_title('Mendelian Randomization Results')
ax.legend()

# 2. Colocalization posterior probabilities
ax = axes[0, 1]
coloc_df.plot(x='gene', y='H4_shared_causal', kind='bar', ax=ax)
ax.axhline(0.8, color='red', linestyle='--', label='PP.H4 > 0.8')
ax.set_ylabel('PP.H4 (Colocalization)')
ax.set_title('Colocalization Evidence')
ax.legend()

# 3. Network visualization
ax = axes[1, 0]
# Simplified network plot
import networkx as nx
pos = nx.spring_layout(causal_network.subgraph(disease_genes[:20]))
nx.draw(causal_network.subgraph(disease_genes[:20]), pos, ax=ax,
        node_color='lightblue', node_size=500, with_labels=True,
        font_size=8)
ax.set_title('Causal Gene Network')

# 4. Biomarker effect sizes
ax = axes[1, 1]
top_biomarkers = biomarkers.head(15)
ax.barh(range(len(top_biomarkers)), top_biomarkers['log2_fc'])
ax.set_yticks(range(len(top_biomarkers)))
ax.set_yticklabels(top_biomarkers['feature'])
ax.set_xlabel('Log2 Fold Change')
ax.set_title('Top Biomarker Candidates')

plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate summary report
print("\n" + "="*60)
print("CAUSAL GENOMICS ANALYSIS SUMMARY")
print("="*60)
print(f"\nTotal genes tested: {len(genes_to_test)}")
print(f"Genes with significant MR evidence: {len(sig_results['gene'].unique())}")
print(f"Genes with colocalization support: {len(colocalized)}")
print(f"High-confidence causal genes: {len(high_confidence)}")
print(f"Disease modules identified: {modules['n_modules']}")
print(f"Patient subtypes: {clustering_result['n_clusters']}")
print(f"Biomarker candidates: {len(biomarkers)}")
print(f"\nPrediction model CV R²: {metrics['cv_r2_mean']:.3f}")
print("\nTop 5 causal genes:")
for i, gene in enumerate(high_confidence['gene'].head(5), 1):
    print(f"{i}. {gene}")
```

## 11. Export Results

```python
# Save all results
mr_results_df.to_csv('results/mendelian_randomization_results.csv', index=False)
coloc_df.to_csv('results/colocalization_results.csv', index=False)
biomarkers.to_csv('results/biomarker_candidates.csv', index=False)
high_confidence.to_csv('results/high_confidence_causal_genes.csv', index=False)
predicted_effects.to_csv('results/predicted_causal_effects.csv', index=False)

print("\nResults exported to 'results/' directory")
```

## Conclusion

This analysis demonstrates a comprehensive workflow for:
1. **Causal Discovery**: Using MR to identify genes causally affecting disease
2. **Validation**: Colocalization to verify shared causal variants
3. **Integration**: Multi-omics data to build causal networks
4. **Translation**: Identifying biomarkers and patient subtypes
5. **Prediction**: ML models to predict effects for unstudied genes

The high-confidence causal genes identified can be prioritized for:
- Functional validation experiments
- Drug target development
- Biomarker validation studies
- Mechanistic investigations
