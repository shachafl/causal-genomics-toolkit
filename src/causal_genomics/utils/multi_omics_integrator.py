"""
Multi-Omics Integration Module

Integrates genetic, transcriptomic, proteomic, and other omics data
to identify causal mechanisms and biomarkers.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, DBSCAN
from typing import Dict, List, Tuple, Optional, Any
import warnings


class MultiOmicsIntegrator:
    """
    Integrate multiple omics layers for systems-level analysis.
    
    Combines GWAS, eQTL, pQTL, metabolomics, and other data types.
    """
    
    def __init__(self):
        self.layers = {}
        self.integrated_network = None
        self.causal_graph = None
    
    def add_layer(self, 
                  name: str,
                  data: pd.DataFrame,
                  layer_type: str = 'association'):
        """
        Add an omics data layer.
        
        Parameters
        ----------
        name : str
            Layer name (e.g., 'gwas', 'eqtl', 'pqtl')
        data : DataFrame
            Association or expression data
        layer_type : str
            'association', 'expression', or 'network'
        """
        self.layers[name] = {
            'data': data,
            'type': layer_type
        }
    
    def build_causal_network(self,
                            method: str = 'multi_layer',
                            significance_threshold: float = 0.05) -> nx.DiGraph:
        """
        Build a multi-layer causal network.
        
        Parameters
        ----------
        method : str
            'multi_layer' or 'integrated'
        significance_threshold : float
            P-value threshold for including edges
        
        Returns
        -------
        networkx.DiGraph
            Directed causal network
        """
        G = nx.DiGraph()
        
        # Add nodes from all layers
        for layer_name, layer_data in self.layers.items():
            if 'gene' in layer_data['data'].columns:
                genes = layer_data['data']['gene'].unique()
                for gene in genes:
                    G.add_node(gene, layer=layer_name)
        
        # Add edges based on associations
        for layer_name, layer_data in self.layers.items():
            data = layer_data['data']
            
            if 'pval' in data.columns and 'gene' in data.columns:
                # Filter significant associations
                sig_data = data[data['pval'] < significance_threshold]
                
                # Add edges
                if 'target' in sig_data.columns:
                    for _, row in sig_data.iterrows():
                        G.add_edge(row['gene'], row['target'],
                                  weight=abs(row.get('beta', 1)),
                                  pval=row['pval'],
                                  layer=layer_name)
        
        self.integrated_network = G
        return G
    
    def identify_disease_modules(self,
                                disease_genes: List[str],
                                method: str = 'community') -> Dict:
        """
        Identify disease-associated network modules.
        
        Parameters
        ----------
        disease_genes : list
            Known disease-associated genes
        method : str
            'community', 'diffusion', or 'steiner_tree'
        
        Returns
        -------
        dict
            Disease modules and their properties
        """
        if self.integrated_network is None:
            self.build_causal_network()
        
        G = self.integrated_network
        
        if method == 'community':
            # Community detection around disease genes
            modules = self._community_detection(G, disease_genes)
        elif method == 'diffusion':
            # Network diffusion from disease genes
            modules = self._network_diffusion(G, disease_genes)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return modules
    
    def _community_detection(self,
                           G: nx.Graph,
                           seed_genes: List[str]) -> Dict:
        """
        Detect communities around seed genes.
        """
        # Find subgraph containing seed genes and neighbors
        seed_nodes = [n for n in seed_genes if n in G.nodes()]
        
        if not seed_nodes:
            return {'modules': [], 'n_modules': 0}
        
        # Get k-hop neighborhood
        subgraph_nodes = set(seed_nodes)
        for seed in seed_nodes:
            neighbors = nx.single_source_shortest_path_length(G, seed, cutoff=2)
            subgraph_nodes.update(neighbors.keys())
        
        subgraph = G.subgraph(subgraph_nodes)
        
        # Use Louvain community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(subgraph.to_undirected())
        except ImportError:
            # Fallback to simple connected components
            partition = {}
            for i, component in enumerate(nx.connected_components(subgraph.to_undirected())):
                for node in component:
                    partition[node] = i
        
        # Organize modules
        modules = {}
        for node, module_id in partition.items():
            if module_id not in modules:
                modules[module_id] = []
            modules[module_id].append(node)
        
        return {
            'modules': list(modules.values()),
            'n_modules': len(modules),
            'partition': partition
        }
    
    def _network_diffusion(self,
                          G: nx.Graph,
                          seed_genes: List[str],
                          alpha: float = 0.5) -> Dict:
        """
        Network diffusion analysis.
        
        Propagates signal from seed genes through network.
        """
        seed_nodes = [n for n in seed_genes if n in G.nodes()]
        
        if not seed_nodes:
            return {'scores': {}, 'top_genes': []}
        
        # Initialize scores
        scores = {node: 0.0 for node in G.nodes()}
        for seed in seed_nodes:
            scores[seed] = 1.0
        
        # Iterative diffusion
        n_iter = 10
        for _ in range(n_iter):
            new_scores = scores.copy()
            
            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                if neighbors:
                    neighbor_score = sum(scores[n] for n in neighbors) / len(neighbors)
                    new_scores[node] = alpha * scores[node] + (1 - alpha) * neighbor_score
            
            scores = new_scores
        
        # Get top scoring genes
        sorted_genes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_genes = [g for g, s in sorted_genes[:100] if s > 0.1]
        
        return {
            'scores': scores,
            'top_genes': top_genes
        }
    
    def pathway_enrichment(self,
                          gene_list: List[str],
                          pathway_database: str = 'GO') -> pd.DataFrame:
        """
        Perform pathway enrichment analysis.
        
        Parameters
        ----------
        gene_list : list
            Genes to test for enrichment
        pathway_database : str
            'GO', 'KEGG', or 'Reactome'
        
        Returns
        -------
        pd.DataFrame
            Enriched pathways with statistics
        """
        # In practice, would query pathway databases
        # This is a simplified implementation
        
        # Mock pathway data
        pathways = {
            'Signal Transduction': ['EGFR', 'MAPK1', 'AKT1', 'PIK3CA'],
            'Immune Response': ['IL6', 'TNF', 'IFNG', 'CD4'],
            'Metabolic Process': ['INS', 'PPARG', 'LDLR', 'APOE']
        }
        
        results = []
        background_size = 20000  # Approximate genome size
        
        for pathway_name, pathway_genes in pathways.items():
            # Count overlaps
            overlap = len(set(gene_list) & set(pathway_genes))
            
            # Fisher's exact test
            a = overlap  # in gene list and pathway
            b = len(gene_list) - overlap  # in gene list, not pathway
            c = len(pathway_genes) - overlap  # in pathway, not gene list
            d = background_size - a - b - c  # neither
            
            oddsratio, pval = stats.fisher_exact([[a, b], [c, d]])
            
            results.append({
                'pathway': pathway_name,
                'overlap': overlap,
                'pathway_size': len(pathway_genes),
                'query_size': len(gene_list),
                'pval': pval,
                'odds_ratio': oddsratio,
                'genes': list(set(gene_list) & set(pathway_genes))
            })
        
        results_df = pd.DataFrame(results)
        
        # FDR correction
        from statsmodels.stats.multitest import multipletests
        if len(results_df) > 0:
            _, results_df['fdr'], _, _ = multipletests(results_df['pval'], method='fdr_bh')
        
        return results_df.sort_values('pval')
    
    def identify_biomarkers(self,
                           case_data: pd.DataFrame,
                           control_data: pd.DataFrame,
                           method: str = 'differential',
                           n_biomarkers: int = 50) -> pd.DataFrame:
        """
        Identify candidate biomarkers.
        
        Parameters
        ----------
        case_data : DataFrame
            Omics data for cases (samples x features)
        control_data : DataFrame
            Omics data for controls
        method : str
            'differential', 'ml', or 'causal'
        n_biomarkers : int
            Number of biomarkers to return
        
        Returns
        -------
        pd.DataFrame
            Ranked candidate biomarkers
        """
        if method == 'differential':
            return self._differential_biomarkers(case_data, control_data, n_biomarkers)
        elif method == 'ml':
            return self._ml_biomarkers(case_data, control_data, n_biomarkers)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _differential_biomarkers(self,
                                case_data: pd.DataFrame,
                                control_data: pd.DataFrame,
                                n_biomarkers: int) -> pd.DataFrame:
        """
        Differential expression/abundance analysis.
        """
        results = []
        
        for feature in case_data.columns:
            case_values = case_data[feature].dropna()
            control_values = control_data[feature].dropna()
            
            # T-test
            t_stat, pval = stats.ttest_ind(case_values, control_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((case_values.std()**2 + control_values.std()**2) / 2)
            cohens_d = (case_values.mean() - control_values.mean()) / pooled_std
            
            # Fold change
            fc = case_values.mean() / control_values.mean()
            log2fc = np.log2(fc) if fc > 0 else np.nan
            
            results.append({
                'feature': feature,
                'case_mean': case_values.mean(),
                'control_mean': control_values.mean(),
                'log2_fc': log2fc,
                'cohens_d': cohens_d,
                'pval': pval,
                't_stat': t_stat
            })
        
        results_df = pd.DataFrame(results)
        
        # FDR correction
        _, results_df['fdr'], _, _ = multipletests(results_df['pval'], method='fdr_bh')
        
        # Rank by p-value and effect size
        results_df['rank_score'] = -np.log10(results_df['pval'] + 1e-300) * np.abs(results_df['cohens_d'])
        
        return results_df.sort_values('rank_score', ascending=False).head(n_biomarkers)
    
    def _ml_biomarkers(self,
                      case_data: pd.DataFrame,
                      control_data: pd.DataFrame,
                      n_biomarkers: int) -> pd.DataFrame:
        """
        ML-based biomarker selection.
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Combine data
        X = pd.concat([case_data, control_data])
        y = np.concatenate([np.ones(len(case_data)), np.zeros(len(control_data))])
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        
        # Get feature importances
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importances.head(n_biomarkers)
    
    def cluster_samples(self,
                       data: pd.DataFrame,
                       n_clusters: int = 3,
                       method: str = 'kmeans') -> Dict:
        """
        Cluster samples to identify subtypes.
        
        Parameters
        ----------
        data : DataFrame
            Sample x feature matrix
        n_clusters : int
            Number of clusters
        method : str
            'kmeans', 'hierarchical', or 'dbscan'
        
        Returns
        -------
        dict
            Cluster assignments and characteristics
        """
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(data)
            
        elif method == 'hierarchical':
            Z = linkage(data, method='ward')
            labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            labels = clusterer.fit_predict(data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Characterize clusters
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_samples = data[labels == cluster_id]
            
            cluster_stats.append({
                'cluster': cluster_id,
                'size': len(cluster_samples),
                'mean_profile': cluster_samples.mean().to_dict()
            })
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'cluster_stats': cluster_stats
        }
    
    def integrate_perturbation_data(self,
                                   crispr_data: pd.DataFrame,
                                   observational_data: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate CRISPR/perturbation data with observational data.
        
        Parameters
        ----------
        crispr_data : DataFrame
            Gene perturbation effects
        observational_data : DataFrame
            Observational associations (e.g., from GWAS)
        
        Returns
        -------
        pd.DataFrame
            Integrated causal evidence
        """
        # Merge datasets
        merged = pd.merge(crispr_data, observational_data,
                         on='gene', suffixes=('_perturb', '_obs'))
        
        # Compare directions
        merged['direction_consistent'] = (
            np.sign(merged['effect_perturb']) == np.sign(merged['effect_obs'])
        )
        
        # Calculate concordance score
        merged['concordance_score'] = (
            merged['direction_consistent'].astype(int) *
            np.minimum(abs(merged['effect_perturb']), abs(merged['effect_obs']))
        )
        
        # Rank by evidence strength
        merged['evidence_rank'] = (
            -np.log10(merged['pval_obs'] + 1e-300) *
            -np.log10(merged['pval_perturb'] + 1e-300) *
            merged['concordance_score']
        )
        
        return merged.sort_values('evidence_rank', ascending=False)


class CausalNetworkInference:
    """
    Infer causal relationships between molecular features.
    """
    
    def __init__(self):
        self.causal_graph = None
    
    def infer_grn(self,
                  expression_data: pd.DataFrame,
                  method: str = 'genie3') -> nx.DiGraph:
        """
        Infer gene regulatory network.
        
        Parameters
        ----------
        expression_data : DataFrame
            Gene expression matrix (samples x genes)
        method : str
            'genie3', 'aracne', or 'correlation'
        
        Returns
        -------
        networkx.DiGraph
            Inferred regulatory network
        """
        if method == 'correlation':
            # Simple correlation-based network
            corr_matrix = expression_data.corr()
            
            G = nx.DiGraph()
            genes = corr_matrix.columns
            
            # Add edges for strong correlations
            threshold = 0.7
            for i, gene1 in enumerate(genes):
                for gene2 in genes[i+1:]:
                    corr = corr_matrix.loc[gene1, gene2]
                    if abs(corr) > threshold:
                        G.add_edge(gene1, gene2, weight=abs(corr))
            
            self.causal_graph = G
            return G
        
        else:
            raise NotImplementedError(f"Method {method} not yet implemented")
    
    def causal_discovery(self,
                        data: pd.DataFrame,
                        method: str = 'pc') -> nx.DiGraph:
        """
        Causal discovery from observational data.
        
        Parameters
        ----------
        data : DataFrame
            Observational data
        method : str
            'pc' (PC algorithm) or 'ges' (Greedy Equivalence Search)
        
        Returns
        -------
        networkx.DiGraph
            Causal DAG
        """
        # Simplified implementation
        # Full implementation would use causal-learn or similar
        
        G = nx.DiGraph()
        features = data.columns
        
        # Compute conditional independencies
        for f1 in features:
            for f2 in features:
                if f1 != f2:
                    # Test for association
                    corr = data[f1].corr(data[f2])
                    if abs(corr) > 0.3:
                        G.add_edge(f1, f2, weight=abs(corr))
        
        return G
