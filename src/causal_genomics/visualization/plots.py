"""
Visualization Module

Functions for creating publication-quality plots of causal genomics analyses.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict, Any
import warnings


class MRVisualizer:
    """
    Visualizations for Mendelian Randomization results.
    """
    
    @staticmethod
    def forest_plot(results: pd.DataFrame,
                   figsize: Tuple[int, int] = (10, 8),
                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create forest plot of MR results across methods.
        
        Parameters
        ----------
        results : DataFrame
            MR results with columns: method, beta, ci_lower, ci_upper
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by effect size
        results = results.sort_values('beta')
        
        y_pos = np.arange(len(results))
        
        # Plot point estimates
        ax.scatter(results['beta'], y_pos, s=100, zorder=3, color='darkblue')
        
        # Plot confidence intervals
        for i, (_, row) in enumerate(results.iterrows()):
            ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 
                   'k-', linewidth=2, zorder=2)
        
        # Add vertical line at null
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(results['method'])
        ax.set_xlabel('Causal Effect Size', fontsize=12)
        ax.set_ylabel('MR Method', fontsize=12)
        ax.set_title('Mendelian Randomization Results', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def scatter_plot(exposure_beta: np.ndarray,
                    outcome_beta: np.ndarray,
                    exposure_se: np.ndarray,
                    outcome_se: np.ndarray,
                    mr_results: Optional[Dict] = None,
                    figsize: Tuple[int, int] = (10, 8),
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Scatter plot of SNP effects on exposure vs outcome.
        
        Parameters
        ----------
        exposure_beta, outcome_beta : array
            SNP effect sizes
        exposure_se, outcome_se : array
            Standard errors
        mr_results : dict, optional
            MR results to plot regression lines
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot with error bars
        ax.errorbar(exposure_beta, outcome_beta,
                   xerr=exposure_se, yerr=outcome_se,
                   fmt='o', alpha=0.6, capsize=3, color='steelblue',
                   label='SNPs')
        
        # Plot MR regression lines
        if mr_results:
            x_range = np.array([exposure_beta.min(), exposure_beta.max()])
            
            for method, result in mr_results.items():
                if method == 'IVW':
                    y_pred = x_range * result['beta']
                    ax.plot(x_range, y_pred, 'r-', linewidth=2, 
                           label=f"IVW: β={result['beta']:.3f}")
                elif method == 'Egger':
                    y_pred = result['intercept'] + x_range * result['beta']
                    ax.plot(x_range, y_pred, 'g--', linewidth=2,
                           label=f"Egger: β={result['beta']:.3f}")
        
        # Formatting
        ax.set_xlabel('SNP Effect on Exposure', fontsize=12)
        ax.set_ylabel('SNP Effect on Outcome', fontsize=12)
        ax.set_title('Mendelian Randomization Scatter Plot', fontsize=14, 
                    fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def funnel_plot(beta: np.ndarray,
                   se: np.ndarray,
                   figsize: Tuple[int, int] = (10, 8),
                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Funnel plot to assess asymmetry and pleiotropy.
        
        Parameters
        ----------
        beta : array
            Effect size estimates
        se : array
            Standard errors
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Inverse of SE on y-axis
        precision = 1 / se
        
        ax.scatter(beta, precision, alpha=0.6, s=50)
        
        # Add vertical line at mean effect
        mean_effect = np.average(beta, weights=1/se**2)
        ax.axvline(mean_effect, color='red', linestyle='--', 
                  label=f'Mean effect: {mean_effect:.3f}')
        
        # Formatting
        ax.set_xlabel('Effect Size', fontsize=12)
        ax.set_ylabel('Precision (1/SE)', fontsize=12)
        ax.set_title('MR Funnel Plot', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class NetworkVisualizer:
    """
    Visualizations for gene networks and pathways.
    """
    
    @staticmethod
    def plot_network(G: Any,
                    node_color: Optional[Dict] = None,
                    node_size: int = 500,
                    figsize: Tuple[int, int] = (12, 10),
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot gene regulatory network.
        
        Parameters
        ----------
        G : networkx.Graph
            Network graph
        node_color : dict, optional
            Mapping of node -> color
        node_size : int
            Size of nodes
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        matplotlib.Figure
        """
        import networkx as nx
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # Node colors
        if node_color:
            colors = [node_color.get(node, 'lightblue') for node in G.nodes()]
        else:
            colors = 'lightblue'
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_size,
                              alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title('Gene Regulatory Network', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_pathway_enrichment(enrichment_results: pd.DataFrame,
                               top_n: int = 15,
                               figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Bar plot of pathway enrichment results.
        
        Parameters
        ----------
        enrichment_results : DataFrame
            Enrichment results with columns: pathway, pval, odds_ratio
        top_n : int
            Number of top pathways to plot
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get top pathways
        top_pathways = enrichment_results.head(top_n).sort_values('pval', 
                                                                   ascending=False)
        
        # Plot
        y_pos = np.arange(len(top_pathways))
        ax.barh(y_pos, -np.log10(top_pathways['pval']), color='steelblue')
        
        # Significance line
        ax.axvline(-np.log10(0.05), color='red', linestyle='--', 
                  label='p=0.05', linewidth=2)
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_pathways['pathway'], fontsize=10)
        ax.set_xlabel('-log10(P-value)', fontsize=12)
        ax.set_title('Pathway Enrichment Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class MultiOmicsVisualizer:
    """
    Visualizations for multi-omics integration.
    """
    
    @staticmethod
    def heatmap(data: pd.DataFrame,
               figsize: Tuple[int, int] = (12, 10),
               save_path: Optional[str] = None) -> plt.Figure:
        """
        Heatmap of multi-omics data.
        
        Parameters
        ----------
        data : DataFrame
            Data matrix (samples x features)
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Standardize data
        from scipy.stats import zscore
        data_z = data.apply(zscore)
        
        # Create heatmap
        sns.heatmap(data_z.T, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Z-score'},
                   ax=ax)
        
        ax.set_title('Multi-Omics Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Samples', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def manhattan_plot(data: pd.DataFrame,
                      chr_col: str = 'CHR',
                      pos_col: str = 'POS',
                      pval_col: str = 'P',
                      figsize: Tuple[int, int] = (16, 6),
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Manhattan plot for GWAS results.
        
        Parameters
        ----------
        data : DataFrame
            GWAS results
        chr_col : str
            Chromosome column name
        pos_col : str
            Position column name
        pval_col : str
            P-value column name
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate -log10 p-values
        data = data.copy()
        data['-log10p'] = -np.log10(data[pval_col])
        
        # Group by chromosome
        grouped = data.groupby(chr_col)
        
        colors = ['steelblue', 'coral']
        x_labels = []
        x_labels_pos = []
        
        # Plot each chromosome
        last_x = 0
        for i, (chr_num, chr_data) in enumerate(grouped):
            chr_data = chr_data.sort_values(pos_col)
            x_coords = last_x + chr_data[pos_col]
            
            ax.scatter(x_coords, chr_data['-log10p'], 
                      c=colors[i % 2], s=10, alpha=0.7)
            
            # Store label position
            x_labels.append(chr_num)
            x_labels_pos.append(last_x + (chr_data[pos_col].max() / 2))
            
            last_x = x_coords.max()
        
        # Significance lines
        ax.axhline(-np.log10(5e-8), color='red', linestyle='--', 
                  label='Genome-wide significance', linewidth=1.5)
        ax.axhline(-np.log10(1e-5), color='blue', linestyle='--', 
                  label='Suggestive', linewidth=1)
        
        # Formatting
        ax.set_xticks(x_labels_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Chromosome', fontsize=12)
        ax.set_ylabel('-log10(P-value)', fontsize=12)
        ax.set_title('Manhattan Plot', fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def volcano_plot(data: pd.DataFrame,
                    beta_col: str = 'beta',
                    pval_col: str = 'pval',
                    label_col: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 8),
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Volcano plot for differential analysis.
        
        Parameters
        ----------
        data : DataFrame
            Results with effect sizes and p-values
        beta_col : str
            Effect size column
        pval_col : str
            P-value column
        label_col : str, optional
            Column with labels for significant points
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        
        Returns
        -------
        matplotlib.Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate -log10 p-values
        data = data.copy()
        data['-log10p'] = -np.log10(data[pval_col])
        
        # Color by significance
        significant = (data[pval_col] < 0.05) & (np.abs(data[beta_col]) > 0.5)
        
        # Plot non-significant points
        ax.scatter(data.loc[~significant, beta_col], 
                  data.loc[~significant, '-log10p'],
                  c='gray', alpha=0.5, s=20, label='Not significant')
        
        # Plot significant points
        ax.scatter(data.loc[significant, beta_col], 
                  data.loc[significant, '-log10p'],
                  c='red', alpha=0.7, s=30, label='Significant')
        
        # Add labels for top significant points
        if label_col and label_col in data.columns:
            top_sig = data[significant].nlargest(10, '-log10p')
            for _, row in top_sig.iterrows():
                ax.annotate(row[label_col], 
                          xy=(row[beta_col], row['-log10p']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.7)
        
        # Significance lines
        ax.axhline(-np.log10(0.05), color='blue', linestyle='--', 
                  linewidth=1, alpha=0.7)
        ax.axvline(-0.5, color='green', linestyle='--', 
                  linewidth=1, alpha=0.7)
        ax.axvline(0.5, color='green', linestyle='--', 
                  linewidth=1, alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Effect Size', fontsize=12)
        ax.set_ylabel('-log10(P-value)', fontsize=12)
        ax.set_title('Volcano Plot', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
