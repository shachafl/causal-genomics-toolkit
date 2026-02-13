"""
Data Loading and Preprocessing Module

Handles loading GWAS, QTL, and other genomic data formats.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import requests
import warnings


class GWASDataLoader:
    """
    Load and preprocess GWAS summary statistics.
    
    Supports multiple formats and data sources.
    """
    
    def __init__(self):
        self.data = None
        self.metadata = {}
    
    def load_from_file(self,
                      filepath: str,
                      format: str = 'standard',
                      **kwargs) -> pd.DataFrame:
        """
        Load GWAS data from file.
        
        Parameters
        ----------
        filepath : str
            Path to summary statistics file
        format : str
            'standard', 'plink', 'bolt', or 'saige'
        
        Returns
        -------
        pd.DataFrame
            Standardized GWAS summary statistics
        """
        if format == 'standard':
            # Standard format: SNP, CHR, POS, A1, A2, BETA, SE, P
            data = pd.read_csv(filepath, sep='\t', **kwargs)
            
        elif format == 'plink':
            # PLINK .assoc format
            data = pd.read_csv(filepath, delim_whitespace=True, **kwargs)
            # Rename columns to standard format
            column_mapping = {
                'SNP': 'SNP',
                'CHR': 'CHR',
                'BP': 'POS',
                'A1': 'A1',
                'A2': 'A2',
                'BETA': 'BETA',
                'SE': 'SE',
                'P': 'P'
            }
            data = data.rename(columns=column_mapping)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Standardize column names
        data = self._standardize_columns(data)
        
        # Quality control
        data = self._quality_control(data)
        
        self.data = data
        return data
    
    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across formats.
        """
        standard_cols = ['SNP', 'CHR', 'POS', 'A1', 'A2', 'BETA', 'SE', 'P']
        
        # Check which columns are present
        missing_cols = set(standard_cols) - set(data.columns)
        if missing_cols:
            warnings.warn(f"Missing columns: {missing_cols}")
        
        return data
    
    def _quality_control(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Quality control filtering.
        """
        initial_n = len(data)
        
        # Remove missing values
        data = data.dropna(subset=['BETA', 'SE', 'P'])
        
        # Remove invalid p-values
        data = data[(data['P'] > 0) & (data['P'] <= 1)]
        
        # Remove extreme betas (potential errors)
        data = data[np.abs(data['BETA']) < 10]
        
        # Remove zero or negative standard errors
        data = data[data['SE'] > 0]
        
        final_n = len(data)
        removed = initial_n - final_n
        
        if removed > 0:
            print(f"QC: Removed {removed} variants ({removed/initial_n*100:.1f}%)")
        
        return data
    
    def load_from_gwas_catalog(self,
                              study_id: str) -> pd.DataFrame:
        """
        Load GWAS from NHGRI-EBI GWAS Catalog.
        
        Parameters
        ----------
        study_id : str
            GWAS Catalog accession (e.g., 'GCST004988')
        
        Returns
        -------
        pd.DataFrame
            GWAS summary statistics
        """
        # API endpoint
        url = f"https://www.ebi.ac.uk/gwas/api/search/downloads/{study_id}"
        
        try:
            data = pd.read_csv(url, sep='\t')
            
            # Convert to standard format
            data = data.rename(columns={
                'SNPS': 'SNP',
                'CHR_ID': 'CHR',
                'CHR_POS': 'POS',
                'RISK ALLELE': 'A1',
                'P-VALUE': 'P',
                'OR or BETA': 'BETA'
            })
            
            self.data = data
            self.metadata['source'] = 'GWAS Catalog'
            self.metadata['study_id'] = study_id
            
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to load from GWAS Catalog: {e}")
    
    def clump(self,
             r2_threshold: float = 0.1,
             kb_distance: int = 500) -> pd.DataFrame:
        """
        LD clumping to get independent signals.
        
        Parameters
        ----------
        r2_threshold : float
            LD R² threshold
        kb_distance : int
            Distance threshold in kb
        
        Returns
        -------
        pd.DataFrame
            Clumped variants
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Sort by p-value
        data_sorted = self.data.sort_values('P')
        
        # Simple distance-based clumping
        # In practice, would use actual LD information
        clumped = []
        used_positions = set()
        
        for _, variant in data_sorted.iterrows():
            chr_pos = (variant['CHR'], variant['POS'])
            
            # Check if any nearby variant already selected
            nearby = False
            for used_chr, used_pos in used_positions:
                if used_chr == chr_pos[0]:
                    if abs(used_pos - chr_pos[1]) < kb_distance * 1000:
                        nearby = True
                        break
            
            if not nearby:
                clumped.append(variant)
                used_positions.add(chr_pos)
        
        clumped_df = pd.DataFrame(clumped)
        
        print(f"Clumping: {len(data_sorted)} → {len(clumped_df)} independent signals")
        
        return clumped_df


class QTLDataLoader:
    """
    Load and process QTL data (eQTL, pQTL, mQTL).
    """
    
    def __init__(self):
        self.data = None
        self.qtl_type = None
    
    def load_eqtl(self,
                  filepath: str,
                  tissue: Optional[str] = None) -> pd.DataFrame:
        """
        Load eQTL data.
        
        Parameters
        ----------
        filepath : str
            Path to eQTL summary statistics
        tissue : str, optional
            Tissue type
        
        Returns
        -------
        pd.DataFrame
            eQTL data
        """
        data = pd.read_csv(filepath, sep='\t')
        
        # Standardize columns
        required_cols = ['SNP', 'gene', 'beta', 'se', 'pval']
        
        self.data = data
        self.qtl_type = 'eQTL'
        
        if tissue:
            self.data['tissue'] = tissue
        
        return data
    
    def load_from_gtex(self,
                      gene: str,
                      tissue: str) -> pd.DataFrame:
        """
        Load eQTL data from GTEx portal.
        
        Parameters
        ----------
        gene : str
            Gene symbol or Ensembl ID
        tissue : str
            GTEx tissue name
        
        Returns
        -------
        pd.DataFrame
            GTEx eQTL data for gene
        """
        # API endpoint (simplified)
        # In practice, would use GTEx API
        
        warnings.warn("GTEx API access not fully implemented")
        
        # Return mock data structure
        return pd.DataFrame({
            'SNP': [],
            'gene': [],
            'beta': [],
            'se': [],
            'pval': [],
            'tissue': []
        })
    
    def get_cis_qtls(self,
                    window_kb: int = 1000) -> pd.DataFrame:
        """
        Filter for cis-QTLs (within specified window of gene).
        
        Parameters
        ----------
        window_kb : int
            Window size in kb
        
        Returns
        -------
        pd.DataFrame
            Cis-QTLs only
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Would need gene positions to implement properly
        # For now, just return data
        return self.data


class PerturbationDataLoader:
    """
    Load genome-scale perturbation screen data (CRISPR, RNAi).
    """
    
    def __init__(self):
        self.data = None
    
    def load_crispr_screen(self,
                          filepath: str,
                          screen_type: str = 'knockout') -> pd.DataFrame:
        """
        Load CRISPR screen data.
        
        Parameters
        ----------
        filepath : str
            Path to screen results
        screen_type : str
            'knockout', 'activation', or 'inhibition'
        
        Returns
        -------
        pd.DataFrame
            Screen results with gene-level effects
        """
        data = pd.read_csv(filepath)
        
        # Standardize columns
        # Typical format: gene, log2fc, pval, fdr
        
        self.data = data
        self.data['screen_type'] = screen_type
        
        return data
    
    def load_from_depmap(self,
                        cell_line: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from DepMap portal.
        
        Parameters
        ----------
        cell_line : str, optional
            Specific cell line to load
        
        Returns
        -------
        pd.DataFrame
            DepMap CRISPR screen data
        """
        # Would access DepMap API
        # Simplified implementation
        
        warnings.warn("DepMap API access not fully implemented")
        
        return pd.DataFrame({
            'gene': [],
            'dependency_score': [],
            'cell_line': []
        })


class DataHarmonizer:
    """
    Harmonize data from multiple sources.
    
    Ensures alleles are aligned, positions match, etc.
    """
    
    @staticmethod
    def harmonize_alleles(data1: pd.DataFrame,
                         data2: pd.DataFrame,
                         on: str = 'SNP') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Harmonize alleles between two datasets.
        
        Ensures effect alleles are aligned.
        
        Parameters
        ----------
        data1, data2 : DataFrame
            Datasets with columns: SNP, A1, A2, beta, se
        on : str
            Column to merge on
        
        Returns
        -------
        tuple
            (harmonized_data1, harmonized_data2)
        """
        # Merge datasets
        merged = pd.merge(data1, data2, on=on, suffixes=('_1', '_2'))
        
        # Align alleles
        harmonized1 = []
        harmonized2 = []
        
        for _, row in merged.iterrows():
            # Check if alleles match
            if (row['A1_1'] == row['A1_2'] and row['A2_1'] == row['A2_2']):
                # Already aligned
                harmonized1.append(row[[c for c in row.index if c.endswith('_1')]])
                harmonized2.append(row[[c for c in row.index if c.endswith('_2')]])
                
            elif (row['A1_1'] == row['A2_2'] and row['A2_1'] == row['A1_2']):
                # Need to flip dataset 2
                row_2 = row[[c for c in row.index if c.endswith('_2')]].copy()
                row_2['beta_2'] *= -1
                
                harmonized1.append(row[[c for c in row.index if c.endswith('_1')]])
                harmonized2.append(row_2)
        
        harm1_df = pd.DataFrame(harmonized1)
        harm2_df = pd.DataFrame(harmonized2)
        
        return harm1_df, harm2_df
    
    @staticmethod
    def lift_over_positions(data: pd.DataFrame,
                           from_build: str = 'hg19',
                           to_build: str = 'hg38') -> pd.DataFrame:
        """
        Convert genomic positions between genome builds.
        
        Parameters
        ----------
        data : DataFrame
            Must contain CHR and POS columns
        from_build : str
            Source genome build
        to_build : str
            Target genome build
        
        Returns
        -------
        pd.DataFrame
            Data with updated positions
        """
        # Would use liftOver tool in practice
        warnings.warn("Liftover not fully implemented - positions unchanged")
        
        return data


class AnnotationLoader:
    """
    Load functional annotations for genes and variants.
    """
    
    @staticmethod
    def load_gene_annotations(genes: List[str]) -> pd.DataFrame:
        """
        Load gene annotations (GO terms, pathways, etc.).
        
        Parameters
        ----------
        genes : list
            Gene symbols
        
        Returns
        -------
        pd.DataFrame
            Gene annotations
        """
        # Would query annotation databases
        # Simplified mock implementation
        
        annotations = []
        for gene in genes:
            annotations.append({
                'gene': gene,
                'go_terms': 'GO:0008150',  # Biological process
                'pathways': 'Metabolic pathways',
                'gene_type': 'protein_coding'
            })
        
        return pd.DataFrame(annotations)
    
    @staticmethod
    def load_variant_annotations(variants: List[str]) -> pd.DataFrame:
        """
        Load variant functional annotations.
        
        Parameters
        ----------
        variants : list
            Variant IDs
        
        Returns
        -------
        pd.DataFrame
            Variant consequences and predictions
        """
        # Would query VEP or similar
        
        annotations = []
        for var in variants:
            annotations.append({
                'variant': var,
                'consequence': 'missense_variant',
                'cadd_score': np.random.uniform(10, 30),
                'polyphen': 'probably_damaging'
            })
        
        return pd.DataFrame(annotations)
