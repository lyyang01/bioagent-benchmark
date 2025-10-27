# SpaGE
## The import form of the SpaGE package
from SpaGE.main import SpaGE
## usage of SpaGE function 
SpaGE(Spatial_data, RNA_data, n_pv, genes_to_predict=None)
### the detail description of parameters:
1. Spatial_data : Dataframe
            Normalized Spatial data matrix (cells X genes).
2. RNA_data : Dataframe
    Normalized scRNA-seq data matrix (cells X genes).
3. n_pv : int
    Number of principal vectors to find from the independently computed
    principal components, and used to align both datasets. This should
    be <= number of shared genes between the two datasets.
4. genes_to_predict : str array 
    list of gene names missing from the spatial data, to be predicted 
    from the scRNA-seq data. Default is the set of different genes 
    (columns) between scRNA-seq and spatial data.
### returns
Imp_Genes: Dataframe
    Matrix containing the predicted gene expressions for the spatial 
    cells. Rows are equal to the number of spatial data rows (cells), 
    and columns are equal to genes_to_predict,  .