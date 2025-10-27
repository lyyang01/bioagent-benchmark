# nichenet
## load nichenet packages
library(nichenetr)
## 1. `get_expressed_genes`
```R
get_expressed_genes(ident, seurat_obj, pct = 0.10, assay_oi = NULL)
```
### Description:
Determines the genes that are expressed in a given cell cluster based on the fraction of cells in that cluster that express the gene.
### Parameters:
1. **`ident`**: character  
   - Name of the cluster identity/identities of cells.
2. **`seurat_obj`**: Seurat object  
   - Single-cell expression dataset as a Seurat v3 object. It should contain a column "aggregate" in the metadata, indicating the condition/sample where cells came from.
3. **`pct`**: numeric (default: 0.10)  
   - The fraction of cells in the cluster that must express the gene for it to be considered "expressed." The choice of this parameter depends largely on the sequencing platform used.
4. **`assay_oi`**: character (default: NULL)  
   - If specified, this parameter allows you to choose which assay to use for determining expressed genes. If NULL, the most advanced assay will be used.
### Value:
A character vector containing the gene symbols of the expressed genes.
### Example:
```R
expressed_genes <- get_expressed_genes(ident = "CD8 T", seurat_obj = seuratObj, pct = 0.10)
```
## 2. `predict_ligand_activities`
```R
predict_ligand_activities(geneset, background_expressed_genes, ligand_target_matrix, potential_ligands, single = TRUE, ...)
```
### Description:
Predicts the activities of ligands in regulating the expression of a gene set of interest. Ligand activities are defined by how well they predict the observed transcriptional response according to the NicheNet model.
### Parameters:
1. **`geneset`**: character vector  
   - Gene symbols of genes whose expression is potentially affected by ligands from the interacting cell.
2. **`background_expressed_genes`**: character vector  
   - Gene symbols of the background, non-affected genes (can include the symbols of the affected genes as well).
3. **`ligand_target_matrix`**: matrix  
   - The NicheNet ligand-target matrix denoting regulatory potential scores between ligands and targets (ligands in columns).
4. **`potential_ligands`**: character vector  
   - Gene symbols of the potentially active ligands for which you want to define ligand activities.
5. **`single`**: logical (default: TRUE)  
   - If TRUE, calculate ligand activity scores by considering every ligand individually (recommended). If FALSE, calculate ligand activity scores as variable importances of a multi-ligand classification model.
6. **`...`**: additional arguments  
   - Additional parameters for `get_multi_ligand_importances` if `single = FALSE`.
### Value:
A tibble containing several ligand activity scores with columns: `$test_ligand`, `$auroc`, `$aupr`, and `$pearson`.
### Example:
```R
ligand_activities <- predict_ligand_activities(
  geneset = c("SOCS2", "SOCS3", "IRF1"),
  background_expressed_genes = c("SOCS2", "SOCS3", "IRF1", "ICAM1", "ID1", "ID2", "ID3"),
  ligand_target_matrix = ligand_target_matrix,
  potential_ligands = c("TNF", "BMP2", "IL4")
)
```
## 3. `prepare_ligand_target_visualization`
```R
prepare_ligand_target_visualization(ligand_target_df, ligand_target_matrix, cutoff = 0.25)
```
### Description:
Prepares a heatmap visualization of the ligand-target links starting from a ligand-target tibble. It retrieves regulatory potential scores between all pairs of ligands and targets documented in the tibble. For better visualization, a quantile cutoff on the ligand-target scores is recommended.
### Parameters:
1. **`ligand_target_df`**: tibble  
   - A tibble with columns 'ligand', 'target', and 'weight' indicating ligand-target regulatory potential scores of interest.
2. **`ligand_target_matrix`**: matrix  
   - The NicheNet ligand-target matrix denoting regulatory potential scores between ligands and targets (ligands in columns).
3. **`cutoff`**: numeric (default: 0.25)  
   - Quantile cutoff on the ligand-target scores of the input weighted ligand-target network. Scores under this cutoff will be set to 0.
### Value:
A matrix containing the ligand-target regulatory potential scores between ligands of interest and their target genes.
### Example:
```R
active_ligand_target_links <- prepare_ligand_target_visualization(
  ligand_target_df = active_ligand_target_links_df,
  ligand_target_matrix = ligand_target_matrix,
  cutoff = 0.25
)
```

These descriptions are based on the official documentation provided in the links. If you have any further questions or need additional clarification, feel free to ask!