# PAGA
## The import form of the paga package
import scanpy
## Generates a Partition-based Graph Abstraction (PAGA) graph, which simplifies the complex manifold structure of single-cell data into a coarse-grained graph of partitions
scanpy.tl.paga(adata, groups, use_rna_velocity, model, copy)
### the detail description of parameters:
1. adata: The annotated data matrix (AnnData object).
2. groups: Key for the categorical annotation in adata.obs that defines the partitions (e.g., clusters). Default is 'leiden' or 'louvain' if available.
3. use_rna_velocity: Boolean. If True, uses RNA velocity to orient edges in the PAGA graph. Requires a directed single-cell graph in adata.uns['velocity_graph'].
4. model: The PAGA connectivity model version. Default is 'v1.2'.
5. copy: Boolean. If True, returns a copy of adata with the PAGA results. Otherwise, modifies adata in place
## Visualizes the PAGA graph, showing the connectivity between partitions
scanpy.pl.paga(adata, threshold, color, show, save)
### the detail description of parameters:
1. adata: The annotated data matrix (AnnData object) containing PAGA results.
2. threshold: Float. Minimum confidence threshold for displaying edges in the PAGA graph.
3. color: List of strings. Annotations to color the nodes by (e.g., cluster labels or gene expression).
4. show: Boolean. If True, displays the plot. Otherwise, returns the plot object.
5. save: Boolean or string. If True, saves the plot to a file. Can also specify a filename
## Compares the PAGA graph with other embeddings or annotations
scanpy.pl.paga_compare(adata, threshold, title, right_margin, size, edge_width_scale, legend_fontsize, frameon, edges, save)
### the detail description of parameters:
1. adata: The annotated data matrix (AnnData object) containing PAGA results.
2. threshold: Float. Minimum confidence threshold for displaying edges in the PAGA graph.
3. title: String or list of strings. Titles for the subplots.
4. right_margin: Float. Additional margin on the right side of the plot.
5. size: Float. Size of the nodes.
6. edge_width_scale: Float. Scaling factor for the edge widths.
7. legend_fontsize: Float. Font size for the legend.
8. fontsize: Float. Font size for labels.
9. frameon: Boolean. If True, draws a frame around the plot.
10. edges: Boolean. If True, displays edges in the PAGA graph.
11. save: Boolean or string. If True, saves the plot to a file. Can also specify a filename