# Celltypist
## The import form of the Celltypist package
import celltypist
## Training function
celltypist.train(X=None, labels: str | list | tuple | ndarray | Series | Index | None = None, genes: str | list | tuple | ndarray | Series | Index | None = None, transpose_input: bool = False, with_mean: bool = True, check_expression: bool = True, C: float = 1.0, solver: str | None = None, max_iter: int | None = None, n_jobs: int | None = None, use_SGD: bool = False, alpha: float = 0.0001, use_GPU: bool = False, mini_batch: bool = False, batch_number: int = 100, batch_size: int = 1000, epochs: int = 10, balance_cell_type: bool = False, feature_selection: bool = False, top_genes: int = 300, date: str = '', details: str = '', url: str = '', source: str = '', version: str = '', **kwargs)
### the detail description of parameters:
1. X – Path to the input count matrix (supported types are csv, txt, tsv, tab and mtx) or AnnData (h5ad). Also accepts the input as an AnnData object, or any array-like objects already loaded in memory. See check_expression for detailed format requirements. A cell-by-gene format is desirable (see transpose_input for more information).
2. labels – Path to the file containing cell type label per line corresponding to the cells in X. Also accepts any list-like objects already loaded in memory (such as an array). If X is specified as an AnnData, this argument can also be set as a column name from cell metadata.
3. genes – Path to the file containing one gene per line corresponding to the genes in X. Also accepts any list-like objects already loaded in memory (such as an array). Note genes will be extracted from X where possible (e.g., X is an AnnData or data frame).
4. transpose_input – Whether to transpose the input matrix. Set to True if X is provided in a gene-by-cell format. (Default: False)
5. with_mean – Whether to subtract the mean values during data scaling. Setting to False can lower the memory usage when the input is a sparse matrix but may slightly reduce the model performance. (Default: True)
6. check_expression – Check whether the expression matrix in the input data is supplied as required. Except the case where a path to the raw count table file is specified, all other inputs for X should be in log1p normalized expression to 10000 counts per cell. Set to False if you want to train the data regardless of the expression formats. (Default: True)
7. C – Inverse of L2 regularization strength for traditional logistic classifier. A smaller value can possibly improve model generalization while at the cost of decreased accuracy. This argument is ignored if SGD learning is enabled (use_SGD = True). (Default: 1.0)
8. solver – Algorithm to use in the optimization problem for traditional logistic classifier. The default behavior is to choose the solver according to the size of the input data. This argument is ignored if SGD learning is enabled (use_SGD = True).
9. max_iter – Maximum number of iterations before reaching the minimum of the cost function. Try to decrease max_iter if the cost function does not converge for a long time. This argument is for both traditional and SGD logistic classifiers, and will be ignored if mini-batch SGD training is conducted (use_SGD = True and mini_batch = True). Default to 200, 500, and 1000 for large (>500k cells), medium (50-500k), and small (<50k) datasets, respectively.
10. n_jobs – Number of CPUs used. Default to one CPU. -1 means all CPUs are used. This argument is for both traditional and SGD logistic classifiers.
11. use_SGD – Whether to implement SGD learning for the logistic classifier. (Default: False)
12. alpha – L2 regularization strength for SGD logistic classifier. A larger value can possibly improve model generalization while at the cost of decreased accuracy. This argument is ignored if SGD learning is disabled (use_SGD = False). (Default: 0.0001)
13. use_GPU – Whether to use GPU for logistic classifier. This argument is ignored if SGD learning is enabled (use_SGD = True). (Default: False)
14. mini_batch – Whether to implement mini-batch training for the SGD logistic classifier. Setting to True may improve the training efficiency for large datasets (for example, >100k cells). This argument is ignored if SGD learning is disabled (use_SGD = False). (Default: False)
15. batch_number – The number of batches used for training in each epoch. Each batch contains batch_size cells. For datasets which cannot be binned into batch_number batches, all batches will be used. This argument is relevant only if mini-batch SGD training is conducted (use_SGD = True and mini_batch = True). (Default: 100)
16. batch_size – The number of cells within each batch. This argument is relevant only if mini-batch SGD training is conducted (use_SGD = True and mini_batch = True). (Default: 1000)
17. epochs – The number of epochs for the mini-batch training procedure. The default values of batch_number, batch_size, and epochs together allow observing ~10^6 training cells. This argument is relevant only if mini-batch SGD training is conducted (use_SGD = True and mini_batch = True). (Default: 10)
18. balance_cell_type – Whether to balance the cell type frequencies in mini-batches during each epoch. Setting to True will sample rare cell types with a higher probability, ensuring close-to-even cell type distributions in mini-batches. This argument is relevant only if mini-batch SGD training is conducted (use_SGD = True and mini_batch = True). (Default: False)
19. feature_selection – Whether to perform two-pass data training where the first round is used for selecting important features/genes using SGD learning. If True, the training time will be longer. (Default: False)
20. top_genes – The number of top genes selected from each class/cell-type based on their absolute regression coefficients. The final feature set is combined across all classes (i.e., union). (Default: 300)
21. date – Free text of the date of the model. Default to the time when the training is completed.
22. details – Free text of the description of the model.
23. url – Free text of the (possible) download url of the model.
24. source – Free text of the source (publication, database, etc.) of the model.
25. version – Free text of the version of the model.
26. **kwargs – Other keyword arguments passed to LogisticRegression (use_SGD = False and use_GPU = False), cuml.LogisticRegression (use_SGD = False and use_GPU = True), or SGDClassifier (use_SGD = True).
### returns
An instance of the Model trained by celltypist.
### return type
Model
## Prediction function
celltypist.annotate(filename: AnnData | str = '', model: str | Model | None = None, transpose_input: bool = False, gene_file: str | None = None, cell_file: str | None = None, mode: str = 'best match', p_thres: float = 0.5, majority_voting: bool = False, over_clustering: str | list | tuple | ndarray | Series | Index | None = None, use_GPU: bool = False, min_prop: float = 0)
### the detail description of parameters:
1. filename – Path to the input count matrix (supported types are csv, txt, tsv, tab and mtx) or AnnData (h5ad). If it’s the former, a cell-by-gene format is desirable (see transpose_input for more information). Also accepts the input as an AnnData object already loaded in memory. Genes should be gene symbols. Non-expressed genes are preferred to be provided as well.
2. model – Model used to predict the input cells. Default to using the ‘Immune_All_Low.pkl’ model. Can be a Model object that wraps the logistic Classifier and the StandardScaler, the path to the desired model file, or the model name. To see all available models and their descriptions, use models_description().
3. transpose_input – Whether to transpose the input matrix. Set to True if filename is provided in a gene-by-cell format. (Default: False)
4. gene_file – Path to the file which stores each gene per line corresponding to the genes used in the provided mtx file. Ignored if filename is not provided in the mtx format.
5. cell_file – Path to the file which stores each cell per line corresponding to the cells used in the provided mtx file. Ignored if filename is not provided in the mtx format.
6. mode – The way cell prediction is performed. For each query cell, the default (‘best match’) is to choose the cell type with the largest score/probability as the final prediction. Setting to ‘prob match’ will enable a multi-label classification, which assigns 0 (i.e., unassigned), 1, or >=2 cell type labels to each query cell. (Default: ‘best match’)
7. p_thres – Probability threshold for the multi-label classification. Ignored if mode is ‘best match’. (Default: 0.5)
8. majority_voting – Whether to refine the predicted labels by running the majority voting classifier after over-clustering. (Default: False)
9. over_clustering – This argument can be provided in several ways: 1) an input plain file with the over-clustering result of one cell per line. 2) a string key specifying an existing metadata column in the AnnData (pre-created by the user). 3) a python list, tuple, numpy array, pandas series or index representing the over-clustering result of the input cells. 4) if none of the above is provided, will use a heuristic over-clustering approach according to the size of input data. Ignored if majority_voting is set to False.
10. use_GPU – Whether to use GPU for over clustering on the basis of rapids-singlecell. This argument is only relevant when majority_voting = True. (Default: False)
11. min_prop – For the dominant cell type within a subcluster, the minimum proportion of cells required to support naming of the subcluster by this cell type. Ignored if majority_voting is set to False. Subcluster that fails to pass this proportion threshold will be assigned ‘Heterogeneous’. (Default: 0)
### returns
An AnnotationResult object. Four important attributes within this class are: 1) predicted_labels, predicted labels from celltypist. 2) decision_matrix, decision matrix from celltypist. 3) probability_matrix, probability matrix from celltypist. 4) adata, AnnData representation of the input data.
### return type
AnnotationResult
## Dot plot function
celltypist.dotplot(predictions: AnnotationResult, use_as_reference: str | list | tuple | ndarray | Series | Index, use_as_prediction: str = 'majority_voting', prediction_order: str | list | tuple | ndarray | Series | Index | None = None, reference_order: str | list | tuple | ndarray | Series | Index | None = None, filter_prediction: float = 0.0, cmap: str = 'RdBu_r', vmin: float | None = 0.0, vmax: float | None = 1.0, colorbar_title: str | None = 'Mean probability', dot_min: float | None = 0.0, dot_max: float | None = 1.0, smallest_dot: float | None = 0.0, size_title: str | None = 'Fraction of cells (%)', swap_axes: bool | None = False, title: str | None = 'CellTypist label transfer', figsize: tuple | None = None, show: bool | None = None, save: str | bool | None = None, ax: _AxesSubplot | None = None, return_fig: bool | None = False, **kwds)
### the detail description of parameters:
1. predictions – An AnnotationResult object containing celltypist prediction result through annotate().
2. use_as_reference – Key (column name) of the input AnnData representing the reference cell types (or clusters) celltypist will assess. Also accepts any list-like objects already loaded in memory (such as an array).
3. use_as_prediction – Column name of predicted_labels specifying the prediction type which the assessment is based on. Set to ‘predicted_labels’ if you want to assess the prediction result without majority voting. (Default: ‘majority_voting’)
4. prediction_order – Order in which to show the predicted cell types. Can be a subset of predicted cell type labels. Default to plotting all predicted labels, with the order of categories as is (alphabetical order in most cases).
5. reference_order – Order in which to show the reference cell types (or clusters). Can be a subset of reference cell types (or clusters). Default to plotting all reference cell types, with an order that ensures the resulting dot plot is diagonal.
6. filter_prediction – Filter out the predicted cell types with the maximal assignment fractions less than filter_prediction. This argument is only effective when prediction_order is not specified, and can be used to reduce the number of predicted cell types displayed in the dot plot. Default to 0 (no filtering).
7. title – Title of the dot plot. (Default: ‘CellTypist label transfer’)
8. size_title – Legend title for the dot sizes. (Default: ‘Fraction of cells (%)’)
9. colorbar_title – Legend title for the dot colors. (Default: ‘Mean probability’)
10. swap_axes – Whether to swap the x and y axes. (Default: False)
11. others – All other parameters are the same as scanpy.pl.dotplot() with selected tags and customized defaults.
### return type
If return_fig is True, returns a scanpy.pl.DotPlot object, else if show is false, return axes dict.
## Model download function
celltypist.models.download_models(force_update: bool = False, model: str | list | tuple | None = None)
### the detail description of parameters:
1. force_update – Whether to fetch a latest JSON index for downloading all available or selected models. Set to True if you want to parallel the latest celltypist model releases. (Default: False)
2. model – Specific model(s) to download. By default, all available models are downloaded. Set to a specific model name or a list of model names to only download a subset of models. For example, set to [“ModelA.pkl”, “ModelB.pkl”] to only download ModelA and ModelB. To check all available models, use models_description().
## Downsampling function
celltypist.samples.downsample_adata(adata: AnnData, mode: str = 'total', n_cells: int | None = None, by: str | None = None, balance_cell_type: bool = False, random_state: int = 0, return_index: bool = True)
### the detail description of parameters:
1. adata – An AnnData object representing the input data.
2. mode – The way downsampling is performed. Default to downsampling the input cells to a total of n_cells. Set to ‘each’ if you want to downsample cells within each cell type to n_cells. (Default: ‘total’)
3. n_cells – The total number of cells (mode = ‘total’) or the number of cells from each cell type (mode = ‘each’) to sample. For the latter, all cells from a given cell type will be selected if its cell number is fewer than n_cells.
4. by – Key (column name) of the input AnnData representing the cell types.
5. balance_cell_type – Whether to balance the cell type frequencies when mode = ‘total’. Setting to True will sample rare cell types with a higher probability, ensuring close-to-even cell type compositions. This argument is ignored if mode = ‘each’. (Default: False)
6. random_state – Random seed for reproducibility.
7. return_index – Only return the downsampled cell indices. Setting to False if you want to get a downsampled version of the input AnnData. (Default: True)
### return type
Depending on return_index, returns the downsampled cell indices or a subset of the input AnnData.