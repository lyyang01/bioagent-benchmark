# scPoli
## The import form of the scPoli package
from scarches.models.scpoli import scPoli
from scarches.models.scpoli._utils import reads_to_fragments
## Data Setup
reads_to_fragments(adata, copy=True)
## Model Initialization
scpoli.scPoli(adata, condition_keys, cell_type_keys, *, embedding_dims=None, recon_loss='nb', **kwargs)
### the detail description of parameters:
1. adata: AnnData
The annotated data matrix that contains the single-cell data to be integrated. It should have the structure (n_obs, n_vars), where n_obs is the number of cells and n_vars is the number of genes.
2. condition_keys: str or Sequence[str]
The column name(s) in adata.obs that specify the batch or condition information. For example, if you have a column named 'batch' in adata.obs that indicates the batch of each cell, you would pass 'batch' as the condition key. To integrate over multiple covariates, you can pass a list of column names.
3. cell_type_keys: str or Sequence[str]
The column name(s) in adata.obs that specify the cell type annotations. This is used for prototype learning. If you have a column named 'cell_type' in adata.obs that indicates the cell type of each cell, you would pass 'cell_type' as the cell type key. Multiple cell type annotations can be passed as a list.
4. embedding_dims: int or Sequence[int] (default: None)
The dimensionality of the embeddings. If an integer is passed, the model will use embeddings of the same dimensionality for each covariate. If different dimensionalities are desired for each covariate, a list of integers can be provided.
5. recon_loss: str (default: 'nb')
The type of reconstruction loss to be used. Common options include 'nb' for negative binomial loss, which is suitable for count data.
6. kwargs:
Additional keyword arguments that can be passed to the scPoli model initialization. These may include parameters specific to the scPoli algorithm, such as hyperparameters for regularization or optimization.
## Model training
scpoli.scPoli.train(n_epochs, pretraining_epochs, *, early_stopping_kwargs=None, eta=1.0, prototype_training=True, unlabeled_prototype_training=True, **kwargs)
### the detail description of parameters:
1. n_epochs: int
The total number of training epochs for the scPoli model.
2. pretraining_epochs: int
The number of epochs for which the model is trained in an unsupervised fashion. The finetuning epochs will be (n_epochs - pretraining_epochs).
3. early_stopping_kwargs: dict (default: None)
Arguments for early stopping during training. This can include parameters such as the patience (number of epochs with no improvement after which training will be stopped) and the minimum delta (minimum change in the monitored quantity to qualify as an improvement).
4. eta: float (default: 1.0)
The weight of the prototype loss in the overall loss function. This parameter controls the importance of the prototype learning component during training.
## get latent representation of reference data
scpoli_model.get_latent(adata_fragments, mean=True)
## get embedding
scpoli_model.get_conditional_embeddings()