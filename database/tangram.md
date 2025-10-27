# Tangram
## The import form of the Tangram package
from scvi.external import Tangram
## Setup data for Tangram
Note that Tangram.setup_anndata() is not implemented, please use setup_mudata:
Tangram.setup_mudata(mdata, density_prior_key='rna_count_based', sc_layer=None, sp_layer=None, modalities=None, **kwargs)
### The detail description of parameters:
1. mdata (MuData) – MuData with scRNA and spatial modalities.
2. sc_layer (str | None (default: None)) – Layer key in scRNA modality to use for training.
3. sp_layer (str | None (default: None)) – Layer key in spatial modality to use for training.
4. density_prior_key (Union[str, Literal['rna_count_based', 'uniform'], None] (default: 'rna_count_based')) – Key in spatial modality obs for density prior.
5. modalities (dict[str, str] | None (default: None)) – Mapping from setup_mudata param name to modality in mdata.
## setup model function of Tangram
Tangram(sc_adata, constrained=False, target_count=None, **model_kwargs)
### the detail description of parameters:
1. mdata – MuData object that has been registered via setup_mudata().
2. constrained (bool (default: False)) – Whether to use the constrained version of Tangram instead of cells mode.
3. target_count (int | None (default: None)) – The number of cells to be filtered. Necessary when constrained is True.
4. **model_kwargs – Keyword args for TangramMapper
## how to train the Tangram model
Tangram.train(max_epochs=1000, accelerator='auto', devices='auto', lr=0.1, plan_kwargs=None)
### the detail description of parameters:
1. max_epochs (int (default: 1000)) – Number of passes through the dataset.
2. accelerator (str (default: 'auto')) – Supports passing different accelerator types (“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps, “auto”) as well as custom accelerator instances.
3. devices (int | list[int] | str (default: 'auto')) – The devices to use. Can be set to a non-negative index (int or str), a sequence of device indices (list or comma-separated str), the value -1 to indicate all available devices, or “auto” for automatic selection based on the chosen 4. accelerator. If set to “auto” and accelerator is not determined to be “cpu”, then devices will be set to the first available device.
5. lr (float (default: 0.1)) – Optimiser learning rate (default optimiser is ClippedAdam). Specifying optimiser via plan_kwargs overrides this choice of lr.
6. plan_kwargs (dict | None (default: None)) – Keyword args for JaxTrainingPlan. Keyword arguments passed to train() will overwrite values present in plan_kwargs, when appropriate.
## how to get results from trained model
Tangram.get_mapper_matrix()

Tangram.project_cell_annotations(adata_sc, adata_sp, mapper, labels)
### the detail description of parameters:
1. adata_sc (AnnData) – AnnData object with single-cell data.
2. adata_sp (AnnData) – AnnData object with spatial data.
3. mapper (ndarray) – Mapping from single-cell to spatial data.
4. labels (Series) – Cell annotations to project.

Tangram.project_genes(adata_sc, adata_sp, mapper)
### the detail description of parameters:
1. adata_sc (AnnData) – AnnData object with single-cell data.
2. adata_sp (AnnData) – AnnData object with spatial data.
3. mapper (ndarray) – Mapping from single-cell to spatial data.
## An Example of Tangram usage
>>> from scvi.external import Tangram
>>> ad_sc = anndata.read_h5ad(path_to_sc_anndata)
>>> ad_sp = anndata.read_h5ad(path_to_sp_anndata)
>>> markers = pd.read_csv(path_to_markers, index_col=0)  # genes to use for mapping
>>> mdata = mudata.MuData(
        {
            "sp_full": ad_sp,
            "sc_full": ad_sc,
            "sp": ad_sp[:, markers].copy(),
            "sc": ad_sc[:, markers].copy()
        }
    )
>>> modalities = {"density_prior_key": "sp", "sc_layer": "sc", "sp_layer": "sp"}
>>> Tangram.setup_mudata(
        mdata, density_prior_key="rna_count_based_density", modalities=modalities
    )
>>> tangram = Tangram(sc_adata)
>>> tangram.train()
>>> ad_sc.obsm["tangram_mapper"] = tangram.get_mapper_matrix()
>>> ad_sp.obsm["tangram_cts"] = tangram.project_cell_annotations(
        ad_sc, ad_sp, ad_sc.obsm["tangram_mapper"], ad_sc.obs["labels"]
    )
>>> projected_ad_sp = tangram.project_genes(ad_sc, ad_sp, ad_sc.obsm["tangram_mapper"])
