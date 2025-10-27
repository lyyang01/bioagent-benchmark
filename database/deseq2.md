# PyDESeq2 API Documentation

## Overview

PyDESeq2 is a Python implementation of the popular DESeq2 R package for differential gene expression analysis. It follows the same statistical approach described in Love et al. (2014).

## Main Classes

PyDESeq2 consists of three main classes:

1. `DeseqDataSet`: For dispersion and log fold-change (LFC) estimation
2. `DeseqStats`: For statistical testing of differential expression
3. `DefaultInference`: Contains the statistical inference methods

## DeseqDataSet

```python
pydeseq2.dds.DeseqDataSet(
    *, 
    adata=None, 
    counts=None, 
    metadata=None, 
    design='~condition', 
    design_factors=None, 
    continuous_factors=None, 
    ref_level=None, 
    fit_type='parametric', 
    size_factors_fit_type='ratio', 
    control_genes=None, 
    min_mu=0.5, 
    min_disp=1e-08, 
    max_disp=10.0, 
    refit_cooks=True, 
    min_replicates=7, 
    beta_tol=1e-08, 
    n_cpus=None, 
    inference=None, 
    quiet=False, 
    low_memory=False
)
```

A class that extends AnnData to implement dispersion and log fold-change (LFC) estimation.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adata` | `AnnData` | `None` | AnnData object with counts ('X') and sample metadata ('obs') |
| `counts` | `pandas.DataFrame` | `None` | Raw counts matrix (required if adata is None) |
| `metadata` | `pandas.DataFrame` | `None` | Sample metadata (required if adata is None) |
| `design` | `str` or `DataFrame` | `'~condition'` | Model formula (e.g., '~condition') or design matrix |
| `fit_type` | `str` | `'parametric'` | Method for fitting dispersions: 'parametric' or 'mean' |
| `size_factors_fit_type` | `str` | `'ratio'` | Normalization method: 'ratio', 'poscounts' or 'iterative' |
| `control_genes` | `list` or `Index` | `None` | Genes to use for size factor estimation |
| `min_mu` | `float` | `0.5` | Threshold for mean estimates |
| `min_disp` | `float` | `1e-08` | Lower threshold for dispersion |
| `max_disp` | `float` | `10.0` | Upper threshold for dispersion |
| `refit_cooks` | `bool` | `True` | Whether to refit Cook's outliers |
| `min_replicates` | `int` | `7` | Minimum replicates for refitting samples |
| `beta_tol` | `float` | `1e-08` | Stopping criterion for IRWLS |
| `n_cpus` | `int` | `None` | Number of CPUs to use |
| `quiet` | `bool` | `False` | Suppress status updates |
| `low_memory` | `bool` | `False` | Remove intermediate data structures after use |

### Key Methods

#### deseq2(fit_type=None)

Perform the full DESeq2 pipeline: fit size factors, estimate dispersions, and fit LFCs.

```python
dds.deseq2()
```

#### fit_size_factors(fit_type=None, control_genes=None)

Fit sample-wise normalization factors using the median-of-ratios method.

```python
dds.fit_size_factors(fit_type='ratio')
```

#### fit_genewise_dispersions(vst=False)

Fit gene-wise dispersion estimates independently.

```python
dds.fit_genewise_dispersions()
```

#### fit_dispersion_trend(vst=False)

Fit the dispersion trend curve.

```python
dds.fit_dispersion_trend()
```

#### fit_MAP_dispersions()

Fit Maximum a Posteriori dispersion estimates.

```python
dds.fit_MAP_dispersions()
```

#### fit_LFC()

Fit log fold change coefficients.

```python
dds.fit_LFC()
```

#### refit()

Refit Cook's outliers.

```python
dds.refit()
```

#### vst(use_design=False, fit_type=None)

Apply variance stabilizing transformation to normalized counts.

```python
dds.vst()
```

#### plot_dispersions(log=True, save_path=None, **kwargs)

Plot dispersions with trend curve.

```python
dds.plot_dispersions()
```

## DeseqStats

```python
pydeseq2.ds.DeseqStats(
    dds, 
    contrast, 
    alpha=0.05, 
    cooks_filter=True, 
    independent_filter=True, 
    prior_LFC_var=None, 
    lfc_null=0.0, 
    alt_hypothesis=None, 
    inference=None, 
    quiet=False
)
```

Performs statistical tests for differential expression analysis.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dds` | `DeseqDataSet` | required | DeseqDataSet with estimated dispersions and LFCs |
| `contrast` | `list` or `ndarray` | required | Either ['variable', 'level_1', 'level_2'] or a contrast vector |
| `alpha` | `float` | `0.05` | Significance threshold for p-values |
| `cooks_filter` | `bool` | `True` | Filter p-values based on Cook's outliers |
| `independent_filter` | `bool` | `True` | Perform independent filtering of p-values |
| `prior_LFC_var` | `ndarray` | `None` | Prior variance for LFCs |
| `lfc_null` | `float` | `0.0` | Log fold change under null hypothesis |
| `alt_hypothesis` | `str` | `None` | Alternative hypothesis for Wald test |
| `quiet` | `bool` | `False` | Suppress status updates |

### Key Methods

#### summary(**kwargs)

Run the statistical analysis and store results in `results_df`.

```python
stats.summary()
```

#### run_wald_test()

Perform a Wald test to get gene-wise p-values.

```python
stats.run_wald_test()
```

#### lfc_shrink(coeff, adapt=True)

Apply LFC shrinkage with an apeGLM prior.

```python
stats.lfc_shrink("condition_B_vs_A")
```

#### plot_MA(log=True, save_path=None, **kwargs)

Create an MA plot of log fold changes vs. mean expression.

```python
stats.plot_MA()
```

### Key Attributes

- `results_df`: DataFrame with results of differential expression analysis
- `p_values`: P-values from Wald test
- `padj`: Adjusted p-values for multiple testing
- `LFC`: Estimated log fold changes
- `SE`: Standard errors for log fold changes
- `statistics`: Wald test statistics
- `base_mean`: Gene-wise means of normalized counts

## DefaultInference

```python
pydeseq2.default_inference.DefaultInference(
    joblib_verbosity=0, 
    batch_size=128, 
    n_cpus=None, 
    backend='loky'
)
```

Contains the statistical inference methods used by PyDESeq2.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `joblib_verbosity` | `int` | `0` | Verbosity level for joblib tasks |
| `batch_size` | `int` | `128` | Number of tasks per worker |
| `n_cpus` | `int` | `None` | Number of CPUs to use |
| `backend` | `str` | `'loky'` | Joblib backend |

### Key Methods

Most methods in this class are used internally by DeseqDataSet and DeseqStats and not typically called directly by users.

