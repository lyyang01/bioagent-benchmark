# clusterProfiler 
## Overview

clusterProfiler is an R package for statistical analysis and visualization of functional profiles for genes and gene clusters. It provides tools for gene ontology (GO) and pathway enrichment analysis.

## Functions

### groupGO

```r
groupGO(
  gene,
  OrgDb,
  keyType = "ENTREZID",
  ont = "CC",
  level = 2,
  readable = FALSE
)
```

Functional profile of a gene set at a specific GO level. Given a vector of genes, this function returns the GO profile at a specific level.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gene` | vector | required | A vector of entrez gene IDs |
| `OrgDb` | object | required | Organism annotation database (e.g., 'org.Hs.eg.db') |
| `keyType` | string | `"ENTREZID"` | Key type of input gene |
| `ont` | string | `"CC"` | One of "MF" (Molecular Function), "BP" (Biological Process), or "CC" (Cellular Component) subontologies |
| `level` | integer | `2` | Specific GO level to analyze |
| `readable` | logical | `FALSE` | If `TRUE`, gene IDs will be mapped to gene symbols |

#### Returns

A `groupGOResult` instance.

#### Example

```r
data(gcSample)
yy <- groupGO(gcSample[[1]], 'org.Hs.eg.db', ont="BP", level=2)
head(summary(yy))
```

### enrichGO

```r
enrichGO(
  gene,
  OrgDb,
  keyType = "ENTREZID",
  ont = "MF",
  pvalueCutoff = 0.05,
  pAdjustMethod = "BH",
  universe,
  qvalueCutoff = 0.2,
  minGSSize = 10,
  maxGSSize = 500,
  readable = FALSE,
  pool = FALSE
)
```

GO enrichment analysis of a gene set. Given a vector of genes, this function returns the enriched GO categories after FDR control.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gene` | vector | required | A vector of entrez gene IDs |
| `OrgDb` | object | required | Organism annotation database |
| `keyType` | string | `"ENTREZID"` | Key type of input gene |
| `ont` | string | `"MF"` | One of "BP", "MF", "CC", or "ALL" for all three subontologies |
| `pvalueCutoff` | numeric | `0.05` | Adjusted p-value cutoff for reporting enrichment |
| `pAdjustMethod` | string | `"BH"` | P-value adjustment method. One of "holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none" |
| `universe` | vector | optional | Background genes. If missing, all genes in the database will be used |
| `qvalueCutoff` | numeric | `0.2` | q-value cutoff for reporting significant terms |
| `minGSSize` | integer | `10` | Minimum size of gene set for testing |
| `maxGSSize` | integer | `500` | Maximum size of gene set for testing |
| `readable` | logical | `FALSE` | Whether to map gene IDs to gene names |
| `pool` | logical | `FALSE` | If `ont='ALL'`, whether to pool 3 GO sub-ontologies |

#### Returns

An `enrichResult` instance.

#### Example

```r
data(geneList, package = "DOSE")
de <- names(geneList)[1:100]
yy <- enrichGO(de, 'org.Hs.eg.db', ont="BP", pvalueCutoff=0.01)
head(yy)
```

### gseGO

```r
gseGO(
  geneList,
  ont = "BP",
  OrgDb,
  keyType = "ENTREZID",
  exponent = 1,
  minGSSize = 10,
  maxGSSize = 500,
  eps = 1e-10,
  pvalueCutoff = 0.05,
  pAdjustMethod = "BH",
  verbose = TRUE,
  seed = FALSE,
  by = "fgsea",
  ...
)
```

Gene Set Enrichment Analysis (GSEA) of Gene Ontology. Analyzes the statistically significant association of gene sets with a ranked list of genes.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geneList` | named vector | required | Ordered ranked gene list (usually by fold change or other metric) |
| `ont` | string | `"BP"` | One of "BP", "MF", "CC", or "ALL" for all three subontologies |
| `OrgDb` | object | required | Organism annotation database |
| `keyType` | string | `"ENTREZID"` | Key type of gene |
| `exponent` | numeric | `1` | Weight of each step |
| `minGSSize` | integer | `10` | Minimum size of gene set for testing |
| `maxGSSize` | integer | `500` | Maximum size of gene set for testing |
| `eps` | numeric | `1e-10` | Boundary for calculating p-values |
| `pvalueCutoff` | numeric | `0.05` | P-value cutoff |
| `pAdjustMethod` | string | `"BH"` | P-value adjustment method |
| `verbose` | logical | `TRUE` | Whether to print messages |
| `seed` | logical | `FALSE` | Set random seed for reproducibility |
| `by` | string | `"fgsea"` | Which algorithm to use: "fgsea" or "DOSE" |
| `...` | - | - | Other parameters |

#### Returns

A `gseaResult` object.

## Common Result Object Methods

The package returns result objects (`groupGOResult`, `enrichResult`, `gseaResult`) that share common methods:

- `summary()`: Returns a data frame with the enrichment results
- `head()`: Shows the first few rows of the results
- `plot()`: Generates a visualization of the results
- `dotplot()`: Creates a dot plot visualization
- `barplot()`: Creates a bar plot visualization
- `cnetplot()`: Creates a gene-concept network plot
- `heatplot()`: Creates a heat plot of enriched terms
- `enrichMap()`: Creates an enrichment map
- `goplot()`: Creates a GO term visualization (for GO results)

