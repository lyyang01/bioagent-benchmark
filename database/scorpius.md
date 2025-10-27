# SCORPIUS
## load SCORPIUS packages
library(SCORPIUS)
## reduce_dimensionality
```R
reduce_dimensionality(x, dist = c("spearman", "pearson", "euclidean", "cosine", "manhattan"), ndim = 3, num_landmarks = 1000)
```
### Description:
Performs dimensionality reduction by performing an eigenanalysis of the given dissimilarity matrix and returns coordinates of the samples in a lower-dimensional space.
### Parameters:
1. **`x`**: numeric matrix  
   - A numeric matrix containing the expression data of genes (rows) in cells (columns).
2. **`dist`**: character (default: "spearman")  
   - The distance metric to be used. Options include "spearman", "pearson", "euclidean", "cosine", "manhattan".
3. **`ndim`**: integer (default: 3)  
   - The number of dimensions in the new space.
4. **`num_landmarks`**: integer (default: 1000)  
   - The number of landmarks to be selected.
### Example:
```R
space <- reduce_dimensionality(expression_data, dist = "spearman", ndim = 2)
```
## 2. `infer_initial_trajectory`
```R
infer_initial_trajectory(space, k)
```
### Description:
Infers an initial trajectory by clustering the points and calculating the shortest path through cluster centers.

### Parameters:
1. **`space`**: numeric matrix  
   - A numeric matrix containing the coordinates of samples in a reduced-dimensional space.
2. **`k`**: integer  
   - The number of clusters to use for clustering the data.

### Example:
```R
initial_traj <- infer_initial_trajectory(space, k = 4)
```

## 3. `infer_trajectory`
```R
infer_trajectory(space, k = 4, thresh = 0.001, maxit = 10, stretch = 0, smoother = "smooth_spline", approx_points = 100)
```
### Description:
Infers a trajectory through samples in a given space using a four-step process: clustering, distance matrix calculation, shortest path finding, and iterative curve fitting.

### Parameters:
1. **`space`**: numeric matrix  
   - A numeric matrix containing the coordinates of samples in a reduced-dimensional space.
2. **`k`**: integer (default: 4)  
   - The number of clusters to use for clustering the data.
3. **`thresh`**: numeric (default: 0.001)  
   - Convergence threshold on shortest distances to the curve.
4. **`maxit`**: integer (default: 10)  
   - Maximum number of iterations.
5. **`stretch`**: numeric (default: 0)  
   - Stretch factor for the endpoints of the curve.
6. **`smoother`**: character (default: "smooth_spline")  
   - Choice of smoother. Options include "smooth_spline", "lowess", "periodic_lowess".
7. **`approx_points`**: integer (default: 100)  
   - Number of points to approximate the curve to.

### Example:
```R
traj <- infer_trajectory(space, k = 4, thresh = 0.001, maxit = 10)
```

## 4. `draw_trajectory_plot`
```R
draw_trajectory_plot(space, progression_group = NULL, path = NULL, contour = FALSE, progression_group_palette = NULL, point_size = 2, point_alpha = 1, path_size = 0.5, path_alpha = 1, contour_alpha = 0.2)
```
### Description:
Plots samples in a reduced-dimensional space with optional coloring by progression group, trajectory path, and contours.

### Parameters:
1. **`space`**: numeric matrix  
   - A numeric matrix containing the coordinates of samples in a reduced-dimensional space.
2. **`progression_group`**: vector or factor (default: NULL)  
   - Groupings of the samples.
3. **`path`**: numeric matrix (default: NULL)  
   - Coordinates of the inferred path.
4. **`contour`**: logical (default: FALSE)  
   - Whether to draw contours around the samples.
5. **`progression_group_palette`**: named vector (default: NULL)  
   - Palette for the progression group.
6. **`point_size`**: numeric (default: 2)  
   - Size of the points.
7. **`point_alpha`**: numeric (default: 1)  
   - Alpha value for the points.
8. **`path_size`**: numeric (default: 0.5)  
   - Size of the path.
9. **`path_alpha`**: numeric (default: 1)  
   - Alpha value for the path.
10. **`contour_alpha`**: numeric (default: 0.2)  
    - Alpha value for the contour.

### Example:
```R
draw_trajectory_plot(space, progression_group = groups, path = traj$path, contour = TRUE)
```

## 5. `draw_trajectory_heatmap`
```R
draw_trajectory_heatmap(x, time, progression_group = NULL, modules = NULL, show_labels_row = FALSE, show_labels_col = FALSE, scale_features = TRUE, progression_group_palette = NULL, ...)
```
### Description:
Draws a heatmap of samples ordered by their position in an inferred trajectory, with optional progression groups and feature modules.

### Parameters:
1. **`x`**: numeric matrix  
   - A numeric matrix with one row per sample and one column per feature.
2. **`time`**: numeric vector  
   - Inferred time points of each sample along a trajectory.
3. **`progression_group`**: vector or factor (default: NULL)  
   - Groupings of the samples.
4. **`modules`**: data frame (default: NULL)  
   - Data frame as returned by `extract_modules`.
5. **`show_labels_row`**: logical (default: FALSE)  
   - Whether to show row labels.
6. **`show_labels_col`**: logical (default: FALSE)  
   - Whether to show column labels.
7. **`scale_features`**: logical (default: TRUE)  
   - Whether to scale the values of each feature.
8. **`progression_group_palette`**: named vector (default: NULL)  
   - Palette for the progression group.
9. **`...`**: additional arguments  
   - Passed to `pheatmap`.

### Example:
```R
draw_trajectory_heatmap(expression_data, time = traj$time, progression_group = groups)
```

## 6. `extract_modules`
```R
extract_modules(x, time = NULL, suppress_warnings = FALSE, verbose = FALSE, ...)
```
### Description:
Extracts modules of features using adaptive branch pruning, typically on smoothed expression data.

### Parameters:
1. **`x`**: numeric matrix  
   - A numeric matrix with one row per sample and one column per feature.
2. **`time`**: numeric vector (default: NULL)  
   - Inferred time points of each sample along a trajectory.
3. **`suppress_warnings`**: logical (default: FALSE)  
   - Whether to suppress warnings.
4. **`verbose`**: logical (default: FALSE)  
   - Whether to print output from Mclust.
5. **`...`**: additional arguments  
   - Passed to Mclust.

### Example:
```R
modules <- extract_modules(expression_data, time = traj$time)
```

## 7. `gene_importances`
```R
gene_importances(x, time, num_permutations = 0, ntree = 10000, ntree_perm = ntree/10, mtry = ncol(x) * 0.01, num_threads = 1, ...)
```
### Description:
Calculates the importance of each feature in predicting the time ordering.

### Parameters:
1. **`x`**: numeric matrix  
   - A numeric matrix with one row per sample and one column per feature.
2. **`time`**: numeric vector  
   - Inferred time points of each sample along a trajectory.
3. **`num_permutations`**: integer (default: 0)  
   - Number of permutations to test against for calculating p-values.
4. **`ntree`**: integer (default: 10000)  
   - Number of trees to grow.
5. **`ntree_perm`**: integer (default: ntree/10)  
   - Number of trees to grow for each permutation.
6. **`mtry`**: integer (default: 1% of features)  
   - Number of variables randomly sampled at each split.
7. **`num_threads`**: integer (default: 1)  
   - Number of threads to use.
8. **`...`**: additional arguments  
   - Passed to `ranger`.

### Example:
```R
gene_importances(expression_data, time = traj$time, num_permutations = 0, ntree = 1000)
```

## 8. `generate_dataset`
```R
generate_dataset(num_samples = 400, num_genes = 500, num_groups = 4)
```
### Description:
Generates a synthetic dataset for visualization purposes.

### Parameters:
1. **`num_samples`**: integer (default: 400)  
   - Number of samples in the dataset.
2. **`num_genes`**: integer (default: 500)  
   - Number of genes in the dataset.
3. **`num_groups`**: integer (default: 4)  
   - Number of groups the samples will be split into.

### Example:
```R
dataset <- generate_dataset(num_samples = 300, num_genes = 500, num_groups = 4)
```

## 9. `reverse_trajectory`
```R
reverse_trajectory(trajectory)
```
### Description:
Reverses the direction of a given trajectory.

### Parameters:
1. **`trajectory`**: list  
   - A trajectory as returned by `infer_trajectory`.

### Example:
```R
reversed_traj <- reverse_trajectory(traj)
```

## 10. `ti_scorpius`
```R
ti_scorpius(distance_method = "spearman", ndim = 3L, k = 4L, thresh = 0.001, maxit = 10L, stretch = 0, smoother = "smooth_spline")
```
### Description:
Creates a trajectory inference method using SCORPIUS, compatible with `dynwrap::infer_trajectory()`.

### Parameters:
1. **`distance_method`**: character (default: "spearman")  
   - Distance metric to use. Options include "spearman", "pearson", "cosine".
2. **`ndim`**: integer (default: 3)  
   - Number of dimensions in the new space.
3. **`k`**: integer (default: 4)  
   - Number of clusters to use for constructing the initial trajectory.
4. **`thresh`**: numeric (default: 0.001)  
   - Convergence threshold for shortest distances to the curve.
5. **`maxit`**: integer (default: 10)  
   - Maximum number of iterations.
6. **`stretch`**: numeric (default: 0)  
   - Stretch factor for the endpoints of the curve.
7. **`smoother`**: character (default: "smooth_spline")  
   - Choice of smoother. Options include "smooth_spline", "lowess", "periodic_lowess".

### Example:
```R
ti_method <- ti_scorpius(distance_method = "spearman", ndim = 3, k = 4)
```

These functions provide a comprehensive toolkit for analyzing single-cell RNA sequencing data to infer developmental trajectories and visualize the results.