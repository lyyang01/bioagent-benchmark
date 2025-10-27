# novoSpaRc
## The import form of the novoSpaRc package
import novosparc
## Create a tissue target space from a given image. The image is assumed to contain a black-colored tissue space in white background.
novosparc.gm.create_target_space_from_image(image)
### the detail description of parameters:
1. image: the location of the image on the disk.
## Create a circular target space for spatial reconstruction in the NovoSpaRc framework
novosparc.gm.construct_circle(num_locations, random=False)
### the detail description of parameters:
1. num_locations: int
The number of spatial locations to generate in the circular pattern. This parameter determines the density of the target space.
2. random: bool (default: False)
If set to True, the locations are randomly distributed within the circular space. If False, the locations are evenly spaced along the circumference of the circle.
## The class that handles the processes for the tissue reconstruction. It is responsible for keeping the data, creating the reconstruction and saving the results.
Initialize the tissue using the dataset and locations:
novosparc.cm.Tissue(dataset, locations, atlas_matrix, markers_to_use, output_folder)
### the detail description of parameters:
1. dataset        -- Anndata object for the single cell data (cells x genes)
2. locations      -- target space locations (locations x dimensions)
3. atlas_matrix   -- optional atlas matrix (atlas locations x markers)
4. markers_to_use -- optional indices of atlas marker genes in dataset
5. output_folder  -- folder path to save the plots and data
## Plots fields (color) of Scanpy AnnData object on spatial coordinates
novosparc.pl.embedding(dataset, color)
### the detail description of parameters:
1. dataset -- Scanpy AnnData with 'spatial' matrix in obsm containing the spatial coordinates of the tissue
2. color -- a list of fields - gene names or columns from obs to use for color
## Plots expression distances vs physical distances over locations
novosparc.pl.plot_exp_loc_dists(exp, locations)
### the detail description of parameters:
1. exp       -- spatial expression over locations (locations x genes)
2. locations -- spatial coordinates of locations (locations x dimensions)
## Calculates the spatial correlation (Moran's I value) and its corresponding one-tailed p-value under a normal distribution assumption (permuting locations)
novosparc.an.get_moran_pvals(sdge, locations, npermut=100, n_neighbors=8)
### the detail description of parameters:
1. sdge        -- expression over locations (locations x genes)
2. locations   -- spatial coordinates (locations x dimensions)
3. npermut     -- number of permutations
4. n_neighbors -- number of neighbors to consider for each location