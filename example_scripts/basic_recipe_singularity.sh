#!/bin/bash

# Specify paths
sf_path=/Image-processing/
singularity_img=/Software-images/sourcefinding_latest.sif

# Run the whole thing on a series of fits images in a folder
images='J*.fits'
declare -a catalogs

for i in $images
do
    source=$(echo $i | cut -d'.' -f 1)

	# Perform sourcefinding on the image
    singularity exec --bind $PWD,$sf_path $singularity_img python3 $sf_path'sourcefinding.py' catalog $i --plot --survey MALS
    working_catalog=$source'_pybdsf'/$source'_bdsfcat.fits'

    # Exclude sources fit by multiple gaussians
    singularity exec --bind $PWD,$sf_path $singularity_img python3 $sf_path'source_catalogue_crusher.py' --CAT_FILE $working_catalog --OUTPUTNAME $working_catalog --RESET_FLAG --DO_SELECT_PYBDSF_TYPE -M

	# Cross match catalogs, sources not selected with the catalog crusher are ignored
    singularity exec --bind $PWD,$sf_path $singularity_img python3 $sf_path'catalog_matching.py' $working_catalog NVSS --flux --astro --output
    singularity exec --bind $PWD,$sf_path $singularity_img python3 $sf_path'catalog_matching.py' $working_catalog FIRST --flux --astro --output
    singularity exec --bind $PWD,$sf_path $singularity_img python3 $sf_path'catalog_matching.py' $working_catalog SUMSS --flux --astro --output

	# Catalog analysis will measure number counts and resolved fraction
    singularity exec --bind $PWD,$sf_path $singularity_img python3  $sf_path'catalog_analysis.py' $working_catalog -r $source'_pybdsf'/$source'_rms.fits'
    catalogs+=($working_catalog)

done

# If multiple images are being catalogued, they can be combined into a single stacked catalog
singularity exec --bind $PWD,$sf_path $singularity_img python3 $sf_path'combine_catalogs.py' ${catalogs[@]} 'MALS_combined_catalogs.fits'
singularity exec --bind $PWD,$sf_path $singularity_img python3 $sf_path'catalog_analysis.py' 'MALS_combined_catalogs.fits' --stacked_catalog
