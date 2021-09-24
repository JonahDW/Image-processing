#!/bin/bash

sf_path=Image-processing/
singularity_img=Software-images/sourcefinding.sif

# Run the whole thing on a series of fits images in a folder
images='*.fits'
declare -a catalogs

for i in $images
do
    source=$(echo $i | cut -d'.' -f 1)

    singularity exec --bind $PWD,$sf_path $singularity_img python3 $sf_path'sourcefinding.py' catalog $i --plot --survey MALS

    cd $source'_pybdsf'

    singularity exec --bind $PWD,$sf_path $singularity_img python3  $sf_path'catalog_matching.py' $source'_catalog.fits' NVSS --flux --astro --output
    singularity exec --bind $PWD,$sf_path $singularity_img python3  $sf_path'catalog_matching.py' $source'_catalog.fits' FIRST --flux --astro --output
    singularity exec --bind $PWD,$sf_path $singularity_img python3  $sf_path'catalog_matching.py' $source'_catalog.fits' SUMSS --flux --astro --output

    # Since catalog_analysis uses latex fonts, singularity can cause problems, try to run with regular python if this happens
    singularity exec --bind $PWD,$sf_path $singularity_img python3  $sf_path'catalog_analysis.py' $source'_catalog.fits' -r $source'_rms.fits'
    catalogs+=($source'_pybdsf/'$source'_catalog.fits')

    cd ../
done

singularity exec --bind $PWD,$sf_path $singularity_img python3 $sf_path'combine_catalogs.py' ${catalogs[@]} 'MALS_combined_catalogs.fits'
singularity exec --bind $PWD,$sf_path $singularity_img python3 $sf_path'catalog_analysis.py' 'MALS_combined_catalogs.fits'
