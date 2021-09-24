#!/bin/bash

sf_path=Image-processing/

# Run the whole thing on a series of fits images in a folder
images='*.fits'
declare -a catalogs

for i in $images
do
    source=$(echo $i | cut -d'.' -f 1)

    python $sf_path'sourcefinding.py' catalog $i --plot --survey MALS

    cd $source'_pybdsf'

    python $sf_path'catalog_matching.py' $source'_catalog.fits' NVSS --flux --astro --output
    python $sf_path'catalog_matching.py' $source'_catalog.fits' FIRST --flux --astro --output
    python $sf_path'catalog_matching.py' $source'_catalog.fits' SUMSS --flux --astro --output

    python $sf_path'catalog_analysis.py' $source'_catalog.fits' -r $source'_rms.fits'
    catalogs+=($source'_pybdsf/'$source'_catalog.fits')

    cd ../
done

python $sf_path'combine_catalogs.py' ${catalogs[@]} 'MALS_combined_catalogs.fits'
python $sf_path'catalog_analysis.py' 'MALS_combined_catalogs.fits'
