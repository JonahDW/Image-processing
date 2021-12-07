#!/bin/bash
set -e

sf_path=Image-processing/

# Run the whole thing on a series of fits images in a folder
images='J*.fits'
declare -a catalogs

for i in $images
do
    source=$(echo $i | cut -d'.' -f 1)

	# Perform sourcefinding on the image
    python $sf_path'sourcefinding.py' catalog $i -o fits:gaul fits:srl --plot --survey MALS

    cd $source'_pybdsf'

    # Exclude sources fit by multiple gaussians
    python $sf_path'source_catalogue_crusher.py' --CAT_FILE $source'_bdsfcat.fits' --OUTPUTNAME $source'_bdsfcat.fits' --RESET_FLAG --DO_SELECT_PYBDSF_TYPE -M

	# Cross match catalogs, sources not selected with the catalog crusher are ignored
    python $sf_path'catalog_matching.py' $source'_bdsfcat.fits' NVSS --flux --astro --output --annotate --annotate_nonmatched
    python $sf_path'catalog_matching.py' $source'_bdsfcat.fits' FIRST --flux --astro --output --annotate --annotate_nonmatched
    python $sf_path'catalog_matching.py' $source'_bdsfcat.fits' SUMSS --flux --astro --output --annotate --annotate_nonmatched

	# Catalog analysis will measure number counts and resolved fraction
    python $sf_path'catalog_analysis.py' $source'_bdsfcat.fits' -r $source'_rms.fits'
    catalogs+=($source'_pybdsf/'$source'_bdsfcat.fits')

    cd ../
done

# If multiple images are being catalogued, they can be combined into a single stacked catalog
python $sf_path'combine_catalogs.py' ${catalogs[@]} 'MALS_combined_catalogs.fits'
python $sf_path'catalog_analysis.py' 'MALS_combined_catalogs.fits' --stacked_catalog
