# Image processing

The purpose of this module is to process a radio astronomical image in sourcefinding.py, where you can choose to either output a catalog of sources or output a mask file compatible with CASA for further data reduction. If the choice is to catalog, catalog_matching.py can match the output catalog (or any catalog created by PyBDSF) to external catalogs such as NVSS, SUMSS or FIRST, or even a user specified file. This allows one to check systematics such as the flux ratio (to check the primary beam) or the astrometric offsets.

This code makes extensive use `astropy` and uses PyBDSF and its dependencies for its sourcefinding, which can be found [here](https://github.com/lofar-astron/PyBDSF). Furthermore the [`regions`](https://github.com/astropy/regions) astropy package is utilised in order to create the CASA mask files.

PyBDSF requires an installation of `python-casacore`, and for it directly work on CASA images measures data is required to be somewhere on your system. As specified on the [`casacore`](https://github.com/casacore/casacore) github:

Various parts of casacore require measures data, which requires regular
updating. You can obtain the WSRT measures archive from the ASTRON FTP server:

ftp://ftp.astron.nl/outgoing/Measures/

Extract this somewhere on a permanent location on your filesystem.

## sourcefinding.py

Choose between outputting a catalog of sources or a mask file of Gaussians. Input parameters for PyBDSF are located in `bdsf_args_cat.json` and `bdsf_args_mask.json` in the `parsets` folder for cataloging and masking respectively.

```
usage: sourcefinding.py [-h] [-s SIZE] [--ds9 [DS9]] [--plot [PLOT]]
                        [--spectral_index]
                        mode image

positional arguments:
  mode                  Purpose of the sourcefinding, choose between
                        cataloging (c) or masking (m). This choice will
                        determine the parameter file that PyBDSF will use, as
                        well as the output files.
  image                 Name of the image to perform sourcefinding on.

optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  If masking, multiply the size of the masks by this
                        amount (default = 1.0).
  --ds9 [DS9]           Write the sources found to a ds9 region file,
                        optionally give a number n, only the n brightest
                        sources will be included in the file (default = do not
                        create a region file).
  --plot [PLOT]         Plot the results of the sourcefinding as a png of the
                        image with sources overlaid, optionally provide an
                        output filename (default = do not plot the results).
  --spectral_index      Measure the spectral indices in the .alpha and
                        alpha.error images. (assuming they are in the same
                        directory as the input image) and include them in the
                        catalog. This requires CASA functionalities (default =
                        do not measure spectral indices).
```

## catalog_matching.py

Match a PyBDSF catalog to an external catalog. Choices are between NVSS, SUMSS and FIRST, or an specified catalog file (mileage may vary for this option). Different types of plots can be made to judge the systematics in the catalog.

```
usage: catalog_matching.py [-h] [-d DPI] [--astro [ASTRO]] [--flux [FLUX]]
                           [--fluxtype FLUXTYPE] [--alpha ALPHA]
                           [--output [OUTPUT]]
                           pointing ext_cat

positional arguments:
  pointing             MeerKAT pointing catalog made by PyBDSF.
  ext_cat              External catalog to match to, choice between NVSS,
                       SUMMS, FIRST or a file. If the external catalog is a
                       PyBDSF catalog, make sure the filename has 'bdsfcat' in
                       it. If a different catalog, the parsets/extcat.json
                       file must be used to specify its details (default
                       NVSS).

optional arguments:
  -h, --help           show this help message and exit
  -d DPI, --dpi DPI    DPI of the output images (default = 300).
  --astro [ASTRO]      Plot the astrometric offset of the matches, optionally
                       provide an output filename (default = don't plot
                       astrometric offsets).
  --flux [FLUX]        Plot the flux ratios of the matches, optionally provide
                       an output filename (default = don't plot flux ratio).
  --fluxtype FLUXTYPE  Whether to use Total or Peak flux for determining the
                       flux ratio (default = Total).
  --alpha ALPHA        The spectral slope to assume for calculating the flux
                       ratio, where Flux_1 = Flux_2 * (freq_1/freq_2)^-alpha
                       (default = 0.7)
  --output [OUTPUT]    Output the result of the matching into a catalog,
                       optionally provide an output filename (default = don't
                       output a catalog).
```
