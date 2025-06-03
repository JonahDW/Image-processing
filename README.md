# Image processing

The purpose of this module is to allow easy source extraction, catalog creation, cross matching and further analysis of a radio astronomical image. This code makes extensive use `astropy` and uses [PyBDSF](https://github.com/lofar-astron/PyBDSF) and its dependencies for its sourcefinding. PyBDSF requires an installation of `python-casacore`, and for it directly work on CASA images measures data is required to be somewhere on your system. As specified on the [`casacore`](https://github.com/casacore/casacore) github:

> Various parts of casacore require measures data, which requires regular updating. You can obtain the WSRT measures archive from the ASTRON FTP server: ftp://ftp.astron.nl/outgoing/Measures/. 
> Extract this somewhere on a permanent location on your filesystem.

Furthermore the [`regions`](https://github.com/astropy/regions) astropy package is utilised in order if creating CASA mask files. The following scripts have a lot of flexibility built in, for a simple recipe of how a full run of this software would look on an image, example bash scripts are present in the `example_scripts` folder.

## Docker/singularity image

If you want to avoid doing all the installations or don't have the permissions to do so, there is an accompanying docker image for this module in my [`sourcefinding-docker`](https://github.com/JonahDW/sourcefinding-docker) repository. Details on how to obtain the image are found there.

## sourcefinding.py

Perform source extraction on an image, outputting a catalog of sources or Gaussian components. Optionally, a mask file of Gaussian components cam be created. Standard input parameters for PyBDSF are located in `bdsf_args_cat.json` in the `parsets` folder. Alternative PyBDSF parameter files can be specified with the `--parfile` option. For example

```python sourcefinding.py <my_image> -o fits:srl ds9 --plot```

Will perform sourcefinding on `<my_image>` and produce both a fits source catalog and DS9 region file. A plot will be produced showing the image and the sources as ellipses overlaid. 

```
usage: sourcefinding.py [-h] [-o OUTPUT_FORMAT [OUTPUT_FORMAT ...]]
                        [--mask [MASK]] [--outdir OUTDIR] [--size SIZE]
                        [--plot [PLOT]] [--plot_isl]
                        [--spectral_index [SPECTRAL_INDEX]]
                        [--max_separation MAX_SEPARATION] [--flag_artefacts]
                        [--rms_image RMS_IMAGE] [--reuse_rmsmean]
                        [--parfile PARFILE] [--survey SURVEY]
                        [--pointing POINTING] [--redo_catalog REDO_CATALOG]
                        image

positional arguments:
  image                 Name of the image to perform sourcefinding on.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FORMAT [OUTPUT_FORMAT ...], --output_format OUTPUT_FORMAT [OUTPUT_FORMAT ...]
                        Output format of the catalog, supported formats are:
                        ds9, fits, star, kvis, ascii, csv. In case of fits,
                        ascii, ds9, and csv, additionally choose output
                        catalog as either source list (srl) or gaussian list
                        (gaul), default srl. Currently, only fits and csv
                        formats source list can be used for further
                        processing. Input can be multiple entries, e.g. -o
                        fits:srl ds9 (default = fits:srl).
  --mask [MASK]         If specified, use mask parameter file
                        'bdsf_args_mask', and writes out mask files. Choices
                        are at the moment between 'crtf' or 'fits'.
  --outdir OUTDIR       Name of directory to place output, default is the
                        image directory.
  --size SIZE           If masking, multiply the size of the masks by this
                        amount (default = 1.0).
  --plot [PLOT]         Plot the results of the sourcefinding as a png of the
                        image with sources overlaid, optionally provide an
                        output filename (default = do not plot the results).
  --plot_isl            Plot island boundaries along with source ellipses.
  --spectral_index [SPECTRAL_INDEX]
                        Measure the spectral indices of the sources using a
                        specified spectral index image. Can be FITS or CASA
                        format. (default = do not measure spectral indices).
  --max_separation MAX_SEPARATION
                        Only include sources in the final catalogue within a
                        specified distance (in degrees) from the image centre.
                        (default = include all)
  --flag_artefacts      Add column for flagging artefacts around bright
                        sources (default = do not flag)
  --rms_image RMS_IMAGE
                        Specify RMS alternative image to use for plotting
                        (default = use RMS image from sourcefinding)
  --reuse_rmsmean       Use already present rms and mean images for
                        sourcefinding.
  --parfile PARFILE     Alternative PyBDSF parameter file, without .json
                        extension.
  --survey SURVEY       Name of the survey to be used in source ids.
  --pointing POINTING   Name of the pointing to be used in the pointing id. If
                        not specified this is taken from the 'OBJECT' header
                        flag.
  --redo_catalog REDO_CATALOG
                        Specify catalog file if you want some part of the
                        process to be redone, but want to skip sourcefinding

```

## catalog_matching.py

Match a catalog produced with `sourcefinding.py` to an external catalog. Currently, available choices are NVSS, SUMSS, FIRST, RACS-low or RACS-mid. Alternatively, a catalog file or VO link can be given. If the name of the other catalog contains `bdsfcat`, the catalog is assumed to be one generated by `sourcefinding.py` as well, and will be automatically handled. If this is not the case, relevant information about the catalog, such as the names of columns and the shape of the beam must be put in the `parsets/extcat.json` file by hand in order for the file to properly handled by the script. Different types of plots can be made to judge the systematics in the catalog. For example

```catalog_matching.py myimage_catalog.fits NVSS --astro --flux```

Will match the catalog `myimage_catalog.fits` to an external catalog, in this case the NVSS. The matched catalog is written out, and additionally plots are produced with the astrometric and flux offsets between the sources in the image. 

```
usage: catalog_matching.py [-h] [--match_sigma_extent MATCH_SIGMA_EXTENT]
                           [--search_dist SEARCH_DIST]
                           [--source_overlap_percent SOURCE_OVERLAP_PERCENT]
                           [--astro [ASTRO]] [--flux [FLUX]] [--plot [PLOT]]
                           [--survey_name SURVEY_NAME] [--fluxtype FLUXTYPE]
                           [--alpha ALPHA] [--output [OUTPUT]]
                           [--annotate [ANNOTATE]] [--annotate_nonmatched]
                           [--ra_center RA_CENTER] [--dec_center DEC_CENTER]
                           [--fov FOV] [-d DPI]
                           input_cat ext_cat

positional arguments:
  input_cat             Catalog made by PyBDSF.
  ext_cat               External catalog to match to. Standard catalogues are
                        NVSS, SUMMS, FIRST, TGSS, RACS-low or RACS-mid
                        (default NVSS). Any other name will be interpreted as
                        a file, or failing that a VO link. If a non-standard
                        catalog is specified, the parsets/extcat.json file
                        must be used to specify its details. Alternatively, if
                        the external catalog is a PyBDSF catalog, it will be
                        parsed as such if the filename contains the substring
                        'bdsfcat'.

optional arguments:
  -h, --help            show this help message and exit
  --match_sigma_extent MATCH_SIGMA_EXTENT
                        The matching extent used for sources, defined in
                        sigma. Any sources within this extent will be
                        considered matches. (default = 3 sigma = 1.27398 times
                        the FWHM)
  --search_dist SEARCH_DIST
                        Additional search distance beyond the source size to
                        be used for matching, in arcseconds (default = 0)
  --source_overlap_percent SOURCE_OVERLAP_PERCENT
                        The percentage is used, of the ratio of size of the
                        intersection area to size of the individual sources,
                        to fine-tune source matches, in percentage (default =
                        80)
  --astro [ASTRO]       Plot the astrometric offset of the matches, optionally
                        provide an output filename (default = don't plot
                        astrometric offsets).
  --flux [FLUX]         Plot the flux ratios of the matches, optionally
                        provide an output filename (default = don't plot flux
                        ratio).
  --plot [PLOT]         Plot the field with the matched ellipses, optionally
                        provide an output filename (default = don't plot the
                        matched ellipses).
  --survey_name SURVEY_NAME
                        Survey name of input catalog to use in matched catalog
                        columns and plots
  --fluxtype FLUXTYPE   Whether to use Total or Peak flux for determining the
                        flux ratio (default = Total).
  --alpha ALPHA         The spectral slope to assume for calculating the flux
                        ratio, where Flux_1 = Flux_2 * (freq_1/freq_2)^alpha
                        (default = -0.8)
  --output [OUTPUT]     Output the result of the matching into a catalog,
                        optionally provide an output filename (default = don't
                        output a catalog).
  --annotate [ANNOTATE]
                        Output the result of the matching into an annotation
                        file, either kvis or ds9 (default = don't output
                        annotation file).
  --annotate_nonmatched
                        Annotation file will include the non-macthed catalogue
                        sources (default = don't show).
  --ra_center RA_CENTER
                        Assumed centre (RA, degrees) for matching to external
                        catalogues. (default = use CRVAL1/2 from image
                        header).
  --dec_center DEC_CENTER
                        Assumed centre (DEC, degrees) for matching to external
                        catalogues. (default = use CRVAL1/2 from image
                        header).
  --fov FOV             Assumed FOV (degrees) for matching to external
                        catalogues. (default = use FOV of input image).
  -d DPI, --dpi DPI     DPI of the output images (default = 300).
```

## source_catalogue_crusher.py

The purpose of this tool is to mark sources in the catalog not to be used. It can be used to clean up the catalog or specify sources to be used in other steps such as catalog matching. It does this through the `Quality_flag` column in the catalog, and this provides an easy way to edit this column in the catalog. 

```
Usage: source_catalogue_crusher.py [options]

Options:
  -h, --help            show this help message and exit
  --CAT_FILE=CATFILE    CAT - file name e.g. CATALOUGE.fits
  --OUTPUTNAME=OUTPUTFILENAME
                        CATALOUGE outputfile name e.g. CATALOUGE.fits
  --RESET_FLAG          Reset the Quality flag column prior to making a
                        selection
  --DO_SELECT_PYBDSF_TYPE=DOSELECTTYPE
                        Select catalouges sources on type. [e.g. M or -M, this
                        would select all except M]
  --DO_SELECT_SOURCE_MAJ_DECON=DOSELECTSOURCES_MAJ_DECON
                        Select  sources on deconvolved major axis = minor axis
                        = zero
  --DO_SELECT_SOURCE_MAJ_FIT=DOSELECTEXTENDEDSOURCES
                        Select sources Major Axis FIT that is larger than the
                        Deconvolved Maj Axis, input value is sigma or -sigma
                        to exclude these source
  --DO_SELECT_SOURCE_TOTFLX_ISLANDFLX=DOSELECTONFLUXDENSITY
                        Select  sources on total flux density matching the
                        total island flux density, input value is sigma or
                        -sigma to exclude these sources
  --DO_SELECT_SOURCE_SAMEINDEX=DOSELECTONSAMEIDX
                        Select sources with the same column index, if [-] in
                        front of column name to exclude these sources
  --DO_SELECT_ON=DOSELECTON
                        Set selection based on table column [e.g. Maj]
  --DO_SELECT_OPERATION=DOSELECTONOPERATION
                        Input operation of selection [e.g. = (default), >, <]
  --DO_SELECT_VALUE=DOSELECTONVALUE
                        Input value of selection
  --INVERT_SELECTION    Invert the selection final selection
  --KVISANNOUTPUT=KVISOUTPUTFILENAME
                        KVIS annotation outputfile name e.g. CATALOUGE.ann
  --KVISCOLOR=KVISCOLOR
                        Change COLOR of KVIS annotation [e.g. RANDOM ]
  --KVISPRINT           use the fitted values Maj,Min,PA or deconvolved values
                        DC_Maj, DC_Min, DC_PA
  --TABLECOLUMNOUTPUT=TCOLUMNOUTFILENAME
                        write table column output as ASCII to be edit by hand
  --TABLECOLUMNINPUT=TCOLUMNINFILENAME
                        read table column ASCII file.
  --DO_PRINT_INFO=DOPRTCATAINFO
                        =BASIC default, =FULL Print statistics of the
                        catalouge
  --DO_PRINT_TAB_COLUMN
                        Print some information of the catalouge
```

## catalog_analysis.py

Analyze a PyBDSF catalog with different metrics regularly applied to radio astronomical data, like source counts and fraction of resolved sources.  For example

```python catalog_analysis.py myimage_catalog.fits -r myimage_rms.fits```

Runs the catalog analysis on `myimage_catalog.fits` and produces plots for the number counts, resolved fraction, and differential number counts. The rms image given is used to correct the differential number counts for area coverage.

```
usage: catalog_analysis.py [-h] [-r RMS_IMAGE] [-c COMP_CORR]
                           [--flux_col FLUX_COL] [--stacked_catalog]
                           [--no_plots] [--fancy] [-d DPI]
                           catalog

positional arguments:
  catalog               Pointing catalog(s) made by PyBDSF.

optional arguments:
  -h, --help            show this help message and exit
  -r RMS_IMAGE, --rms_image RMS_IMAGE
                        Specify input rms image for creating an rms coverage
                        plot. In the absence of a completeness correction
                        file, will also be used to correct for completeness.
  -c COMP_CORR, --comp_corr COMP_CORR
                        Specify input json file containing completeness
                        fractions for correcting differential number counts.
                        the file is assumed to contain at least the arrays of
                        flux bins, completeness fraction.
  --flux_col FLUX_COL   Name of flux column to use for analysis (default =
                        Total_flux).
  --stacked_catalog     Indicate if catalog is built up from multiple
                        catalogs, for example with combine_catalogs script.
  --no_plots            Run the scripts without making any plots.
  --fancy               Output plots with latex font and formatting.
  -d DPI, --dpi DPI     DPI of the output images (default = 300).
```

## combine_catalogs.py

Combine a list of catalogs into one output catalog, options to combine are vstack, hstack, or different join types, such as inner join and outer join.

```
usage: combine_catalogs.py [-h] [-m MODE] [-k KEYS [KEYS ...]]
                           input_cats [input_cats ...] output_cat

positional arguments:
  input_cats            Pointing catalogs made by PyBDSF, to be combined.
  output_cat            Name of the full output catalog

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  How to combine tables. Supports vstack, hstack, or
                        different join types (inner join, outer join, etc.)
                        default = vstack
  -k KEYS [KEYS ...], --keys KEYS [KEYS ...]
                        Key column(s) for join
```
