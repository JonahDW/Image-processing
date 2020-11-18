import os
import sys
import json
import ast

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord

from regions import EllipseSkyRegion, read_crtf, write_crtf

import bdsf

def run_bdsf(image, argfile):
    '''
    Run PyBDSF on an image

    Keyword arguments:
    image -- Name of image
    catalog_type -- How sources are combined in catalog, i.e.
                    'gaul' for a list of Gaussians (for masking)
                    'srl' for a list of sources (for cataloging)
    output_imgs -- Specify additional output images produced by
                   bdsf. Options are: 'gaus_resid',
                  'shap_resid', 'rms', 'mean', 'gaus_model',
                  'shap_model', 'ch0', 'pi', 'psf_major', 'psf_minor',
                  'psf_pa', 'psf_ratio', 'psf_ratio_aper', 'island_mask'
    kwargs -- Extra keyword arguments for pybdsf source finding
    '''
    inp_image = os.path.splitext(image)[0]
    outcatalog = inp_image+'_bdsfcat.fits'

    with open(argfile) as f:
        args_dict = json.load(f)

    img = bdsf.process_image(image, **args_dict['process_image'])

    for img_type in args_dict['export_image']:
        if args_dict['export_image'][img_type]:
            img.export_image(outfile=inp_image+'_'+img_type+'.fits', clobber=True, img_type=img_type)

    img.write_catalog(outfile = outcatalog, **args_dict['write_catalog'])

    return outcatalog

def transform_cat(catalog):
    '''
    Add names for sources in the catalog
    '''
    coordinates = SkyCoord([source['RA'] for source in catalog],
                           [source['DEC'] for source in catalog],
                           unit=(u.deg,u.deg))

    ids = ['MALS J{0}{1}'.format(coord.ra.to_string(unit=u.hourangle,
                                                     sep='',
                                                     precision=0,
                                                     pad=True),
                                  coord.dec.to_string(sep='',
                                                      precision=0,
                                                      alwayssign=True,
                                                      pad=True)) for coord in coordinates]

    c = Column(ids, name='MALS_id')
    catalog.add_column(c, index=0)

    catalog.remove_column('Source_id')
    catalog.remove_column('Isl_id')

    return catalog

def catalog_to_regions(catalog, ra='RA', dec='DEC', majax='Maj', minax='Min', PA='PA'):
    '''
    Convert catalog to a list of regions

    Keyword arguments:
    catalog -- Input catalog
    ra, dec, majax, minax, PA -- Column names of containing required variables
    '''
    regions = [
        EllipseSkyRegion(center=SkyCoord(source[ra], source[dec], unit='deg'),
                         height=2*source[majax]*u.deg, width=2*source[minax]*u.deg,
                         angle=source[PA]*u.deg) for source in catalog]
    return regions

def write_mask(outfile, regions, size=1.0):
    """
    Write an output file containing sources to mask

    Keyword arguments:
    outfile -- Name of the output mask file (CRTF)
    regions -- Region or list of regions to write
    size -- Multiply input major and minor axes by this amount
    """
    if size != 1.0:
        for region in regions:
            region.height *= size
            region.width *= size

    print(f'Wrote mask file {outfile}')
    write_crtf(regions, outfile)

def plot_sf_results(image_file, rms_image, regions, plot):
    '''
    Plot the results of the sourcefinding
    '''
    image = fits.open(image_file)
    rms = fits.open(rms_image)

    img = image[0].data[0,0,:,:]
    rms_img = rms[0].data[0,0,:,:]
    wcs = WCS(image[0].header, naxis=2)

    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(projection=wcs)
    ax.imshow(img/rms_img, origin='lower', cmap='gist_gray', vmin=0, vmax=5)
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')

    for region in regions:
        patch = region.to_pixel(wcs).as_artist(facecolor='none', edgecolor='m', lw=0.25)
        ax.add_patch(patch)

    if plot is True:
        plt.savefig(os.path.splitext(image_file)[0]+'.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(plot, dpi=300, bbox_inches='tight')
    plt.close()

def main():

    parser = new_argument_parser()
    args = parser.parse_args()

    inpimage = args.image
    mode = args.mode
    size = args.size
    plot = args.plot

    if mode in 'cataloging':
        outcat = run_bdsf(inpimage, argfile='parsets/bdsf_args_cat.json')
        bdsf_cat = Table.read(outcat)
        bdsf_cat = transform_cat(bdsf_cat)
        bdsf_regions = catalog_to_regions(bdsf_cat)

        outfile = os.path.splitext(inpimage)[0]+'_catalog.fits'
        print(f'Wrote catalog to {outfile}')
        bdsf_cat.write(outfile, overwrite=True)

    elif mode in 'masking':
        outcat = run_bdsf(inpimage, argfile='parsets/bdsf_args_mask.json')
        bdsf_cat = Table.read(outcat)
        bdsf_regions = catalog_to_regions(bdsf_cat)
        write_mask(outfile = inpimage+'_mask.crtf', regions=bdsf_regions, size=size)
    else:
        print(f'Invalid mode {mode}, please choose between c(ataloging) or m(atching)')

    if plot:
        inpimage = os.path.splitext(inpimage)[0]
        plot_sf_results(f'{inpimage}_ch0.fits', f'{inpimage}_rms.fits', bdsf_regions, plot)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("mode",
                        help="""Purpose of the sourcefinding, choose between
                                cataloging (c) or masking (m). This choice will determine
                                the parameter file that PyBDSF will use, as well as the
                                output files.""")
    parser.add_argument("image",
                        help="""Name of the image to perform sourcefinding on.""")
    parser.add_argument("-s", "--size", default=1.0,
                        help="""If masking, multiply the size of the masks by this
                                amount (default = 1.0).""")
    parser.add_argument("--plot", nargs="?", const=True,
                        help="""Plot the results of the sourcefinding as a png
                                of the image with sources overlaid, optionally
                                provide an output filename (default = do
                                not plot the results).""")
    return parser

if __name__ == '__main__':
    main()