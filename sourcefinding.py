#!/usr/bin/env python

import os
import sys
import json
import ast

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from pathlib import Path

from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord

from regions import EllipseSkyRegion, write_crtf, write_ds9

import bdsf

def run_bdsf(image, argfile):
    '''
    Run PyBDSF on an image

    Keyword arguments:
    image -- Name of image
    argfile -- Input json file containing arguments
               for bdsf functions
    '''
    imname = os.path.join(os.path.dirname(image),
                          os.path.basename(image).split('.')[0]+'_sf_output',
                          os.path.basename(image).split('.')[0])
    outcatalog = imname+'_bdsfcat.fits'

    path = Path(__file__).parent / argfile
    with open(path) as f:
        args_dict = json.load(f)

    img = bdsf.process_image(image,
                            rms_box=(150,15),
                            rms_box_bright=(50,15),
                            **args_dict['process_image'])

    for img_type in args_dict['export_image']:
        if args_dict['export_image'][img_type]:
            img.export_image(outfile=imname+'_'+img_type+'.fits', clobber=True, img_type=img_type)

    img.write_catalog(outfile = outcatalog, **args_dict['write_catalog'])

    return outcatalog

def read_alpha(imname, catalog, regions):
    '''
    Determine spectral indices of the sources
    '''
    path = Path(__file__).parent

    os.system(f'casa --nologfile -c {os.path.join(path,"smooth_alpha.py")} {imname}')
    dirname = os.path.dirname(imname)

    alpha = fits.open(os.path.join(dirname,'smooth_alpha.fits'))
    alpha_err = fits.open(os.path.join(dirname,'smooth_alpha_error.fits'))

    wcs = WCS(alpha[0].header)

    pixel_regions = [region.to_pixel(wcs) for region in regions]
    image = alpha[0].data[0,0,:,:]
    err_image = alpha_err[0].data[0,0,:,:]

    alpha_list = []
    alpha_err_list = []
    for i, source in enumerate(catalog):
        pixel_region = pixel_regions[i]

        mask = pixel_region.to_mask(mode='center')
        mask_data = mask.to_image(image.shape).astype(bool)

        alpha_values = image[mask_data]
        alpha_err_values = err_image[mask_data]

        alpha_values = alpha_values[~np.isnan(alpha_values)]
        alpha_err_values = alpha_err_values[~np.isnan(alpha_err_values)]

        alpha_values = alpha_values[alpha_err_values < 1.0]
        alpha_err_values = alpha_err_values[alpha_err_values < 1.0]

        if len (alpha_values) > 0:
            alpha_tot = np.sum(alpha_values/alpha_err_values)/np.sum(1/alpha_err_values)
            alpha_err_tot = np.mean(alpha_err_values)

            alpha_list.append(alpha_tot)
            alpha_err_list.append(alpha_err_tot)
        else:
            alpha_list.append(np.ma.masked)
            alpha_err_list.append(np.ma.masked)

    catalog['alpha'] = alpha_list
    catalog['alpha_err'] = alpha_err_list

    # Clean up
    os.remove(os.path.join(dirname,'smooth_alpha.fits'))
    os.remove(os.path.join(dirname,'smooth_alpha_error.fits'))

    files = os.listdir('.')
    for f in files:
        if f.endswith('.last') or f.endswith('false'):
            os.remove(f)

    return catalog

def transform_cat(catalog):
    '''
    Add names for sources in the catalog following IAU naming conventions
    '''
    header = dict([x.split(' = ') for x in catalog.meta['comments'][4:]])

    pointing_center = SkyCoord(float(header['OBSRA'])*u.degree,
                               float(header['OBSDEC'])*u.degree)
    pointing_name = ['PT-'+header['OBJECT'].replace("'","")] * len(catalog)

    source_coord = SkyCoord([source['RA'] for source in catalog],
                            [source['DEC'] for source in catalog],
                            unit=(u.deg,u.deg))

    ids = ['MALS J{0}{1}'.format(coord.ra.to_string(unit=u.hourangle,
                                                     sep='',
                                                     precision=0,
                                                     pad=True),
                                 coord.dec.to_string(sep='',
                                                      precision=0,
                                                      alwayssign=True,
                                                      pad=True)) for coord in source_coord]

    dra, ddec = pointing_center.spherical_offsets_to(source_coord)
    
    # Remove unnecessary columns
    catalog.remove_column('Source_id')
    catalog.remove_column('Isl_id')

    # Add columns at appropriate indices
    col_a = Column(pointing_name, 'Pointing_id')
    col_b = Column(ids, name='MALS_id')
    col_c = Column(dra, name='dRA')
    col_d = Column(ddec, name='dDEC')
    catalog.add_columns([col_a, col_b, col_c, col_d],
                         indexes=[0,0,2,4])

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

    print(f'Wrote mask file to {outfile}')
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
    ax.imshow(img/rms_img, origin='lower', cmap='bone', vmin=0, vmax=5)
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
    ds9 = args.ds9
    spectral_index = args.spectral_index

    if mode in 'cataloging':
        bdsf_args = 'parsets/bdsf_args_cat.json'
    elif mode in 'masking':
        bdsf_args = 'parsets/bdsf_args_mask.json'
    else:
        print(f'Invalid mode {mode}, please choose between c(ataloging) or m(asking)')

    output_folder = os.path.join(os.path.dirname(inpimage),
                                 os.path.basename(inpimage).split('.')[0]+'_sf_output')
    imname = os.path.join(output_folder, os.path.basename(inpimage).split('.')[0])
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    outcat = run_bdsf(inpimage, argfile=bdsf_args)
    bdsf_cat = Table.read(outcat)
    bdsf_regions = catalog_to_regions(bdsf_cat)

    if plot:
        plot_sf_results(f'{imname}_ch0.fits', f'{imname}_rms.fits', bdsf_regions, plot)

    if spectral_index:
        bdsf_cat = read_alpha(imname, bdsf_cat, bdsf_regions)

    if ds9:
        if ds9 is True:
            pass
        else:
            bdsf_regions = [bdsf_regions[i] for i in np.argpartition(-bdsf_cat['Peak_flux'], int(ds9))[:5]]
        outfile = imname+'.reg'
        print(f'Wrote ds9 region file to {outfile}')
        write_ds9(bdsf_regions, outfile)

    # Determine output by mode
    if mode in 'cataloging':
        outfile = imname+'_catalog.fits'
        bdsf_cat = transform_cat(bdsf_cat)
        print(f'Wrote catalog to {outfile}')
        bdsf_cat.write(outfile, overwrite=True)

    if mode in 'masking':
        bdsf_cat.write(outcat, overwrite=True)
        write_mask(outfile=imname+'_mask.crtf', regions=bdsf_regions, size=size)

    # Make sure the log file is in the output folder
    logname = os.path.join(os.path.dirname(inpimage),
                           os.path.basename(inpimage)+'.pybdsf.log')
    os.system(f'mv {logname} {output_folder}')

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
    parser.add_argument("--ds9", nargs="?", const=True,
                        help="""Write the sources found to a ds9 region file,
                                optionally give a number n, only the n brightest
                                sources will be included in the file
                                (default = do not create a region file).""")
    parser.add_argument("--plot", nargs="?", const=True,
                        help="""Plot the results of the sourcefinding as a png
                                of the image with sources overlaid, optionally
                                provide an output filename (default = do
                                not plot the results).""")
    parser.add_argument("--spectral_index", action='store_true',
                        help="""Measure the spectral indices in the .alpha and
                                alpha.error images. (assuming they are in the
                                same directory as the input image) and include
                                them in the catalog. This requires CASA
                                functionalities (default = do not measure
                                spectral indices).""")
    return parser

if __name__ == '__main__':
    main()
