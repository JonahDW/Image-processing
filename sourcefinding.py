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
from astropy.io import fits, ascii
from astropy import units as u
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle

from regions import EllipseSkyRegion, Regions

import bdsf
import helpers

def run_bdsf(image, output_dir, argfile, output_format, reuse_rmsmean=False):
    '''
    Run PyBDSF on an image

    Keyword arguments:
    image -- Name of image
    argfile -- Input json file containing arguments
               for bdsf functions
    '''
    imname = os.path.basename(image).rsplit('.',1)[0]
    impath = os.path.join(output_dir,imname)

    path = Path(__file__).parent / argfile
    with open(path) as f:
        args_dict = json.load(f)

    # Make sure tuples are correctly parsed
    if 'rms_box' in args_dict['process_image']:
        if args_dict['process_image']['rms_box'] is not None:
            args_dict['process_image']['rms_box'] = ast.literal_eval(args_dict['process_image']['rms_box'])
    if 'rms_box_bright' in args_dict['process_image']:
        if args_dict['process_image']['rms_box_bright'] is not None:
            args_dict['process_image']['rms_box_bright'] = ast.literal_eval(args_dict['process_image']['rms_box_bright'])

    if reuse_rmsmean:
        img = bdsf.process_image(image, **args_dict['process_image'],
                                 rmsmean_map_filename=[imname+'_mean.fits',
                                                       imname+'_rms.fits'])
    else:
        img = bdsf.process_image(image, **args_dict['process_image'])

    for img_type in args_dict['export_image']:
        if args_dict['export_image'][img_type]:
            img.export_image(outfile=impath+'_'+img_type+'.fits', 
                             clobber=True, img_type=img_type)

    out_srl_cat = None
    out_gaus_cat = None
    for of in output_format:
        fmt = of.lower().split(':')
        if len(fmt) == 1:
            fmt = fmt[0]
            cat_type = 'srl'
        if len(fmt) == 2:
            fmt, cat_type = fmt

        if fmt == 'ds9':
            outcatalog = impath+'_'+cat_type+'_bdsfcat.ds9.reg'
            img.write_catalog(outfile=outcatalog,
                              format=fmt,
                              catalog_type=cat_type,
                              clobber=True)
        elif fmt == 'kvis':
            outcatalog = impath+'_bdsfcat.kvis.ann'
            img.write_catalog(outfile=outcatalog,
                              format=fmt,
                              catalog_type='gaul',
                              clobber=True)
        elif fmt == 'star':
            outcatalog = impath+'_bdsfcat.star'
            img.write_catalog(outfile=outcatalog,
                              format=fmt,
                              catalog_type='gaul',
                              clobber=True)
        else:
            outcatalog = impath+'_'+cat_type+'_bdsfcat.'+fmt
            img.write_catalog(outfile=outcatalog,
                              format=fmt,
                              catalog_type=cat_type,
                              clobber=True)
            if fmt == 'fits' and cat_type == 'srl':
                out_srl_cat = outcatalog
            if fmt == 'fits' and cat_type == 'gaul':
                out_gaus_cat = outcatalog

    return out_srl_cat, out_gaus_cat, img

def fake_run_bdsf(image):
    '''
    Fake run PyBDSF on an image, only getting image parameters

    Keyword arguments:
    image -- Name of image
    argfile -- Input json file containing arguments
               for bdsf functions
    '''
    img = bdsf.process_image(image, advanced_opts=True,
                             stop_at='read')

    return img

def read_alpha(inpimage, alpha_image, catalog, regions):
    '''
    Determine spectral indices of the sources
    '''
    weight = helpers.open_fits_casa(inpimage)
    alpha = helpers.open_fits_casa(alpha_image)

    # Remove degenerate axes if any
    alpha_image = np.squeeze(alpha[0].data)
    weight_image = np.squeeze(weight[0].data)

    # Get WCS from header and drop freq and stoke axes
    alpha_wcs = WCS(alpha[0].header, naxis=2)
    alpha_regions = [region.to_pixel(alpha_wcs) for region in regions]

    weight_wcs = WCS(weight[0].header, naxis=2)
    weight_regions = [region.to_pixel(weight_wcs) for region in regions]

    alpha_list, alpha_err_list = helpers.measure_image_regions(alpha_regions, alpha_image,
                                                               weight_image=weight_image,
                                                               weight_regions=weight_regions)

    a = Column(alpha_list, name='Spectral_index')
    b = Column(alpha_err_list, name='E_Spectral_index')
    catalog.add_columns([a,b], indexes=[10,10]) 

    return catalog

def transform_cat(catalog, img, max_separation, flag_artefacts, pointing_name, survey_name):
    '''
    Add names for sources in the catalog following IAU naming conventions
    '''
    header = {}
    for i, x in enumerate(img.header):
        if x != 'HISTORY':
            header[x]=img.header[i]

    pointing_center = SkyCoord(float(header['CRVAL1'])*u.degree,
                               float(header['CRVAL2'])*u.degree)

    source_coord = SkyCoord([source['RA'] for source in catalog],
                            [source['DEC'] for source in catalog],
                            unit=(u.deg,u.deg))

    if survey_name:
        survey_name = survey_name.ljust(len(survey_name)+1)
    else:
        survey_name = ''

    ids = [survey_name+'J{0}{1}'.format(coord.ra.to_string(unit=u.hourangle,
                                                           sep='',
                                                           precision=2,
                                                           pad=True),
                                        coord.dec.to_string(sep='',
                                                            precision=1,
                                                            alwayssign=True,
                                                            pad=True)) for coord in source_coord]

    sep = pointing_center.separation(source_coord)
    quality_flag = [1] * len(catalog)

    # Add columns at appropriate indices
    col_a = Column(ids, name='Source_name')
    col_b = Column(sep, name='Sep_PC')
    col_c = Column(quality_flag, name='Quality_flag')
    catalog.add_columns([col_a, col_b, col_c],
                         indexes=[0,6,-1])

    # Add identifier
    if 'OBJECT' in header and pointing_name is None:
        pointing_name = [header['OBJECT'].replace("'","")] * len(catalog)
    if pointing_name:
        catalog.add_column(pointing_name, name='Pointing_id', index=0)

    if flag_artefacts:
        flag_close = flag_artifacts(catalog, img)
        catalog.add_column(flag_close, name='Flag_Artifact')
        # Set quality flag of artefacts
        catalog['Quality_flag'][catalog['Flag_Artifact']] = 0

    # Remove sources beyond maximum separation
    if max_separation is not None:
        catalog = catalog[catalog['Sep_PC'] < max_separation]

    # Update catalog meta
    catalog.meta['comments'] = catalog.meta['comments'][:2]
    catalog.meta.update(header)

    # Change NAXIS keywords so that astropy doesn't complain
    for key in ['NAXIS','NAXIS1','NAXIS2','NAXIS3','NAXIS4']:
        replacement = {key:key.replace('N','')}
        for k, v in list(catalog.meta.items()):
            catalog.meta[replacement.get(k, k)] = catalog.meta.pop(k)

    # Put beam and freq in header in case they're not already there
    catalog.meta['SF_BMAJ'] = img.beam[0]
    catalog.meta['SF_BMIN'] = img.beam[1]
    catalog.meta['SF_BPA'] = img.beam[2]
    catalog.meta['SF_TELE'] = img._telescope

    return catalog

def flag_artifacts(catalog, img):
    '''
    Identify and flag artifacts
    '''
    bright_idx = catalog['Peak_flux']*catalog['Sep_PC']**2/catalog['Isl_rms'] > 100
    idx = helpers.id_artifacts(catalog[bright_idx], catalog, img.beam[0])

    flag_close = np.zeros(len(catalog), dtype=bool)
    flag_close[idx] = True

    print(f'Identified {len(idx)} possible artifacts in the image')

    return flag_close

def catalog_to_regions(catalog, ra='RA', dec='DEC', majax='Maj', minax='Min', PA='PA'):
    '''
    Convert catalog to a list of regions

    Keyword arguments:
    catalog -- Input catalog
    ra, dec, majax, minax, PA -- Column names of containing required variables
    '''
    regions = Regions([
        EllipseSkyRegion(center=SkyCoord(source[ra], source[dec], unit='deg'),
                         height=source[majax]*u.deg, width=source[minax]*u.deg,
                         angle=source[PA]*u.deg) for source in catalog])
    return regions

def write_fits_mask(imname, image_file, thresh_isl, rms_image=None):
    """
    Write an output fits containing binary mask

    Keyword arguments:
    outfile -- Name of the output mask file (CRTF)
    regions -- Region or list of regions to write
    size -- Multiply input major and minor axes by this amount
    """
    image = helpers.open_fits_casa(image_file)
    if rms_image is None:
        rms_image = imname+'_rms.fits'
    rms = helpers.open_fits_casa(rms_image)

    snr = np.squeeze(image[0].data)/np.squeeze(rms[0].data)
    mask = np.zeros(snr.shape)
    mask[snr > thresh_isl] = 1

    # Store in existing HDU and write to fits
    rms[0].data[0,0,:,:] = mask

    outfile = imname+'_mask.fits'
    print(f'Wrote mask file to {outfile}')
    rms.writeto(outfile, overwrite=True)

def write_crtf_mask(imname, regions, size=1.0):
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

    outfile = imname+'_mask.crtf'
    print(f'Wrote mask file to {outfile}')
    regions.write(outfile, format='crtf')

def plot_sf_results(image_file, imname, regions, max_sep, plot, plot_isl, thresh_isl, rms_image=None, flag_regions=None):
    '''
    Plot the results of the sourcefinding
    '''
    image = helpers.open_fits_casa(image_file)
    img = np.squeeze(image[0].data)
    vmax = np.percentile(img, 95)

    if rms_image is None:
        rms_image = imname+'_rms.fits'

    # If rms image exists, divide and set max to 5
    if os.path.exists(rms_image):
        rms = helpers.open_fits_casa(rms_image)
        rms_img = np.squeeze(rms[0].data)
        img /= rms_img
        vmax = 5

    wcs = WCS(image[0].header, naxis=2)

    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot(projection=wcs)
    ax.imshow(img, origin='lower', cmap='bone', vmin=0, vmax=vmax)
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')

    for region in regions:
        patch = region.to_pixel(wcs).as_artist(facecolor='none', edgecolor='m', lw=0.25)
        ax.add_patch(patch)

    if plot_isl:
        ax.contour(img, origin='lower', colors='royalblue', levels=[thresh_isl], linewidths=0.25)

    if flag_regions is not None:
        for region in flag_regions:
            patch = region.to_pixel(wcs).as_artist(facecolor='none', edgecolor='g', lw=0.25)
            ax.add_patch(patch)

    if max_sep is not None:
        center = (image[0].header['CRVAL1'] * u.deg, image[0].header['CRVAL2'] * u.deg)
        s = SphericalCircle(center, max_sep * u.deg,
                            edgecolor='white', facecolor='none', lw=1,
                            transform=ax.get_transform('fk5'))
        ax.add_patch(s)

    if plot is True:
        plt.savefig(imname+'.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(plot, dpi=300, bbox_inches='tight')
    plt.close()

def main():

    parser = new_argument_parser()
    args = parser.parse_args()

    inpimage = args.image

    # Output options
    outdir = args.outdir
    output_format = args.output_format
    mask = args.mask

    # Additional input options
    rms_image = args.rms_image
    spectral_index = args.spectral_index
    reuse_rmsmean = args.reuse_rmsmean
    redo_catalog = args.redo_catalog
    parfile = args.parfile

    # Catalog and mask options
    max_separation = args.max_separation
    flag_artefacts = args.flag_artefacts
    size = args.size
    survey = args.survey
    pointing = args.pointing

    # Plotting options
    plot = args.plot
    plot_isl = args.plot_isl

    if parfile:
        bdsf_args = f'parsets/{parfile}.json'
    elif mask:
        bdsf_args = 'parsets/bdsf_args_mask.json'
    else:
        bdsf_args = 'parsets/bdsf_args_cat.json'

    if outdir is None:
        output_dir = os.path.join(os.path.dirname(inpimage),
                                  os.path.basename(inpimage).rsplit('.',1)[0]+'_pybdsf')
    else:
        output_dir = outdir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    imname = os.path.join(output_dir, os.path.basename(inpimage).rsplit('.',1)[0])

    # Standard output is source list, Gaussian list for masking
    if output_format is None:
        if mask:
            output_format = ['fits:gaus']
        else:
            output_format = ['fits:srl']

    if redo_catalog:
        print(f'Using previously generated catalog and skipping sourcefinding')
        img = fake_run_bdsf(inpimage)
        if mask:
            out_srl_cat = None
            out_gaus_cat = redo_catalog
        else:
            out_srl_cat = redo_catalog
            out_gaus_cat = None
    else:
        out_srl_cat, out_gaus_cat, img = run_bdsf(inpimage, output_dir,
                                                  argfile=bdsf_args,
                                                  output_format=output_format,
                                                  reuse_rmsmean=reuse_rmsmean)

    if not out_srl_cat and not out_gaus_cat:
        print('No FITS catalog generated, no further operations are performed')
        sys.exit()

    if mask:
        # Just output mask files and be done
        bdsf_gaus_cat = helpers.open_catalog(out_gaus_cat)
        if mask.lower() == 'crtf':
            bdsf_regions = catalog_to_regions(bdsf_gaus_cat)
            write_crtf_mask(imname, bdsf_regions, size)
        elif mask.lower() == 'fits':
            write_fits_mask(imname, inpimage, thresh_isl, rms_image)
        else:
            bdsf_regions = catalog_to_regions(bdsf_gaus_cat)
            write_crtf_mask(imname, bdsf_regions, size)
            write_fits_mask(imname, inpimage, thresh_isl, rms_image)
        sys.exit()

    # Do catalogs
    for outcat in [out_gaus_cat,out_srl_cat]:
        if outcat:
            bdsf_cat = helpers.open_catalog(outcat)

            bdsf_cat = transform_cat(bdsf_cat, img, max_separation, 
                                     flag_artefacts, pointing, survey)
            bdsf_regions = catalog_to_regions(bdsf_cat)
            if spectral_index:
                bdsf_cat = read_alpha(inpimage, spectral_index, bdsf_cat, bdsf_regions)

            print('Wrote updated catalog to '+outcat)
            bdsf_cat.write(outcat, overwrite=True)

    # Get the island threshold for plotting
    path = Path(__file__).parent / bdsf_args
    with open(path) as f:
        args_dict = json.load(f)
    thresh_isl = args_dict['process_image']['thresh_isl']

    if plot:
        if flag_artefacts:
            flag_regions = catalog_to_regions(bdsf_cat[bdsf_cat['Flag_Artifact'] == True])
            plot_sf_results(inpimage, imname, bdsf_regions, max_separation, 
                            plot, plot_isl, thresh_isl, rms_image, flag_regions)
        else:
            plot_sf_results(inpimage, imname, bdsf_regions, max_separation, 
                            plot, plot_isl, thresh_isl, rms_image)

    # Make sure the log file is in the output folder
    logname = inpimage+'.pybdsf.log'
    os.system(f'mv {logname} {output_dir}')

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("image", type=str,
                        help="""Name of the image to perform sourcefinding on.""")
    parser.add_argument("-o", "--output_format", nargs='+', default=None,
                        help="""Output format of the catalog, supported formats
                                are: ds9, fits, star, kvis, ascii, csv. In case of fits,
                                ascii, ds9, and csv, additionally choose output catalog as either
                                source list (srl) or gaussian list (gaul), default srl. Currently, only
                                fits and csv formats source list can be used for further processing. 
                                Input can be multiple entries, e.g. -o fits:srl ds9 (default = fits:srl).""")
    parser.add_argument("--mask", nargs='?', const=True,
                        help="""If specified, use mask parameter file 'bdsf_args_mask', and writes out
                                mask files. Choices are at the moment between 'crtf' or 'fits'.""")
    parser.add_argument("--outdir", default=None, type=str, 
                        help="Name of directory to place output, default is the image directory.")
    parser.add_argument("--size", default=1.0, type=float,
                        help="""If masking, multiply the size of the masks by this
                                amount (default = 1.0).""")
    parser.add_argument("--plot", nargs="?", const=True,
                        help="""Plot the results of the sourcefinding as a png
                                of the image with sources overlaid, optionally
                                provide an output filename (default = do
                                not plot the results).""")
    parser.add_argument("--plot_isl", action='store_true',
                        help="""Plot island boundaries along with source ellipses.""")
    parser.add_argument("--spectral_index", nargs="?", const=True,
                        help="""Measure the spectral indices of the sources using a specified
                                spectral index image. Can be FITS or CASA format.
                                (default = do not measure spectral indices).""")
    parser.add_argument("--max_separation", default=None, type=float,
                        help="""Only include sources in the final catalogue within a specified
                                distance (in degrees) from the image centre. (default = include all)""")
    parser.add_argument("--flag_artefacts", action='store_true', 
                        help="""Add column for flagging artefacts around 
                                bright sources (default = do not flag)""")
    parser.add_argument("--rms_image", default=None, 
                        help="""Specify RMS alternative image to use for plotting 
                                (default = use RMS image from sourcefinding)""")
    parser.add_argument("--reuse_rmsmean", action='store_true',
                        help="""Use already present rms and mean images for sourcefinding.""")
    parser.add_argument("--parfile", default=None, type=str,
                        help="""Alternative PyBDSF parameter file, without .json extension.""")
    parser.add_argument("--survey", default=None, help="Name of the survey to be used in source ids.")
    parser.add_argument("--pointing", default=None, type=str,
                        help="""Name of the pointing to be used in the pointing id. 
                                If not specified this is taken from the 'OBJECT' header flag.""")
    parser.add_argument("--redo_catalog", default=None,
                        help="""Specify catalog file if you want some part of the process
                                to be redone, but want to skip sourcefinding""")
    return parser

if __name__ == '__main__':
    main()
