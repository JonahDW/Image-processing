import os
import json
import pickle
import numpy as np
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import astropy.wcs as WCS

import casacore.images as pim
from matplotlib.patches import Ellipse

def pickle_to_file(data, fname):
    fh = open(fname, 'wb')
    pickle.dump(data, fh)
    fh.close()

def pickle_from_file(fname):
    fh = open(fname, 'rb')
    data = pickle.load(fh)
    fh.close()
    return data

def open_fits_casa(file):
    '''
    Open an image in fits or CASA format and return the image
    '''
    if '.fits' in file.lower():
        imagedata = fits.open(file)
    else:
        image = pim.image(file)
        image.putmask(False)
        image.tofits('temp.fits', velocity=False)
        imagedata = fits.open('temp.fits')
        # Clean up
        os.system('rm temp.fits')

    return imagedata

def make_header(catheader):
    """
    generates a header structure for WCS 
    to work with
    """
    wcsheader = {}

    wcsheader['NAXIS'] = 2                                             # number of axis 
    wcsheader['NAXI1'] = float(catheader['AXIS1'])                     # number of elements along the axis (e.g. number of pixel)
    wcsheader['CTYPE1'] = str(catheader['CTYPE1']).replace('\'','')    # axis type
    wcsheader['CRVAL1'] = float(catheader['CRVAL1'])                   # Coordinate value at reference
    wcsheader['CRPIX1'] = float(catheader['CRPIX1'])                   # pixel value at reference
    wcsheader['CDELT1'] = float(catheader['CDELT1'])                   # coordinate increment
    wcsheader['CUNIT1'] = 'deg'
    if catheader.__contains__('CUNIT1') == True:
        wcsheader['CUNIT1'] = str(catheader['CUNIT1']).replace('\'',''),  # axis unit

    wcsheader['NAXI2'] = float(catheader['AXIS2'])                     # number of elements along the axis (e.g. number of pixel)
    wcsheader['CTYPE2'] = str(catheader['CTYPE2']).replace('\'','')    # axis type
    wcsheader['CRVAL2'] = float(catheader['CRVAL2'])                   # Coordinate value at reference
    wcsheader['CRPIX2'] = float(catheader['CRPIX2'])                   # pixel value at reference
    wcsheader['CDELT2'] = float(catheader['CDELT2'])                   # coordinate increment
    wcsheader['CUNIT2'] = 'deg'
    if catheader.__contains__('CUNIT2') == True:
        wcsheader['CUNIT2'] = str(catheader['CUNIT2']).replace('\'',''),  # axis unit

    return wcsheader

def get_properties(identity, ra_center, dec_center):
    '''
    Get the beam and frequency of a given survey. As for some surveys
    the beam shape changes per position, the position needs to be given as well

    Keyword arguments:
    identity (string) -- The name of the survey (NVSS,SUMSS,FIRST,TGSS,RACS)
    ra_center (float) -- Right ascension to get the beam at
    dec_center (float) -- Declination to get the beam at
    '''
    path = Path(__file__).parent / 'parsets/surveys.json'
    with open(path) as f:
        open_dict = json.load(f)

    if identity in open_dict.keys():
        cat_dict = open_dict[identity]
    else:
        path = Path(__file__).parent / 'parsets/extcat.json'
        with open(path) as f:
            cat_dict = json.load(f)

    BMaj = cat_dict['properties']['BMAJ']
    BMin = cat_dict['properties']['BMIN']
    BPA = cat_dict['properties']['BPA']

    freq = cat_dict['properties']['freq']

    if identity == 'FIRST':
        if (dec_center > 4.5558333):
            BMaj = cat_dict['properties']['BMAJ2']
        elif (dec_center < -2.506944):
                if (ra_center > (21*15)) or (ra_center < (3*15)):
                    BMaj = cat_dict['properties']['BMAJ3']
    if identity == 'SUMSS':
        BMaj = BMaj/np.sin(np.radians(dec_center))
    if identity == 'TGSS':
        if dec_center < 19:
            BMaj = BMaj/np.cos(np.radians(dec_center-19))

    beam = [BMaj,BMin,BPA]
    columns = cat_dict['data_columns']
    return beam, freq, columns

def measure_image_regions(pixel_regions, image, weight_image=None, weight_regions=None):
    '''
    Measure values in images an from given regions

    Keyword arguments:
    pixel_ragions -- Regions of pixels to get values from
    image         -- Image to measure values from
    weight_image  -- Optional image containing weights
    '''
    values = []
    err_values = []
    # Measure value for each source
    for i, region in enumerate(pixel_regions):
        mask = region.to_mask(mode='center')
        mask_data = mask.to_image(image.shape).astype(bool)

        image_values = image[mask_data]
        if np.ma.isMaskedArray(image_values):
            image_values = image_values.filled(np.nan)

        nan_values = np.isnan(image_values)
        image_values = image_values[~nan_values]
        if weight_image is None:
            weights = np.ones(image_values.shape)
        else:
            weight_mask = weight_regions[i].to_mask(mode='center')
            weight_mask_data = weight_mask.to_image(weight_image.shape).astype(bool)

            weights = weight_image[weight_mask_data]
            try:
                weights = weights[~nan_values]
            except IndexError:
                weights = np.ones(image_values.shape)

        # Get weighted mean and standard deviations
        if len(image_values) > 0.5*np.sum(mask_data):
            mean = np.nansum(image_values*weights)/np.sum(weights)
            std = np.sqrt(np.nansum(weights*(image_values-mean)**2) /
                         (np.sum(weights)*(len(weights)-1 / len(weights))))
            values.append(mean)
            err_values.append(std)
        else:
            values.append(np.ma.masked)
            err_values.append(np.ma.masked)

    return values, err_values

def ellipse_skyprojection(ra, dec, Bmaj, Bmin, PA, header=None):
    """
    Provide real pixel values for deprojected Ellipse in
    tha tangent plane

    CAUTION: Definition of an ellipse in matplotlib is 
    width horizontal axis, height vertical axis, angle is anti-clockwise
    in order to match the astronomical definition PA from North clockwise
    height is major axis, width is minor axis and angle is -PA
    """
    if header != None:
        wcs = WCS.WCS(header)
        source_centre_position = SkyCoord(ra*u.deg,dec*u.deg, frame='icrs')
        source_centre_position_pix_xy = list(np.array(WCS.utils.skycoord_to_pixel(source_centre_position,wcs)).flatten())

        # calculate the Ellipse in pixels
        # 
        degtopix = abs(1/header['CDELT1'])

        # CAUTION: the IMAGE has a reverse sense of RA so if the 
        # increment is negative we need to compensate 
        #
        PA_sense = -1
        if header['CDELT1'] < 0:
            PA_sense = 1

        Ellipse_tangent_plane_pix = Ellipse(source_centre_position_pix_xy,
                                            height=degtopix*Bmaj,
                                            width=degtopix*Bmin,
                                            angle=PA_sense*PA).get_verts()
        Ellipse_Sky_deg           = WCS.utils.pixel_to_skycoord(Ellipse_tangent_plane_pix[:,0],
                                                                Ellipse_tangent_plane_pix[:,1],
                                                                wcs)
        Ellipse_Sky_deg_reshaped  = np.column_stack((Ellipse_Sky_deg.ra.deg,
                                                     Ellipse_Sky_deg.dec.deg))
        Ellipse_SKY               = Ellipse_Sky_deg_reshaped 

    else:
        Ellipse_SKY = Ellipse([ra,dec],
                              height=Bmaj,
                              width=Bmin,
                              angle=-PA).get_verts()

    return Ellipse_SKY

def ellipse_RA_check(radec):
    """
    Split the polygons into sub-polygons to be checked
    """
    new_polygons = []

    # check sources if they go over 360 degrees
    ra_check       = abs(np.diff(radec[:,0])) > 300
    ra_check_where = np.where(ra_check)[0].flatten()

    selit          = np.ones(len(radec)).astype('bool')

    if len(ra_check_where) == 2:
        # ellipse crosses twice the RA border
        selit[ra_check_where[0]+1:ra_check_where[1]+1] = False

        radec_list_1 = radec[selit].tolist()

        if radec_list_1[0] == radec_list_1[-1]:
            new_polygons.append(radec_list_1)
        else:
            radec_list_1.append(radec_list_1[0])
            new_polygons.append(radec_list_1)

        radec_list_2 = radec[np.invert(selit)].tolist()
        if radec_list_2[0] == radec_list_2[-1]:
            new_polygons.append(radec_list_2)
        else:
            radec_list_2.append(radec_list_2[0])
            new_polygons.append(radec_list_2)

    elif len(ra_check_where) == 1:
        # ellipse crosses once the RA border
        selit[ra_check_where[0]+1] = False
        new_polygons.append(radec[selit].tolist())
    elif len(ra_check_where) == 0:
        # ellipse crosses never the RA border
        new_polygons.append(radec)
    else:
        print('strange polygon',radec)
        sys.exit(-1)

    return new_polygons

def runningmedian(X, Y, window, stepsize):
    """
    Find the median for the points in a sliding window (odd number in size) 
    as it is moved from left to right by one point at a time.
              
    Also find the std! 
              
    Keyword arguments:
    X -- Y is ordered according to the values of this array, such that median is taken
            over M neighbouring datapoints in X. 
    Y -- list containing items for which a running median (in a sliding window) 
            is to be calculated
    window  -- number of items in window (window size) -- must be an integer > 1
    stepsize

   Note:
     1. The median of a finite list of numbers is the "center" value when this list
        is sorted in ascending order. 
     2. If M is an even number the two elements in the window that
        are close to the center are averaged to give the median (this
        is not by definition)
    """   
    inds = X.argsort()
    sorted_X = X[inds[::-1]]
    sorted_Y = Y[inds[::-1]]

    n_steps = int(len(X)/stepsize)

    medians_x = []
    medians_y = []
    stds_y = []
    for i in range(n_steps):
        min_i = max(0, int(i*stepsize-window/2))
        max_i = min(len(X), int(i*stepsize+window/2))

        medians_x.append(sorted_X[int(i*stepsize)])
        medians_y.append(np.median(sorted_Y[min_i:max_i]))
        stds_y.append(np.std(sorted_Y[min_i:max_i]))

    return np.asarray(medians_x), np.asarray(medians_y), np.asarray(stds_y)

def id_artifacts(bright_sources, catalog, bmaj):
    '''
    Identify artifacts near bright sources
    '''
    catalog_coord = SkyCoord(catalog['RA'], catalog['DEC'], unit='deg', frame='icrs')

    indices = []
    for source in bright_sources:
        source_coord = SkyCoord(source['RA'], source['DEC'], unit='deg', frame='icrs')

        d2d = source_coord.separation(catalog_coord)
        close = d2d < 10*bmaj*u.deg

        indices.append(np.where(np.logical_and(close, catalog['Peak_flux'] < 0.05*source['Peak_flux']))[0])

    indices = np.concatenate(indices)
    unique_indices = np.unique(indices)
    return unique_indices

def size_error_condon(catalog, beam_maj, beam_min):
    # Implement errors for elliptical gaussians in the presence of correlated noise
    # as per Condon (1998), MNRAS.
    ncorr = beam_maj*beam_min

    rho_maj = (np.sqrt(catalog['Maj']*catalog['Min']/(4*ncorr))
               * (1 + ncorr/catalog['Maj']**2)**1.25
               * (1 + ncorr/catalog['Min']**2)**0.25
               * (catalog['Peak_flux']/catalog['Isl_rms']))
    rho_min = (np.sqrt(catalog['Maj']*catalog['Min']/(4*ncorr))
               * (1 + ncorr/catalog['Maj']**2)**0.25
               * (1 + ncorr/catalog['Min']**2)**1.25
               * (catalog['Peak_flux']/catalog['Isl_rms']))

    e_maj = np.sqrt(2)*catalog['Maj']/rho_maj + 0.02*catalog['Maj']
    e_min = np.sqrt(2)*catalog['Min']/rho_min + 0.02*catalog['Min']

    return e_maj, e_min