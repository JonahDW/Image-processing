from astropy.coordinates import SkyCoord
from astropy import units as u
import astropy.wcs as WCS

from matplotlib.patches import Ellipse

from pathlib import Path
import numpy as np
import pickle
import json

def pickle_to_file(data, fname):
    fh = open(fname, 'wb')
    pickle.dump(data, fh)
    fh.close()

def pickle_from_file(fname):
    fh = open(fname, 'rb')
    data = pickle.load(fh)
    fh.close()
    return data

def meerkat_lpb(a, b, freq, offset):
    '''
    MeerKAT L-band primary beam from Mauch et al 2019

    Keyword arguments:
    a (float) -- Constant scaling for the angle (in degrees)
    b (float) -- Constant scaling for the ratio offset/angle
    freq (float) -- Central frequency in GHz
    offset (array) -- Offsets for which to calculate the beam amplitude
    '''
    theta_beam = a*(1.5/freq)
    x = offset/theta_beam
    a_beam = (np.cos(b*np.pi*x)/(1-4*(b*x)**2))**2

    return a_beam

def flux_correction(center_dist, freq, dfreq, alpha):
    # Correct flux modified beam pattern induced by spectral shape
    freqs = np.linspace(freq-dfreq/2,freq+dfreq/2,100)/1e3
    flux = freqs**(-alpha)

    ref_beam = meerkat_lpb(0.985, 1.189, freqs, 0)
    ref_flux = np.trapz(flux*ref_beam, freqs)

    total_flux = []
    attenuation = []
    for offset in center_dist:
        beam = meerkat_lpb(0.985, 1.189, freqs, offset)
        total_flux.append(np.trapz(flux*beam, freqs)/ref_flux)

        beam_cen = meerkat_lpb(0.985, 1.189, freq/1e3, offset)
        attenuation.append(beam_cen)

    correction = np.array(total_flux)/np.array(attenuation)

    return correction

def make_header(catheader):
    """
    generates a header structure for WCS 
    to work with
    """
    wcsheader  = { 'NAXIS'  : 2,                                       # number of axis 
                'NAXIS1' : float(catheader['AXIS1']),                  # number of elements along the axis (e.g. number of pixel)
                'CTYPE1' : str(catheader['CTYPE1']).replace('\'',''),  # axis type
                'CRVAL1' : float(catheader['CRVAL1']),                 # Coordinate value at reference
                'CRPIX1' : float(catheader['CRPIX1']),                 # pixel value at reference
                'CUNIT1' : str(catheader['CUNIT1']).replace('\'',''),  # axis unit
                'CDELT1' : float(catheader['CDELT1']),                 # coordinate increment

                'NAXIS2' : float(catheader['AXIS2']),                  # number of elements along the axis (e.g. number of pixel)
                'CTYPE2' : str(catheader['CTYPE2']).replace('\'',''),  # axis type
                'CRVAL2' : float(catheader['CRVAL2']),                 # Coordinate value at reference
                'CRPIX2' : float(catheader['CRPIX2']),                 # pixel value at reference
                'CUNIT2' : str(catheader['CUNIT2']).replace('\'',''),  # axis unit
                'CDELT2' : float(catheader['CDELT2']),                 # coordinate increment
         }

    return wcsheader

def get_beam(identity, ra_center, dec_center):
    '''
    Get the beam and frequency of a given survey. As for some surveys
    the beam shape changes per position, the position needs to be given as well

    Keyword arguments:
    identity (string) -- The name of the survey (NVSS,SUMSS,FIRST)
    ra_center (float) -- Right ascension to get the beam at
    dec_center (float) -- Declination to get the beam at
    '''
    path = Path(__file__).parent / 'parsets/surveys.json'
    with open(path) as f:
        beam_dict = json.load(f)

    BMaj = beam_dict[identity]['Maj']
    BMin = beam_dict[identity]['Min']
    BPA = beam_dict[identity]['PA']

    freq = beam_dict[identity]['Freq']

    if identity=='FIRST':
        if (dec_center > 4.5558333):
            BMaj = beam_dict[identity]['Maj2']
        elif (dec_center < -2.506944):
                if (ra_center > (21*15)) or (ra_center < (3*15)):
                    BMaj = beam_dict[identity]['Maj3']
    if identity=='SUMSS':
        BMaj = BMaj/np.sin(np.radians(dec_center))

    beam = [BMaj,BMin,BPA]
    return beam, freq

def measure_image_regions(pixel_regions, image, weight_image=None):
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
    for region in pixel_regions:
        mask = region.to_mask(mode='center')
        mask_data = mask.to_image(image.shape).astype(bool)

        image_values = image[mask_data]
        image_values = image_values.filled(np.nan)

        nan_values = np.isnan(image_values)
        image_values = image_values[~nan_values]
        if weight_image is None:
            weights = np.ones(image_values.shape)
        else:
            weights = weight_image[mask_data]
            weights = weights[~nan_values]

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