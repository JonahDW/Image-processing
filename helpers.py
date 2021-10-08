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