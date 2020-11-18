import numpy as np
import json

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

def get_beam(identity, ra_center, dec_center):
    with open('parsets/surveys.json') as f:
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
        BMaj = Maj/np.sin(np.radians(dec_center))

    beam = [BMaj,BMin,BPA]
    return beam, freq