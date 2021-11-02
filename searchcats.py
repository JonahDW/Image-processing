import string
import os
import sys
import urllib
import gzip
import time

import numpy as np
from math import ceil
from itertools import cycle, islice

from astropy import units as u
from astropy.io import ascii
from astropy.table import Table, vstack, join, Column
from astropy.coordinates import SkyCoord, Angle

FIRSTCATURL='http://sundog.stsci.edu/cgi-bin/searchfirst'
NVSSCATURL='http://www.cv.nrao.edu/cgi-bin/NVSSlist.pl'
SUMSSCATURL='http://www.astrop.physics.usyd.edu.au/sumsscat/sumsscat.Mar-11-2008'

def geturl(url,params,timeout,tries=5):
    '''
    Attempt to obtain open a url

    Keyword arguments:
    url (string) -- The url to be queried
    params (string) -- Optional query string
    timeout (int) -- Timeout time in seconds
    tries (int) -- Number of tries before giving up

    Return:
    The data from the url as a string
    '''
    params = params.encode('utf-8')

    for i in range(1,tries+1):
        try:
            urlsocket=urllib.request.urlopen(url,params,timeout)
            urldata=urlsocket.read()
        except:
            time.sleep(5)
        else:
            break

    if i==tries:
        print("Cannot access "+url)
        return False
    else:
        return urldata.decode('utf-8')

def getsumssdata(ra,dec,offset):
    '''
    Get the SUMSS catalogue at a given position.
    '''
    def reduce_sumssfile(sumssfile, ra, dec, offset):
        '''
        Reduce the SUMSS file by filtering RA as the catalog
        is only strictly increasing in Right Ascension
        '''
        ra = Angle(ra, unit='deg')
        dec = Angle(dec, unit='deg')

        min_ra = ra - offset/np.cos(dec.radian)
        max_ra = ra + offset/np.cos(dec.radian)

        min_ra_idx = None
        max_ra_idx = None

        # Iterate through list, account for 
        for i, line in cycle(enumerate(sumssfile)):
            if int(line[0:2]) >= min_ra.hms[0] and int(line[3:5]) >= min_ra.hms[1] and min_ra_idx is None:
                min_ra_idx = i
            if int(line[0:2]) >= max_ra.hms[0] and int(line[3:5]) >= max_ra.hms[1] and max_ra_idx is None:
                max_ra_idx = i
            if max_ra_idx and min_ra_idx:
                break

        start = min_ra_idx
        stop = max_ra_idx
        if start > stop:
            stop += len(sumssfile)

        sumssfile = list(islice(cycle(sumssfile), start, stop))
        return sumssfile

    center = SkyCoord(ra,dec)

    # Define SUMSS columns
    sumss_columns = ['RA(h)','RA(m)','RA(s)',
                     'DEC(d)','DEC(m)','DEC(s)',
                     'E_RA','E_DEC','Peak_flux',
                     'E_Peak_flux','Total_flux',
                     'E_Total_Flux','Maj','Min',
                     'PA', 'DC_Maj','DC_Min','DC_PA',
                     'Mosaic','nMosaics','YPix','XPix']
    exclude_columns = ['RA(h)','RA(m)','RA(s)',
                      'DEC(d)','DEC(m)','DEC(s)']

    sumssoutput=[]
    sumssfile=geturl(SUMSSCATURL,'',30)

    sumssfile = sumssfile.split('\n')[:-1]
    sumssfile = reduce_sumssfile(sumssfile, ra, dec, offset)

    if sumssfile:
        sumsstable = ascii.read(sumssfile, names=sumss_columns, exclude_names=exclude_columns)

        sumss_coordinates = SkyCoord([source[0:11] for source in sumssfile],
                                     [source[13:24] for source in sumssfile],
                                     unit=(u.hourangle,u.deg))

        sumssids = ['SUMSS J{0}{1}'.format(coord.ra.to_string(unit=u.hourangle,
                                                              sep='',
                                                              precision=0,
                                                              pad=True),
                                           coord.dec.to_string(sep='',
                                                               precision=0,
                                                               alwayssign=True,
                                                               pad=True)) for coord in sumss_coordinates]

        sumssids = Column(sumssids, name='Source_name')
        sumss_ra = Column(sumss_coordinates.ra, name='RA')
        sumss_dec = Column(sumss_coordinates.dec, name='DEC')
        sumsstable.add_columns([sumssids,sumss_ra,sumss_dec],
                           indexes=[0,0,0])

        # Match SUMSS catalog to RA and DEC within offset
        d2d = center.separation(sumss_coordinates)
        catalogmask = d2d < offset
        sumssoutput = sumsstable[catalogmask]

    return sumssoutput

def getfirstdata(ra,dec,offset):
    '''
    At a given ra,dec and required search radius, search the FIRST catalogue and
    produce a list of all of the FIRST data at given position. An empty array is returned
    if there is nothing. Else the returned array contains [[source data 1],[source data 2],...]
    '''

    firstdata=[]

    firstparams = urllib.parse.urlencode({'RA': ' '.join(ra+dec),
                                          'Radius': offset,
                                          'Text':1,
                                          'Equinox': 'J2000'})
    firsturl = geturl(FIRSTCATURL,firstparams,10)

    if not firsturl:
        print("No FIRST data available.")
        return None

    urldata=firsturl.split('\n')
    numsources=0
    for tempstr in urldata:
        if len(tempstr)>0 and tempstr.split()[len(tempstr.split())-1]=='arcsec':
            if tempstr.split()[1]=='No':
                print("No FIRST data available.")
                return None
            else:
                numsources=float(tempstr.split()[1])
                break

    # Define FIRST columns
    first_columns = ['Distance','RA(h)','RA(m)','RA(s)',
                     'DEC(d)','DEC(m)','DEC(s)','Side lobe prob',
                     'Peak_flux','Total_flux','RMS','DC_Maj',
                     'DC_Min','DC_PA','Maj','Min','PA','Field name',
                     'SDSS Match <8 arcsec','Closest SDSS sep (arcsec)',
                     'SDSS i','SD Cl','2MASS Match <8 arcsec',
                     'Closest 2MASS sep (arcsec)','2MASS K','Mean Epoch (year)',
                     'Mean Epoch (MJD)','RMS Epoch (MJD)']
    exclude_columns = ['Distance','RA(h)','RA(m)','RA(s)',
                      'DEC(d)','DEC(m)','DEC(s)', 'Side lobe prob']

    # FIRST cat only returns 500 sources at a time so we need to iterate every 500 sources.
    numloops=int(ceil(numsources/500))

    for x in range(numloops):
        firstparams = urllib.parse.urlencode({'RA': ' '.join(ra+dec),
                                              'Radius': offset,
                                              'Text':1,
                                              'Equinox': 'J2000',
                                              'PStart':x*500,
                                              '.cgifields': 'Text'})
        firsturl = geturl(FIRSTCATURL,firstparams,10)
        urldata = firsturl.split('\n')

        table = ascii.read(urldata,
                           data_start=0,
                           names=first_columns,
                           exclude_names=exclude_columns,
                           guess=False)

        # Find first line that is not commented
        for i, line in enumerate(urldata):
            if not line.startswith('#'):
                data_start = i
                break

        first_coordinates = SkyCoord([source[11:23] for source in urldata[data_start:-1]],
                                     [source[24:35] for source in urldata[data_start:-1]],
                                     unit=(u.hourangle,u.deg))

        firstids = ['FIRST J{0}{1}'.format(coord.ra.to_string(unit=u.hourangle,
                                                              sep='',
                                                              precision=0,
                                                              pad=True),
                                           coord.dec.to_string(sep='',
                                                               precision=0,
                                                               alwayssign=True,
                                                               pad=True)) for coord in first_coordinates]

        firstids = Column(firstids, name='Source_name')
        first_ra = Column(first_coordinates.ra, name='RA')
        first_dec = Column(first_coordinates.dec, name='DEC')
        table.add_columns([firstids,first_ra,first_dec],
                           indexes=[0,0,0])

        if x == 0:
            firsttable = table
        else:
            firsttable = vstack([firsttable,table])

    return firsttable

def getnvssdata(ra,dec,offset):
    '''
    At a given ra,dec and required search radius, search the NVSS catalogue and
    produce a list of all of the NVSS data at given position. An empty array is returned
    if there is nothing. Else the reutrned array contains [[source data 1],[source data 2],...]
    '''
    def filternvssfile(nvssurl):
        '''
        Extact only the lines containing data from the url data
        '''
        nvssdata=[]
        tempdata=iter(nvssurl)
        count=0;
        templine=next(tempdata)
        while True:
            if templine and templine[0].isdigit():
                nvssdata=nvssdata+[templine]
            try:
                templine=next(tempdata)
            except:
                break

        return nvssdata

    def get_table(sizes, col_names):
        '''
        Get a table from the NVSS

        Keyword arguments:
        sizes (int) -- Source sizes, 0 for deconvolved values,
                                     1 for fitted values
                                     2 for raw values
        col_names (list) -- Names of the NVSS columns
        '''

        col_starts = [0,12,24,29,37,43,49,55,59,64,71,81,89]
        col_ends = [11,23,28,36,42,48,54,58,63,70,79,87,95]

        nvssparams = urllib.parse.urlencode({'Equinox': 3,
                                             'DecFit': sizes,
                                             'FluxDensity': 0,
                                             'PolFluxDensity': 0,
                                             'RA': ' '.join(ra),
                                             'Dec': ' '.join(dec),
                                             'searchrad':offset})
        nvssfiturl = geturl(NVSSCATURL,nvssparams,10)
        nvssfiturldata = nvssfiturl.split('\n')

        nvssdata=filternvssfile(nvssfiturldata)

        if not nvssdata:
            return None

        nvsstable = ascii.read(nvssdata,
                               format='fixed_width_no_header',
                               names=col_names,
                               col_starts=col_starts,
                               col_ends=col_ends,
                               fill_values=[('Blank','0')],
                               guess=False)
        return nvsstable

    nvsstable = []
    # Define NVSS columns for deconvolved values
    nvss_dc_columns = ['RA','DEC','Distance','Total_flux',
                       'DC_Maj','DC_Min','DC_PA','Res',
                       'Pol_flux','Pol_ang','Field',
                       'YPix','XPix']

    # Define NVSS columns for fitted values
    nvss_fit_columns = ['RA','DEC','Distance','Peak_flux',
                        'Maj','Min','PA','Res',
                        'Pol_flux','Pol_ang','Field',
                        'YPix','XPix']

    # Get both deconvolve and fitted tables
    nvssdctable = get_table(0, nvss_dc_columns)
    nvssfittable = get_table(1, nvss_fit_columns)

    if nvssdctable is None and nvssfittable is None:
        print("No NVSS data available.")
        sys.exit()

    nvsstable = join(nvssdctable,
                     nvssfittable,
                     keys=['RA','DEC','Distance','Field','YPix','XPix'],
                     table_names=['DC','FIT'])

    nvss_coordinates = SkyCoord(ra=nvsstable['RA'],
                                dec=nvsstable['DEC'],
                                unit=(u.hourangle, u.deg))

    nvssids = ['NVSS J{0}{1}'.format(coord.ra.to_string(unit=u.hourangle,
                                                        sep='',
                                                        precision=0,
                                                        pad=True),
                                     coord.dec.to_string(sep='',
                                                         precision=0,
                                                         alwayssign=True,
                                                         pad=True)) for coord in nvss_coordinates]

    nvsstable['RA'] = nvss_coordinates.ra
    nvsstable['DEC'] = nvss_coordinates.dec

    c = Column(nvssids, name='Source_name')
    nvsstable.add_column(c, index=0)

    return nvsstable
