import string
import os
import sys
import urllib
import gzip
import time

from math import ceil

from astropy import units as u
from astropy.io import ascii
from astropy.table import Table, vstack, join, Column
from astropy.coordinates import SkyCoord

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

    if sumssfile:
        sumsstable = ascii.read(sumssfile, names=sumss_columns, exclude_names=exclude_columns)

        sumss_coordinates = SkyCoord([[source[0:11]] for source in sumssfile],
                                     [[source[13:24]] for source in sumssfile],
                                     unit=(u.hourangle,u.deg))

        sumssids = ['SUMSS J{0}{1}'.format(coord.ra.to_string(unit=u.hourangle,
                                                              sep='',
                                                              precision=0,
                                                              pad=True),
                                           coord.dec.to_string(sep='',
                                                               precision=0,
                                                               alwayssign=True,
                                                               pad=True)) for coord in sumss_coordinates]

        sumssids = Column(sumssids, name='SUMSS_id')
        sumss_ra = Column(sumss_coordinates.ra, name='RA')
        sumss_dec = Column(sumss_coordinates.dec, name='DEC')
        sumsstable.add_columns([sumssids,sumss_ra,sumss_dec],
                           indexes=[0,0,0])

        # Match SUMSS catalog to RA and DEC within offset
        d2d = center.separation(sumss_coordinates)
        catalogmask = d2d < offset
        sumssoutput = sumsstable[catalogmask[:,0]]

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

        first_coordinates = SkyCoord([[source[11:23]] for source in urldata[data_start:-1]],
                                     [[source[24:35]] for source in urldata[data_start:-1]],
                                     unit=(u.hourangle,u.deg))

        firstids = ['FIRST J{0}{1}'.format(coord.ra.to_string(unit=u.hourangle,
                                                              sep='',
                                                              precision=0,
                                                              pad=True),
                                           coord.dec.to_string(sep='',
                                                               precision=0,
                                                               alwayssign=True,
                                                               pad=True)) for coord in first_coordinates]

        firstids = Column(firstids, name='FIRST_id')
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

        col_starts = [0,12,24,30,37,43,49,55,59,64,71,81,89]
        col_ends = [11,23,29,36,42,48,54,58,63,70,79,87,95]

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
            print("No NVSS data available.")
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

    if nvssdctable and nvssfittable:
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

        c = Column(nvssids, name='NVSS_id')
        nvsstable.add_column(c, index=0)
        nvsstable.remove_column('Distance')

    return nvsstable