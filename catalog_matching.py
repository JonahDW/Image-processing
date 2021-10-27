#!/usr/bin/env python

import gc
import os
import sys
import numpy as np

import json
from numpyencoder import NumpyEncoder
from pathlib import Path
from argparse import ArgumentParser

from astropy import units as u
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
import astropy.wcs as WCS

import matplotlib
matplotlib.use('Agg')
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from shapely.geometry import Polygon

import searchcats as sc
import helpers

gc.enable()

class SourceEllipse:

    def __init__(self, catalog_entry, column_dict):
        self.RA = catalog_entry[column_dict['ra']]
        self.DEC = catalog_entry[column_dict['dec']]
        self.Maj = catalog_entry[column_dict['majax']]
        self.Min = catalog_entry[column_dict['minax']]
        self.PA  = catalog_entry[column_dict['pa']]

        if column_dict['peak_flux']:
            self.PeakFlux = catalog_entry[column_dict['peak_flux']]
        if column_dict['total_flux']:
            self.IntFlux = catalog_entry[column_dict['total_flux']]

        self.skycoord = SkyCoord(self.RA, self.DEC, unit='deg')


    def match(self, ra_list, dec_list, maj_list, min_list, pa_list, sigma_extent, addsearchdistance,header):
        '''
        Match the ellipse with a (list of) source(s)

        Keyword arguments:
        ra_list -- Right ascension of sources
        dec_list -- Declination of sources
        separation (float) - Additional range in degrees

        '''

        offset_coord   = SkyCoord(ra_list, dec_list, unit='deg')

        sky_separation = self.skycoord.separation(offset_coord) 

        # this factor just increases the number of sources to check
        # so no harm to the process
        # is of the order of sub-arcsec
        #
        dra, ddec           = self.skycoord.spherical_offsets_to(offset_coord)

        deprojection_factor = (np.sqrt( (dra)**2 + (ddec)**2 ) / sky_separation).value

        # The Gaussians are given in FWHM Bmaj, Bmin
        # in order to obtain a source extend we use the 3 sigma extend
        #
        # https://ned.ipac.caltech.edu/level5/Leo/Stats2_3.html
        #
        convert_factor_FWHM_to_sigma_extend = sigma_extent / (2*np.sqrt(2*np.log(2)))

        # Check if sources match within Bmaj/2 boundaries
        # these source could match or not this needs to
        # be checked
        #
        # CAUTION: self is related to the external catalogue
        #
        maj_match = sky_separation < convert_factor_FWHM_to_sigma_extend * deprojection_factor * (self.Maj/2. + maj_list/2.) + addsearchdistance

        # Check if sources match within Bmin/2 boundaries
        # these are the source we do not need to check 
        # further, they always match
        #
        min_match = sky_separation <= convert_factor_FWHM_to_sigma_extend * (self.Min/2. + min_list/2.) + addsearchdistance
        

        # Check out the source between the maj_match and min_match boundaries
        #
        if max(np.array(np.where(np.logical_xor(min_match,maj_match))).shape) > 0:

            msour_idx = np.where(np.logical_xor(min_match,maj_match))[0].flatten()
            
            for s in msour_idx:

                check_source = Ellipse_skyprojection(self.RA,self.DEC,convert_factor_FWHM_to_sigma_extend*self.Maj,convert_factor_FWHM_to_sigma_extend*self.Min,self.PA,header)

                to_sources   = Ellipse_skyprojection(ra_list[s],dec_list[s],convert_factor_FWHM_to_sigma_extend*maj_list[s]+addsearchdistance,convert_factor_FWHM_to_sigma_extend*min_list[s]+addsearchdistance,pa_list[s],header)


                # this is to check if sources are crossing the 360 deg in RA
                #
                ra_check_s       = abs(np.diff(check_source[:,0])) > 300
                ra_check_s_where = np.where(ra_check_s)[0]

                ra_to_s       = abs(np.diff(to_sources[:,0])) > 300
                ra_to_s_where = np.where(ra_to_s)[0]

                if max(ra_check_s_where.shape) > 0 or max(ra_to_s_where.shape) > 0:

                    do_they_overlap = False
                    split_check_s_source  = Ellipse_RA_check(check_source)
                    split_to_s_sources    = Ellipse_RA_check(to_sources)

                    for a in range(len(split_check_s_source)):
                        for b in range(len(split_to_s_sources)):
                            do_they_overlap_split = np.invert(Polygon(split_check_s_source[a]).intersection(Polygon(split_to_s_sources[b])).is_empty)
                            if do_they_overlap_split == True:
                                do_they_overlap = True
                            #
                            del do_they_overlap_split
                            gc.collect()

                else:
                    # this is a realy cool thing
                    # use vertices to a Polygon and shapely to check if they intersect
                    # https://gis.stackexchange.com/questions/243459/drawing-ellipse-with-shapely/243462#243462
                    do_they_overlap = np.invert(Polygon(check_source).intersection(Polygon(to_sources)).is_empty)
                    #


                # adjust the matching of the major_matches and exclude sources 
                #
                maj_match[s]    = do_they_overlap


                del do_they_overlap,check_source,to_sources
                gc.collect()
              
        return np.where(maj_match)[0]


    def to_artist(self):
        '''
        Convert the ellipse to a matplotlib artist

        CAUTION definition in matplotlib is 
        width is horizointal axis, height vertical axis, angle is anti-clockwise
        in order to match the astronomical definition PA from North clockwise
        height is major axis, width is minor axis and angle is -PA
        '''
        return Ellipse(xy = (self.RA, self.DEC),
                        width = self.Min,
                        height = self.Maj,
                        angle = -self.PA)

class ExternalCatalog:

    def __init__(self, name, catalog, center):
        self.name = name
        self.cat = catalog

        if name in ['NVSS','SUMSS','FIRST']:
            columns = {'ra':'RA','dec':'DEC','majax':'Maj','minax':'Min',
                       'pa':'PA','peak_flux':'Peak_flux','total_flux':'Total_flux'}
            self.sources = [SourceEllipse(source, columns) for source in self.cat]
            beam, freq = helpers.get_beam(name, center.ra.deg, center.dec.deg)

            self.BMaj  = beam[0]
            self.BMin  = beam[1]
            self.BPA   = beam[2]
            self.freq  = freq
        else:
            path = Path(__file__).parent / 'parsets/extcat.json'
            with open(path) as f:
                cat_info = json.load(f)

            if cat_info['data_columns']['quality_flag']:
                self.cat = self.cat[self.cat[cat_info['data_columns']['quality_flag']] > 0]
            n_rejected = len(self.cat[self.cat[cat_info['data_columns']['quality_flag']] > 0])
            if  n_rejected > 0:
                print(f'Excluding {n_rejected} sources that have a negative quality flag')

            self.sources = [SourceEllipse(source, cat_info['data_columns']) for source in self.cat]
            self.BMaj = cat_info['properties']['BMAJ']
            self.BMin = cat_info['properties']['BMIN']
            self.BPA = cat_info['properties']['BPA']
            self.freq = cat_info['properties']['freq']

class Pointing:

    def __init__(self, catalog, filename):
        self.dirname = os.path.dirname(filename)

        self.cat = catalog[catalog['Quality_flag'] > 0]
        if len(self.cat) < len(catalog):
            print(f'Excluding {len(catalog) - len(self.cat)} sources that have a negative quality flag')

        columns = {'ra':'RA','dec':'DEC','majax':'Maj','minax':'Min',
                   'pa':'PA','peak_flux':'Peak_flux','total_flux':'Total_flux'}
        self.sources = [SourceEllipse(source, columns) for source in self.cat]

        # Parse meta
        header = catalog.meta

        # HRK
        self.header = catalog.meta

        self.telescope = header['SF_TELE'].replace("'","")
        self.BMaj = float(header['SF_BMAJ'])*3600 #arcsec
        self.BMin = float(header['SF_BMIN'])*3600 #arcsec
        self.BPA = float(header['SF_BPA'])

        # Determine frequency axis:
        for i in range(1,5):
            if 'FREQ' in header['CTYPE'+str(i)]:
                freq_idx = i
                break
        self.freq = float(header['CRVAL'+str(freq_idx)])/1e6 #MHz
        self.dfreq = float(header['CDELT'+str(freq_idx)])/1e6 #MHz

        self.center = SkyCoord(float(header['CRVAL1'])*u.degree,
                               float(header['CRVAL2'])*u.degree)
        dec_fov = abs(float(header['CDELT1']))*float(header['CRPIX1'])*2
        self.fov = dec_fov/np.cos(self.center.dec.rad) * u.degree

        try:
            self.name = header['OBJECT'].replace("'","")
        except KeyError:
            self.name = os.path.basename(filename).split('.')[0]

    def query_NVSS(self):
        '''
        Match the pointing to the NVSS catalog
        '''
        nvsstable = sc.getnvssdata(ra = [self.center.ra.to_string(u.hourangle, sep=' ')],
                                   dec = [self.center.dec.to_string(u.deg, sep=' ')],
                                   offset = 0.5*self.fov.to(u.arcsec))

        if not nvsstable:
            sys.exit()

        nvsstable['Maj'].unit = u.arcsec
        nvsstable['Min'].unit = u.arcsec

        nvsstable['Maj'] = nvsstable['Maj'].to(u.deg)
        nvsstable['Min'] = nvsstable['Min'].to(u.deg)
        nvsstable['PA']  = nvsstable['PA']

        nvsstable['RA']  = nvsstable['RA'].to(u.deg)
        nvsstable['DEC'] = nvsstable['DEC'].to(u.deg)

        nvsstable['Peak_flux'] /= 1e3 #convert to Jy
        nvsstable['Total_flux'] /= 1e3 #convert to Jy

        return nvsstable

    def query_FIRST(self):
        '''
        Match the pointing to the FIRST catalog
        '''
        firsttable = sc.getfirstdata(ra = [self.center.ra.to_string(u.hourangle, sep=' ')],
                                     dec = [self.center.dec.to_string(u.deg, sep=' ')],
                                     offset = 0.5*self.fov.to(u.arcsec))

        if not firsttable:
            sys.exit()

        firsttable['Maj'].unit = u.arcsec
        firsttable['Min'].unit = u.arcsec

        firsttable['Maj'] = firsttable['Maj'].to(u.deg)
        firsttable['Min'] = firsttable['Min'].to(u.deg)
        firsttable['PA']  = firsttable['PA']
        firsttable['RA']  = firsttable['RA'].to(u.deg)
        firsttable['DEC'] = firsttable['DEC'].to(u.deg)

        firsttable['Peak_flux'] /= 1e3 #convert to Jy
        firsttable['Total_flux'] /= 1e3 #convert to Jy

        return firsttable

    def query_SUMSS(self):
        '''
        Match the pointing to the SUMSS catalog. This is very slow since
        SUMSS does not offer a catalog search so we match to the entire catalog
        '''
        if self.center.dec.deg > -29.5:
            print('Catalog outside of SUMSS footprint')
            sys.exit()

        sumsstable = sc.getsumssdata(ra = self.center.ra,
                                     dec = self.center.dec,
                                     offset = 0.5*self.fov)

        sumsstable['Maj'].unit = u.arcsec
        sumsstable['Min'].unit = u.arcsec

        sumsstable['Maj'] = sumsstable['Maj'].to(u.deg)
        sumsstable['Min'] = sumsstable['Min'].to(u.deg)
        sumsstable['PA']  = sumsstable['PA']

        sumsstable['RA'] = sumsstable['RA'].to(u.deg)
        sumsstable['DEC'] = sumsstable['DEC'].to(u.deg)

        sumsstable['Peak_flux'] /= 1e3 #convert to Jy
        sumsstable['Total_flux'] /= 1e3 #convert to Jy

        return sumsstable

def match_catalogs(pointing, ext,sigma_extent,addsearchdistance):
    '''
    Match the sources of the chosen external catalog to the sources in the pointing
    '''
    print(f'Matching {len(ext.sources)} sources in {ext.name} to {len(pointing.cat)} sources in the pointing')

    matches = []
    for source in ext.sources:
        matches.append(source.match(pointing.cat['RA'], pointing.cat['DEC'],pointing.cat['Maj'],pointing.cat['Min'],pointing.cat['PA'],sigma_extent,addsearchdistance,make_header(pointing.header)))

    return matches

def info_match(pointing, ext, matches,fluxtype,alpha,output):
    """
    Provide information of the matches
    """

    match_info = {}

    dDEC      = []
    dRA       = []
    n_matches = []
    
    for i, match in enumerate(matches):
        if len(match) > 0:
            for m in match:
                # determine the offset
                dra, ddec = ext.sources[i].skycoord.spherical_offsets_to(pointing.sources[m].skycoord)
                dRA.append(dra.arcsec)
                dDEC.append(ddec.arcsec)
                # determine the fluxes
                n_matches.append(len(match))
    
    match_info['offset'] = {}
    match_info['offset']['dRA']       = np.array(dRA)
    match_info['offset']['dDEC']      = np.array(dDEC)
    match_info['offset']['n_matches'] = n_matches

    stats_data     = ['dRA','dDEC']
    get_stats      = [np.min,np.max,np.std,np.mean,np.median,len]
    #
    matching_class = list(np.unique(n_matches))
    matching_class.append('Full')

    match_info['offset']['stats'] = {}
    # get stats
    for mmdat in stats_data:
        match_info['offset']['stats'][mmdat]={}
        for cl in matching_class:
            match_info['offset']['stats'][mmdat][str(cl)] = {}

            if cl == 'Full':
                stats_select = np.ones(len(n_matches)).astype(dtype=bool)
            else:
                stats_select = match_info['offset']['n_matches'] == cl

            for getst in get_stats:            
                match_info['offset']['stats'][mmdat][str(cl)][getst.__name__] = getst((match_info['offset'][mmdat][stats_select]))

        

    match_info['fluxes'] = {}

    ext_flux    = []
    int_flux    = []
    separation  = []
    n_matches  = []
    if fluxtype == 'Total':
        for i, match in enumerate(matches):
            if len(match) > 0:
                ext_flux.append(ext.sources[i].IntFlux)
                int_flux.append(np.sum([pointing.sources[m].IntFlux for m in match]))
                source_coord = SkyCoord(ext.sources[i].RA, ext.sources[i].DEC, unit='deg')
                separation.append(source_coord.separation(pointing.center).deg)
                n_matches.append(len(match))
    elif fluxtype == 'Peak':
        for i, match in enumerate(matches):
            if len(match) > 0:
                ext_flux.append(ext.sources[i].PeakFlux)
                int_flux.append(np.sum([pointing.sources[m].PeakFlux for m in match]))
                source_coord = SkyCoord(ext.sources[i].RA, ext.sources[i].DEC, unit='deg')
                separation.append(source_coord.separation(pointing.center).deg)
                n_matches.append(len(match))
    else:
        print(f'Invalid fluxtype {fluxtype}, choose between Total or Peak flux')
        sys.exit()


    match_info['fluxes'][fluxtype] = {}
    match_info['fluxes'][fluxtype]['ext_flux']   = ext_flux
    match_info['fluxes'][fluxtype]['int_flux']   = int_flux
    match_info['fluxes'][fluxtype]['separation'] = separation
    match_info['fluxes'][fluxtype]['n_matches']  = n_matches
 

    # Scale flux density to proper frequency
    ext_flux_corrected = np.array(ext_flux) * (pointing.freq/ext.freq)**-alpha
    dFlux = np.array(int_flux)/ext_flux_corrected

    match_info['fluxes'][fluxtype]['alpha'] = alpha
    match_info['fluxes'][fluxtype]['dFlux'] = dFlux



    stats_data     = ['dFlux']
    get_stats      = [np.min,np.max,np.std,np.mean,np.median,len]
    #
    matching_class = list(np.unique(n_matches))
    matching_class.append('Full')

    match_info['fluxes']['stats'] = {}
    # get stats
    for mmdat in stats_data:
        match_info['fluxes']['stats'][mmdat]={}
        for cl in matching_class:
            match_info['fluxes']['stats'][mmdat][str(cl)] = {}

            if cl == 'Full':
                stats_select = np.ones(len(n_matches)).astype(dtype=bool)
            else:
                stats_select = match_info['fluxes'][fluxtype]['n_matches'] == cl

            for getst in get_stats:            
                match_info['fluxes']['stats'][mmdat][str(cl)][getst.__name__] = getst((match_info['fluxes'][fluxtype][mmdat][stats_select]))


    return match_info


def plot_catalog_match(pointing, ext, matches, plot, dpi):
    '''
    Plot the field with all the matches in it as ellipses
    '''
    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot()

    for i, match in enumerate(matches):
        ext_ell = ext.sources[i].to_artist()
        ax.add_artist(ext_ell)
        ext_ell.set_facecolor('b')
        ext_ell.set_alpha(0.5)

        if len(match) > 0:
            for ind in match:
                ell = pointing.sources[ind].to_artist()
                ax.add_artist(ell)
                ell.set_facecolor('r')
                ell.set_alpha(0.5)
        else:
            ext_ell.set_facecolor('k')

    non_matches = np.setdiff1d(np.arange(len(pointing.sources)), np.concatenate(matches).ravel())
    for i in non_matches:
        ell = pointing.sources[i].to_artist()
        ax.add_artist(ell)
        ell.set_facecolor('g')
        ell.set_alpha(0.5)

    ax.set_xlim(pointing.center.ra.deg-0.5*pointing.fov.value,
                pointing.center.ra.deg+0.5*pointing.fov.value)
    ax.set_ylim(pointing.center.dec.deg-0.5*pointing.fov.value*np.cos(pointing.center.dec.rad),
                pointing.center.dec.deg+0.5*pointing.fov.value*np.cos(pointing.center.dec.rad))
    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('DEC (degrees)')

    if plot is True:
        plt.savefig(os.path.join(pointing.dirname,f'match_{ext.name}_{pointing.name}_shapes.png'), dpi=dpi, bbox_inches='tight')
    else:
        plt.savefig(plot, dpi=dpi)

    plt.close()

def plot_astrometrics(pointing, ext, matches, astro, dpi):
    '''
    Plot astrometric offsets of sources to the reference catalog
    '''
    dDEC = []
    dRA = []
    n_matches = []
    for i, match in enumerate(matches):
        if len(match) > 0:
            for m in match:
                dra, ddec = ext.sources[i].skycoord.spherical_offsets_to(pointing.sources[m].skycoord)
                dRA.append(dra.arcsec)
                dDEC.append(ddec.arcsec)
                n_matches.append(len(match))

    cmap = colors.ListedColormap(["navy", "crimson", "limegreen", "gold"])
    norm = colors.BoundaryNorm(np.arange(0.5, 5, 1), cmap.N)

    fig = plt.figure(figsize=(15,15))
    ax  = plt.subplot()
    ax.axis('equal')

    sc = ax.scatter(dRA, dDEC, zorder=2,
                    marker='.', s=5,
                    c=n_matches, cmap=cmap, norm=norm)

    # Add colorbar to set matches
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(sc, cax=cax)

    cbar.set_ticks([1,2,3,4])
    cax.set_yticklabels(['1','2','3','>3'])
    cax.set_title('Matches')

    ext_beam_ell = Ellipse(xy=(0,0),
                           width=ext.BMin,
                           height=ext.BMaj,
                           angle=-ext.BPA,
                           facecolor='none',
                           edgecolor='b',
                           linestyle='dashed',
                           label=f'{ext.name} beam')
    int_beam_ell = Ellipse(xy=(0,0),
                           width=pointing.BMin,
                           height=pointing.BMaj,
                           angle=-pointing.BPA,
                           facecolor='none',
                           edgecolor='k',
                           label=f'{pointing.telescope} beam')

    ax.add_patch(ext_beam_ell)
    ax.add_patch(int_beam_ell)

    ymax_abs = abs(max(ax.get_ylim(), key=abs))
    xmax_abs = abs(max(ax.get_xlim(), key=abs))

    # Equalise the axis to get no distortion of the beams
    xymax    = abs(max(xmax_abs,ymax_abs))
    xmax_abs = xymax
    ymax_abs = xymax

    ax.set_ylim(ymin=-ymax_abs, ymax=ymax_abs)
    ax.set_xlim(xmin=-xmax_abs, xmax=xmax_abs)

    ax.axhline(0,-xmax_abs,xmax_abs, color='k', zorder=1)
    ax.axvline(0,-ymax_abs,ymax_abs, color='k', zorder=1)

    ax.annotate('Median offsets', xy=(0.05,0.95), xycoords='axes fraction', fontsize=8)
    # Determine mean and standard deviation of points Dec
    ax.axhline(np.median(dDEC),-xmax_abs,xmax_abs, color='grey', linestyle='dashed', zorder=1)
    ax.axhline(np.median(dDEC)-np.std(dDEC),-xmax_abs,xmax_abs, color='grey', linestyle='dotted', zorder=1)
    ax.axhline(np.median(dDEC)+np.std(dDEC),-ymax_abs,ymax_abs, color='grey', linestyle='dotted', zorder=1)
    ax.axhspan(np.median(dDEC)-np.std(dDEC),np.median(dDEC)+np.std(dDEC), alpha=0.2, color='grey')
    ax.annotate(f'dDEC = {np.median(dDEC):.2f}+-{np.std(dDEC):.2f}',
                xy=(0.05,0.90), xycoords='axes fraction', fontsize=8)

    # Determine mean and standard deviation of points in RA
    ax.axvline(np.median(dRA),-ymax_abs,ymax_abs, color='grey', linestyle='dashed', zorder=1)
    ax.axvline(np.median(dRA)-np.std(dRA),-xmax_abs,xmax_abs, color='grey', linestyle='dotted', zorder=1)
    ax.axvline(np.median(dRA)+np.std(dRA),-ymax_abs,ymax_abs, color='grey', linestyle='dotted', zorder=1)
    ax.axvspan(np.median(dRA)-np.std(dRA),np.median(dRA)+np.std(dRA), alpha=0.2, color='grey')
    ax.annotate(f'dRA = {np.median(dRA):.2f}+-{np.std(dRA):.2f}',
                xy=(0.05,0.85), xycoords='axes fraction', fontsize=8)


    n_match_array = np.array(n_matches)
    dRA_array     = np.array(dRA)
    sel_single_match = n_match_array == 1.0
    # HRK
    #print(n_match_array[sel_single_match])
    #print(np.median(dRA_array[sel_single_match]))
    #print(np.std(dRA_array[sel_single_match]))
    #print(len(n_match_array[sel_single_match]))

    ax.set_title(f'Astrometric offset of {len(dRA)} sources')
    ax.set_xlabel('RA offset (arcsec)')
    ax.set_ylabel('DEC offset (arcsec)')
    ax.legend(loc='upper right')

    if astro is True:
        plt.savefig(os.path.join(pointing.dirname,f'match_{ext.name}_{pointing.name}_astrometrics.png'), dpi=dpi)
    else:
        plt.savefig(astro, dpi=dpi)
    plt.close()

def plot_fluxes(pointing, ext, matches, fluxtype, flux, alpha, dpi):
    '''
    Plot flux offsets of sources to the reference catalog
    '''
    ext_flux = []
    int_flux = []
    separation = []
    n_matches = []
    if fluxtype == 'Total':
        for i, match in enumerate(matches):
            if len(match) > 0:
                ext_flux.append(ext.sources[i].IntFlux)
                int_flux.append(np.sum([pointing.sources[m].IntFlux for m in match]))
                source_coord = SkyCoord(ext.sources[i].RA, ext.sources[i].DEC, unit='deg')
                separation.append(source_coord.separation(pointing.center).deg)
                n_matches.append(len(match))
    elif fluxtype == 'Peak':
        for i, match in enumerate(matches):
            if len(match) > 0:
                ext_flux.append(ext.sources[i].PeakFlux)
                int_flux.append(np.sum([pointing.sources[m].PeakFlux for m in match]))
                source_coord = SkyCoord(ext.sources[i].RA, ext.sources[i].DEC, unit='deg')
                separation.append(source_coord.separation(pointing.center).deg)
                n_matches.append(len(match))
    else:
        print(f'Invalid fluxtype {fluxtype}, choose between Total or Peak flux')
        sys.exit()

    # Scale flux density to proper frequency
    ext_flux_corrected = np.array(ext_flux) * (pointing.freq/ext.freq)**-alpha
    dFlux = np.array(int_flux)/ext_flux_corrected

    cmap = colors.ListedColormap(["navy", "crimson", "limegreen", "gold"])
    norm = colors.BoundaryNorm(np.arange(0.5, 5, 1), cmap.N)

    fig, ax = plt.subplots()
    sc = ax.scatter(separation, dFlux,
                    marker='.', s=5,
                    c=n_matches, cmap=cmap, norm=norm)

    # Add colorbar to set matches
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(sc, cax=cax)

    cbar.set_ticks([1,2,3,4])
    cax.set_yticklabels(['1','2','3','>3'])
    cax.set_title('Matches')

    ax.set_title(f'Flux ratio of {len(dFlux)} sources')
    ax.set_xlabel('Distance from pointing center (degrees)')
    ax.set_ylabel(f'Flux ratio ({fluxtype} flux)')
    ax.set_yscale('log')

    if flux is True:
        plt.savefig(os.path.join(pointing.dirname,f'match_{ext.name}_{pointing.name}_fluxes.png'), dpi=dpi)
    else:
        plt.savefig(flux, dpi=dpi)
    plt.close()

def write_to_kvis(pointing, ext, matches, annotate, annotate_nonmatchedcat,sigma_extent):
    '''
    Write the results to a kvis annotation file

    CAUTION: KVIS uses the semimajor and semiminor axes for the Ellipse
             Bmaj, Bmin FWHM needs to be divided by a factor of 2

    '''

    # here define the source sizes and match to the semiminor and semimajor axis
    convert_factor_FWHM_to_sigma_extend  = sigma_extent / (2*np.sqrt(2*np.log(2)))
    FWHMtosemimajmin_factor              = 0.5  * convert_factor_FWHM_to_sigma_extend


    match_ext_lines = []
    match_int_lines = []
    non_match_ext_lines = []
    for i, match in enumerate(matches):
        if len(match) > 0:
            source = ext.sources[i]
            toprt = f'ELLIPSE {source.RA:.6f} {source.DEC:.6f} {source.Maj*FWHMtosemimajmin_factor:.6f} {source.Min*FWHMtosemimajmin_factor:.6f} {source.PA:.4f} \n'
            match_ext_lines.append(toprt)
            for ind in match:
                source = pointing.sources[ind]
                toprt = f'ELLIPSE {source.RA:.6f} {source.DEC:.6f} {source.Maj*FWHMtosemimajmin_factor:.6f} {source.Min*FWHMtosemimajmin_factor:.6f} {source.PA:.4f} \n'
                match_int_lines.append(toprt)
        else:
            source = ext.sources[i]
            toprt = f'ELLIPSE {source.RA:.6f} {source.DEC:.6f} {source.Maj*FWHMtosemimajmin_factor:.6f} {source.Min*FWHMtosemimajmin_factor:.6f} {source.PA:.4f} \n'
            non_match_ext_lines.append(toprt)

    non_matches = np.setdiff1d(np.arange(len(pointing.sources)), np.concatenate(matches).ravel())
    non_match_int_lines = []
    for i in non_matches:
        source = pointing.sources[i]
        toprt = f'ELLIPSE {source.RA:.6f} {source.DEC:.6f} {source.Maj*FWHMtosemimajmin_factor:.6f} {source.Min*FWHMtosemimajmin_factor:.6f} {source.PA:.4f} \n'
        non_match_int_lines.append(toprt)

    if annotate is True:
        outputfilename = os.path.join(pointing.dirname,f'match_{ext.name}_{pointing.name}.ann')
    else:
        outputfilename = annotate

    kvisfile = open(outputfilename,'w')
    kvisfile.writelines('# Annotation file used for KVIS\n')
    kvisfile.writelines('# \n')

    kvisfile.writelines('# Catalogues: '+ext.name+' and '+pointing.name+' \n')
    kvisfile.writelines('# \n')

    kvisfile.writelines('COORD W\n')
    kvisfile.writelines('PA SKY\n')
    kvisfile.writelines('FONT hershey14\n')

    # Write different sources with different colors
    kvisfile.writelines('# Matched sources from external catalog\n')
    kvisfile.writelines('# \n')
    kvisfile.writelines('COLOR BLUE\n')
    for line in match_ext_lines:
        kvisfile.writelines(line)
    kvisfile.writelines('# Matched sources from internal catalog\n')
    kvisfile.writelines('# \n')
    kvisfile.writelines('COLOR RED\n')
    for line in match_int_lines:
        kvisfile.writelines(line)

    if annotate_nonmatchedcat:

        kvisfile.writelines('# Non matched sources from external catalog\n')
        kvisfile.writelines('# \n')
        kvisfile.writelines('COLOR WHITE\n')
        for line in non_match_ext_lines:
            kvisfile.writelines(line)

        kvisfile.writelines('# Non matched sources from internal catalog\n')
        kvisfile.writelines('# \n')
        kvisfile.writelines('COLOR GREEN\n')
        for line in non_match_int_lines:
            kvisfile.writelines(line)

    kvisfile.close()

def write_to_catalog(pointing, ext, matches, output):
    '''
    Write the matched catalogs to a fits file
    '''
    ext.cat['idx'] = np.arange(len(ext.cat))
    pointing.cat['idx'] = np.inf

    for i, match in enumerate(matches):
        for j in match:
            pointing.cat[j]['idx'] = i

    out = join(ext.cat, pointing.cat, keys='idx')

    if output is True:
        filename = os.path.join(pointing.dirname, f'match_{ext.name}_{pointing.name}.fits')
        out.write(filename, overwrite=True, format='fits')
    else:
        out.write(output, overwrite=True, format='fits')


def write_info(pointing,ext,match_info,output):
    """
    wrte the information into a json file
    """

    filename = os.path.join(pointing.dirname, f'match_{ext.name}_{pointing.name}_info.json')

    # Write JSON file
    with open(filename, 'w') as outfile:
            json.dump(match_info,outfile,
                                indent=4, sort_keys=True,
                                separators=(',', ': '), ensure_ascii=False,cls=NumpyEncoder)


    # here just print out the single source astrometry
    # for the full and single matches
    with open(filename, 'r') as json_file:
        matchinfofile = json.load(json_file)
        print('FULL-DEC ',matchinfofile['offset']['stats']['dDEC']['Full'])
        print('FULL-RA ',matchinfofile['offset']['stats']['dRA']['Full'])
        print('Single-DEC ',matchinfofile['offset']['stats']['dDEC']['1'])
        print('Single-RA ',matchinfofile['offset']['stats']['dRA']['1'])
        print('FLUX MATCHES ',matchinfofile['fluxes']['stats'])

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


def Ellipse_skyprojection(ra,dec,Bmaj,Bmin,PA,header=None):
    """
    Provide real pixel values for deprojected Ellipse in
    tha tangent plane

    CAUTION definition in matplotlib is 
    width is horizointal axis, height vertical axis, angle is anti-clockwise
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

        # CAUTION the IMAGE has a reverse sense of RA so if the 
        # increment is negative we need to compensate 
        #
        PA_sense = -1
        if header['CDELT1'] < 0:
            PA_sense = 1

        Ellipse_tangent_plane_pix = Ellipse(source_centre_position_pix_xy,height=degtopix*Bmaj,width=degtopix*Bmin,angle=PA_sense*PA).get_verts()
        Ellipse_Sky_deg           = WCS.utils.pixel_to_skycoord(Ellipse_tangent_plane_pix[:,0],Ellipse_tangent_plane_pix[:,1],wcs)
        Ellipse_Sky_deg_reshaped  = np.column_stack((Ellipse_Sky_deg.ra.deg,Ellipse_Sky_deg.dec.deg))
        Ellipse_SKY               = Ellipse_Sky_deg_reshaped 

    else:
        Ellipse_SKY = Ellipse([ra,dec],height=Bmaj,width=Bmin,angle=-PA).get_verts()
        
    return Ellipse_SKY

def Ellipse_RA_check(radec):
    """
    split the polygons into sub-polygons to be checked
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


def main():

    parser = new_argument_parser()
    args = parser.parse_args()

    pointing = args.pointing
    ext_cat = args.ext_cat
    fluxtype = args.fluxtype
    dpi = args.dpi

    astro = args.astro
    flux = args.flux
    plot = args.plot
    alpha = args.alpha
    output = args.output
    annotate = args.annotate
    annotate_nonmatchedcat = args.annotate_nonmatchedcat

    sigma_extent      = args.match_sigma_extend
    addsearchdistance = args.addsearchdistance

    pointing_cat = Table.read(pointing)
    pointing = Pointing(pointing_cat, pointing)

    if ext_cat == 'NVSS':
        ext_table = pointing.query_NVSS()
        ext_catalog = ExternalCatalog(ext_cat, ext_table, pointing.center)
    elif ext_cat == 'SUMSS':
        ext_table = pointing.query_SUMSS()
        ext_catalog = ExternalCatalog(ext_cat, ext_table, pointing.center)
    elif ext_cat == 'FIRST':
        ext_table = pointing.query_FIRST()
        ext_catalog = ExternalCatalog(ext_cat, ext_table, pointing.center)
    elif os.path.exists(ext_cat):
        ext_table = Table.read(ext_cat)
        if 'bdsfcat' in ext_cat:
            ext_catalog = Pointing(ext_table, ext_cat)
        else:
            ext_catalog = ExternalCatalog(ext_cat, ext_table, pointing.center)
    else:
        print('Invalid input table!')
        exit()

    if len(ext_table) == 0:
        print('No sources were found to match, most likely the external catalog has no coverage here')
        exit()

    matches      = match_catalogs(pointing, ext_catalog,sigma_extent,addsearchdistance)

    matches_info = info_match(pointing, ext_catalog, matches,fluxtype,alpha,output)

    matches_info['INPUT'] = {}
    matches_info['INPUT']['alpha'] = alpha 
    matches_info['INPUT']['match_sigma_extend'] = sigma_extent 
    matches_info['INPUT']['addsearchdistance'] = addsearchdistance 


    if plot:
        plot_catalog_match(pointing, ext_catalog, matches, plot, dpi)
    if astro:
        plot_astrometrics(pointing, ext_catalog, matches, astro, dpi)
    if flux:
        plot_fluxes(pointing, ext_catalog, matches, fluxtype, flux, alpha, dpi)
    if output:
        write_to_catalog(pointing, ext_catalog, matches, output)
        write_info(pointing,ext_catalog,matches_info,output)

    if annotate: 
        write_to_kvis(pointing, ext_catalog, matches, annotate,annotate_nonmatchedcat,sigma_extent)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("pointing",
                        help="""Pointing catalog made by PyBDSF.""")
    parser.add_argument("ext_cat", default="NVSS",
                        help="""External catalog to match to, choice between
                                NVSS, SUMMS, FIRST or a file. If the external
                                catalog is a PyBDSF catalog, make sure the filename
                                has 'bdsfcat' in it. If a different catalog, the
                                parsets/extcat.json file must be used to specify its
                                details (default NVSS).""")
    parser.add_argument('-d', '--dpi', default=300,
                        help="""DPI of the output images (default = 300).""")
    parser.add_argument("--astro", nargs="?", const=True,
                        help="""Plot the astrometric offset of the matches,
                                optionally provide an output filename
                                (default = don't plot astrometric offsets).""")
    parser.add_argument("--flux", nargs="?", const=True,
                        help="""Plot the flux ratios of the matches,
                                optionally provide an output filename
                                (default = don't plot flux ratio).""")
    parser.add_argument("--plot", nargs="?", const=True,
                        help="""Plot the field with the matched ellipses,
                                optionally provide an output filename
                                (default = don't plot the matched ellipses).""")
    parser.add_argument("--fluxtype", default="Total",
                        help="""Whether to use Total or Peak flux for determining
                                the flux ratio (default = Total).""")
    parser.add_argument("--alpha", default=0.8,type=float,
                        help="""The spectral slope to assume for calculating the
                                flux ratio, where Flux_1 = Flux_2 * (freq_1/freq_2)^-alpha
                                (default = 0.8)""")
    parser.add_argument("--output", nargs="?", const=True,
                        help="""Output the result of the matching into a catalog,
                                optionally provide an output filename
                                (default = don't output a catalog).""")
    parser.add_argument("--annotate", nargs="?", const=True,
                        help="""Output the result of the matching into a kvis
                                annotation file, optionally provide an output filename
                                (default = don't output a catalog).""")
    parser.add_argument("--annotate_nonmatchedcat", action='store_true',default=False,
                        help="""Annotation file will include the non-macthed catalouge sources 
                                (default = don't show).""")

    parser.add_argument("--addsearchdistance", default=0,type=float,
                        help="""increase the search distance for matching default = 0 deg""")

    parser.add_argument("--match_sigma_extend", default=3,type=float,
                        help="""use sigma to increase the source extend, 3 sigma would be 1.27398 times the FWHM""")


    return parser

if __name__ == '__main__':
    main()
