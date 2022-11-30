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

import matplotlib
matplotlib.use('Agg')
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from shapely.geometry import Polygon

import kvis_write_lib as kvis
import searchcats as sc
import helpers

gc.enable()

class SourceEllipse:

    def __init__(self, catalog_entry, column_dict):
        self.RA  = catalog_entry[column_dict['ra']]
        self.DEC = catalog_entry[column_dict['dec']]
        self.Maj = catalog_entry[column_dict['majax']]
        self.Min = catalog_entry[column_dict['minax']]
        self.PA  = catalog_entry[column_dict['pa']]

        if column_dict['peak_flux']:
            self.PeakFlux = catalog_entry[column_dict['peak_flux']]
        if column_dict['total_flux']:
            self.IntFlux = catalog_entry[column_dict['total_flux']]

        self.skycoord = SkyCoord(self.RA, self.DEC, unit='deg',frame='icrs')

    def match(self, ra_list, dec_list, maj_list, min_list, pa_list, sigma_extent, search_dist, header, overlap_percentage):
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
        # in order to obtain a source extend we use the 3 sigma extent
        #
        # https://ned.ipac.caltech.edu/level5/Leo/Stats2_3.html
        #
        FWHM_to_sigma_extent = sigma_extent / (2*np.sqrt(2*np.log(2)))

        # Check if sources match within Bmaj/2 boundaries
        # these source could match or not this needs to
        # be checked
        #
        # CAUTION: self is related to the external catalogue
        #
        maj_match = sky_separation < FWHM_to_sigma_extent * deprojection_factor * (self.Maj/2. + maj_list/2.) + search_dist

        # Check if sources match within the smallest source extend
        # these are the source we do not check further
        #
        get_minor_ext = self.Min < min_list
        minor_ext     = np.where(get_minor_ext, self.Min, min_list) 
        #
        min_match     = sky_separation.value <=  FWHM_to_sigma_extent *  minor_ext

        # Check out the source between the maj_match and min_match boundaries
        #
        if max(np.array(np.where(np.logical_xor(min_match,maj_match))).shape) > 0:

            msource_idx = np.where(np.logical_xor(min_match,maj_match))[0].flatten()
            for s in msource_idx:
                check_source = helpers.ellipse_skyprojection(self.RA,self.DEC,
                                                     FWHM_to_sigma_extent*self.Maj,
                                                     FWHM_to_sigma_extent*self.Min,
                                                     self.PA, header)

                to_sources   = helpers.ellipse_skyprojection(ra_list[s],dec_list[s],
                                                     FWHM_to_sigma_extent*maj_list[s]+search_dist,
                                                     FWHM_to_sigma_extent*min_list[s]+search_dist,
                                                     pa_list[s], header)

                # this is to check if sources are crossing the 360 deg in RA
                #
                ra_check_s       = abs(np.diff(check_source[:,0])) > 300
                ra_check_s_where = np.where(ra_check_s)[0]

                ra_to_s       = abs(np.diff(to_sources[:,0])) > 300
                ra_to_s_where = np.where(ra_to_s)[0]

                if max(ra_check_s_where.shape) > 0 or max(ra_to_s_where.shape) > 0:

                    do_they_overlap       = False

                    split_check_s_source  = helpers.ellipse_RA_check(check_source)
                    split_to_s_sources    = helpers.ellipse_RA_check(to_sources)

                    for a in range(len(split_check_s_source)):
                        for b in range(len(split_to_s_sources)):
                            do_they_overlap_split = np.invert(Polygon(split_check_s_source[a]).intersection(Polygon(split_to_s_sources[b])).is_empty)

                            #
                            # evaluate the sub components of the sources
                            #
                            if do_they_overlap_split == True:
                                
                                intersection_area = Polygon(split_check_s_source[a]).intersection(Polygon(split_to_s_sources[b])).area
                                check_source_area = Polygon(split_check_s_source[a]).area
                                to_sources_area   = Polygon(split_to_s_sources[b]).area

                                if (intersection_area != check_source_area) and  (intersection_area != to_sources_area):

                                    areas_percentage = [intersection_area/check_source_area, intersection_area/to_sources_area]

                                    if max(areas_percentage) * 100 > overlap_percentage:
                                        do_they_overlap = True

                                del intersection_area, check_source_area, to_sources_area

                            #
                            del do_they_overlap_split
                            gc.collect()

                else:
                    # this is a realy cool thing
                    # use vertices to a Polygon and shapely to check if they intersect
                    # https://gis.stackexchange.com/questions/243459/drawing-ellipse-with-shapely/243462#243462

                    do_they_overlap = np.invert(Polygon(check_source).intersection(Polygon(to_sources)).is_empty)
                    #

                    #
                    # evaluates the overlap in area 
                    #
                    if do_they_overlap == True:

                        intersection_area = Polygon(check_source).intersection(Polygon(to_sources)).area
                        check_source_area = Polygon(check_source).area
                        to_sources_area   = Polygon(to_sources).area

                        if (intersection_area != check_source_area) and  (intersection_area != to_sources_area):

                            areas_percentage = [intersection_area/check_source_area, intersection_area/to_sources_area]

                            if max(areas_percentage) * 100 < overlap_percentage:
                                do_they_overlap = False

                        del intersection_area, check_source_area, to_sources_area

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

        beam, freq, columns = helpers.get_properties(name, center.ra.deg, center.dec.deg)
        self.BMaj  = beam[0]
        self.BMin  = beam[1]
        self.BPA   = beam[2]
        self.freq  = freq

        if 'quality_flag' in columns:
            if columns['quality_flag']:
                self.cat = self.cat[self.cat[columns['quality_flag']] == 1]
                n_rejected = len(self.cat[self.cat[columns['quality_flag']] == 1])
                if  n_rejected > 0:
                    print(f'Excluding {n_rejected} sources that have a negative quality flag')

        self.sources = [SourceEllipse(source, columns) for source in self.cat]

class Pointing:

    def __init__(self, catalog, filename, survey_name=None, ra_center=None, dec_center=None, fov=None):
        self.dirname = os.path.dirname(filename)

        self.cat = catalog[catalog['Quality_flag'] == 1]
        if len(self.cat) < len(catalog):
            print(f'Excluding {len(catalog) - len(self.cat)} sources that have a negative quality flag')

        columns = {'ra':'RA','dec':'DEC','majax':'Maj','minax':'Min',
                   'pa':'PA','peak_flux':'Peak_flux','total_flux':'Total_flux'}
        self.sources = [SourceEllipse(source, columns) for source in self.cat]

        # Parse meta
        self.header = catalog.meta

        self.BMaj = float(self.header['SF_BMAJ'])*3600 #arcsec
        self.BMin = float(self.header['SF_BMIN'])*3600 #arcsec
        self.BPA = float(self.header['SF_BPA'])

        # Determine frequency axis:
        for i in range(1,5):
            if 'FREQ' in self.header['CTYPE'+str(i)]:
                freq_idx = i
                break
        self.freq = float(self.header['CRVAL'+str(freq_idx)])/1e6 #MHz
        self.dfreq = float(self.header['CDELT'+str(freq_idx)])/1e6 #MHz

        # Determine center and fov, either from input or header
        if ra_center is None:
            self.center = SkyCoord(float(self.header['CRVAL1'])*u.degree,
                                   float(self.header['CRVAL2'])*u.degree)
        else:
            self.center = SkyCoord(ra_center*u.degree,dec_center*u.degree)

        if fov is None:
            dec_fov = abs(float(self.header['CDELT1']))*float(self.header['CRPIX1'])*2
            self.fov = dec_fov/np.cos(self.center.dec.rad) * u.degree
        else:
            self.fov = fov*u.degree

        try:
            self.name = self.header['OBJECT'].replace("'","")
        except KeyError:
            self.name = os.path.basename(filename).rsplit('.',1)[0]

        if survey_name is not None:
            self.survey_name = survey_name
        else:
            self.survey_name = self.name

    def query_NVSS(self):
        '''
        Match the pointing to the NVSS catalog
        '''
        nvsstable = sc.getnvssdata(ra = [self.center.ra.to_string(u.hourangle, sep=' ')],
                                   dec = [self.center.dec.to_string(u.deg, sep=' ')],
                                   offset = 0.5*self.fov.to(u.arcsec))

        nvsstable['Major'].unit = u.arcsec
        nvsstable['Minor'].unit = u.arcsec

        nvsstable['Major'] = nvsstable['Major'].to(u.deg)
        nvsstable['Minor'] = nvsstable['Minor'].to(u.deg)
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

        firsttable['Majax'].unit = u.arcsec
        firsttable['Minax'].unit = u.arcsec

        firsttable['Majax'] = firsttable['Majax'].to(u.deg)
        firsttable['Minax'] = firsttable['Minax'].to(u.deg)
        firsttable['PosAng']  = firsttable['PosAng']

        firsttable['RA']  = firsttable['RA'].to(u.deg)
        firsttable['DEC'] = firsttable['DEC'].to(u.deg)

        firsttable['Peak flux'] /= 1e3 #convert to Jy
        firsttable['Int flux'] /= 1e3 #convert to Jy

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

        sumsstable['Fit_Major_Axis'].unit = u.arcsec
        sumsstable['Fit_Minor_Axis'].unit = u.arcsec

        sumsstable['Fit_Major_Axis'] = sumsstable['Fit_Major_Axis'].to(u.deg)
        sumsstable['Fit_Minor_Axis'] = sumsstable['Fit_Minor_Axis'].to(u.deg)
        sumsstable['Fit_Position_Angle']  = sumsstable['Fit_Position_Angle']

        sumsstable['RA']  = sumsstable['RA'].to(u.deg)
        sumsstable['DEC'] = sumsstable['DEC'].to(u.deg)

        sumsstable['Flux_36_cm']  /= 1e3 #convert to Jy
        sumsstable['Int_Flux_36_cm'] /= 1e3 #convert to Jy

        return sumsstable

    def query_TGSS(self):
        '''
        Match the pointing to the TGSS catalog
        '''
        tgsstable = sc.gettgssdata(central_coord = self.center,
                                   offset = 0.5*self.fov)

        tgsstable['MAJAX']   = tgsstable['MAJAX'].to(u.deg)
        tgsstable['MINAX']   = tgsstable['MINAX'].to(u.deg)
        tgsstable['e_MAJAX'] = tgsstable['e_MAJAX'].to(u.deg)
        tgsstable['e_MINAX'] = tgsstable['e_MINAX'].to(u.deg)

        # Convert to Jy
        tgsstable['Sint']       /= 1e3
        tgsstable['Spk']        /= 1e3
        tgsstable['e_Sint']     /= 1e3
        tgsstable['e_Spk']      /= 1e3
        tgsstable['Island_RMS'] /= 1e3

        return tgsstable

    def query_RACS(self):
        '''
        Match the pointing to the RACS catalog
        '''
        racstable = sc.getracsdata(central_coord = self.center,
                                   offset = 0.5*self.fov)

        racstable['maj_axis']   = racstable['maj_axis'].to(u.deg)
        racstable['min_axis']   = racstable['min_axis'].to(u.deg)
        racstable['e_maj_axis'] = racstable['e_maj_axis'].to(u.deg)
        racstable['e_min_axis'] = racstable['e_min_axis'].to(u.deg)

        racstable['dc_maj']   = racstable['dc_maj'].to(u.deg)
        racstable['dc_min']   = racstable['dc_min'].to(u.deg)
        racstable['e_dc_maj'] = racstable['e_dc_maj'].to(u.deg)
        racstable['e_dc_min'] = racstable['e_dc_min'].to(u.deg)

        # Convert to Jy
        racstable['total_flux_source']          /= 1e3
        racstable['peak_flux']                  /= 1e3
        racstable['e_total_flux_source_pybdsf'] /= 1e3
        racstable['e_total_flux_source']        /= 1e3
        racstable['e_peak_flux']                /= 1e3
        racstable['noise']                      /= 1e3

        return racstable

def match_catalogs(pointing, ext, sigma_extent, search_dist, overlap_percentage):
    '''
    Match the sources of the chosen external catalog to the sources in the pointing
    '''
    print(f'Matching {len(ext.sources)} sources in {ext.name} to {len(pointing.cat)} sources in the pointing')

    matches = []
    for source in ext.sources:
        matches.append(source.match(pointing.cat['RA'], pointing.cat['DEC'],
                                    pointing.cat['Maj'],pointing.cat['Min'],
                                    pointing.cat['PA'],
                                    sigma_extent, search_dist,
                                    helpers.make_header(pointing.header), 
                                    overlap_percentage))

    return matches

def info_match(pointing, ext, matches, fluxtype, alpha, output):
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
                # Determine the offset
                dra, ddec = ext.sources[i].skycoord.spherical_offsets_to(pointing.sources[m].skycoord)
                dRA.append(dra.arcsec)
                dDEC.append(ddec.arcsec)
                # Determine the matches
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
    match_alpha = []
    if fluxtype == 'Total':
        for i, match in enumerate(matches):
            if len(match) > 0:
                ext_flux.append(ext.sources[i].IntFlux)
                int_flux.append(np.sum([pointing.sources[m].IntFlux for m in match]))

                flux_ratio = ext.sources[i].IntFlux/np.sum([pointing.sources[m].IntFlux for m in match])
                match_alpha.append(np.log(flux_ratio)/np.log(ext.freq/pointing.freq))

                source_coord = SkyCoord(ext.sources[i].RA, ext.sources[i].DEC, unit='deg')
                separation.append(source_coord.separation(pointing.center).deg)
                n_matches.append(len(match))
    elif fluxtype == 'Peak':
        for i, match in enumerate(matches):
            if len(match) > 0:
                ext_flux.append(ext.sources[i].PeakFlux)
                int_flux.append(np.sum([pointing.sources[m].PeakFlux for m in match]))

                flux_ratio = ext.sources[i].PeakFlux/np.sum([pointing.sources[m].PeakFlux for m in match])
                match_alpha.append(np.log(flux_ratio)/np.log(ext.freq/pointing.freq))

                source_coord = SkyCoord(ext.sources[i].RA, ext.sources[i].DEC, unit='deg')
                separation.append(source_coord.separation(pointing.center).deg)
                n_matches.append(len(match))
    else:
        print(f'Invalid fluxtype {fluxtype}, choose between Total or Peak flux')
        sys.exit()

    match_info['fluxes'][fluxtype] = {}
    match_info['fluxes'][fluxtype]['ext_flux']    = ext_flux
    match_info['fluxes'][fluxtype]['int_flux']    = int_flux
    match_info['fluxes'][fluxtype]['separation']  = separation
    match_info['fluxes'][fluxtype]['n_matches']   = n_matches
    match_info['fluxes'][fluxtype]['match_alpha'] = match_alpha

    # Scale flux density to proper frequency
    ext_flux_corrected = np.array(ext_flux) * (pointing.freq/ext.freq)**alpha
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
        outfile = os.path.join(pointing.dirname,f'match_{ext.name}_{pointing.name}_shapes.png')
    else:
        outfile = plot

    print(f"--> Saving plot of source ellipses '{outfile}'")
    plt.savefig(astro, dpi=dpi, bbox_inches='tight')

    plt.close()

def plot_astrometrics(match_info, pointing, ext, astro, dpi):
    '''
    Plot astrometric offsets of sources to the reference catalog
    '''
    cmap = colors.ListedColormap(["navy", "crimson", "limegreen", "gold"])
    norm = colors.BoundaryNorm(np.arange(0.5, 5, 1), cmap.N)

    fig = plt.figure(figsize=(8,8))
    ax  = plt.subplot()
    ax.axis('equal')

    sc = ax.scatter(match_info['offset']['dRA'], match_info['offset']['dDEC'], zorder=2,
                    marker='.', s=5,
                    c=match_info['offset']['n_matches'], cmap=cmap, norm=norm)

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
                           label=f'{pointing.survey_name} beam')

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
    dDEC_stats = match_info['offset']['stats']['dDEC']['Full']
    ax.axhline(dDEC_stats['median'],-xmax_abs,xmax_abs, color='grey', linestyle='dashed', zorder=1)
    ax.axhline(dDEC_stats['median']-dDEC_stats['std'],-xmax_abs,xmax_abs, color='grey', linestyle='dotted', zorder=1)
    ax.axhline(dDEC_stats['median']+dDEC_stats['std'],-ymax_abs,ymax_abs, color='grey', linestyle='dotted', zorder=1)
    ax.axhspan(dDEC_stats['median']-dDEC_stats['std'],dDEC_stats['median']+dDEC_stats['std'], alpha=0.2, color='grey')
    ax.annotate(f"dDEC = {dDEC_stats['median']:.2f}+-{dDEC_stats['std']:.2f}",
                xy=(0.05,0.925), xycoords='axes fraction', fontsize=8)

    # Determine mean and standard deviation of points in RA
    dRA_stats = match_info['offset']['stats']['dRA']['Full']
    ax.axvline(dRA_stats['median'],-ymax_abs,ymax_abs, color='grey', linestyle='dashed', zorder=1)
    ax.axvline(dRA_stats['median']-dRA_stats['std'],-xmax_abs,xmax_abs, color='grey', linestyle='dotted', zorder=1)
    ax.axvline(dRA_stats['median']+dRA_stats['std'],-ymax_abs,ymax_abs, color='grey', linestyle='dotted', zorder=1)
    ax.axvspan(dRA_stats['median']-dRA_stats['std'],dRA_stats['median']+dRA_stats['std'], alpha=0.2, color='grey')
    ax.annotate(f"dRA = {dRA_stats['median']:.2f}+-{dRA_stats['std']:.2f}",
                xy=(0.05,0.90), xycoords='axes fraction', fontsize=8)

    ax.set_title(f"Astrometric offset of {len(match_info['offset']['dRA'])} sources")
    ax.set_xlabel('RA offset (arcsec)')
    ax.set_ylabel('DEC offset (arcsec)')
    ax.legend(loc='upper right')

    if astro is True:
        outfile = os.path.join(pointing.dirname,f'match_{ext.name}_{pointing.name}_astrometrics.png')
    else:
        outfile = astro

    print(f"--> Saving astrometry plot '{outfile}'")
    plt.savefig(outfile, dpi=dpi)
    plt.close()

def plot_fluxes(match_info, pointing, ext, fluxtype, flux, dpi):
    '''
    Plot flux offsets of sources to the reference catalog
    '''
    flux_matches = match_info['fluxes'][fluxtype]

    cmap = colors.ListedColormap(["navy", "crimson", "limegreen", "gold"])
    norm = colors.BoundaryNorm(np.arange(0.5, 5, 1), cmap.N)

    fig, ax = plt.subplots()

    # Log scale before plotting
    if np.log10(flux_matches['dFlux'].max()/flux_matches['dFlux'].min()) > 2:
        ax.set_yscale('log')

    sc = ax.scatter(flux_matches['separation'], 
                    flux_matches['dFlux'],
                    marker='.', s=5,
                    c=flux_matches['n_matches'], cmap=cmap, norm=norm)

    # Determine running median
    medians_x, medians_y, stds_y= helpers.runningmedian(np.array(flux_matches['separation']),
                                                        np.array(flux_matches['dFlux']), 
                                                        window=int(len(flux_matches['dFlux'])/5), 
                                                        stepsize=1)
    ax.plot(medians_x, medians_y, color='k')

    # Add colorbar to set matches
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(sc, cax=cax)

    cbar.set_ticks([1,2,3,4])
    cax.set_yticklabels(['1','2','3','>3'])
    cax.set_title('Matches')

    ax.set_title(f"Flux ratio of {len(flux_matches['dFlux'])} sources")
    ax.set_xlabel('Distance from pointing center (degrees)')
    ax.set_ylabel(f'Flux ratio ({fluxtype} flux)')

    if flux is True:
        outfile = os.path.join(pointing.dirname,f'match_{ext.name}_{pointing.name}_fluxes.png')
    else:
        outfile = flux

    print(f"--> Saving flux ratio plot '{outfile}'")
    plt.savefig(outfile, dpi=dpi)
    plt.close()

def write_to_catalog(pointing, ext, matches, match_info, fluxtype, output):
    '''
    Write the matched catalogs to a fits file
    '''
    ext.cat['Match_alpha'] = 0.0
    ext.cat['idx'] = np.arange(len(ext.cat))
    pointing.cat['idx'] = np.inf

    match_alpha_iter = iter(match_info['fluxes'][fluxtype]['match_alpha'])
    for i, match in enumerate(matches):
        if len(match) > 0:
            ext.cat[i]['Match_alpha'] = next(match_alpha_iter)
            for j in match:
                pointing.cat[j]['idx'] = i

    out = join(ext.cat, pointing.cat, keys='idx', table_names=[ext.name,pointing.survey_name])

    if output is True:
        outfile = os.path.join(pointing.dirname, f'match_{ext.name}_{pointing.name}.fits')
    else:
        outfile = output

    print(f"--> Saving output FITS catalog '{outfile}'")
    out.write(outfile, overwrite=True, format='fits')

def write_info(pointing, ext, match_info, output):
    """
    Write the information into a json file
    """
    filename = os.path.join(pointing.dirname, f'match_{ext.name}_{pointing.name}_info.json')

    # Write JSON file
    print(f"--> Saving info json file '{filename}'")
    with open(filename, 'w') as outfile:
            json.dump(match_info,outfile,
                      indent=4, sort_keys=True,
                      separators=(',', ': '),
                      ensure_ascii=False,
                      cls=NumpyEncoder)

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
    survey_name = args.survey_name
    annotate = args.annotate
    annotate_nonmatched = args.annotate_nonmatched
    ra_center = args.ra_center
    dec_center = args.dec_center
    fov = args.fov

    sigma_extent   = args.match_sigma_extent
    search_dist    = args.search_dist/3600 # to degrees
    source_overlap = args.source_overlap_percent

    pointing_cat = Table.read(pointing)
    pointing = Pointing(pointing_cat, pointing, survey_name, ra_center, dec_center, fov)

    if ext_cat == 'NVSS':
        ext_table = pointing.query_NVSS()
        ext_catalog = ExternalCatalog(ext_cat, ext_table, pointing.center)
    elif ext_cat == 'SUMSS':
        ext_table = pointing.query_SUMSS()
        ext_catalog = ExternalCatalog(ext_cat, ext_table, pointing.center)
    elif ext_cat == 'FIRST':
        ext_table = pointing.query_FIRST()
        ext_catalog = ExternalCatalog(ext_cat, ext_table, pointing.center)
    elif ext_cat == 'TGSS':
        ext_table = pointing.query_TGSS()
        ext_catalog = ExternalCatalog(ext_cat, ext_table, pointing.center)
    elif ext_cat == 'RACS':
        ext_table = pointing.query_RACS()
        ext_catalog = ExternalCatalog(ext_cat, ext_table, pointing.center)
    elif os.path.exists(ext_cat):
        ext_table = Table.read(ext_cat)
        if 'bdsfcat' in ext_cat:
            ext_catalog = Pointing(ext_table, ext_cat, ra_center=ra_center, dec_center=dec_center, fov=fov)
        else:
            ext_catalog = ExternalCatalog(ext_cat, ext_table, pointing.center)
    else:
        print('Invalid input table!')
        exit()

    if len(ext_table) == 0:
        print('No sources were found to match, most likely the external catalog has no coverage here')
        exit()

    matches      = match_catalogs(pointing, ext_catalog, sigma_extent, search_dist, source_overlap)
    matches_info = info_match(pointing, ext_catalog, matches, fluxtype, alpha, output)

    matches_info['INPUT'] = {}
    matches_info['INPUT']['alpha'] = alpha 
    matches_info['INPUT']['match_sigma_extend'] = sigma_extent 
    matches_info['INPUT']['search_dist'] = search_dist
    matches_info['INPUT']['source_overlap_percent'] = source_overlap

    if plot:
        plot_catalog_match(pointing, ext_catalog, matches, plot, dpi)
    if astro:
        plot_astrometrics(matches_info, pointing, ext_catalog, astro, dpi)
    if flux:
        plot_fluxes(matches_info, pointing, ext_catalog, fluxtype, flux, dpi)
    if output:
        write_to_catalog(pointing, ext_catalog, matches, matches_info, fluxtype, output)
        write_info(pointing, ext_catalog, matches_info, output)

    if annotate == 'kvis':
        kvis.matches_to_kvis(pointing, ext_catalog, matches, annotate, annotate_nonmatched, sigma_extent)
    if annotate == 'ds9':
        kvis.matches_to_ds9(pointing, ext_catalog, matches, annotate, annotate_nonmatched, sigma_extent)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("pointing", type=str,
                        help="""Pointing catalog made by PyBDSF.""")
    parser.add_argument("ext_cat", default="NVSS", type=str,
                        help="""External catalog to match to, choice between
                                NVSS, SUMMS, FIRST, TGSS, RACS or a file. If the external
                                catalog is a PyBDSF catalog, make sure the filename
                                has 'bdsfcat' in it. If a different catalog, the
                                parsets/extcat.json file must be used to specify its
                                details (default NVSS).""")
    parser.add_argument("--match_sigma_extent", default=3, type=float,
                        help="""The matching extent used for sources, defined in sigma.
                                Any sources within this extent will be considered matches.
                                (default = 3 sigma = 1.27398 times the FWHM)""")
    parser.add_argument("--search_dist", default=0, type=float,
                        help="""Additional search distance beyond the source size to be
                                used for matching, in arcseconds (default = 0)""")
    parser.add_argument("--source_overlap_percent", default=80, type=float,
                        help="""The percentage is used, of the ratio of size of the intersection 
                                area to size of the individual sources, to fine-tune source matches, 
                                in percentage (default = 80)""")
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
    parser.add_argument("--survey_name", default=None, type=str,
                        help="""Survey name of input catalog to use in matched catalog 
                                columns and plots""")
    parser.add_argument("--fluxtype", default="Total",
                        help="""Whether to use Total or Peak flux for determining
                                the flux ratio (default = Total).""")
    parser.add_argument("--alpha", default=-0.8, type=float,
                        help="""The spectral slope to assume for calculating the
                                flux ratio, where Flux_1 = Flux_2 * (freq_1/freq_2)^alpha
                                (default = -0.8)""")
    parser.add_argument("--output", nargs="?", const=True,
                        help="""Output the result of the matching into a catalog,
                                optionally provide an output filename
                                (default = don't output a catalog).""")
    parser.add_argument("--annotate", nargs="?",
                        help="""Output the result of the matching into an
                                annotation file, either kvis or ds9
                                (default = don't output annotation file).""")
    parser.add_argument("--annotate_nonmatched", action='store_true', default=False,
                        help="""Annotation file will include the non-macthed catalogue sources 
                                (default = don't show).""")
    parser.add_argument("--ra_center", default=None, type=float,
                        help="""Assumed centre (RA, degrees) for matching to external catalogues.
                                (default = use CRVAL1/2 from image header).""")
    parser.add_argument("--dec_center", default=None, type=float,
                        help="""Assumed centre (DEC, degrees) for matching to external catalogues.
                                (default = use CRVAL1/2 from image header). """)
    parser.add_argument("--fov", default=None, type=float,
                        help="""Assumed FOV (degrees) for matching to external catalogues.
                                (default = use FOV of input image).""")        
    parser.add_argument('-d', '--dpi', default=300,
                        help="""DPI of the output images (default = 300).""")

    return parser

if __name__ == '__main__':
    main()
