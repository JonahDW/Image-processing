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

    def __init__(self, catalog_entry):
        self.RA  = catalog_entry['ra']
        self.DEC = catalog_entry['dec']
        self.Maj = catalog_entry['majax']
        self.Min = catalog_entry['minax']
        self.PA  = catalog_entry['pa']

        if 'peak_flux' in catalog_entry.colnames:
            self.PeakFlux = catalog_entry['peak_flux']
        if 'total_flux' in catalog_entry.colnames:
            self.IntFlux = catalog_entry['total_flux']

        self.skycoord = SkyCoord(self.RA, self.DEC, unit='deg',frame='icrs')

    def match(self, ra_list, dec_list, maj_list, min_list, pa_list, sigma_extent, search_dist, header, overlap_percentage):
        """
        Match the ellipse with a (list of) source(s)

        Keyword arguments:
        ra_list (array)  -- Right ascension of sources
        dec_list (array) -- Declination of sources
        maj_list (array) -- Major axis of FWHM of sources
        min_list (array) -- Minor axis of FWHM of sources
        pa_list (array)  -- Position angle of sources
        sigma_extent (float) -- Source extent to match, as factor of sigma
        search_dist (float)  -- Additional search radius, in degrees
        header (wcsheader)   -- Image header
        overlap_percentage (int) -- Minimum percentage overlap between ellipses 
                                    required for two sources to be considered a match

        Returns:
        List of matches for source ellipse
        """
        offset_coord   = SkyCoord(ra_list, dec_list, unit='deg')
        sky_separation = self.skycoord.separation(offset_coord)

        # this factor just increases the number of sources to check
        # so no harm to the process, it is of the order of sub-arcsec
        dra, ddec           = self.skycoord.spherical_offsets_to(offset_coord)
        deprojection_factor = (np.sqrt( (dra)**2 + (ddec)**2 ) / sky_separation).value

        # The Gaussians are given in FWHM Bmaj, Bmin
        # in order to obtain a source extend we use the 3 sigma extent
        FWHM_to_sigma_extent = sigma_extent / (2*np.sqrt(2*np.log(2)))

        # Check if sources match within Bmaj/2 boundaries
        maj_match = sky_separation.value < FWHM_to_sigma_extent * deprojection_factor * (self.Maj/2. + maj_list/2.) + search_dist

        # Check if sources match within the smallest source extend
        # These are the source we do not check further
        get_minor_ext = self.Min < min_list
        minor_ext     = np.where(get_minor_ext, self.Min, min_list) 
        min_match     = sky_separation.value <=  FWHM_to_sigma_extent *  minor_ext

        # Check out the sources between the maj_match and min_match boundaries
        if max(np.array(np.where(np.logical_xor(min_match,maj_match))).shape) > 0:

            msource_idx = np.where(np.logical_xor(min_match,maj_match))[0].flatten()
            for s in msource_idx:
                check_source = helpers.ellipse_skyprojection(self.RA,self.DEC,
                                                     FWHM_to_sigma_extent*self.Maj,
                                                     FWHM_to_sigma_extent*self.Min,
                                                     self.PA, header)

                to_sources = helpers.ellipse_skyprojection(ra_list[s],dec_list[s],
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
                            if len(split_check_s_source[a]) > 1 and len(split_to_s_sources[b]) > 1:
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
        """
        Convert the ellipse to a matplotlib artist

        CAUTION: Definition in matplotlib is 
        width is horizontal axis, height vertical axis, angle is anti-clockwise
        in order to match the astronomical definition PA from North clockwise
        height is major axis, width is minor axis and angle is -PA
        """
        return Ellipse(xy = (self.RA, self.DEC),
                        width = self.Min,
                        height = self.Maj,
                        angle = -self.PA)

class ExternalCatalog:

    def __init__(self, name, catalog, center):
        self.name = Path(name).stem
        self.cat = catalog

        # Get properties
        beam, freq, column_dict, unit_dict = helpers.get_properties(name, center.ra.deg, center.dec.deg)
        self.BMaj  = beam[0]
        self.BMin  = beam[1]
        self.BPA   = beam[2]
        self.freq  = freq

        # If quality flag is present, exclude flagged sources
        if column_dict['quality_flag']:
            self.cat = self.cat[self.cat[column_dict['quality_flag']] == 1]
            n_rejected = len(self.cat[self.cat[column_dict['quality_flag']] == 0])
            if  n_rejected > 0:
                print(f'Excluding {n_rejected} sources that have a negative quality flag')

        # Create catalog for creating source ellipses
        self.reduced_cat = Table()
        for column in column_dict:
            self.reduced_cat[column] = self.cat[column_dict[column]]
            if column in unit_dict:
                self.reduced_cat[column].unit = u.Unit(unit_dict[column])

        # Sort out column units
        self.reduced_cat['majax'] = self.reduced_cat['majax'].to(u.deg)
        self.reduced_cat['minax'] = self.reduced_cat['minax'].to(u.deg)
        self.reduced_cat['ra']    = self.reduced_cat['ra'].to(u.deg)
        self.reduced_cat['dec']   = self.reduced_cat['dec'].to(u.deg)
        if column_dict['peak_flux']:
            self.reduced_cat['peak_flux'] = self.reduced_cat['peak_flux'].to(u.Jy/u.beam)
        if column_dict['total_flux']:
            self.reduced_cat['total_flux'] = self.reduced_cat['total_flux'].to(u.Jy)

        self.sources = [SourceEllipse(source) for source in self.reduced_cat]

class BDSFCatalog:

    def __init__(self, catalog, filename, survey_name=None, ra_center=None, dec_center=None, fov=None):
        self.dirname = os.path.dirname(filename)
        self.cat = catalog

        # If quality flag is present, exclude flagged sources
        if 'Quality_flag' in catalog.colnames:
            self.cat = catalog[catalog['Quality_flag'] == 1]
            if len(self.cat) < len(catalog):
                print(f'Excluding {len(catalog) - len(self.cat)} sources that have a negative quality flag')

        # Define columns and units for PyBDSF catalog
        column_dict = {'ra':'RA','dec':'DEC','majax':'Maj','minax':'Min',
                       'pa':'PA','peak_flux':'Peak_flux','total_flux':'Total_flux'}
        unit_dict = {'ra':'deg','dec':'deg','majax':'deg','minax':'deg',
                     'pa':'deg','peak_flux':'Jy/beam','total_flux':'Jy'}

        # Create catalog for creating source ellipses and other stuff
        self.reduced_cat = Table()
        for column in column_dict:
            self.reduced_cat[column] = self.cat[column_dict[column]]
            self.reduced_cat[column].unit = u.Unit(unit_dict[column])

        # Sort out column units
        self.reduced_cat['majax'] = self.reduced_cat['majax'].to(u.deg)
        self.reduced_cat['minax'] = self.reduced_cat['minax'].to(u.deg)
        self.reduced_cat['ra'] = self.reduced_cat['ra'].to(u.deg)
        self.reduced_cat['dec'] = self.reduced_cat['dec'].to(u.deg)
        if column_dict['peak_flux']:
            self.reduced_cat['peak_flux'] = self.reduced_cat['peak_flux'].to(u.Jy/u.beam)
        if column_dict['total_flux']:
            self.reduced_cat['total_flux'] = self.reduced_cat['total_flux'].to(u.Jy)
        self.sources = [SourceEllipse(source) for source in self.reduced_cat]

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
            self.uncorrected_fov = dec_fov
        else:
            self.fov = fov*u.degree

        try:
            self.name = self.header['OBJECT'].replace("'","")
        except KeyError:
            self.name = os.path.basename(filename).rsplit('.',1)[0]

        if survey_name is not None:
            self.name = survey_name

    def query_catalog(self, ext_cat_name):
        """
        Query external catalog on the internet

        Keyword arguments:
        ext_cat_name (string) -- Name of or (VO) link to external catalogue

        Returns:
        ext_cat (Table) -- Table of external catalogue
        """

        # Query NVSS
        if ext_cat_name == 'NVSS':
            ext_table = sc.getnvssdata(ra = [self.center.ra.to_string(u.hourangle, sep=' ')],
                                       dec = [self.center.dec.to_string(u.deg, sep=' ')],
                                       offset = 0.5*self.fov.to(u.arcsec))
        # Query SUMSS
        elif ext_cat_name == 'SUMSS':
            if self.center.dec.deg > -29.5:
                print('Catalog outside of SUMSS footprint')
                sys.exit()
            ext_table = sc.getsumssdata(ra = self.center.ra,
                                         dec = self.center.dec,
                                         offset = 0.5*self.fov)
        # Query FIRST
        elif ext_cat_name == 'FIRST':
            ext_table = sc.getfirstdata(ra = [self.center.ra.to_string(u.hourangle, sep=' ')],
                                        dec = [self.center.dec.to_string(u.deg, sep=' ')],
                                        offset = 0.5*self.fov.to(u.arcsec))
        # Query TGSS
        elif ext_cat_name == 'TGSS':
            tgss_url = 'https://vo.astron.nl/tgssadr/q/cone/scs.xml'
            tgsstable = sc.getvodata(tgss_url, 
                                     central_coord = self.center,
                                     offset = 0.5*self.fov)
            # Replace object columns because TGSS is horrible
            for col in tgsstable.colnames:
                if tgsstable[col].dtype == object:
                    tgsstable[col] = tgsstable[col].astype('str')
            ext_table = tgsstable
        # Query RACS-low
        elif ext_cat_name == 'RACS-low':
            ext_table = sc.getracslowdata(central_coord = self.center,
                                          offset = 0.5*self.uncorrected_fov)
        # Query RACS-mid
        elif ext_cat_name == 'RACS-mid':
            racsmid_url='https://casda.csiro.au/casda_vo_tools/scs/racs_mid_sources_v01'
            ext_table = sc.getvodata(racsmid_url, 
                                     central_coord = self.center,
                                     offset = 0.5*self.fov)
        # If name is different, try input as VO url
        else:
            try:
                ext_table = sc.getvodata(ext_cat_name, 
                                         central_coord = self.center,
                                         offset = 0.5*self.fov)
            except SomeError:
                print(f"Nothing found at {ext_cat_name}, make sure you have entered the correct name.")
                sys.exit()

        return ext_table

def match_catalogs(internal, external, extbig, sigma_extent, search_dist, overlap_percentage):
    """
    Match the sources of the chosen external catalog to the sources in the pointing

    Keyword arguments:
    internal (class object) -- Internal catalogue
    external (class object) -- External catalogue
    extbig (bool) -- If True, the external catalogue is 
                     considered to have the biggger beam
    sigma_extent (float) -- Source extent to match, as factor of sigma
    search_dist (float)  -- Additional search radius, in degrees
    header (wcsheader)   -- Image header
    overlap_percentage (int) -- Minimum percentage overlap between ellipses 
                                required for two sources to be considered a match

    Returns:
    matches (list of lists) -- List of matches for each source in catalog with bigger beam
    """
    print(f'Matching {len(external.sources)} sources in {external.name} to {len(internal.cat)} sources in input catalog')

    # Determine which catalog has the bigger beam to match properly
    if extbig:
        big = external
        small = internal
    else:
        big = internal
        small = external

    # Get matches for each source in 'big' catalog
    matches = []
    n_matches = 0
    for source in big.sources:
        matched_sources = source.match(small.reduced_cat['ra'],
                                       small.reduced_cat['dec'],
                                       small.reduced_cat['majax'],
                                       small.reduced_cat['minax'],
                                       small.reduced_cat['pa'],
                                       sigma_extent, search_dist,
                                       helpers.make_header(internal.header),
                                       overlap_percentage)
        n_matches += len(matched_sources)
        matches.append(matched_sources)

    #If no sources were matched, exit
    if n_matches == 0:
        print('No sources were matched, exiting')
        sys.exit()

    return matches

def info_match(internal, external, matches, extbig, fluxtype, alpha):
    """
    Provide information of the matches

    Keyword arguments:
    internal (class object) -- Internal catalogue
    external (class object) -- External catalogue
    matches (list of lists) -- Matched sources between catalogues
    extbig (bool) -- If True, the external catalogue is 
                     considered to have the biggger beam
    fluxtype (string) -- Flux type to compare, either 'Total' or 'Peak'
    alpha (float)     -- Spectral index to assum for flux comparison

    Returns:
    match_info (dict) -- Dictionary of astrometric, flux offsets, and
                         spectral indices, including summary statistics
    """

    # Determine which catalog has the bigger beam to parse matches properly
    if extbig:
        big_sources = external.sources
        small_sources = internal.sources
    else:
        big_sources = internal.sources
        small_sources = external.sources

    match_info = {}

    # Astrometric offsets
    dDEC      = []
    dRA       = []
    n_matches = []
    for i, match in enumerate(matches):
        if len(match) > 0:
            for m in match:
                # Determine the offset
                dra, ddec = big_sources[i].skycoord.spherical_offsets_to(small_sources[m].skycoord)
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

    # Flux offsets
    match_info['fluxes'] = {}

    ext_flux    = []
    int_flux    = []
    separation  = []
    n_matches   = []
    match_alpha = []

    for i, match in enumerate(matches):
        if len(match) > 0:
            # Determine based on chosen flux type
            if fluxtype == 'Total':
                big_flux = big_sources[i].IntFlux
                small_flux = np.sum([small_sources[m].IntFlux for m in match])
            elif fluxtype == 'Peak':
                big_flux = big_sources[i].PeakFlux
                small_flux = np.sum([small_sources[m].PeakFlux for m in match])
            else:
                print(f'Invalid fluxtype {fluxtype}, choose between Total or Peak flux')
                sys.exit()

            # Assign fluxes based on bigger beam
            if extbig:
                ext_flux.append(big_flux)
                int_flux.append(small_flux)
                flux_ratio = big_flux/small_flux
            else:
                ext_flux.append(small_flux)
                int_flux.append(big_flux)
                flux_ratio = small_flux/big_flux

            match_alpha.append(np.log(flux_ratio)/np.log(external.freq/internal.freq))

            source_coord = SkyCoord(big_sources[i].RA, big_sources[i].DEC, unit='deg')
            separation.append(source_coord.separation(internal.center).deg)
            n_matches.append(len(match))

    match_info['fluxes'][fluxtype] = {}
    match_info['fluxes'][fluxtype]['ext_flux']    = ext_flux
    match_info['fluxes'][fluxtype]['int_flux']    = int_flux
    match_info['fluxes'][fluxtype]['separation']  = separation
    match_info['fluxes'][fluxtype]['n_matches']   = n_matches
    match_info['fluxes'][fluxtype]['match_alpha'] = match_alpha

    # Scale flux density to proper frequency
    ext_flux_corrected = np.array(ext_flux) * (internal.freq/external.freq)**alpha
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

def plot_catalog_match(internal, external, matches, extbig, plot, dpi):
    """
    Plot the field with all the matches in it as ellipses

    Keyword arguments:
    internal (class object) -- Internal catalogue
    external (class object) -- External catalogue
    matches (list of lists) -- Matched sources between catalogues
    extbig (bool) -- If True, the external catalogue is 
                     considered to have the biggger beam
    plot (bool or string) -- If string, plot will be written here
    dpi (int) -- DPI of plot
    """

    # Determine which catalog has the bigger beam
    if extbig:
        big_sources = external.sources
        small_sources = internal.sources
    else:
        big_sources = internal.sources
        small_sources = external.sources

    fig = plt.figure(figsize=(20,20))
    ax = plt.subplot()

    # For each big source plot and plot all matches
    for i, match in enumerate(matches):
        big_ell = big_sources[i].to_artist()
        ax.add_artist(big_ell)
        big_ell.set_facecolor('b')
        big_ell.set_alpha(0.5)

        if len(match) > 0:
            for ind in match:
                small_ell = small_sources[ind].to_artist()
                ax.add_artist(small_ell)
                small_ell.set_facecolor('r')
                small_ell.set_alpha(0.5)
        else:
            big_ell.set_facecolor('k')

    non_matches = np.setdiff1d(np.arange(len(small_sources)), np.concatenate(matches).ravel())
    for i in non_matches:
        small_ell = small_sources[i].to_artist()
        ax.add_artist(small_ell)
        small_ell.set_facecolor('g')
        small_ell.set_alpha(0.5)

    ax.set_xlim(internal.center.ra.deg-0.5*internal.fov.value,
                internal.center.ra.deg+0.5*internal.fov.value)
    ax.set_ylim(internal.center.dec.deg-0.5*internal.fov.value*np.cos(internal.center.dec.rad),
                internal.center.dec.deg+0.5*internal.fov.value*np.cos(internal.center.dec.rad))
    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('DEC (degrees)')

    if plot is True:
        outfile = os.path.join(internal.dirname,f'match_{external.name}_{internal.name}_shapes.png')
    else:
        outfile = plot

    print(f"--> Saving plot of source ellipses '{outfile}'")
    plt.savefig(outfile, dpi=dpi, bbox_inches='tight')

    plt.close()

def plot_astrometrics(match_info, internal, external, astro, dpi):
    """
    Plot astrometric offsets of sources to the reference catalog

    Keyword arguments:
    match_info (dict) -- Match information generated by info_match
    internal (class object) -- Internal catalogue
    external (class object) -- External catalogue
    astro (bool or string)  -- If string, plot will be written here
    dpi (int) -- DPI of plot
    """
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
                           width=external.BMin,
                           height=external.BMaj,
                           angle=-external.BPA,
                           facecolor='none',
                           edgecolor='b',
                           linestyle='dashed',
                           label=f'{external.name} beam')
    int_beam_ell = Ellipse(xy=(0,0),
                           width=internal.BMin,
                           height=internal.BMaj,
                           angle=-internal.BPA,
                           facecolor='none',
                           edgecolor='k',
                           label=f'{internal.name} beam')

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
        outfile = os.path.join(internal.dirname,f'match_{external.name}_{internal.name}_astrometrics.png')
    else:
        outfile = astro

    print(f"--> Saving astrometry plot '{outfile}'")
    plt.savefig(outfile, dpi=dpi)
    plt.close()

def plot_fluxes(match_info, internal, external, fluxtype, flux, dpi):
    """
    Plot flux offsets of sources to the reference catalog

    Keyword arguments:
    match_info (dict) -- Match information generated by info_match
    internal (class object) -- Internal catalogue
    external (class object) -- External catalogue
    fluxtype (string) -- Flux type to compare, either 'Total' or 'Peak'
    flux (bool or string) -- If string, plot will be written here
    dpi (int) -- DPI of plot
    """
    flux_matches = match_info['fluxes'][fluxtype]

    cmap = colors.ListedColormap(["navy", "crimson", "limegreen", "gold"])
    norm = colors.BoundaryNorm(np.arange(0.5, 5, 1), cmap.N)

    fig, ax = plt.subplots()

    # Log scale before plotting
    if np.log10(flux_matches['dFlux'].max()/flux_matches['dFlux'].min()) > 1:
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
    ax.set_xlabel('Distance from image center (degrees)')
    ax.set_ylabel(f'Flux ratio ({fluxtype} flux)')

    if flux is True:
        outfile = os.path.join(internal.dirname,f'match_{external.name}_{internal.name}_fluxes.png')
    else:
        outfile = flux

    print(f"--> Saving flux ratio plot '{outfile}'")
    plt.savefig(outfile, dpi=dpi)
    plt.close()

def write_to_catalog(internal, external, matches, extbig, match_info, fluxtype, output):
    """
    Write the matched catalogs to a fits file

    Keyword arguments:
    internal (class object) -- Internal catalogue
    external (class object) -- External catalogue
    matches (list of lists) -- Matched sources between catalogues
    extbig (bool) -- If True, the external catalogue is 
                     considered to have the biggger beam
    match_info (dict) -- Match information generated by info_match
    fluxtype (string) -- Flux type to compare, either 'Total' or 'Peak'
    output (bool or string) -- If string, plot will be written here
    """

    # Bigger beam has matches
    if extbig:
        big = external
        small = internal
    else:
        big = internal
        small = external

    big.cat['Match_alpha'] = 0.0
    big.cat['idx'] = np.arange(len(big.cat))
    small.cat['idx'] = np.inf

    match_alpha_iter = iter(match_info['fluxes'][fluxtype]['match_alpha'])
    for i, match in enumerate(matches):
        if len(match) > 0:
            big.cat[i]['Match_alpha'] = next(match_alpha_iter)
            for j in match:
                small.cat[j]['idx'] = i

    if big.name == small.name:
        small.name += '_1'
    out = join(big.cat, small.cat, keys='idx', 
               table_names=[big.name,small.name])

    if output is True:
        outfile = os.path.join(internal.dirname, f'match_{external.name}_{internal.name}.fits')
    else:
        outfile = output

    print(f"--> Saving output FITS catalog '{outfile}'")
    out.write(outfile, overwrite=True, format='fits')

def write_info(internal, external, match_info, output):
    """
    Write the information into a json file

    Keyword arguments:
    internal (class object) -- Internal catalogue
    external (class object) -- External catalogue
    match_info (dict) -- Match information generated by info_match
    output (bool or string) -- If string, plot will be written here
    """
    filename = os.path.join(internal.dirname, f'match_{external.name}_{internal.name}_info.json')

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

    input_cat_file = args.input_cat
    ext_cat = args.ext_cat

    # Matching options
    fluxtype = args.fluxtype
    alpha = args.alpha
    sigma_extent = args.match_sigma_extent
    search_dist = args.search_dist/3600 # to degrees
    source_overlap = args.source_overlap_percent

    # Output options
    output = args.output
    astro = args.astro
    flux = args.flux
    plot = args.plot
    dpi = args.dpi
    annotate = args.annotate
    annotate_nonmatched = args.annotate_nonmatched

    # Catalog options
    survey_name = args.survey_name
    ra_center = args.ra_center
    dec_center = args.dec_center
    fov = args.fov

    input_cat = Table.read(input_cat_file)
    int_catalog = BDSFCatalog(input_cat, input_cat_file, survey_name, ra_center, dec_center, fov)

    if os.path.exists(ext_cat):
        ext_table = Table.read(ext_cat)
        if 'bdsfcat' in ext_cat:
            ext_catalog = BDSFCatalog(ext_table, ext_cat, ra_center=ra_center, dec_center=dec_center, fov=fov)
        else:
            ext_catalog = ExternalCatalog(ext_cat, ext_table, int_catalog.center)
    else:
        ext_table = int_catalog.query_catalog(ext_cat)
        ext_catalog = ExternalCatalog(ext_cat, ext_table, int_catalog.center)

    if len(ext_table) == 0:
        print('No sources were found to match, most likely the external catalog has no coverage here')
        exit()

    # Before matching, determine which catalogue has larger resolution (so is probably smaller)
    extbig = True
    if int_catalog.BMaj > ext_catalog.BMaj:
        extbig = False

    matches = match_catalogs(int_catalog, ext_catalog, extbig, sigma_extent, search_dist, source_overlap)
    matches_info = info_match(int_catalog, ext_catalog, matches, extbig, fluxtype, alpha)

    matches_info['INPUT'] = {}
    matches_info['INPUT']['alpha'] = alpha 
    matches_info['INPUT']['match_sigma_extend'] = sigma_extent 
    matches_info['INPUT']['search_dist'] = search_dist
    matches_info['INPUT']['source_overlap_percent'] = source_overlap

    if plot:
        plot_catalog_match(int_catalog, ext_catalog, matches, extbig, plot, dpi)
    if astro:
        plot_astrometrics(matches_info, int_catalog, ext_catalog, astro, dpi)
    if flux:
        plot_fluxes(matches_info, int_catalog, ext_catalog, fluxtype, flux, dpi)
    if output:
        write_to_catalog(int_catalog, ext_catalog, matches, extbig, matches_info, fluxtype, output)
        write_info(int_catalog, ext_catalog, matches_info, output)

    if annotate == 'kvis':
        kvis.matches_to_kvis(int_catalog, ext_catalog, matches, extbig, annotate, annotate_nonmatched, sigma_extent)
    if annotate == 'ds9':
        kvis.matches_to_ds9(int_catalog, ext_catalog, matches, extbig, annotate, annotate_nonmatched, sigma_extent)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("input_cat", type=str,
                        help="""Catalog made by PyBDSF.""")
    parser.add_argument("ext_cat", default="NVSS", type=str,
                        help="""External catalog to match to. Standard catalogues are
                                NVSS, SUMMS, FIRST, TGSS, RACS-low or RACS-mid (default NVSS).
                                Any other name will be interpreted as a file, or failing
                                that a VO link. If a non-standard catalog is specified,
                                the parsets/extcat.json file must be used to specify its
                                details. Alternatively, if the external catalog is a 
                                PyBDSF catalog, it will be parsed as such if the filename 
                                contains the substring 'bdsfcat'.""")
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
