import os
import sys
import numpy as np

import json
from pathlib import Path
from argparse import ArgumentParser

from astropy import units as u
from astropy.table import Table, join
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import searchcats as sc
import helpers

class SourceEllipse:

    def __init__(self, catalog_entry, ra='RA', dec='DEC', majax='Maj', minax='Min', pa='PA', peak_flux='Peak_flux', total_flux='Total_flux'):
        self.RA = catalog_entry[ra]
        self.DEC = catalog_entry[dec]
        self.Maj = catalog_entry[majax]
        self.Min = catalog_entry[minax]
        self.PA = catalog_entry[pa]
        if peak_flux:
            self.PeakFlux = catalog_entry[peak_flux]
        if total_flux:
            self.IntFlux = catalog_entry[total_flux]

    def match(self, ra_list, dec_list, separation = 0):
        '''
        Match the ellipse with a (list of) source(s)

        Keyword arguments:
        ra_list -- Right ascension of sources
        dec_list -- Declination of sources
        separation (float) - Additional range in degrees
        '''
        PA = np.radians(self.PA)
        bool_points = ((np.cos(PA)*(ra_list-self.RA)
                      +np.sin(PA)*(dec_list-self.DEC))**2
                      /(self.Maj+separation)**2
                      +(np.sin(PA)*(ra_list-self.RA)
                      -np.cos(PA)*(dec_list-self.DEC))**2
                      /(self.Min+separation)**2) <= 1

        return np.where(bool_points)[0]

    def to_artist(self):
        '''
        Convert the ellipse to a matplotlib artist
        '''
        return Ellipse(xy = (self.RA, self.DEC),
                        width = 2*self.Min,
                        height = 2*self.Maj,
                        angle = -self.PA)

class ExternalCatalog():

    def __init__(self, name, catalog, center):
        self.name = name
        self.cat = catalog

        if name in ['NVSS','SUMSS','FIRST']:
            self.sources = [SourceEllipse(source) for source in catalog]
            beam, freq = helpers.get_beam(name, center.ra.deg, center.dec.deg)

            self.BMaj = beam[0]
            self.BMin = beam[1]
            self.BPA = beam[2]
            self.freq = freq
        else:
            path = Path(__file__).parent / 'parsets/extcat.json'
            with open(path) as f:
                cat_info = json.load(f)

            self.sources = [SourceEllipse(source, **cat_info['data_columns']) for source in catalog]
            self.BMaj = cat_info['properties']['BMAJ']
            self.BMin = cat_info['properties']['BMIN']
            self.BPA = cat_info['properties']['BPA']
            self.freq = cat_info['properties']['freq']


class Pointing():

    def __init__(self, catalog):
        self.cat = catalog
        self.sources = [SourceEllipse(source) for source in catalog]

        self.RAmax = catalog['RA'].max()
        self.RAmin = catalog['RA'].min()
        self.DECmax = catalog['DEC'].max()
        self.DECmin = catalog['DEC'].min()

        self.dDEC = (self.DECmax - self.DECmin)*u.degree
        self.dRA = (self.RAmax - self.RAmin)*u.degree
        self.fov = max(self.dDEC,self.dRA)

        # Parse meta
        header = dict([x.split(' = ') for x in catalog.meta['comments'][4:]])

        self.BMaj = float(header['BMAJ'])*3600 #arcsec
        self.BMin = float(header['BMIN'])*3600 #arcsec
        self.BPA = float(header['BPA'])
        self.freq = float(header['RESTFRQ'])/1e6 #MHz

        self.center = SkyCoord(float(header['OBSRA'])*u.degree,
                               float(header['OBSDEC'])*u.degree)

    def query_NVSS(self):
        '''
        Match the pointing to the NVSS catalog
        '''
        nvsstable = sc.getnvssdata(ra= [self.center.ra.to_string(u.hourangle, sep=' ')],
                                   dec = [self.center.dec.to_string(u.deg, sep=' ')],
                                   offset = 0.5*self.fov.to(u.arcsec))

        nvsstable['Maj'].unit = u.arcsec
        nvsstable['Min'].unit = u.arcsec

        nvsstable['Maj'] = nvsstable['Maj'].to(u.deg)
        nvsstable['Min'] = nvsstable['Min'].to(u.deg)
        nvsstable['RA'] = nvsstable['RA'].to(u.deg)
        nvsstable['DEC'] = nvsstable['DEC'].to(u.deg)

        nvsstable['Peak_flux'] /= 1e3 #convert to Jy
        nvsstable['Total_flux'] /= 1e3 #convert to Jy

        return nvsstable

    def query_FIRST(self):
        '''
        Match the pointing to the FIRST catalog
        '''
        firsttable = sc.getfirstdata(ra= [self.center.ra.to_string(u.hourangle, sep=' ')],
                                     dec = [self.center.dec.to_string(u.deg, sep=' ')],
                                     offset = 0.5*self.fov.to(u.arcsec))

        firsttable['Maj'].unit = u.arcsec
        firsttable['Min'].unit = u.arcsec

        firsttable['Maj'] = firsttable['Maj'].to(u.deg)
        firsttable['Min'] = firsttable['Min'].to(u.deg)
        firsttable['RA'] = firsttable['RA'].to(u.deg)
        firsttable['DEC'] = firsttable['DEC'].to(u.deg)

        firsttable['Peak_flux'] /= 1e3 #convert to Jy
        firsttable['Total_flux'] /= 1e3 #convert to Jy

        return firsttable

    def query_SUMSS(self):
        '''
        Match the pointing to the SUMSS catalog. This is very slow since
        SUMSS does not offer a catalog search so we match to the entire catalog
        '''
        sumsstable = sc.getsumssdata(ra= self.center.ra,
                                     dec = self.center.dec,
                                     offset = 0.5*self.fov)

        sumsstable['Maj'].unit = u.arcsec
        sumsstable['Min'].unit = u.arcsec

        sumsstable['Maj'] = sumsstable['Maj'].to(u.deg)
        sumsstable['Min'] = sumsstable['Min'].to(u.deg)
        sumsstable['RA'] = sumsstable['RA'].to(u.deg)
        sumsstable['DEC'] = sumsstable['DEC'].to(u.deg)

        sumsstable['Peak_flux'] /= 1e3 #convert to Jy
        sumsstable['Total_flux'] /= 1e3 #convert to Jy

        return sumsstable

def match_catalogs(pointing, ext):
    '''
    Match the sources of the chosen external catalog to the sources in the pointing
    '''
    print(f'Matching {len(ext.sources)} sources in {ext.name} to {len(pointing.cat)} sources in the pointing')

    matches = []
    for source in ext.sources:
        matches.append(source.match(pointing.cat['RA'], pointing.cat['DEC']))

    return matches

def plot_catalog_match(pointing, ext, matches, datadir, dpi):
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

    ax.set_xlim(pointing.RAmin, pointing.RAmax)
    ax.set_ylim(pointing.DECmin, pointing.DECmax)
    ax.set_xlabel('RA (degrees)')
    ax.set_ylabel('DEC (degrees)')

    plt.savefig(datadir+'/catalog_match.png', dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_astrometrics(pointing, ext, matches, datadir, astro, dpi):
    '''
    Plot astrometric offsets of sources to the reference catalog
    '''
    dDEC = []
    dRA = []
    for i, match in enumerate(matches):
        if len(match) == 1:
            dRA.append(ext.sources[i].RA - pointing.sources[match[0]].RA)
            dDEC.append(ext.sources[i].DEC - pointing.sources[match[0]].DEC)

    dRA = np.array(dRA)*3600
    dDEC = np.array(dDEC)*3600

    fig, ax = plt.subplots()
    ax.scatter(dRA, dDEC, zorder=2, color='k', marker='.', s=5)

    ext_beam_ell = Ellipse(xy=(0,0),
                           width=2*ext.BMin,
                           height=2*ext.BMaj,
                           facecolor='none',
                           edgecolor='b',
                           linestyle='dashed',
                           label=f'{ext.name} beam')
    int_beam_ell = Ellipse(xy=(0,0),
                           width=2*pointing.BMin,
                           height=2*pointing.BMaj,
                           facecolor='none',
                           edgecolor='k',
                           label='MeerKAT beam')
    ax.add_patch(ext_beam_ell)
    ax.add_patch(int_beam_ell)

    ymax_abs = abs(max(ax.get_ylim(), key=abs))
    xmax_abs = abs(max(ax.get_xlim(), key=abs))
    ax.set_ylim(ymin=-ymax_abs, ymax=ymax_abs)
    ax.set_xlim(xmin=-xmax_abs, xmax=xmax_abs)

    ax.axhline(0,-xmax_abs,xmax_abs, color='k', zorder=1)
    ax.axvline(0,-ymax_abs,ymax_abs, color='k', zorder=1)
    ax.axhline(np.median(dDEC),-xmax_abs,xmax_abs, color='grey', linestyle='dashed', zorder=1)
    ax.axvline(np.median(dRA),-ymax_abs,ymax_abs, color='grey', linestyle='dashed', zorder=1)

    ax.set_title(f'Astrometric offset of {len(dRA)} sources')
    ax.set_xlabel('RA offset (arcsec)')
    ax.set_ylabel('DEC offset (arcsec)')
    ax.legend()

    if astro is True:
        plt.savefig(datadir+'/match_astrometrics.png', dpi=dpi)
    else:
        plt.savefig(astro, dpi=dpi)
    plt.close()

def plot_fluxes(pointing, ext, matches, datadir, fluxtype, flux, dpi):
    '''
    Plot flux offsets of sources to the reference catalog
    '''
    ext_flux = []
    int_flux = []
    RA_off = []
    DEC_off = []
    if fluxtype == 'Total':
        for i, match in enumerate(matches):
            if len(match) > 0:
                ext_flux.append(ext.sources[i].IntFlux)
                int_flux.append(np.sum([pointing.sources[m].IntFlux for m in match]))
                RA_off.append(ext.sources[i].RA - pointing.center.ra.deg)
                DEC_off.append(ext.sources[i].DEC - pointing.center.dec.deg)
    elif fluxtype == 'Peak':
        for i, match in enumerate(matches):
            if len(match) > 0:
                ext_flux.append(ext.sources[i].PeakFlux)
                int_flux.append(np.sum([pointing.sources[m].PeakFlux for m in match]))
                RA_off.append(ext.sources[i].RA - pointing.center.ra.deg)
                DEC_off.append(ext.sources[i].DEC - pointing.center.dec.deg)
    else:
        print(f'Invalid fluxtype {fluxtype}, choose between Total or Peak flux')
        sys.exit()

    # Scale flux density to proper frequency
    ext_flux_corrected = np.array(ext_flux) * (pointing.freq/ext.freq)**-0.7
    dFlux = np.array(int_flux)/np.array(ext_flux)
    center_dist = np.sqrt(np.array(RA_off)**2 + np.array(DEC_off)**2)

    fig, ax = plt.subplots()

    ax.set_yscale('log')
    ax.scatter(center_dist, dFlux, color='k', marker='.', s=5)

    ax.set_title(f'Flux ratio of {len(dFlux)} sources')
    ax.set_xlabel('Distance from pointing center (degrees)')
    ax.set_ylabel('Flux ratio')

    if flux is True:
        plt.savefig(datadir+'/match_fluxes.png', dpi=dpi)
    else:
        plt.savefig(flux, dpi=dpi)
    plt.close()

def write_to_catalog(pointing, ext, matches, output, datadir, pointing_name):
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
        filename = f'{datadir}/match_{ext.name}_{pointing_name}.fits'
        out.write(filename, overwrite=True, format='fits')
    else:
        out.write(output, overwrite=True, format='fits')

def main():

    parser = new_argument_parser()
    args = parser.parse_args()

    pointing = args.pointing
    ext_cat = args.ext_cat
    fluxtype = args.fluxtype
    dpi = args.dpi

    astro = args.astro
    flux = args.flux
    output = args.output

    datadir = os.path.dirname(pointing)
    filename = os.path.splitext(os.path.basename(pointing))[0]

    pointing_cat = Table.read(pointing)
    pointing = Pointing(pointing_cat)

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
            ext_catalog = Pointing(ext_table)
        else:
            ext_catalog = ExternalCatalog(ext_cat, ext_table, pointing.center)
    else:
        print('Invalid input table!')
        exit()

    if len(ext_table) == 0:
        print('No sources were found to match, most likely the external catalog has no coverage here')
        exit()

    matches = match_catalogs(pointing, ext_catalog)
    plot_catalog_match(pointing, ext_catalog, matches, datadir, dpi)

    if astro:
        plot_astrometrics(pointing, ext_catalog, matches, datadir, astro, dpi)
    if flux:
        plot_fluxes(pointing, ext_catalog, matches, datadir, fluxtype, flux, dpi)
    if output:
        write_to_catalog(pointing, ext_catalog, matches, output, datadir, filename)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("pointing",
                        help="""MeerKAT pointing catalog made by PyBDSF.""")
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
    parser.add_argument("--fluxtype", default="Total",
                        help="""Whether to use Total or Peak flux for determining
                                the flux ratio (default = Total).""")
    parser.add_argument("--alpha", default=0.7,
                        help="""The spectral slope to assume for calculating the
                                flux ratio, where Flux_1 = Flux_2 * (freq_1/freq_2)^-alpha
                                (default = 0.7)""")
    parser.add_argument("--output", nargs="?", const=True,
                        help="""Output the result of the matching into a catalog,
                                optionally provide an output filename
                                (default = don't output a catalog).""")
    return parser

if __name__ == '__main__':
    main()