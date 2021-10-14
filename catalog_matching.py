#!/usr/bin/env python

import os
import sys
import numpy as np

import json
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

import searchcats as sc
import helpers

class SourceEllipse:

    def __init__(self, catalog_entry, column_dict):
        self.RA = catalog_entry[column_dict['ra']]
        self.DEC = catalog_entry[column_dict['dec']]
        self.Maj = catalog_entry[column_dict['majax']]
        self.Min = catalog_entry[column_dict['minax']]
        self.PA = catalog_entry[column_dict['pa']]
        if column_dict['peak_flux']:
            self.PeakFlux = catalog_entry[column_dict['peak_flux']]
        if column_dict['total_flux']:
            self.IntFlux = catalog_entry[column_dict['total_flux']]

        self.skycoord = SkyCoord(self.RA, self.DEC, unit='deg')

    def match(self, ra_list, dec_list, separation = 0):
        '''
        Match the ellipse with a (list of) source(s)

        Keyword arguments:
        ra_list -- Right ascension of sources
        dec_list -- Declination of sources
        separation (float) - Additional range in degrees
        '''
        offset_coord = SkyCoord(ra_list, dec_list, unit='deg')
        dra, ddec = self.skycoord.spherical_offsets_to(offset_coord)

        PA = np.radians(-self.PA) + 0.5*np.pi
        bool_points = ((np.cos(PA)*(dra.deg)
                      +np.sin(PA)*(ddec.deg))**2
                      /(self.Maj/2+separation)**2
                      +(np.sin(PA)*(dra.deg)
                      -np.cos(PA)*(ddec.deg))**2
                      /(self.Min/2+separation)**2) <= 1

        return np.where(bool_points)[0]

    def to_artist(self):
        '''
        Convert the ellipse to a matplotlib artist
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

            self.BMaj = beam[0]
            self.BMin = beam[1]
            self.BPA = beam[2]
            self.freq = freq
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
        dec_fov = abs(float(header['CDELT1']))*float(header['CRPIX1'])
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
                                   offset = self.fov.to(u.arcsec))

        if not nvsstable:
            sys.exit()

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
        firsttable = sc.getfirstdata(ra = [self.center.ra.to_string(u.hourangle, sep=' ')],
                                     dec = [self.center.dec.to_string(u.deg, sep=' ')],
                                     offset = self.fov.to(u.arcsec))

        if not firsttable:
            sys.exit()

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
        if self.center.dec.deg > -29.5:
            print('Catalog outside of SUMSS footprint')
            sys.exit()

        sumsstable = sc.getsumssdata(ra = self.center.ra,
                                     dec = self.center.dec,
                                     offset = self.fov)

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

    ax.set_xlim(pointing.center.ra.deg-pointing.fov.value,
                pointing.center.ra.deg+pointing.fov.value)
    ax.set_ylim(pointing.center.dec.deg-pointing.fov.value*np.cos(pointing.center.dec.rad),
                pointing.center.dec.deg+pointing.fov.value*np.cos(pointing.center.dec.rad))
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

    fig, ax = plt.subplots()
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
                           facecolor='none',
                           edgecolor='b',
                           linestyle='dashed',
                           label=f'{ext.name} beam')
    int_beam_ell = Ellipse(xy=(0,0),
                           width=pointing.BMin,
                           height=pointing.BMaj,
                           facecolor='none',
                           edgecolor='k',
                           label=f'{pointing.telescope} beam')
    ax.add_patch(ext_beam_ell)
    ax.add_patch(int_beam_ell)

    ymax_abs = abs(max(ax.get_ylim(), key=abs))
    xmax_abs = abs(max(ax.get_xlim(), key=abs))
    ax.set_ylim(ymin=-ymax_abs, ymax=ymax_abs)
    ax.set_xlim(xmin=-xmax_abs, xmax=xmax_abs)

    ax.axhline(0,-xmax_abs,xmax_abs, color='k', zorder=1)
    ax.axvline(0,-ymax_abs,ymax_abs, color='k', zorder=1)

    ax.annotate('Median offsets', xy=(0.05,0.95), xycoords='axes fraction', fontsize=8)
    # Determine mean and standard deviation of points RA
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

def write_to_kvis(pointing, ext, matches, annotate):
    '''
    Write the results to a kvis annotation file
    '''
    match_ext_lines = []
    match_int_lines = []
    non_match_ext_lines = []
    for i, match in enumerate(matches):
        if len(match) > 0:
            source = ext.sources[i]
            toprt = f'ELLIPSE {source.RA:.6f} {source.DEC:.6f} {source.Maj/2:.6f} {source.Min/2:.6f} {source.PA:.4f} \n'
            match_ext_lines.append(toprt)
            for ind in match:
                source = pointing.sources[ind]
                toprt = f'ELLIPSE {source.RA:.6f} {source.DEC:.6f} {source.Maj/2:.6f} {source.Min/2:.6f} {source.PA:.4f} \n'
                match_int_lines.append(toprt)
        else:
            source = ext.sources[i]
            toprt = f'ELLIPSE {source.RA:.6f} {source.DEC:.6f} {source.Maj/2:.6f} {source.Min/2:.6f} {source.PA:.4f} \n'
            non_match_ext_lines.append(toprt)

    non_matches = np.setdiff1d(np.arange(len(pointing.sources)), np.concatenate(matches).ravel())
    non_match_int_lines = []
    for i in non_matches:
        source = pointing.sources[i]
        toprt = f'ELLIPSE {source.RA:.6f} {source.DEC:.6f} {source.Maj/2:.6f} {source.Min/2:.6f} {source.PA:.4f} \n'
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
    kvisfile.writelines('# Non matched sources from external catalog\n')
    kvisfile.writelines('# \n')
    kvisfile.writelines('COLOR WHITE\n')
    '''
    for line in non_match_ext_lines:
        kvisfile.writelines(line)
    kvisfile.writelines('# Non matched sources from internal catalog\n')
    kvisfile.writelines('# \n')
    kvisfile.writelines('COLOR GREEN\n')
    for line in non_match_int_lines:
        kvisfile.writelines(line)
    '''

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

    matches = match_catalogs(pointing, ext_catalog)
    if plot:
        plot_catalog_match(pointing, ext_catalog, matches, plot, dpi)
    if astro:
        plot_astrometrics(pointing, ext_catalog, matches, astro, dpi)
    if flux:
        plot_fluxes(pointing, ext_catalog, matches, fluxtype, flux, alpha, dpi)
    if output:
        write_to_catalog(pointing, ext_catalog, matches, output)
    if annotate:
        write_to_kvis(pointing, ext_catalog, matches, annotate)

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
    parser.add_argument("--alpha", default=0.8,
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
    return parser

if __name__ == '__main__':
    main()
