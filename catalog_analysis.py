#!/usr/bin/env python

import os
import sys
import json
import warnings
import numpy as np

from astropy import units as u
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

from pathlib import Path
from argparse import ArgumentParser
from numpyencoder import NumpyEncoder

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from utils import helpers

class Catalog:

    def __init__(self, catalog_file, flux_col, stacked_cat, outdir=None):
        if outdir is None:
            self.dirname = os.path.dirname(catalog_file)
        else:
            self.dirname = outdir
        self.cat_name = os.path.basename(catalog_file).rsplit('.',1)[0]
        print(f'Reading in catalog: {self.cat_name}')

        self.table = Table.read(catalog_file)
        if not flux_col in self.table.colnames:
            print(f'Invalid flux column name {flux_col}')
            sys.exit()
        self.flux_col = flux_col

        # Parse meta
        header = self.table.meta
        if not stacked_cat:
            self.pix_area = np.pi*float(header['AXIS1'])*float(header['AXIS2'])/4
            self.pix_size = float(max(header['CDELT1'],header['CDELT2']))

            self.center = SkyCoord(float(header['CRVAL1'])*u.degree,
                                   float(header['CRVAL2'])*u.degree)

        # Initialize empty values
        self.dN = None
        self.edges = None
        self.npix = None

    def get_flux_bins(self, nbins):
        '''
        Define flux bins for the catalog

        Keyword arguments:
        flux_col -- Which table column to use for flux
        nbins    -- Number of bins
        '''
        f_low = np.min(self.table[self.table[self.flux_col] > 0][self.flux_col])
        f_high = np.max(self.table[self.flux_col])
        log_bin = np.logspace(np.log10(f_low),np.log10(f_high),nbins)

        self.dN, self.edges = np.histogram(self.table[self.flux_col], bins=log_bin)

        # Remove high flux bins starting from the first empty bin
        if any(self.dN[int(nbins/2):] == 0):
            cutoff_high = np.where(self.dN[int(nbins/2):] == 0)[0][0] + int(nbins/2)
            self.edges = self.edges[:cutoff_high+1]
            self.dN = self.dN[:cutoff_high]

    def plot_number_counts(self, fancy, dpi, no_plot=False):
        '''
        Plot number count bins and fit a power law
        '''
        cutoff_high = np.argmax(self.dN)
        fit_edges = self.edges[cutoff_high:]
        fit_dN = self.dN[cutoff_high:]

        if not no_plot:
            plt.bar(self.edges[:-1], self.dN, width=np.diff(self.edges), edgecolor='k', alpha=0.5, align='edge')
            plt.xscale('log')
            plt.yscale('log')
            plt.autoscale(enable=True, axis='x', tight=True)

            if fancy:
                plt.ylabel('$N$')
                plt.xlabel('$S (\\mathrm{Jy})$')
            else:
                plt.ylabel('N')
                plt.xlabel('S (Jy)')

            outfile = os.path.join(self.dirname,self.cat_name+'_number_counts.png')
            print(f"--> Saving plot of number counts '{outfile}'")

            plt.savefig(outfile, dpi=dpi)
            plt.close()

    def rms_statistics(self, rms_image, fancy, no_plot=False):
        '''
        Determine radial profile and coverage from rms image

        Keyword arguments:
        rms_image -- Filename of rms image
        '''
        image = fits.open(rms_image)[0]

        # Determine coverage
        rms_data = image.data.flatten()
        rms_range = np.logspace(np.log10(np.nanmin(rms_data)), np.log10(np.nanmax(rms_data)), 100)
        coverage = [np.sum([rms_data < rms])/np.count_nonzero(~np.isnan(rms_data)) for rms in rms_range]

        # Define a spline and interpolate the values
        rms_coverage = interp1d(rms_range, coverage, bounds_error=False, fill_value=(0,1))
        self.pix_area = np.count_nonzero(~np.isnan(rms_data))

        # Put sigma_20 into header
        sigma20 = rms_range[np.argmin(np.abs(np.array(coverage)-0.2))]
        self.table.meta['sigma_20'] = sigma20

        # Save to json file
        rms_stats = {}
        rms_stats['rms'] = rms_range
        rms_stats['coverage'] = coverage
        rms_stats['sigma20'] = sigma20

        filename = os.path.join(self.dirname, self.cat_name+'_rms_coverage.json')
        print(f"--> Saving json file of rms coverage '{filename}'")
        with open(filename, 'w') as outfile:
            json.dump(rms_stats,outfile,
                      indent=4, sort_keys=True,
                      separators=(',', ': '),
                      ensure_ascii=False,
                      cls=NumpyEncoder)

        if not no_plot:
            # Plot rms coverage
            plt.plot(rms_range, coverage, linewidth=2, color='k')
            plt.fill_between(rms_range, 0, coverage, color='k', alpha=0.2)

            plt.xscale('log')
            plt.autoscale(enable=True, axis='x', tight=True)

            if fancy:
                plt.xlabel('$\\sigma$ (Jy/beam)')
            else:
                plt.xlabel('RMS (Jy/beam)')
            plt.ylabel('Coverage')

            outfile = os.path.join(self.dirname, self.cat_name+'_rms_coverage.png')
            print(f"--> Saving plot of rms coverage '{outfile}'")

            plt.savefig(outfile, dpi=300)
            plt.close()

        return rms_coverage

    def plot_diff_number_counts(self, fancy, dpi, model_counts=None, rms_coverage=None, 
                                completeness=None, no_plot=False):
        '''
        Compute and plot differential number counts
        input RMS image is used to account for sky coverage

        Keyword arguments:
        flux_col     -- Which table column to use for flux, should be the same
                        as the one used to define the bins
        rms_coverage -- Input rms coverage function
        completeness -- File to correct for completeness
        '''
        S = (self.edges[:-1] + self.edges[1:]) / 2
        dS = np.diff(self.edges)
        solid_angle = self.pix_area*self.pix_size**2*(np.pi/180)**2

        count_correction = 1
        if completeness:
            # Get data from specified file
            with open(completeness, 'r') as infile:
                data = json.load(infile)
            flux_means = np.array([(data['flux_bins'][i]+data['flux_bins'][i+1])/2 for i in range(len(data['flux_bins'])-1)])

            comp_frac = interp1d(flux_means, np.mean(data['detected_fraction'], axis=0), 
                                 bounds_error=False, fill_value=(0,1))
            count_correction = comp_frac(S)

        if rms_coverage:
            count_correction = rms_coverage(S/5.0)

        diff_counts = S**(5/2) * (self.dN/count_correction) / dS / solid_angle
        diff_counts_err = S**(5/2) * (np.sqrt(self.dN)/count_correction) / dS / solid_angle

        plt.errorbar(S, diff_counts, yerr=diff_counts_err,
                     fmt='o', color='k', label='Catalog')

        # Save diff number counts to json file
        counts = {}
        counts['flux'] = S
        counts['dflux'] = dS
        counts['counts'] = self.dN
        counts['diff_counts'] = diff_counts
        counts['diff_counts_err'] = diff_counts_err
        counts['solid_angle'] = solid_angle

        filename = os.path.join(self.dirname, self.cat_name+'_diff_counts.json')
        print(f"--> Saving json file of differential number counts '{filename}'")
        with open(filename, 'w') as outfile:
            json.dump(counts,outfile,
                      indent=4, sort_keys=True,
                      separators=(',', ': '),
                      ensure_ascii=False,
                      cls=NumpyEncoder)

        if not no_plot:
            if model_counts is not None:
                # Get differential number counts from SKADS and plot
                path = Path(__file__).parent / 'data' / (model_counts+'.json')
                with open(path) as f:
                    model_counts = json.load(f)

            plt.errorbar(model_counts['flux'], model_counts['counts'],
                        yerr=model_counts['counts_err'],
                        fmt=':.', color='r', lw=0.5, label=model_counts['desc'])

            plt.xscale('log')
            plt.yscale('log')

            if fancy:
                plt.ylabel('$S^{5/2}\\  \\mathrm{d}N/\\mathrm{d}S \\ (\\mathrm{Jy}^{3/2} \\mathrm{sr}^{-1})$')
                plt.xlabel('$S (\\mathrm{Jy})$')
            else:
                plt.ylabel('S^5/2 dN/dS (Jy^3/2 / sr)')
                plt.xlabel('S (Jy)')
            plt.legend()

            outfile = os.path.join(self.dirname, self.cat_name+'_diff_counts.png')
            print(f"--> Saving plot of differential number counts '{outfile}'")

            plt.savefig(outfile, dpi=dpi)
            plt.close()

    def plot_resolved_fraction(self, resolved_sigma, resolved_offset, stacked_cat, fancy, dpi, no_plot=False):
        '''
        Plot fraction of resolved sources
        '''
        if stacked_cat:
            resolved_idx = self.table['Resolved']
        else:
            # Calculate errors on total flux and peak flux
            sigma_s = np.sqrt(self.table['E_'+self.flux_col]**2 + (resolved_offset*self.table[self.flux_col])**2)

            sigma_r = np.sqrt((sigma_s/self.table[self.flux_col])**2
                            - (self.table['E_Peak_flux']/self.table['Peak_flux'])**2
                            + (self.table['Isl_rms']/self.table['Peak_flux'])**2)

            f = resolved_sigma
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                # Check how much of the resolved sources are captured
                missed_unresolved = np.sum(np.log(self.table[self.flux_col]/self.table['Peak_flux']) < -f*sigma_r)
                defs_unresolved = np.sum(np.log(self.table[self.flux_col]/self.table['Peak_flux']) < 0)
                print(f'Current envelope captures {(1 - (missed_unresolved/defs_unresolved))*100:.2f}% of the unresolved sources, '
                      +'consider changing the envelope parameters if this is not satisfactory')

                resolved_idx = np.log(self.table[self.flux_col]/self.table['Peak_flux']) > f*sigma_r

        resolved = self.table[resolved_idx]
        unresolved = self.table[~resolved_idx]

        if not no_plot:
            # Log scale before plotting
            plt.xscale('log')
            plt.yscale('log')

            alpha = min(1000 / (len(unresolved)+len(resolved)), 1)
            alpha = max(alpha, 1/255)
            plt.scatter(resolved['Peak_flux']/resolved['Isl_rms'],
                        resolved[self.flux_col]/resolved['Peak_flux'],
                        color='crimson', s=5, label=f'Resolved ({len(resolved)})',
                        alpha=alpha, marker='.')
            plt.scatter(unresolved['Peak_flux']/unresolved['Isl_rms'],
                        unresolved[self.flux_col]/unresolved['Peak_flux'],
                        color='navy', s=5, label=f'Unresolved ({len(unresolved)})',
                        alpha=alpha, marker='.')

            plt.xlim(left=4)

            if fancy:
                plt.ylabel('$S_{tot}/S_{peak}$')
            else:
                plt.ylabel('Total flux / Peak flux')
            plt.xlabel('S/N')

            leg = plt.legend()
            for lh in leg.legend_handles: 
                lh.set_alpha(1)

            outfile = os.path.join(self.dirname,self.cat_name+'_resolved.png')
            print(f"--> Saving plot of resolved sources '{outfile}'")

            plt.savefig(outfile, dpi=dpi)
            plt.close()

        return resolved_idx

def main():
    parser = new_argument_parser()
    args = parser.parse_args()

    # Input options
    catalog_file = args.catalog
    rms_image = args.rms_image
    comp_corr = args.comp_corr
    model_counts = args.model_counts

    flux_col = args.flux_col
    resolved_sigma = args.resolved_sigma
    resolved_offset = args.resolved_offset
    stacked_cat = args.stacked_catalog

    outdir = args.outdir
    no_plots = args.no_plots
    fancy = args.fancy
    dpi = args.dpi

    if fancy:
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rc('text', usetex=True)
        plt.rcParams.update({'font.size': 14})

    catalog = Catalog(catalog_file, flux_col, stacked_cat, outdir)
    catalog.get_flux_bins(nbins=50)

    catalog.plot_number_counts(fancy, dpi, no_plot=no_plots)

    if rms_image:
        rms_coverage = catalog.rms_statistics(rms_image, fancy, no_plot=no_plots)
        catalog.plot_diff_number_counts(fancy, dpi, model_counts, rms_coverage=rms_coverage, no_plot=no_plots)
    if comp_corr:
        catalog.plot_diff_number_counts(fancy, dpi, model_counts, completeness=comp_corr, no_plot=no_plots)

    resolved = catalog.plot_resolved_fraction(resolved_sigma, resolved_offset, stacked_cat, fancy, dpi, no_plot=no_plots)
    if not stacked_cat:
        catalog.table['Resolved'] = resolved
        catalog.table.write(catalog_file, overwrite=True)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("catalog", type=str,
                        help="""Pointing catalog(s) made by PyBDSF.""")
    parser.add_argument('-r', '--rms_image', default=None,
                        help="""Specify input rms image for creating an rms coverage
                                plot. In the absence of a completeness correction file,
                                will also be used to correct for completeness.""")
    parser.add_argument('-c', '--comp_corr', default=None,
                        help="""Specify input json file containing completeness
                                fractions for correcting differential number counts.
                                the file is assumed to contain at least the arrays of 
                                flux bins, completeness fraction.""")
    parser.add_argument('--resolved_sigma', default=1.25, type=float,
                        help="""Selection parameter for resolved sources, higher values more
                                stringently select resolved sources (default=1.25).""")
    parser.add_argument('--resolved_offset', default=0.0, type=float,
                        help="""Offset from 1-to-1 relation of integrated/peak flux
                                of unresolved sources (default=0.0).""")
    parser.add_argument('--model_counts', default='SKADS_1285MHz_counts', type=str,
                        help="""Input file for model differential source counts to compare
                                to observed counts. Must be present as a json file in the
                                'data' directory (default='SKADS_1285MHz_counts').""")
    parser.add_argument('--flux_col', default='Total_flux', type=str,
                        help="""Name of integrated flux column to use for analysis
                                (default = Total_flux).""")
    parser.add_argument('--stacked_catalog', action='store_true',
                        help="""Indicate if catalog is built up from multiple catalogs,
                                for example with combine_catalogs script.""")
    parser.add_argument('--outdir', default=None, type=str,
                        help="Output directory. Default is same directory as input catalog.")
    parser.add_argument('--no_plots', action='store_true',
                        help="""Run the scripts without making any plots.""")
    parser.add_argument('--fancy', action='store_true',
                        help="Output plots with latex font and formatting.")
    parser.add_argument('-d', '--dpi', default=300, type=int,
                        help="DPI of the output images (default = 300).")

    return parser


if __name__ == '__main__':
    main()