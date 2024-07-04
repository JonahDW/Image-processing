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

import helpers

class Catalog:

    def __init__(self, catalog_file, flux_col, stacked_cat):
        self.dirname = os.path.dirname(catalog_file)
        self.cat_name = os.path.basename(catalog_file).rsplit('.',1)[0]
        print(f'Reading in catalog: {self.cat_name}')

        self.table = Table.read(catalog_file)
        if not flux_col in self.table.colnames:
            print(f'Invalid flux column name {flux_col}')
            sys.exit(1)
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

        # Determine radial profile
        filename = os.path.join(self.dirname, self.cat_name+'_rms_radial.json')

        if os.path.exists(filename):
            with open(filename, 'r') as infile:
                radialprofile = json.load(infile)
        else:
            radialprofile = {}
            center = (image.header['CRPIX1'],image.header['CRPIX2'])
            data = np.squeeze(image.data)

            y, x = np.indices((data.shape))
            r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            r = r.astype(int)

            r_values, indices, inverse = np.unique(r.ravel(), return_index=True, return_inverse=True)
            rms = np.array([np.median(data.ravel()[inverse == r]) for r in r_values])

            radialprofile['dist'] = r_values*max(image.header['CDELT1'],image.header['CDELT2'])
            radialprofile['rms'] = rms

            with open(filename, 'w') as outfile:
                json.dump(radialprofile,outfile,
                          indent=4, sort_keys=True,
                          separators=(',', ': '),
                          ensure_ascii=False,
                          cls=NumpyEncoder)

        radial_profile = interp1d(radialprofile['dist'], radialprofile['rms']/np.nanmin(radialprofile['rms']), fill_value=1)

        # Determine coverage
        rms_data = image.data.flatten()

        rms_range = np.logspace(np.log10(np.nanmin(rms_data)), np.log10(np.nanmax(rms_data)), 100)
        coverage = [np.sum([rms_data < rms])/np.count_nonzero(~np.isnan(rms_data)) for rms in rms_range]
        # Put sigma_20 into header
        self.table.meta['sigma_20'] = rms_range[np.argmin(np.abs(np.array(coverage)-0.2))]

        # Define a spline and interpolate the values
        rms_coverage = interp1d(rms_range, coverage, fill_value='extrapolate')
        self.npix = np.count_nonzero(~np.isnan(rms_data))

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

        return radial_profile, rms_coverage

    def plot_diff_number_counts(self, flux_col, fancy, dpi, rms_coverage=None, completeness=None, no_plot=False):
        '''
        Compute and plot differential number counts
        input RMS image is used to account for sky coverage

        Keyword arguments:
        flux_col     -- Which table column to use for flux, should be the same
                        as the one used to define the bins
        rms_coverage -- Input rms coverage function
        completeness -- File to correct for completeness
        '''
        counts = {}

        bin_means = [np.mean(self.table[self.flux_col][np.logical_and(self.table[self.flux_col] > self.edges[i],
                             self.table[self.flux_col] < self.edges[i+1])]) for i in range(len(self.edges)-1)]
        counts['solid_angle'] = self.pix_area*self.pix_size**2*(np.pi/180)**2
        counts['dN'] = self.dN
        counts['dS'] = np.diff(self.edges)
        counts['S'] = np.array(bin_means)

        count_correction = 1
        if completeness:
            # Get data from specified file
            with open(completeness, 'r') as infile:
                data = json.load(infile)
            flux_means = np.array([(data['flux_bins'][i]+data['flux_bins'][i+1])/2 for i in range(len(data['flux_bins'])-1)])

            comp_frac = interp1d(flux_means, np.mean(data['detected_fraction'], axis=0), bounds_error=False, fill_value=(0,1))
            count_correction = comp_frac(counts['S'])

        if rms_coverage:
            count_correction = rms_coverage(counts['S']/5.0)
            counts['solid_angle'] = self.npix*self.pix_size**2*(np.pi/180)**2

        # Save diff number counts to json file
        filename = os.path.join(self.dirname, self.cat_name+'_diff_counts.json')
        print(f"--> Saving json file of differential number counts '{filename}'")
        with open(filename, 'w') as outfile:
            json.dump(counts,outfile,
                      indent=4, sort_keys=True,
                      separators=(',', ': '),
                      ensure_ascii=False,
                      cls=NumpyEncoder)

        counts['dN'] = counts['dN']/count_correction
        plt.errorbar(counts['S'], counts['S']**(5/2)*counts['dN']/counts['dS']/counts['solid_angle'],
                     yerr=counts['S']**(5/2)*np.sqrt(counts['dN'])/counts['dS']/counts['solid_angle'],
                     fmt='o', color='k', label='Catalog')

        # Get differential number counts from SKADS and plot
        path = Path(__file__).parent / 'parsets/SKADS_10muJy_diff_counts.json'
        with open(path) as f:
            SKADS = json.load(f)

        if not no_plot:
            SKADS['flux_means'] = np.array([(SKADS['flux_bins'][i]+SKADS['flux_bins'][i+1])/2 for i in range(len(SKADS['flux_bins'])-1)])
            plt.errorbar(SKADS['flux_means'], SKADS['counts_1285'],
                        yerr=SKADS['counts_1285_err'],
                        fmt=':.', color='r', lw=0.5, label='SKADS @ 1285 MHz')

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

    def plot_resolved_fraction(self, resolved_sigma, stacked_cat, fancy, dpi, no_plot=False):
        '''
        Plot fraction of resolved sources
        '''
        cal_error = 0.0

        if stacked_cat:
            resolved_idx = self.table['Resolved']
        else:
            # Calculate errors on total flux and peak flux
            sigma_s = np.sqrt(self.table['E_'+self.flux_col]**2 + (cal_error*self.table[self.flux_col])**2)

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
            for lh in leg.legendHandles: 
                lh.set_alpha(1)

            outfile = os.path.join(self.dirname,self.cat_name+'_resolved.png')
            print(f"--> Saving plot of resolved sources '{outfile}'")

            plt.savefig(outfile, dpi=dpi)
            plt.close()

        return resolved_idx

def main():
    parser = new_argument_parser()
    args = parser.parse_args()

    catalog_file = args.catalog
    rms_image = args.rms_image
    comp_corr = args.comp_corr
    resolved_sigma = args.resolved_sigma
    flux_col = args.flux_col
    stacked_cat = args.stacked_catalog
    no_plots = args.no_plots
    fancy = args.fancy
    dpi = args.dpi

    if fancy:
        plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        plt.rc('text', usetex=True)
        plt.rcParams.update({'font.size': 14})

    catalog = Catalog(catalog_file, flux_col, stacked_cat)
    catalog.get_flux_bins(nbins=50)

    catalog.plot_number_counts(fancy, dpi, no_plot=no_plots)

    if rms_image:
        radial_profile, rms_coverage = catalog.rms_statistics(rms_image, fancy, no_plot=no_plots)
        catalog.plot_diff_number_counts(flux_col, fancy, dpi, rms_coverage=rms_coverage, no_plot=no_plots)
    if comp_corr:
        catalog.plot_diff_number_counts(flux_col, fancy, dpi, completeness=comp_corr, no_plot=no_plots)

    resolved = catalog.plot_resolved_fraction(resolved_sigma, stacked_cat, fancy, dpi, no_plot=no_plots)
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
    parser.add_argument('--flux_col', default='Total_flux', type=str,
                        help="""Name of integrated flux column to use for analysis
                                (default = Total_flux).""")
    parser.add_argument('--stacked_catalog', action='store_true',
                        help="""Indicate if catalog is built up from multiple catalogs,
                                for example with combine_catalogs script.""")
    parser.add_argument('--no_plots', action='store_true',
                        help="""Run the scripts without making any plots.""")
    parser.add_argument('--fancy', action='store_true',
                        help="Output plots with latex font and formatting.")
    parser.add_argument('-d', '--dpi', default=300, type=int,
                        help="DPI of the output images (default = 300).")

    return parser


if __name__ == '__main__':
    main()