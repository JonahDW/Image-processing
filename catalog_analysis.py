#!/usr/bin/env python

import os
import sys
import pickle
import numpy as np

from astropy import units as u
from astropy.io import fits
from astropy.table import Table, join
from astropy.coordinates import SkyCoord

from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

#plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

plt.rcParams.update({'font.size': 14})

def power_law(x, k, gamma):
    return k*x**(-gamma)

class Catalog:

    def __init__(self, catalog_file):
        self.dirname = os.path.dirname(catalog_file)
        self.cat_name = os.path.basename(catalog_file).split('.')[0]
        print(f'Reading in catalog: {self.cat_name}')

        self.table = Table.read(catalog_file)

        self.dN = None
        self.edges = None

    def get_flux_bins(self, flux_col, nbins):
        '''
        Define flux bins for the catalog

        Keyword arguments:
        flux_col -- Which table column to use for flux
        nbins -- Number of bins
        '''
        f_low = np.min(self.table[flux_col])
        f_high = np.max(self.table[flux_col])
        log_bin = np.logspace(np.log10(f_low),np.log10(f_high),nbins)

        self.dN, self.edges = np.histogram(self.table[flux_col], bins=log_bin)

        # Remove high flux bins starting from the first empty bin
        cutoff_high = np.where(self.dN[int(nbins/2):] == 0)[0][0] + int(nbins/2)
        self.edges = self.edges[:cutoff_high+1]
        self.dN = self.dN[:cutoff_high]

    def plot_number_counts(self, dpi):
        '''
        Plot number count bins and fit a power law
        '''
        cutoff_high = np.argmax(self.dN)
        fit_edges = self.edges[cutoff_high:]
        fit_dN = self.dN[cutoff_high:]

        popt, pcov = curve_fit(power_law, fit_edges[:-1], fit_dN, p0=[1,1])
        print(f'Fit power law to number counts with value alpha={popt[1]:.2f}')

        plt.plot(self.edges, power_law(self.edges, *popt), color='k', ls='--', label=f'Power law $\\alpha={popt[1]:.2f}$')
        plt.bar(self.edges[:-1], self.dN, width=np.diff(self.edges), edgecolor='k', alpha=0.5, align='edge')
        plt.xscale('log')
        plt.yscale('log')
        plt.autoscale(enable=True, axis='x', tight=True)

        plt.ylabel('$N$')
        plt.xlabel('$S (\\mathrm{Jy})$')
        plt.legend()

        plt.savefig(os.path.join(self.dirname,self.cat_name+'_number_counts.png'), dpi=dpi)
        plt.close()

    def plot_diff_number_counts(self, rms_image, flux_col, dpi):
        '''
        Compute and plot differential number counts
        input RMS image is used to account for sky coverage

        Keyword arguments:
        rms_image -- Input RMS image
        flux_col -- Which table column to use for flux, should be the same
                    as the one used to define the bins
        '''
        image = fits.open(rms_image)[0]
        rms_data = image.data.flatten()

        # Convert rms to Jy/pixel
        bmaj = float(image.header['BMAJ'])
        bmin = float(image.header['BMIN'])
        pix_size = abs(image.header['CDELT1'])

        beam = np.pi*bmaj*bmin/(4*np.log(2))
        pix_per_beam = beam/pix_size**2
        #rms_data = rms_data/pix_per_beam

        rms_range = np.logspace(np.log10(np.nanmin(rms_data)), np.log10(np.nanmax(rms_data)), 100)
        coverage = [np.sum([rms_data < rms]) for rms in rms_range]

        # Define a splin and interpolate the values
        data_spline = UnivariateSpline(rms_range, coverage, s=0, k=3, ext=3)
        bin_means = [np.mean(self.table[flux_col][np.logical_and(self.table[flux_col] > self.edges[i],
                                                        self.table[flux_col] < self.edges[i+1])]) for i in range(len(self.edges)-1)]
        interp_bins = data_spline(np.array(bin_means)/5.0)

        dS = np.diff(self.edges)
        S = np.array(bin_means)
        angular_size = interp_bins*pix_size**2/3283

        plt.errorbar(S, S**(5/2)*self.dN/dS/angular_size,
                     yerr=S**(5/2)*np.sqrt(self.dN)/dS/angular_size,
                     fmt='o', color='k', label='Catalog')

        # Get differential number counts from SKADS and plot
        path = Path(__file__).parent / 'parsets/intflux_SKADS.pkl'
        with open(path, 'rb') as f:
            intflux = pickle.load(f)
        SKADS_total_flux = 10**intflux

        SKADS_dN, SKADS_edges = np.histogram(SKADS_total_flux, bins=self.edges)
        SKADS_dS = np.diff(SKADS_edges)
        SKADS_angle = 10.0**2/3283

        plt.plot(S, S**(5/2)*SKADS_dN/SKADS_dS/SKADS_angle, 
                 ':rx', lw=0.5, label='SKADS Simulation')

        plt.xscale('log')
        plt.yscale('log')

        plt.ylabel('$S^{5/2}\\  \\mathrm{d}N/\\mathrm{d}S \\ (\\mathrm{Jy}^{3/2} \\mathrm{sr}^{-1})$')
        plt.xlabel('$S (\\mathrm{Jy})$')
        plt.legend()
        plt.savefig(os.path.join(self.dirname, self.cat_name+'_diff_counts.png'), dpi=dpi)
        plt.close()

    def plot_resolved_fraction(self, dpi):
        '''
        Plot ratio of peak flux to integrated flux
        '''
        sigma = np.sqrt(self.table['E_Peak_flux']**2 + self.table['E_Total_flux']**2)
        resolved_idx = self.table['Total_flux'] > self.table['Peak_flux'] + 3*sigma

        resolved = self.table[resolved_idx]
        unresolved = self.table[~resolved_idx]

        plt.scatter(unresolved['Peak_flux'], 
                    unresolved['Total_flux']/unresolved['Peak_flux'], 
                    color='b', s=5, label='Unresolved')
        plt.scatter(resolved['Peak_flux'], 
                    resolved['Total_flux']/resolved['Peak_flux'], 
                    color='r', s=5, label='Resolved')
        plt.xscale('log')
        plt.yscale('log')

        plt.ylabel('$S_{tot}/S_{peak}$')
        plt.xlabel('$S_{peak} (\\mathrm{Jy}$')
        plt.legend()

        plt.savefig(os.path.join(self.dirname,self.cat_name+'_resolved.png'), dpi=dpi)
        plt.close()

def main():

    parser = new_argument_parser()
    args = parser.parse_args()

    catalog_file = args.catalog
    rms_image = args.rms_image
    dpi = args.dpi

    flux_col = 'Total_flux'
    catalog = Catalog(catalog_file)
    catalog.get_flux_bins(flux_col, nbins=50)

    catalog.plot_number_counts(dpi)
    catalog.plot_resolved_fraction(dpi)

    if rms_image:
        catalog.plot_diff_number_counts(rms_image, flux_col, dpi)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("catalog",
                        help="""Pointing catalog made by PyBDSF.""")
    parser.add_argument('-r', '--rms_image', default=None,
                        help="""Specify input rms image for the calculation
                                of differential number counts.""")
    parser.add_argument('-d', '--dpi', default=300,
                        help="""DPI of the output images (default = 300).""")

    return parser


if __name__ == '__main__':
    main()