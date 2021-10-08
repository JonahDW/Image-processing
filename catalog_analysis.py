#!/usr/bin/env python

import os
import sys
import pickle
import numpy as np

from astropy import units as u
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

from pathlib import Path
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, interp1d

import helpers

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

plt.rcParams.update({'font.size': 14})

class Catalog:

    def __init__(self, catalog_file):
        self.dirname = os.path.dirname(catalog_file)
        self.cat_name = os.path.basename(catalog_file).split('.')[0]
        print(f'Reading in catalog: {self.cat_name}')

        self.table = Table.read(catalog_file)

        # Parse meta
        header = self.table.meta
        if len(header['OBJECT'].split(',')) > 1:
            self.obj_name = header['OBJECT'].split(',')
        else:
            self.obj_name = header['OBJECT'].replace("'","")
        self.pix_area = np.pi*np.array(header['AXIS1'].split(','), dtype=float)*np.array(header['AXIS2'].split(','), dtype=float)/4
        self.pix_size = float(max(header['CDELT1'],header['CDELT2']))
        self.bmaj = np.array(header['BMAJ'].split(','), dtype=float)
        self.bmin = np.array(header['BMIN'].split(','), dtype=float)

        # Determine frequency axis:
        for i in range(1,5):
            if 'FREQ' in header['CTYPE'+str(i)]:
                freq_idx = i
                break
        self.freq = np.array(header['CRVAL'+str(freq_idx)].split(','), dtype=float)/1e6 #MHz
        self.dfreq = np.array(header['CDELT'+str(freq_idx)].split(','), dtype=float)/1e6 #MHz

        self.center = SkyCoord(np.array(header['CRVAL1'].split(','), dtype=float)*u.degree,
                               np.array(header['CRVAL2'].split(','), dtype=float)*u.degree)

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

        plt.bar(self.edges[:-1], self.dN, width=np.diff(self.edges), edgecolor='k', alpha=0.5, align='edge')
        plt.xscale('log')
        plt.yscale('log')
        plt.autoscale(enable=True, axis='x', tight=True)

        plt.ylabel('$N$')
        plt.xlabel('$S (\\mathrm{Jy})$')

        plt.savefig(os.path.join(self.dirname,self.cat_name+'_number_counts.png'), dpi=dpi)
        plt.close()

    def plot_diff_number_counts(self, flux_col, dpi, rms_image=None, completeness=None):
        '''
        Compute and plot differential number counts
        input RMS image is used to account for sky coverage

        Keyword arguments:
        rms_image -- Input RMS image
        flux_col -- Which table column to use for flux, should be the same
                    as the one used to define the bins
        '''
        # Correct fluxes for primary beam pattern
        source_coord = SkyCoord(self.table['RA'], self.table['DEC'], unit='deg')
        offsets = source_coord.separation(self.center).deg
        flux_corr = helpers.flux_correction(offsets, self.freq[0], self.dfreq[0], 0.8)

        self.table[flux_col] /= flux_corr

        bin_means = [np.mean(self.table[flux_col][np.logical_and(self.table[flux_col] > self.edges[i],
                             self.table[flux_col] < self.edges[i+1])]) for i in range(len(self.edges)-1)]
        solid_angle = self.pix_area*self.pix_size**2*(np.pi/180)**2
        dS = np.diff(self.edges)
        S = np.array(bin_means)

        count_correction = 1
        if completeness:
            # Get data from specified file
            data = helpers.pickle_from_file(completeness)
            flux_means = np.array([(data[0][i]+data[0][i+1])/2 for i in range(len(data[0])-1)])

            comp_frac = interp1d(flux_means, np.mean(data[1], axis=0), bounds_error=False, fill_value=(0,1))
            count_correction = comp_frac(S)

        if rms_image:
            image = fits.open(rms_image)[0]
            rms_data = image.data.flatten()

            rms_range = np.logspace(np.log10(np.nanmin(rms_data)), np.log10(np.nanmax(rms_data)), 100)
            coverage = [np.sum([rms_data < rms])/np.count_nonzero(~np.isnan(rms_data)) for rms in rms_range]

            # Define a splin and interpolate the values
            rms_coverage = interp1d(rms_range, coverage, fill_value='extrapolate')
            count_correction = rms_coverage(S/5.0)
            solid_angle = np.count_nonzero(~np.isnan(rms_data))*self.pix_size**2*(np.pi/180)**2

            # Plot rms coverage
            plt.plot(rms_range, coverage, linewidth=2, color='k')
            plt.fill_between(rms_range, 0, coverage, color='k', alpha=0.2)

            plt.xscale('log')
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.xlabel('$\\sigma$ (Jy/beam)')
            plt.ylabel('Coverage')

            plt.savefig(os.path.join(self.dirname, self.cat_name+'_rms_coverage.png'), dpi=300)
            plt.close()

        # Save diff number counts to pickle file
        data = S, dS, self.dN, solid_angle
        helpers.pickle_to_file(data, os.path.join(self.dirname, self.cat_name+'_diff_counts.pkl'))

        self.dN = self.dN/count_correction
        plt.errorbar(S, S**(5/2)*self.dN/dS/solid_angle,
                     yerr=S**(5/2)*np.sqrt(self.dN)/dS/solid_angle,
                     fmt='o', color='k', label='Catalog')

        # Get differential number counts from SKADS and plot
        path = Path(__file__).parent / 'parsets/intflux_SKADS.pkl'
        intflux = helpers.pickle_from_file(path)
        SKADS_total_flux = 10**intflux

        SKADS_dN, SKADS_edges = np.histogram(SKADS_total_flux, bins=self.edges)
        SKADS_dS = np.diff(SKADS_edges)
        SKADS_angle = 10.0**2*(np.pi/180)**2

        plt.errorbar(S, S**(5/2)*SKADS_dN/SKADS_dS/SKADS_angle,
                    yerr=S**(5/2)*np.sqrt(SKADS_dN)/SKADS_dS/SKADS_angle,
                    fmt=':.', color='r', lw=0.5, label='SKADS Simulation')

        plt.xscale('log')
        plt.yscale('log')

        plt.ylabel('$S^{5/2}\\  \\mathrm{d}N/\\mathrm{d}S \\ (\\mathrm{Jy}^{3/2} \\mathrm{sr}^{-1})$')
        plt.xlabel('$S (\\mathrm{Jy})$')
        plt.legend()
        plt.savefig(os.path.join(self.dirname, self.cat_name+'_diff_counts.png'), dpi=dpi)
        plt.close()

    def plot_resolved_fraction(self, dpi):
        '''
        Plot fraction of resolved sources
        '''
        def isresolved(catalog, bmaj, bmin):
            # Check if this source is resolved (>2.33sigma beam; 98% confidence)
            sizeerror = size_error_condon(catalog, bmaj, bmin)
            majcompare = bmaj+(2.33*sizeerror[0])
            mincompare = bmin+(2.33*sizeerror[1])

            resolved_idx = catalog['Maj'] > majcompare
            return resolved_idx

        def size_error_condon(catalog, beam_maj, beam_min):
            # Implement errors for elliptical gaussians in the presence of correlated noise
            # as per Condon (1998), MNRAS.
            ncorr = beam_maj*beam_min

            rho_maj = ((catalog['Maj']*catalog['Min'])/(4*ncorr)
                       *(1 + (ncorr/catalog['Maj'])**2)**2.5
                       *(1 + (ncorr/catalog['Min'])**2)**0.5
                       *(catalog['Peak_flux']/catalog['Isl_rms'])**2)
            rho_min = ((catalog['Maj']*catalog['Min'])/(4*ncorr)
                       *(1 + ncorr/(catalog['Maj'])**2)**0.5
                       *(1 + ncorr/(catalog['Min'])**2)**2.5
                       *(catalog['Peak_flux']/catalog['Isl_rms'])**2)
            majerr = np.sqrt(2*(catalog['Maj']/rho_maj)**2 + (0.02*beam_maj)**2)
            minerr = np.sqrt(2*(catalog['Min']/rho_min)**2 + (0.02*beam_min)**2)

            return majerr, minerr

        if isinstance(self.obj_name, list):
            resolved = Table()
            unresolved = Table()
            for i, name in enumerate(self.obj_name):
                table_in = self.table[self.table['Pointing_id'] == 'PT-'+name.replace("'","")]
                resolved_idx = isresolved(table_in, self.bmaj[i], self.bmin[i])

                resolved = vstack([resolved,table_in[resolved_idx]])
                unresolved = vstack([unresolved,table_in[~resolved_idx]])
        else:
            resolved_idx = isresolved(self.table, self.bmaj, self.bmin)

            resolved = self.table[resolved_idx]
            unresolved = self.table[~resolved_idx]

        plt.scatter(unresolved['Peak_flux']/unresolved['Isl_rms'],
                    unresolved['Total_flux']/unresolved['Peak_flux'],
                    color='b', s=5, label=f'Unresolved ({len(unresolved)})',
                    alpha=0.1)
        plt.scatter(resolved['Peak_flux']/resolved['Isl_rms'],
                    resolved['Total_flux']/resolved['Peak_flux'],
                    color='r', s=5, label=f'Resolved ({len(resolved)})',
                    alpha=0.1)
        plt.xscale('log')
        plt.yscale('log')

        plt.ylabel('$S_{tot}/S_{peak}$')
        plt.xlabel('S/N')
        leg = plt.legend()
        for lh in leg.legendHandles: 
            lh.set_alpha(1)

        plt.savefig(os.path.join(self.dirname,self.cat_name+'_resolved.png'), dpi=dpi)
        plt.close()

        return resolved_idx

def main():
    parser = new_argument_parser()
    args = parser.parse_args()

    catalog_file = args.catalog
    rms_image = args.rms_image
    comp_corr = args.comp_corr
    dpi = args.dpi

    flux_col = 'Total_flux'
    catalog = Catalog(catalog_file)
    catalog.get_flux_bins(flux_col, nbins=50)

    catalog.plot_number_counts(dpi)
    resolved = catalog.plot_resolved_fraction(dpi)

    if rms_image or comp_corr:
        catalog.plot_diff_number_counts(flux_col, dpi, rms_image, comp_corr)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("catalog",
                        help="""Pointing catalog(s) made by PyBDSF.""")
    parser.add_argument('-r', '--rms_image', default=None,
                        help="""Specify input rms image for creating an rms coverage
                                plot. In the absence of a completeness correction file,
                                will also be used to correct for completeness.""")
    parser.add_argument('-c', '--comp_corr', default=None,
                        help="""Specify input pickle file containing completeness
                                fractions for correcting differential number counts.
                                the file is assumed to contain at least the arrays of 
                                flux bins, completeness fraction.""")
    parser.add_argument('-d', '--dpi', default=300,
                        help="""DPI of the output images (default = 300).""")

    return parser


if __name__ == '__main__':
    main()