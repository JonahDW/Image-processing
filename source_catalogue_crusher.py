#
# Hans-Rainer Kloeckner
# MPIfR 2021
# hrk@mpifr-bonn.mpg.de
#
import sys
from optparse import OptionParser
from math import *
import json

from astropy.io import fits
from astropy.io import ascii
from astropy.table import join,setdiff,Table

import numpy as np
from matplotlib import pyplot as plt

from kvis_write_lib import *

# ----
# This tool is an add on to the
# https://github.com/JonahDW/Image-processing
# cataloger software
# ----

#
# Use this software to mark source that may not be included in your further analysis
#

# 
# Information about the PYBDFS catalouge entries
# can be found:
#
# https://www.astron.nl/citt/pybdsf/write_catalog.html
#

# Isl_Total_flux: the total, integrated Stokes I flux density of the island in which the source is located, in Jy. 
# This value is calculated from the sum of all non-masked pixels in the island with values above thresh_isl
# E_Isl_Total_flux: the 1-σ error on the total flux density of the island in which the source is located, in Jy

# Total_flux
# E_Total_flux 

# Info about fits handleing
# https://docs.astropy.org/en/stable/io/fits/index.html
#




def main():

    # argument parsing
    #
    print('\n== Source Catalogue Crusher == \n')
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)

    parser.add_option('--CAT_FILE', dest='catfile', type=str,
                      help='CAT - file name e.g. CATALOUGE.fits')


    parser.add_option('--OUTPUTNAME', dest='outputfilename', type=str,default='',
                      help='CATALOUGE outputfile name e.g. CATALOUGE.fits')


    parser.add_option('--DO_SELECT_PYBDSF_TYPE', dest='doselecttype', type=str, default='',
                      help='Select catalouges sources on type. [e.g. M or -M, this would select all except M]')

    parser.add_option('--DO_SELECT_SOURCE_MAJ_DECON', dest='doselectsources_maj_decon',type=float, default=0,
                      help='Select  sources on deconvolved major axis = minor axis = zero')

    parser.add_option('--DO_SELECT_SOURCE_MAJ_FIT', dest='doselectextendedsources', type=float, default=0,
                      help='Select sources Major Axis FIT that is larger than the Deconvolved Maj Axis, input value is sigma or -sigma to exclude these source')

    parser.add_option('--DO_SELECT_SOURCE_TOTFLX_ISLANDFLX', dest='doselectonfluxdensity', type=float, default=0,
                      help='Select  sources on total flux density matching the total island flux density, input value is sigma or -sigma to exclude these sources')


    parser.add_option('--DO_SELECT_ON', dest='doselecton',type=str, default='',
                      help='Set selection based on table column [e.g. Maj]')

    parser.add_option('--DO_SELECT_OPERATION', dest='doselectonoperation',type=str, default='=',
                      help='Input operation of selection [e.g. = (default), >, <]')

    parser.add_option('--DO_SELECT_VALUE', dest='doselectonvalue',type=str, default='',
                      help='Input value of selection')

    parser.add_option('--KVISANNOUTPUT', dest='kvisoutputfilename', type=str,default='',
                      help='KVIS annotation outputfile name e.g. CATALOUGE.ann')

    parser.add_option('--KVISCOLOR', dest='kviscolor',type=str,default='ORANGE',
                      help='Change COLOR of KVIS annotation [e.g. RANDOM ]')

    parser.add_option('--KVISPRINT', dest='kvisprint',action='store_false',default=True,
                      help='use the fitted values Maj,Min,PA or deconvolved values DC_Maj, DC_Min, DC_PA')

    parser.add_option('--TABLECOLUMNOUTPUT', dest='tcolumnoutfilename', type=str,default='',
                      help='write table column output as ASCII to be edit by hand')

    parser.add_option('--TABLECOLUMNINPUT', dest='tcolumninfilename', type=str,default='',
                      help='read table column ASCII file.')

    parser.add_option('--DO_PRINT_INFO', dest='doprtcatainfo', action='store_true', default=False,
                      help='Print some statistics of the catalouge')

    parser.add_option('--DO_PRINT_TAB_COLUMN', dest='doprtcatacol', action='store_true', default=False,
                      help='Print some information of the catalouge')



    # ----

    (opts, args)         = parser.parse_args()

    if opts.catfile == None:
    
        parser.print_help()
        sys.exit()


    # set the parmaters
    #
    fits_tab           = opts.catfile
    outputfilename     = opts.outputfilename
    kvisoutputfilename = opts.kvisoutputfilename
    tcolumnoutfilename = opts.tcolumnoutfilename
    tcolumninfilename  = opts.tcolumninfilename

    kvisprint          = opts.kvisprint
    kviscolor          = opts.kviscolor
    doprtcatainfo      = opts.doprtcatainfo
    doprtcatacols      = opts.doprtcatacol
    #
    doselecttype              = opts.doselecttype
    doselectextendedsources   = opts.doselectextendedsources
    doselectsources_maj_decon = opts.doselectsources_maj_decon
    doselectonfluxdensity     = opts.doselectonfluxdensity
    doselecton                = opts.doselecton
    doselectonvalue           = opts.doselectonvalue
    doselectonoperation       = opts.doselectonoperation
    #
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------


    hdul = fits.open(fits_tab)

    hdr  = hdul[0].header

    # cata data
    data  = hdul[1].data
    #print(data[0])

    # column information
    cols  = hdul[1].columns
    
    # print cataloge colomn
    #
    if doprtcatacols:
        #print(cols.info())
        #print(cols)
        #print(dir(cols))
        #print(cols.dtype)
        #print(cols.names)
        #print(cols.units)

        c = 1
        print('\n\n === Catalog Column  ===')
        for s in cols:
            print('column ',c,'\t',s.name,'\t[',s.unit,']')
            c +=1
        print('\n\n === ')
        sys.exit(-1)


    # get the pre-selection flag
    Quality_Flag     = hdul[1].data['Quality_flag']
    pre_select       = Quality_Flag.astype(dtype=bool)

    # get the data
    catalog = hdul[1].data

    if doprtcatainfo:

        info ={}

        # determine preselection counts
        pre_excluded_sources = np.count_nonzero(np.invert(pre_select))

        # determinne information of the catalouge
        #
        minmax_data  = ['Total_flux','RA','DEC']
        if len(doselecton) > 0:
            minmax_data.append(doselecton)
        get_stats    = [np.min,np.max,np.sum,len]
        #
        #
        total_no_of_sources = len(list(catalog['S_Code']))
        info['number_of_catalouged_sources'] = total_no_of_sources

        info['pre_excluded_sources'] = pre_excluded_sources

        # get stats
        for mmdat in minmax_data:
                info[mmdat] = {}
                for getst in get_stats:
                        info[mmdat][getst.__name__]=getst((catalog[mmdat])[pre_select])

        # determine the number of different types of sources
        source_prop          = np.unique(list(catalog['S_Code']))
        info['Source_types'] = source_prop 
        for sp in source_prop:
            sel_s_code    = catalog['S_Code'] == sp
            sel_s_sources = np.logical_and(pre_select,sel_s_code)
            #
            info[sp] = {}
            info[sp]['number_of_catalouged_sources'] = list((catalog['S_Code'])[sel_s_code]).count(sp)
            # get stats of individual types
            for mmdat in minmax_data:
                    info[sp][mmdat] = {}
                    for getst in get_stats:
                        info[sp][mmdat][getst.__name__] = getst(catalog[mmdat][sel_s_sources])

        print('\n=== Catalogue Information ===') 
        for k in info:
            print(k,' ',info[k])

        sys.exit(-1)


    # Read in and write out the Quality_flag of the table
    if len(tcolumnoutfilename) > 0:
        ascii_data = Table()
        ascii_data['Source_id']    = catalog['Source_id']
        ascii_data['S_Code']       = catalog['S_Code']
        ascii_data['Quality_flag'] = catalog['Quality_flag']

        ascii.write(ascii_data,tcolumnoutfilename, overwrite=True)
        print('ASCII file ',tcolumnoutfilename,'has been written')
        sys.exit(-1)


    # Read in and write out the Quality_flag of the table
    if len(tcolumninfilename) > 0:
        ascii_data   = ascii.read(tcolumninfilename)
        column_names = ascii_data.columns.keys()

        if column_names.count('Source_id') > 0 and column_names.count('Quality_flag') > 0:

            Quality_Flag     = ascii_data['Quality_flag'].data
            Source_ID        = ascii_data['Source_id'].data
            
            if list(Source_ID) == list(hdul[1].data['Source_id']):
                pre_select       = Quality_Flag.astype(dtype=bool)
            else:
                print('Input file ',tcolumninfilename,'does not match the table entries.')
                sys.exit(-1)


    # General selection on input column 
    #
    if len(doselecton) > 0:
        inp_sel_data = catalog[doselecton]

        print(doselecton,' Column',inp_sel_data)

        if doselectonoperation == '>':
            sel_inp      = inp_sel_data > eval(doselectonvalue)
        elif doselectonoperation == '<':
            sel_inp      = inp_sel_data < eval(doselectonvalue)
        elif doselectonoperation == '=':
            sel_inp      = inp_sel_data == eval(doselectonvalue)
        elif doselectonoperation == '!=':
            sel_inp      = inp_sel_data != eval(doselectonvalue)
        else:
            print('Operation ',doselectonoperation,' unknown')

        print('Source based on ',doselecton,' using ',doselectonoperation,' of ', doselectonvalue,' select',np.count_nonzero(sel_inp))
    else:
        sel_inp = pre_select


    # Select the source on catalouges type
    #
    if len(doselecttype) > 0:
        #
        if list(np.unique(list(catalog['S_Code']))).count(doselecttype.replace('-','')) > 0:
            # Select source based on S_Code
            #
            # S == a single-Gaussian source that is the only source in the island 
            # M == a multi-Gaussian source
            # C == a single-Gaussian source in an island with other sources
            #
            if doselecttype.count('-'):
                sel_s_code = catalog['S_Code'] != doselecttype
                #
                print('Select source',np.count_nonzero(sel_s_code),' not of type ',doselecttype)
            else:
                sel_s_code = catalog['S_Code'] == doselecttype
                #
                print('Select source',np.count_nonzero(sel_s_code),' of type ',doselecttype)
            # ---
        else:
            print('NOTE: No source type ',doselecttype,' catalouged, use full list of sources.')
            sel_s_code = pre_select
    else:
        sel_s_code = pre_select


    # Select point sources based on deconvolved major axis
    #
    if doselectsources_maj_decon != 0:
        sel_DC_point_source  =  np.logical_and(catalog['DC_Maj'] == 0.0,\
                                            catalog['DC_Min'] == 0.0)
        if doselectsources_maj_decon > 0:
            print('Select sources Maj = Min = 0 Deconvolved Axis ',np.count_nonzero(sel_DC_point_source))
        else:
            sel_DC_point_source  = np.invert(sel_DC_point_source)
            print('Select sources that are not Maj = Min = 0 Deconvolved Axis  ',np.count_nonzero(sel_DC_point_source))

    else:
        sel_DC_point_source = pre_select

    # ----


    # Select extended sources based on Fiting error and deconvolved major axis
    #
    if doselectextendedsources != 0:
        sigma_fitting            = abs(doselectextendedsources)
        sour_fit_error           =  sigma_fitting * catalog['E_Maj'] + sigma_fitting * catalog['E_DC_Maj']
        sel_FIT_extended_source  =  np.logical_and(catalog['Maj'] + sour_fit_error > catalog['DC_Maj'], catalog['DC_Maj'] != 0.0)

        if doselectextendedsources > 0:
            print('Select extended sources based on MAJ fitting information',np.count_nonzero(sel_FIT_extended_source))
        else:
            sel_FIT_extended_source  = np.invert(sel_FIT_extended_source)
            print('Select point sources based on MAJ fitting information',np.count_nonzero(sel_FIT_extended_source))
        # ----
    else:
        sel_FIT_extended_source = pre_select

    # ----


    # Select sources based on the total flux density and the total integrated flux density of the island
    #
    if doselectonfluxdensity != 0:

        sigma_flx   = abs(doselectonfluxdensity)
        flx_error   = sigma_flx * catalog['E_Isl_Total_flux'] + sigma_flx * catalog['E_Total_flux']
        sel_tot_flx =  np.logical_and(catalog['Isl_Total_flux'] + flx_error >= catalog['Total_flux'],\
                                          catalog['Isl_Total_flux'] - flx_error <= catalog['Total_flux'])
        #
        if doselectonfluxdensity > 0:
            print('Source flux density does match the island flux density ',np.count_nonzero(sel_tot_flx))
        else:
            sel_tot_flx = np.invert(sel_tot_flx)
            print('Source flux density does not match the island flux density ',np.count_nonzero(sel_tot_flx))
        # ----

    else:
        sel_tot_flx = pre_select

    # ----

    # Combine all preselection criteria
    #
    total_selection = np.logical_and(pre_select,pre_select)
    selections     = [sel_inp,sel_s_code,sel_DC_point_source,sel_FIT_extended_source,sel_tot_flx]

    for sels in selections:
        total_selection = np.logical_and(total_selection,sels)

    tot_selected_sources = np.count_nonzero(total_selection)

    print('Total selected sources ',tot_selected_sources)

    # ----------------------------


    if tot_selected_sources > 0:
        if len(outputfilename) > 0:

            if outputfilename.count('.FITS') == 0 or outputfilename.count('.fits') == 0:
                outputfilename = outputfilename+'.FITS'


            new_Q_select   = total_selection.astype(dtype=int)

            hdul[1].data['Quality_flag'] = new_Q_select

            newhdu = fits.BinTableHDU(data=hdul[1].data)

            newhdu.writeto(outputfilename)

            hdul.close()
            print('New Catalouge file has been written',outputfilename)

    else:
        print('No sources have been selected a new cataloge file has not been written')



    if len(kvisoutputfilename) > 0:
        if kvisoutputfilename.count('.ANN') == 0 and kvisoutputfilename.count('.ann') == 0:
                kvisoutputfilename = kvisoutputfilename+'.ann'

        if kvisprint:
            kvisprt = ['Maj','Min','PA']
        else:
            kvisprt = ['DC_Maj','DC_Min','DC_PA']

        
        write_annotation(kvisoutputfilename,catalog['RA'][total_selection],catalog['DEC'][total_selection],\
                             source_Bmaj=catalog[kvisprt[0]][total_selection],source_Bmin=catalog[kvisprt[1]][total_selection],\
                             source_PA=catalog[kvisprt[2]][total_selection],source_ID=catalog['Source_id'][total_selection],\
                             INFORMATIONLINE=' Used parameter '+str(kvisprt)+'\n'+'# \n'+'#  No Sources: '+str(tot_selected_sources),COLOR=kviscolor,CATALOGUE_NAME=fits_tab)


    sys.exit(-1)
    

if __name__ == "__main__":
    main()


