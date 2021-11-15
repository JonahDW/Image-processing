#
# HRK
#
#  2021
#
import os

from astropy.io import fits
from astropy.io import ascii
from astropy.table import join,setdiff
import numpy as np
from matplotlib import pyplot as plt
from math import *

def write_annotation(outputfilename,source_ra,source_dec,**kwargs):
    """
    source_ID
    source_Bmaj
    source_Bmin
    source_PA
    source_selection
    CROSSSIZE
    """
    # Information about the annotation file can be found via
    # https://www.atnf.csiro.au/computing/software/karma/user-manual/node17.html

    cataloguename = ''
    if kwargs.__contains__('CATALOGUE_NAME') == True:
        cataloguename = kwargs['CATALOGUE_NAME']

    informationline = ''
    if kwargs.__contains__('INFORMATIONLINE') == True:
        informationline = kwargs['INFORMATIONLINE']

    annotate_type = 0
    if kwargs.__contains__('source_Bmaj') == True:
        Bmaj = kwargs['source_Bmaj']
        annotate_type =+1

    if kwargs.__contains__('source_Bmin') == True:
        Bmin = kwargs['source_Bmin']
        annotate_type = annotate_type + 2

    if kwargs.__contains__('source_PA') == True:
        PA   = kwargs['source_PA']
        annotate_type = annotate_type + 3

    dosid = 0
    if kwargs.__contains__('source_ID') == True:
        Sid   = kwargs['source_ID']
        dosid = 1

    crosssize = 5.0/3600. # default size is 5 arcsec 
    if kwargs.__contains__('CROSSSIZE') == True:
        crosssize = kwargs['CROSSSIZE']     # need to be in degrees

    kvis_colour = 'ORANGE'
    if kwargs.__contains__('COLOR') == True:
        if kwargs['COLOR'] != 'RANDOM':
            kvis_colour =  kwargs['COLOR']
        else:
            kvis_colour = np.random.choice(['RED','GREEN','CYAN','ORANGE','YELLOW','WHITE','BLUE']).upper()

    kvis_font = 'hershey14'
    if kwargs.__contains__('FONT') == True:
        kvis_font = kwargs['FONT']
       
    write_line = []
    if annotate_type == 0:
        if dosid > 0:
            for ra,dec,sid in zip(source_ra,source_dec,Sid):
                toprt = str('CROSS %3.6f'%ra)+'  '+str('%3.6f'%dec)+'  '+str('%3.6f'%crosssize)+'\n'
                write_line.append(toprt)
                toprt = str('TEXT %3.6f'%ra)+' '+str('%3.6f'%dec)+'  '+str(sid)+'\n'
                write_line.append(toprt)
        else:
            for ra,dec in zip(source_ra,source_dec):
                toprt = str('CROSS %3.6f'%ra)+'  '+str('%3.6f'%dec)+'  '+str('%3.6f'%crosssize)+'\n'
                write_line.append(toprt)

    elif annotate_type == 1:
        if dosid > 0:
            for ra,dec,bmaj,sid in zip(source_ra,source_dec,Bmaj,Sid):
                toprt = str('CIRCLE %3.6f'%ra)+' '+str('%3.6f'%dec)+'  '+str('%3.6f'%(bmaj/2.))+'\n'
                write_line.append(toprt)
                toprt = str('TEXT %3.6f'%ra)+' '+str('%3.6f'%dec)+'  '+str(sid)+'\n'
                write_line.append(toprt)
        else:
            for ra,dec,bmaj in zip(source_ra,source_dec,Bmaj):
                toprt = str('CIRCLE %3.6f'%ra)+' '+str('%3.6f'%dec)+'  '+str('%3.6f'%(bmaj/2.))+'\n'
                write_line.append(toprt)

    elif annotate_type == 6:
        if dosid > 0:

            for ra,dec,bmaj,bmin,pa,sid in zip(source_ra,source_dec,Bmaj,Bmin,PA,Sid):
                toprt = str('ELLIPSE %3.6f'%ra)+' '+str('%3.6f'%dec)+' '+str('%3.6f'%(bmaj))+' '+\
                  str('%3.6f'%(bmin))+' '+str('%3.4f'%(pa))+'\n'
                write_line.append(toprt)
                toprt = str('TEXT %3.6f'%ra)+' '+str('%3.6f'%dec)+'  '+str(sid)+'\n'
                write_line.append(toprt)
        else:

            for ra,dec,bmaj,bmin,pa in zip(source_ra,source_dec,Bmaj,Bmin,PA):
                toprt = str('ELLIPSE %3.6f'%ra)+' '+str('%3.6f'%dec)+' '+str('%3.6f'%(bmaj))+' '+\
                  str('%3.6f'%(bmin))+' '+str('%3.4f'%(pa))+'\n'
                write_line.append(toprt)

    else:
        print('INPUT IS WRONG ',annotate_type)
        sys.exit(-1)

 
    # Generate output file
    #
    if outputfilename.count('.ANN') == 0 and outputfilename.count('.ann') == 0:
                outputfilename = outputfilename+'.ann'
    
    kvisfile = open(outputfilename,'w')
    kvisfile.writelines('# Annotation file used for KVIS\n')
    kvisfile.writelines('# \n')

    if len(cataloguename) > 0:
        kvisfile.writelines('# Catalouge Name: '+cataloguename+' \n')
        kvisfile.writelines('# \n')

    if len(informationline) > 0:
        kvisfile.writelines('# '+informationline+' \n')
        kvisfile.writelines('# \n')

    kvisfile.writelines('COORD W\n')
    kvisfile.writelines('PA SKY\n')
    kvisfile.writelines('COLOR '+kvis_colour+'\n')
    kvisfile.writelines('FONT '+kvis_font+'\n')
    for line in list(write_line):
        kvisfile.writelines(line)
    kvisfile.close()

    return(outputfilename)

def matches_to_kvis(pointing, ext, matches, annotate, annotate_nonmatchedcat, sigma_extent):
    '''
    Write the results to a kvis annotation file

    CAUTION: KVIS uses the semimajor and semiminor axes for the Ellipse
             Bmaj, Bmin FWHM needs to be divided by a factor of 2
    '''
    # Define the source sizes and match to the semiminor and semimajor axis
    FWHM_to_sigma_extent  = sigma_extent / (2*np.sqrt(2*np.log(2)))
    FWHMtosemimajmin      = 0.5 * FWHM_to_sigma_extent

    match_ext_lines = []
    match_int_lines = []
    non_match_ext_lines = []
    for i, match in enumerate(matches):
        if len(match) > 0:
            source = ext.sources[i]
            toprt = f'ELLIPSE {source.RA:.6f} {source.DEC:.6f} {source.Maj*FWHMtosemimajmin:.6f} {source.Min*FWHMtosemimajmin:.6f} {source.PA:.4f} \n'
            match_ext_lines.append(toprt)
            for ind in match:
                source = pointing.sources[ind]
                toprt = f'ELLIPSE {source.RA:.6f} {source.DEC:.6f} {source.Maj*FWHMtosemimajmin:.6f} {source.Min*FWHMtosemimajmin:.6f} {source.PA:.4f} \n'
                match_int_lines.append(toprt)
        else:
            source = ext.sources[i]
            toprt = f'ELLIPSE {source.RA:.6f} {source.DEC:.6f} {source.Maj*FWHMtosemimajmin:.6f} {source.Min*FWHMtosemimajmin:.6f} {source.PA:.4f} \n'
            non_match_ext_lines.append(toprt)

    non_matches = np.setdiff1d(np.arange(len(pointing.sources)), np.concatenate(matches).ravel())
    non_match_int_lines = []
    for i in non_matches:
        source = pointing.sources[i]
        toprt = f'ELLIPSE {source.RA:.6f} {source.DEC:.6f} {source.Maj*FWHMtosemimajmin:.6f} {source.Min*FWHMtosemimajmin:.6f} {source.PA:.4f} \n'
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