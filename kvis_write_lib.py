#
# HRK
#
#  2021
#

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
