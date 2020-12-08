import sys
import os
import shutil

imname = sys.argv[3]

dirname = os.path.dirname(imname)

hlist = imhead(imname+'.alpha', mode='list')

major = str(hlist['beammajor']['value'])+hlist['beammajor']['unit']
minor = str(hlist['beamminor']['value'])+hlist['beamminor']['unit']
pa = str(hlist['beampa']['value'])+hlist['beampa']['unit']

imsmooth(imagename=imname+'.alpha',
         outfile=dirname+'/smooth.alpha',
         kernel='gauss',
         beam={"major":major, "minor":minor, "pa":pa})

imsmooth(imagename=imname+'.alpha.error',
         outfile=dirname+'/smooth.alpha.error',
         kernel='gauss',
         beam={"major":major, "minor":minor, "pa":pa})

exportfits(imagename=dirname+'/smooth.alpha',
           fitsimage=dirname+'/smooth_alpha.fits')

exportfits(imagename=dirname+'/smooth.alpha.error',
           fitsimage=dirname+'/smooth_alpha_error.fits')

shutil.rmtree(dirname+'/smooth.alpha')
shutil.rmtree(dirname+'/smooth.alpha.error')