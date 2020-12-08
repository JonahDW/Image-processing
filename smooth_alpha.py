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
         outfile=os.path.join(dirname,'smooth.alpha'),
         kernel='gauss',
         beam={"major":major, "minor":minor, "pa":pa})

imsmooth(imagename=imname+'.alpha.error',
         outfile=os.path.join(dirname,'smooth.alpha.error'),
         kernel='gauss',
         beam={"major":major, "minor":minor, "pa":pa})

exportfits(imagename=os.path.join(dirname,'smooth.alpha'),
           fitsimage=os.path.join(dirname,'smooth_alpha.fits'))

exportfits(imagename=os.path.join(dirname,'smooth.alpha.error'),
           fitsimage=os.path.join(dirname,'smooth_alpha_error.fits'))

shutil.rmtree(os.path.join(dirname,'smooth.alpha'))
shutil.rmtree(os.path.join(dirname,'smooth.alpha.error'))