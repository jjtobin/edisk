import analysisUtils as au
import analysisUtils as aU
import string
import os
import glob
import numpy as np
import sys
import pickle
execfile('../reduction_utils3.py', globals())


###############################################################
################ SETUP/METADATA SPECIFICATION #################
################ USERS NEED TO SET STUFF HERE #################
###############################################################

### Use MPI CASA for faster imaging (start casa with mpicasa -n XX CASA; where XX is the number of processes >= 2)
parallel=True

### if True, can run script non-interactively if later parameters properly set
skip_plots = False	

### Add field names (corresponding to the field in the MS) here and prefix for 
### filenameing (can be different but try to keep same)
### Only make different if, for example, the field name has a space
field   = 'L1489IRS'
prefix  = 'L1489IRS' 

### always include trailing slashes!!
WD_path = '/lustre/cv/projects/edisk/L1489IRS/'
SB_path = WD_path+'SB/'
LB_path = WD_path+'LB/'

### scales for multi-scale clean
SB_scales = [0, 5] #[0, 5, 10, 20]
LB_scales = [0, 5, 30]  #[0, 5, 30, 100, 200]

### automasking parameters for very extended emission
#sidelobethreshold=2.0
#noisethreshold=3.75
#lownoisethreshold=1.0
#smoothfactor=2.0
### automasking parameters for compact emission (uncomment to use)
sidelobethreshold=2.0
noisethreshold=4.0
lownoisethreshold=1.5 
smoothfactor=1.0

import glob

selectedVis='vis'

#pick up extracted MSes from the archived data
vislist=glob.glob('*spectral_line.ms')
vislist_sb=glob.glob('*SB*spectral_line.ms')
vislist_lb=glob.glob('*LB*spectral_line.ms')

#from data_params['SBX/LBX']['cont_spws'] in continuum script
cont_spws_LB=np.array([0, 1, 2, 3, 4, 5, 6])
cont_spws_SB=np.array([0, 1, 2, 3, 4, 5, 6])
cont_spws={}

for vis in vislist_lb:
   cont_spws[vis]=cont_spws_LB.copy()
for vis in vislist_sb:
   cont_spws[vis]=cont_spws_SB.copy()

#from data_params[i]['common_dir'] in continuum script
phasecenter='J2000 04h04m43.080s 26d18m56.104s'

### Dictionary defining the spectral line imaging parameters.
image_list = {
        ### 12CO images
        "12CO":dict(chanstart='-100.0km/s', chanwidth='0.635km/s', 
            nchan=315, linefreq='230.538GHz', linespw='6',
            robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### SO Images
        "SO":dict(chanstart='-10.0km/s', chanwidth='0.167km/s', 
            nchan=270, linefreq='219.94944200GHz', linespw='2',
            robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
       }

for line in image_list:
    print(line)
    for robust in image_list[line]["robust"]:
        imagename = prefix+f'_SBLB_'+line+'_robust_'+str(robust)

        sigma = get_sensitivity_nodata_params(vislist,cont_spws, specmode='cube', \
                spw=[image_list[line]["linespw"]], chan=450,robust=robust,)

        tclean_spectral_line_wrapper(vislist, imagename,
                image_list[line]["chanstart"], image_list[line]["chanwidth"], 
                image_list[line]["nchan"], image_list[line]["linefreq"], 
                image_list[line]["linespw"], SB_scales, threshold=3.0*sigma,
                imsize=image_list[line]["imsize"], cellsize=image_list[line]["cellsize"],
                robust=robust, uvtaper=image_list[line]["uvtaper"],
                sidelobethreshold=sidelobethreshold, noisethreshold=noisethreshold,
                lownoisethreshold=lownoisethreshold,smoothfactor=smoothfactor,parallel=parallel,
                phasecenter=phasecenter.replace('J2000','ICRS'))
    if selectedVis=='vis_shift':
       tclean_spectral_line_wrapper(vislist[0], imagename.replace(prefix,'temporary.pbfix'),
        image_list[line]["chanstart"], image_list[line]["chanwidth"], 
        image_list[line]["nchan"], image_list[line]["linefreq"], 
        image_list[line]["linespw"][1], LB_scales, threshold=3.0*sigma,
        imsize=image_list[line]["imsize"],
        cellsize=image_list[line]["cellsize"],
        robust=robust, uvtaper=image_list[line]["uvtaper"],
        sidelobethreshold=sidelobethreshold, noisethreshold=noisethreshold,
        lownoisethreshold=lownoisethreshold, smoothfactor=smoothfactor,
        parallel=parallel,
        phasecenter=phasecenter.replace('J2000','ICRS'))
       os.system('mv '+imagename+'.pb orig_pbimages/')
       os.system('cp -r '+imagename.replace(prefix,'temporary.pbfix')+'.pb '+imagename+'.pb')
       os.system('rm -rf '+imagename.replace(prefix,'temporary.pbfix')+'*')


import glob
### Remove extra image products
os.system('rm -rf *.residual* *.psf* *.model* *dirty* *.sumwt* *.gridwt* *.workdirectory')

### Remove fits files and pbcor files from previous iterations. 
os.system("rm -rf *.pbcor* *.fits") 

imagelist=glob.glob('*.image') + glob.glob('*.image.tt0')
for image in imagelist:
   if selectedVis=='vis_shift':
       immath(imagename=[image,image.replace('image', 'pb')],expr='IM0/IM1',outfile=image.replace('image', 'pbcor'),imagemd=image)
   else:
       impbcor(imagename=image, pbimage=image.replace('image', 'pb'),
            outfile=image.replace('image', 'pbcor'))

   exportfits(imagename=image.replace('image','pbcor'),fitsimage=image.replace('image','pbcor')+'.fits',overwrite=True,dropdeg=True)
   exportfits(imagename=image,fitsimage=image+'.fits',overwrite=True,dropdeg=True)

imagelist=glob.glob('*.mask')
for image in imagelist:
   exportfits(imagename=image,fitsimage=image+'.fits',overwrite=True,dropdeg=True)
   os.system('gzip '+image+'.fits')

os.system('rm -rf *initcont*.pb')
imagelist=glob.glob('*.pb') + glob.glob('*.pb.tt0')
for image in imagelist:
   exportfits(imagename=image,fitsimage=image+'.fits',overwrite=True,dropdeg=True)
   os.system('gzip '+image+'.fits')

### Remove rescaled selfcal MSfiles

### Remove extra image products
os.system('rm -rf *.residual* *.psf* *.model* *dirty* *.sumwt* *.gridwt* *.workdirectory')

### Make a directory to put the final products
os.system('rm -rf export')
os.system('mkdir export')
os.system('mv *.fits export/')
os.system('mv *.fits.gz export/')
os.system('mv *.tgz export/')
os.system('mv *.pdf export/')

