"""
This script was written for CASA 5.6.2
Based on reduction script for DSHARP data

Datasets calibrated (in order of date observed):
SB1: 2013.1.01086.S
     (1 execution blocks)
LB1: 

reducer: J. Tobin
"""

""" Starting matter """
sys.path.append('/home/casa/contrib/AIV/science/analysis_scripts/') #CHANGE THIS TO YOUR PATH TO THE SCRIPTS!
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
field   = 'IRAS15398'
prefix  = 'IRAS15398' 

### always include trailing slashes!!
WD_path = '/lustre/cv/projects/edisk/IRAS15398/'
SB_path = WD_path+'SB/'
LB_path = WD_path+'LB/'

### scales for multi-scale clean
SB_scales = [0, 5] #[0, 5, 10, 20]
LB_scales = [0, 5, 30]  #[0, 5, 30, 100, 200]

#read in final data_params from continuum to ensure we get the phase centers for each MS
with open(prefix+'.pickle', 'rb') as handle:
    data_params = pickle.load(handle)


###############################################################
#################### SHIFT PHASE CENTERS ######################
###############################################################

for i in data_params.keys():
   data_params[i]['vis_shift']=prefix+'_'+i+'_shift.ms'
   os.system('rm -rf '+data_params[i]['vis_shift']+'*')
   fixvis(vis=data_params[i]['vis'], outputvis=data_params[i]['vis_shift'], 
       field=data_params[i]['field'], 
       phasecenter='J2000 '+data_params[i]['phasecenter'])
   fixplanets(vis=data_params[i]['vis_shift'], field=data_params[i]['field'], 
           direction=data_params[i]['common_dir'])

###############################################################
############### SCALE DATA RELATIVE TO ONE EB #################
###############################################################

### Uses scaling from continuum data
for i in data_params.keys():
   rescale_flux(data_params[i]['vis_shift'], [data_params[i]['gencal_scale']])
   data_params[i]['vis_shift_rescaled']=data_params[i]['vis_shift'].replace('.ms','_rescaled.ms')

with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################
##### APPLY SELF-CALIBRATION SOLUTIONS TO LINE DATA ###########
###############################################################

### Gain tables and spw mapping saved to data dictionaries during selfcal and used as arguments here
for i in data_params.keys():
   applycal(vis=data_params[i]['vis_shift_rescaled'], spw='', 
         gaintable=data_params[i]['selfcal_tables'],spwmap=data_params[i]['selfcal_spwmap'], interp='linearPD', 
         calwt=True, applymode='calonly')
   split(vis=data_params[i]['vis_shift_rescaled'],outputvis=data_params[i]['vis_shift_rescaled'].replace('.ms','.ms.selfcal'),datacolumn='corrected')
   data_params[i]['vis_selfcal']=data_params[i]['vis_shift_rescaled'].replace('.ms','.ms.selfcal')
   ### cleanup
   os.system('rm -rf '+data_params[i]['vis_shift_rescaled']+' '+data_params[i]['vis_shift'])

with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


###############################################################
################## DO CONTINUUM SUBTRACTION ###################
###############################################################

### Get channels to exclude for continuum fitting (same as the ones 
### we flagged for doing making continuum MS)
for i in data_params.keys():
   flagchannels_string = get_flagchannels(data_params[i], prefix)
   print(flagchannels_string)

   ### Get spws for argument list to uvcontsub
   spws_string = get_contsub_spws_indivdual_ms(data_params[i], prefix,only_cont_spws=True)
   print(spws_string)

   ### Run uvcontsub on combined, self-cal applied dataset; THIS WILL TAKE MANY HOURS PER EB
   contsub(data_params[i]['vis_selfcal'], prefix, spw=spws_string,flagchannels=flagchannels_string,excludechans=True)
   os.system('rm -rf '+prefix+'_'+i+'_spectral_line.ms')  ### remove existing spectral line MS if present
   os.system('mv '+data_params[i]['vis_selfcal'].replace('.selfcal','.selfcal.contsub')+' '+prefix+'_'+i+'_spectral_line.ms')
   data_params[i]['vis_contsub']=prefix+'_'+i+'_spectral_line.ms'

with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################
################ TAR UP FINAL CONTSUBBED DATA #################
###############################################################

for i in data_params.keys():
   if 'SB' in i:
      os.system('rm -rf '+data_params[i]['vis_contsub']+'.tar.gz')
      os.system('tar czf '+data_params[i]['vis_contsub']+'.tar.gz '+data_params[i]['vis_contsub'])

###############################################################
############ RUN A FINAL SPECTRAL LINE IMAGE SET ##############
###############################################################

parallel=True     # only set to True if running with mpicasa
prefix='IRAS15398'

### generate list of MS files to image
vislist=[]
for i in data_params.keys():
   if 'SB' in i:
      vislist.append(data_params[i]['vis_contsub'])

### C18O images
chanstart = '-5.5km/s'
chanwidth = '0.167km/s'
nchan = 120
linefreq='219.56035410GHz'
linespw='3'


imagename = prefix+'_SB_C18O_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=2.5,parallel=parallel)


### 13CO images
chanstart = '-5.5km/s'
chanwidth = '0.167km/s'
nchan = 120
linefreq='220.39868420GHz'
linespw='1'


imagename = prefix+'_SB_13CO_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=2.5,parallel=parallel)


### 12CO images
chanstart = '-100.0km/s'
chanwidth = '0.635km/s'
nchan = 315
linefreq='230.538GHz'
linespw='6'

imagename = prefix+'_SB_12CO_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             threshold='0.005Jy',imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=2.5,parallel=parallel)

imagename = prefix+'_SB_12CO_robust_0.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             threshold='0.005Jy',imsize=2600,cellsize='0.015arcsec',robust=0.0, sidelobethreshold=2.0,
                             noisethreshold=2.5,parallel=parallel)
### SO Images
chanstart = '-5.5km/s'
chanwidth = '0.167km/s'
nchan = 120
linefreq='219.94944200GHz'
linespw='2'

imagename = prefix+'_SB_SO_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=3.0,parallel=parallel)

### H2CO 3(2,1)-2(2,0) Images
chanstart = '-5.5km/s'
chanwidth = '0.167km/s'
nchan = 120
linefreq='218.76006600GHz'
linespw='0'

imagename = prefix+'_SB_H2CO_3_21-2_20_218.76GHz_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=3.0,parallel=parallel)


### H2CO 3(0,3)-2(0,2) Images
chanstart = '-10km/s'
chanwidth = '1.34km/s'
nchan = 23
linefreq='218.22219200GHz'
linespw='4'

imagename = prefix+'_SB_H2CO_3_03-2_02_218.22GHz_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=0.5, sidelobethreshold=2.0,
                             noisethreshold=3.0,parallel=parallel)

parallel=True
chanstart = '-10km/s'
chanwidth = '1.34km/s'
nchan = 23
H2CO_spw='4'

imagename = prefix+'_SB_H2CO_3_03-2_02_218.47GHz_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=3.0,parallel=parallel)


### H2CO 3(2,2)-2(2,1) Images
parallel=True
chanstart = '-10km/s'
chanwidth = '1.34km/s'
nchan = 23
linefreq='218.47563200GHz'
linespw='4'

imagename = prefix+'_SB_H2CO_3_03-2_02_218.47GHz_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=3.0,parallel=parallel)

### c-C3H2 217.82 GHz Images
chanstart = '-10km/s'
chanwidth = '1.34km/s'
nchan = 23
linefreq='217.82215GHz'
linespw='4'

imagename = prefix+'_SB_c-C3H2_217.82_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=3.0,parallel=parallel)


### c-C3H2 217.94 GHz Images
chanstart = '-10km/s'
chanwidth = '1.34km/s'
nchan = 23
linefreq='217.94005GHz'
linespw='4'

imagename = prefix+'_SB_cC3H2_217.94_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=3.0,parallel=parallel)


### c-C3H2 218.16 GHz Images
chanstart = '-10km/s'
chanwidth = '1.34km/s'
nchan = 23
linefreq='218.16044GHz'
linespw='4'

imagename = prefix+'_SB_cC3H2_218.16_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=3.0,parallel=parallel)


### DCN Images
chanstart = '-10km/s'
chanwidth = '1.34km/s'
nchan = 23
linefreq='217.2386GHz'
linespw='4'

imagename = prefix+'_SB_DCN_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=3.0,parallel=parallel)

### CH3OH Images
chanstart = '-10km/s'
chanwidth = '1.34km/s'
nchan = 23
linefreq='218.44006300GHz'
linespw='4'

imagename = prefix+'_SB_CH3OH_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=3.0,parallel=parallel)

### SiO Images
parallel=True
chanstart = '-100km/s'
chanwidth = '1.34km/s'
nchan = 150
linefreq='217.10498000GHz'
linespw='4'

imagename = prefix+'_SB_SiO_robust_2.0'
tclean_spectral_line_wrapper(vislist,imagename,chanstart,chanwidth,nchan,linefreq,linespw,SB_scales,
                             nsigma=3.0,imsize=1600,cellsize='0.025arcsec',robust=2.0, sidelobethreshold=2.0,
                             noisethreshold=3.0,parallel=parallel)



#CLEANUP
import glob
### Remove extra image products
os.system('rm -rf *.residual* *.psf* *.model* *dirty* *.sumwt* *.gridwt* *.workdirectory')

### put selfcalibration intermediate images somewhere safe
os.system('mkdir initial_images')
os.system('mv *initcont*.image *contp*.image *contap*.image initial_images')


imagelist=glob.glob('*.image*')
for image in imagelist:
   impbcor(imagename=image,pbimage=image.replace('image','pb'),outfile=image.replace('image','pbcor'))
   exportfits(imagename=image.replace('image','pbcor'),fitsimage=image.replace('image','pbcor')+'.fits',overwrite=True)
   exportfits(imagename=image,fitsimage=image+'.fits',overwrite=True)


### Remove intermediate selfcal MSfiles
os.system("rm -rf *p{0..99}.ms")
### Remove rescaled selfcal MSfiles
os.system('rm -rf *rescaled.ms')
### Remove rescaled selfcal MSfiles
os.system('rm -rf *initcont*.ms')





os.system('rm -rf initial_images/*.ms initial_images/*.residual* initial_images/*.psf* initial_images/*.model* initial_images/*.mask initial_images/*dirty* initial_images/*.sumwt* initial_images/*.gridwt*' )
os.system('rm -rf *.fits.fits')
os.system('cp *.fits export/')
os.system('cp *.ms* export/')


