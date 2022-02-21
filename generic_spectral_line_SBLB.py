"""
eDisk data reduction script
This script was written for CASA 6.1.1/6.2
Originally derived from DSHARP reduction scripts

Datasets calibrated (in order of date observed):
SB1: 

LB1: 

reducer: 
"""

""" Starting matter """
#sys.path.append('/home/casa/contrib/AIV/science/analysis_scripts/') #CHANGE THIS TO YOUR PATH TO THE SCRIPTS!
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
field   = 'fieldName'
prefix  = 'filename prefix' 

### always include trailing slashes!!
WD_path = '/lustre/cv/projects/edisk/sourceDirectory/'
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

#read in final data_params from continuum to ensure we get the phase centers for each MS
with open(prefix+'.pickle', 'rb') as handle:
    data_params = pickle.load(handle)


###############################################################
#################### SHIFT PHASE CENTERS ######################
###############################################################
selectedVis='vis'
#selectedVis='vis_shift'

if selectedVis == 'vis_shift':
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
   rescale_flux(data_params[i][selectedVis].replace(WD_path,''), [data_params[i]['gencal_scale']])
   data_params[i]['vis_rescaled']=data_params[i][selectedVis].replace('.ms','_rescaled.ms')

with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################
##### APPLY SELF-CALIBRATION SOLUTIONS TO LINE DATA ###########
###############################################################

### Gain tables and spw mapping saved to data dictionaries during selfcal and used as arguments here
for i in data_params.keys():
   n_tables=len(data_params[i]['selfcal_tables'])
   interp_list=['linearPD']*n_tables
   applycal(vis=data_params[i]['vis_rescaled'], spw='', 
         gaintable=data_params[i]['selfcal_tables'],spwmap=data_params[i]['selfcal_spwmap'], interp=interp_list, 
         calwt=True, applymode='calonly')
   split(vis=data_params[i]['vis_rescaled'],outputvis=data_params[i]['vis_rescaled'].replace('.ms','.ms.selfcal'),datacolumn='corrected')
   data_params[i]['vis_selfcal']=data_params[i]['vis_rescaled'].replace('.ms','.ms.selfcal')
   ### cleanup
   os.system('rm -rf '+data_params[i]['vis_rescaled'])
   if selectedVis=='vis_shift':
      os.system('rm -rf '+data_params[i]['vis_shift'])

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
   os.system('rm -rf '+data_params[i]['vis_selfcal'])
   data_params[i]['vis_contsub']=prefix+'_'+i+'_spectral_line.ms'

with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################
################ TAR UP FINAL CONTSUBBED DATA #################
###############################################################

for i in data_params.keys():
      os.system('rm -rf '+data_params[i]['vis_contsub']+'.tgz')
      os.system('tar czf '+data_params[i]['vis_contsub']+'.tgz '+data_params[i]['vis_contsub'])

###############################################################
############ RUN A FINAL SPECTRAL LINE IMAGE SET ##############
###############################################################


### generate list of MS files to image
vislist=[]
for i in data_params.keys():
   if 'SB' in i:
      vislist.append(data_params[i]['vis_contsub'])

### Dictionary defining the spectral line imaging parameters.

### generate list of MS files to image
vislist=[]
vislist_sb=[]
for i in data_params.keys():
   vislist.append(data_params[i]['vis_contsub'])
   if 'SB' in i:
      vislist_sb.append(data_params[i]['vis_contsub'])

### Dictionary defining the spectral line imaging parameters.

image_list = {
        ### C18O images
        "C18O":dict(chanstart='-5.5km/s', chanwidth='0.167km/s',
            nchan=120, linefreq='219.56035410GHz', linespw='3',
            robust=[0.5,-0.5,0.],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### 13CO images
        "13CO":dict(chanstart='-5.5km/s', chanwidth='0.167km/s',
            nchan=120, linefreq='220.39868420GHz', linespw='1', 
            robust=[0.5,-0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### 12CO images
        "12CO":dict(chanstart='-100.0km/s', chanwidth='0.635km/s', 
            nchan=315, linefreq='230.538GHz', linespw='6',
            robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### SO Images
        "SO":dict(chanstart='-5.5km/s', chanwidth='0.167km/s', 
            nchan=120, linefreq='219.94944200GHz', linespw='2',
            robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### H2CO 3(2,1)-2(2,0) Images
        "H2CO_3_21-2_20_218.76GHz":dict(chanstart='-5.5km/s', 
            chanwidth='0.167km/s', nchan=120, linefreq='218.76006600GHz', 
            linespw='0', robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### H2CO 3(0,3)-2(0,2) Images
        "H2CO_3_03-2_02_218.22GHz":dict(chanstart='-10km/s',
            chanwidth='1.34km/s', nchan=23, linefreq='218.22219200GHz', 
            linespw='4', robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### H2CO 3(2,2)-2(2,1) Images
        "H2CO_3_22-2_21_218.47GHz":dict(chanstart='-10km/s', 
            chanwidth='1.34km/s', nchan=23, linefreq='218.47563200GHz',
            linespw='4', robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### c-C3H2 217.82 GHz Images
        "c-C3H2_217.82":dict(chanstart='-10km/s', chanwidth='1.34km/s', 
            nchan=23, linefreq='217.82215GHz', linespw='4', robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### c-C3H2 217.94 GHz Images
        "cC3H2_217.94":dict(chanstart='-10km/s', chanwidth='1.34km/s', 
            nchan=23, linefreq='217.94005GHz', linespw='4', robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### c-C3H2 218.16 GHz Images
        "cC3H2_218.16":dict(chanstart='-10km/s', chanwidth='1.34km/s', 
            nchan=23, linefreq='218.16044GHz', linespw='4', robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### DCN Images
        "DCN":dict(chanstart='-10km/s', chanwidth='1.34km/s', nchan=23, 
            linefreq='217.2386GHz', linespw='4', robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### CH3OH Images
        "CH3OH":dict(chanstart='-10km/s', chanwidth='1.34km/s', nchan=23, 
            linefreq='218.44006300GHz', linespw='4', robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda']),
        ### SiO Images
        "SiO":dict(chanstart='-100km/s', chanwidth='1.34km/s', nchan=150, 
            linefreq='217.10498000GHz', linespw='4', robust=[0.5],imsize=4000,cellsize='0.01arcsec',uvtaper=['2000klambda'])
        }

'''
image_list_sb = {
        ### C18O images
        "C18O":dict(chanstart='-5.5km/s', chanwidth='0.167km/s',
            nchan=120, linefreq='219.56035410GHz', linespw='3',
            robust=[1.0],imsize=1600,cellsize='0.025arcsec'),
        ### 13CO images
        "13CO":dict(chanstart='-5.5km/s', chanwidth='0.167km/s',
            nchan=120, linefreq='220.39868420GHz', linespw='1', 
            robust=[1.0],imsize=1600,cellsize='0.025arcsec'),
        ### 12CO images
        "12CO":dict(chanstart='-100.0km/s', chanwidth='0.635km/s', 
            nchan=315, linefreq='230.538GHz', linespw='6',
            robust=[2.0],imsize=1600,cellsize='0.025arcsec'),
        ### SO Images
        "SO":dict(chanstart='-5.5km/s', chanwidth='0.167km/s', 
            nchan=120, linefreq='219.94944200GHz', linespw='2',
            robust=[2.0],imsize=1600,cellsize='0.025arcsec'),
        ### H2CO 3(2,1)-2(2,0) Images
        "H2CO_3_21-2_20_218.76GHz":dict(chanstart='-5.5km/s', 
            chanwidth='0.167km/s', nchan=120, linefreq='218.76006600GHz', 
            linespw='0', robust=[2.0],imsize=1600,cellsize='0.025arcsec'),
        ### H2CO 3(0,3)-2(0,2) Images
        "H2CO_3_03-2_02_218.22GHz":dict(chanstart='-10km/s',
            chanwidth='1.34km/s', nchan=23, linefreq='218.22219200GHz', 
            linespw='4', robust=[2.0],imsize=1600,cellsize='0.025arcsec'),
        ### H2CO 3(2,2)-2(2,1) Images
        "H2CO_3_22-2_21_218.47GHz":dict(chanstart='-10km/s', 
            chanwidth='1.34km/s', nchan=23, linefreq='218.47563200GHz',
            linespw='4', robust=[2.0],imsize=1600,cellsize='0.025arcsec'),
        ### c-C3H2 217.82 GHz Images
        "c-C3H2_217.82":dict(chanstart='-10km/s', chanwidth='1.34km/s', 
            nchan=23, linefreq='217.82215GHz', linespw='4', robust=[2.0],imsize=1600,cellsize='0.025arcsec'),
        ### c-C3H2 217.94 GHz Images
        "cC3H2_217.94":dict(chanstart='-10km/s', chanwidth='1.34km/s', 
            nchan=23, linefreq='217.94005GHz', linespw='4', robust=[2.0],imsize=1600,cellsize='0.025arcsec'),
        ### c-C3H2 218.16 GHz Images
        "cC3H2_218.16":dict(chanstart='-10km/s', chanwidth='1.34km/s', 
            nchan=23, linefreq='218.16044GHz', linespw='4', robust=[2.0],imsize=1600,cellsize='0.025arcsec'),
        ### DCN Images
        "DCN":dict(chanstart='-10km/s', chanwidth='1.34km/s', nchan=23, 
            linefreq='217.2386GHz', linespw='4', robust=[2.0],imsize=1600,cellsize='0.025arcsec'),
        ### CH3OH Images
        "CH3OH":dict(chanstart='-10km/s', chanwidth='1.34km/s', nchan=23, 
            linefreq='218.44006300GHz', linespw='4', robust=[2.0],imsize=1600,cellsize='0.025arcsec'),
        ### SiO Images
        "SiO":dict(chanstart='-100km/s', chanwidth='1.34km/s', nchan=150, 
            linefreq='217.10498000GHz', linespw='4', robust=[2.0],imsize=1600,cellsize='0.025arcsec')
        }
'''

### Loop through the spectral line images and make images.

'''

for line in image_list_sb:
    for robust in image_list_sb[line]["robust"]:
        imagename = prefix+f'_SB_'+line+'_robust_'+str(robust)
        data_params_sb = {k: v for k, v in data_params.items() if k.startswith('SB')}
        sigma = get_sensitivity(data_params_sb, specmode='cube', \
                spw=[image_list_sb[line]["linespw"]], chan=450)

        tclean_spectral_line_wrapper(vislist_sb, imagename,
                image_list_sb[line]["chanstart"], image_list_sb[line]["chanwidth"], 
                image_list_sb[line]["nchan"], image_list_sb[line]["linefreq"], 
                image_list_sb[line]["linespw"], SB_scales, threshold=3.0*sigma,
                imsize=image_list_sb[line]["imsize"], cellsize=image_list_sb[line]["cellsize"],robust=robust, 
                sidelobethreshold=sidelobethreshold, noisethreshold=noisethreshold,
                lownoisethreshold=lownoisethreshold,smoothfactor=smoothfactor,parallel=parallel,
                phasecenter=data_params['SB1']['common_dir'].replace('J2000','ICRS'))
    if selectedVis=='vis_shift':
       tclean_spectral_line_wrapper(data_params['SB1']['vis'], imagename.replace(prefix,'temporary.pbfix'),
        image_list_sb[line]["chanstart"], image_list_sb[line]["chanwidth"], 
        image_list_sb[line]["nchan"], image_list_sb[line]["linefreq"], 
        image_list_sb[line]["linespw"], SB_scales, threshold=3.0*sigma,
        imsize=image_list_sb[line]["imsize"],
        cellsize=image_list_sb[line]["cellsize"], robust=robust, 
        sidelobethreshold=sidelobethreshold, noisethreshold=noisethreshold,
        lownoisethreshold=lownoisethreshold, smoothfactor=smoothfactor,
        parallel=parallel,niter=0,
        phasecenter=data_params['SB1']['common_dir'].replace('J2000','ICRS'))
       os.system('mv '+imagename+'.pb orig_pbimages/')
       os.system('cp -r '+imagename.replace(prefix,'temporary.pbfix')+'.pb '+imagename+'.pb')
       os.system('rm -rf '+imagename.replace(prefix,'temporary.pbfix')+'*')
'''

for line in image_list:
    print(line)
    for robust in image_list[line]["robust"]:
        imagename = prefix+f'_SBLB_'+line+'_robust_'+str(robust)

        sigma = get_sensitivity(data_params, specmode='cube', \
                spw=[image_list[line]["linespw"]], chan=450,robust=robust,)

        tclean_spectral_line_wrapper(vislist, imagename,
                image_list[line]["chanstart"], image_list[line]["chanwidth"], 
                image_list[line]["nchan"], image_list[line]["linefreq"], 
                image_list[line]["linespw"], SB_scales, threshold=3.0*sigma,
                imsize=image_list[line]["imsize"], cellsize=image_list[line]["cellsize"],
                robust=robust, uvtaper=image_list[line]["uvtaper"],
                sidelobethreshold=sidelobethreshold, noisethreshold=noisethreshold,
                lownoisethreshold=lownoisethreshold,smoothfactor=smoothfactor,parallel=parallel,
                phasecenter=data_params['SB1']['common_dir'].replace('J2000','ICRS'))
    if selectedVis=='vis_shift':
       tclean_spectral_line_wrapper(data_params['LB1']['vis'], imagename.replace(prefix,'temporary.pbfix'),
        image_list[line]["chanstart"], image_list[line]["chanwidth"], 
        image_list[line]["nchan"], image_list[line]["linefreq"], 
        image_list[line]["linespw"][1], LB_scales, threshold=3.0*sigma,
        imsize=image_list[line]["imsize"],
        cellsize=image_list[line]["cellsize"],
        robust=robust, uvtaper=image_list[line]["uvtaper"],
        sidelobethreshold=sidelobethreshold, noisethreshold=noisethreshold,
        lownoisethreshold=lownoisethreshold, smoothfactor=smoothfactor,
        parallel=parallel,
        phasecenter=data_params['SB1']['common_dir'].replace('J2000','ICRS'))
       os.system('mv '+imagename+'.pb orig_pbimages/')
       os.system('cp -r '+imagename.replace(prefix,'temporary.pbfix')+'.pb '+imagename+'.pb')
       os.system('rm -rf '+imagename.replace(prefix,'temporary.pbfix')+'*')
###############################################################
################ CLEANUP AND FITS CONVERSION ##################
###############################################################


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

   impbcor(imagename=image,pbimage=image.replace('image','pb'),outfile=image.replace('image','pbcor'))
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
###############################################################
################# Make Plots of Everything ####################
###############################################################
import sys
sys.argv = ['../edisk/plot_final_images_SBLB.py', prefix]
execfile('../edisk/plot_final_images_SBLB.py')

### Remove rescaled selfcal MSfiles
os.system('rm -rf *rescaled.ms.*')
os.system('rm -rf scale*')

### Remove extra image products
os.system('rm -rf *.residual* *.psf* *.model* *dirty* *.sumwt* *.gridwt* *.workdirectory')

### Make a directory to put the final products
os.system('rm -rf export')
os.system('mkdir export')
os.system('mv *.fits export/')
os.system('mv *.fits.gz export/')
os.system('mv *.tgz export/')
os.system('mv *.pdf export/')

