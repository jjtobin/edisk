"""
eDisk data reduction script
This script was written for CASA 6.1.1/6.2
Originally derived from DSHARP reduction scripts


Datasets calibrated (in order of date observed):
SB1: 
 
LB1: 
     
LB2: 

reducer: 
"""

### Import statements
#sys.path.append('/home/casa/contrib/AIV/science/analysis_scripts/')
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
prefix  = 'fieldName prefix' 

### always include trailing slashes!!
WD_path = '/lustre/cv/projects/edisk/sourceDirectory/'
SB_path = WD_path+'SB/'
LB_path = WD_path+'LB/'

### scales for multi-scale clean
SB_scales = [0, 5] #[0, 5, 10, 20]
LB_scales = [0, 5, 30]  #[0, 5, 30, 100, 200]

### Add additional dictionary entries if need, i.e., SB2, SB3, LB1, LB2, etc. for each execution
### Note that C18O and 13CO have different spws in the DDT vis LP os the spw ordering
### is different for data that were originally part of the DDT than the LP
### DDT 2019.A.00034.S SB data need 'spws': '25,31,29,27,33,35,37'
### LP  2019.1.00261.L SB data need 'spws': '25,27,29,31,33,35,37'
pl_data_params={'SB1': {'vis': SB_path+'uid___A002_Xeba1ac_Xcc28.ms',
                        'spws': '25,27,29,31,33,35,37'},
                'LB1': {'vis': LB_path+'uid___A002_Xeef629_X3191.ms',
                        'spws': '25,27,29,31,33,35,37'},
                'LB2': {'vis': LB_path+'uid___A002_Xeef629_X3b39.ms',
                        'spws': '25,27,29,31,33,35,37'},
               }

### Dictionary defining necessary metadata for each execution
### SiO at 217.10498e9 excluded because of non-detection
### Only bother specifying simple species that are likely present in all datasets
### Hot corino lines (or others) will get taken care of by using the cont.dat
data_params = {'SB1': {'vis' : WD_path+prefix+'_SB1.ms',
                       'name' : 'SB1',
                       'field': field,
                       'line_spws': np.array([0,1,2,3,4,6,4,4,4,4,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9,220.39868420e9,219.94944200e9,219.56035410e9,
                                               217.82215e9,230.538e9,217.94005e9,218.16044e9,217.2386e9,
                                               218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO','13CO','SO','C18O','c-C3H2','12CO','c-C3H2','c-C3H2','DCN','H2CO','H2CO'], #restfreqs
                       'flagrange': np.array([[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],
                                              [-5.5,14.5],[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],
                                              [-5.5,14.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {25:0, 31:1, 29:2, 27:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/04/24/01:45:00~2021/04/24/03:00:00',
                       'contdotdat' : 'SB/cont.dat'
                      }, 
               'LB1': {'vis' : WD_path+prefix+'_LB1.ms',
                       'name' : 'LB1',
                       'field': field,
                       'line_spws': np.array([0,1,2,3,4,6,4,4,4,4,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9,220.39868420e9,219.94944200e9,219.56035410e9,
                                               217.82215e9,230.538e9,217.94005e9,218.16044e9,217.2386e9,
                                               218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO','13CO','SO','C18O','c-C3H2','12CO','c-C3H2','c-C3H2','DCN','H2CO','H2CO'], #restfreqs
                       'flagrange': np.array([[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],
                                              [-5.5,14.5],[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],
                                              [-5.5,14.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/08/10/01:47:00~2021/08/10/02:50:00',
                       'contdotdat' : 'SB/cont.dat'
                      }, 
               'LB2': {'vis' : WD_path+prefix+'_LB2.ms',
                       'name' : 'LB2',
                       'field': field,
                       'line_spws': np.array([0,1,2,3,4,6,4,4,4,4,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9,220.39868420e9,219.94944200e9,219.56035410e9,
                                               217.82215e9,230.538e9,217.94005e9,218.16044e9,217.2386e9,
                                               218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO','13CO','SO','C18O','c-C3H2','12CO','c-C3H2','c-C3H2','DCN','H2CO','H2CO'], #restfreqs
                       'flagrange': np.array([[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],
                                              [-5.5,14.5],[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],
                                              [-5.5,14.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/08/10/04:32:00~2021/08/10/05:32:00',
                       'contdotdat' : 'SB/cont.dat'
                      }, 
               }


### Flag range corresponds to velocity range in each spw that should be flagged. 
### Velocity range should correspond to 
### approximate width of the line contamination

#save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################
#################### DATA PREPARATION #########################
###############################################################

### split out each pipeline-calibrated dataset into an MS only containing the target data 
for i in pl_data_params.keys():
   if os.path.exists(prefix+'_'+i+'.ms'):
      flagmanager(vis=prefix+'_'+i+'.ms', mode="restore", \
                  versionname="starting_flags")
   else:
      split(vis=pl_data_params[i]['vis'],outputvis=prefix+'_'+i+'.ms',spw=pl_data_params[i]['spws'],field=field,datacolumn='corrected')

### Backup the the flagging state at start of reduction
for i in data_params.keys():
    if not os.path.exists(data_params[i]['vis']+\
            ".flagversions/flags.starting_flags"):
       flagmanager(vis=data_params[i]['vis'], mode = 'save', versionname = 'starting_flags', comment = 'Flag states at start of reduction')

### Inspect data in each spw for each dataset
#### OPTIONAL #####
if not skip_plots:
    for i in data_params.keys():
        plotms(vis=data_params[i]['vis'], xaxis='frequency', yaxis='amplitude', 
               field=data_params[i]['field'], ydatacolumn='data', 
               avgtime='1e8', avgscan=True, avgbaseline=True, iteraxis='spw',
               transform=True,freqframe='LSRK')
        input("Press Enter key to advance to next MS/Caltable...")
#### END OPTIONAL ###

### Flag spectral regions around lines and do spectral averaging to make a smaller continuum MS 
for i in data_params.keys():      
    flagchannels_string = get_flagchannels(data_params[i], prefix)
    s=' '  # work around for Python 3 port of following string generating for loops
    print(i) 
    avg_cont(data_params[i], prefix, flagchannels=flagchannels_string,contspws=s.join(str(elem) for elem in data_params[i]['cont_spws'].tolist()).replace(' ',','),width_array=data_params[i]['cont_avg_width'])
    data_params[i]['vis_avg']=prefix+'_'+i+'_initcont.ms'
      

###############################################################
############## INITIAL IMAGING FOR ALIGNMENT ##################
###############################################################


### Image each dataset individually to get source position in each image
### Images are saved in the format prefix+'_name_initcont_exec#.ms'
outertaper='2000klambda' # taper if necessary to align using larger-scale uv data, small-scale may have subtle shifts from phase noise
for i in data_params.keys():
       print('Imaging MS: ',i) 
       if 'LB' in i:
          image_each_obs(data_params[i], prefix, scales=LB_scales,  uvtaper=outertaper,
                   nsigma=5.0, sidelobethreshold=2.5, smoothfactor=1.5,interactive=False,parallel=parallel) 
       else:
          image_each_obs(data_params[i], prefix, scales=SB_scales, 
                   nsigma=5.0, sidelobethreshold=2.5, interactive=False,parallel=parallel)

       #check masks to ensure you are actually masking the image, lower sidelobethreshold if needed

""" Fit Gaussians to roughly estimate centers, inclinations, PAs """
""" Loops through each dataset specified """
###default fit region is blank for an obvious single source
fit_region=''

###specify manual mask on brightest source if Gaussian fitting fails due to confusion

mask_ra  =  '19h01m56.419s'.replace('h',':').replace('m',':').replace('s','')
mask_dec = '-36d57m28.690s'.replace('d','.').replace('m','.').replace('s','')
mask_pa  = 90.0 	# position angle of mask in degrees
mask_maj = 0.76	# semimajor axis of mask in arcsec
mask_min = 0.75 	# semiminor axis of mask in arcsec
fit_region = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
              (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)

for i in data_params.keys():
       print(i)
       data_params[i]['phasecenter']=fit_gaussian(prefix+'_'+i+'_initcont_exec0.image', region=fit_region,mask=prefix+'_'+i+'_initcont_exec0.mask')


### Check phase center fits in viewer, tf centers appear too shifted from the Gaussian fit, 
### manually set the phase center dictionary entry by eye


""" The emission centers are slightly misaligned.  So we split out the 
    individual executions, shift the peaks to the phase center, and reassign 
    the phase centers to a common direction. """

### Set common direction for each EB using one as reference (typically best looking LB image)

for i in data_params.keys():
       #################### MANUALLY SET THIS ######################
       data_params[i]['common_dir']='J2000 19h01m56.419063s -36d57m28.67292s'

### save updated data params to a pickle

with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


###############################################################
#################### SHIFT PHASE CENTERS ######################
###############################################################

for i in data_params.keys():
   print(i)
   data_params[i]['vis_avg_shift']=prefix+'_'+i+'_initcont_shift.ms'
   os.system('rm -rf '+data_params[i]['vis_avg_shift'])
   fixvis(vis=data_params[i]['vis_avg'], outputvis=data_params[i]['vis_avg_shift'], 
       field=data_params[i]['field'], 
       phasecenter='J2000 '+data_params[i]['phasecenter'])
   ### fix planets may throw an error, usually safe to ignore
   fixplanets(vis=data_params[i]['vis_avg_shift'], field=data_params[i]['field'], 
           direction=data_params[i]['common_dir'])

###############################################################
############### REIMAGING TO CHECK ALIGNMENT ##################
###############################################################
for i in data_params.keys():
       print(i)
       if 'SB' in i:
          scales=SB_scales
       else:
          scales=LB_scales
       for suffix in ['image','mask','mode','psf','pb','residual','sumwt']:
          os.system('rm -rf '+prefix+'_'+i+'_initcont_shift.'+suffix)
       image_each_obs_shift(data_params[i]['vis_avg_shift'], prefix, scales=scales, 
                   nsigma=5.0, sidelobethreshold=2.5, interactive=False,parallel=parallel)

for i in data_params.keys():
      print(i)     
      data_params[i]['phasecenter_new']=fit_gaussian(prefix+'_'+i+'_initcont_shift.image',\
                                                     region=fit_region,mask=prefix+'_'+i+'_initcont_shift.mask')
      print('Phasecenter new: ',data_params[i]['phasecenter_new'])
      print('Phasecenter old: ',data_params[i]['phasecenter'])

### save updated data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



###############################################################
############### PLOT UV DATA TO CHECK SCALING #################
###############################################################

### Assign rough emission geometry parameters; keep 0, 0
PA, incl = 0, 0

### Export MS contents into Numpy save files 
export_vislist=[]
for i in data_params.keys():
   export_MS(data_params[i]['vis_avg_shift'])
   export_vislist.append(data_params[i]['vis_avg_shift'].replace('.ms','.vis.npz'))

if not skip_plots:
    ### Plot deprojected visibility profiles for all data together """
    plot_deprojected(export_vislist,
                     fluxscale=[1.0]*len(export_vislist), PA=PA, incl=incl, 
                     show_err=False)

### Now inspect offsets by comparing against a reference 
### Set reference data using the dictionary key.
### Using SB1 as reference because it looks the nicest by far

#################### MANUALLY SET THIS ######################
refdata='LB1'

reference=prefix+'_'+refdata+'_initcont_shift.vis.npz'
for i in data_params.keys():
   print(i)
   if i != refdata:
      data_params[i]['gencal_scale']=estimate_flux_scale(reference=reference, 
                        comparison=prefix+'_'+i+'_initcont_shift.vis.npz', 
                        incl=incl, PA=PA)
   else:
      data_params[i]['gencal_scale']=1.0
   print(' ')

#No rescaling here since just one dataset
#Go ahead with rescaling anyway to keep the flow of the script


###############################################################
############### SCALE DATA RELATIVE TO ONE EB #################
###############################################################

os.system('rm -rf *_rescaled.ms')
for i in data_params.keys():
   rescale_flux(data_params[i]['vis_avg_shift'], [data_params[i]['gencal_scale']])
   rescale_flux(data_params[i]['vis_avg'], [data_params[i]['gencal_scale']])
   data_params[i]['vis_avg_shift_rescaled']=data_params[i]['vis_avg_shift'].replace('.ms','_rescaled.ms')
   data_params[i]['vis_avg_rescaled']=data_params[i]['vis_avg'].replace('.ms','_rescaled.ms')

###############################################################
############## PLOT UV DATA TO CHECK RE-SCALING ###############
###############################################################

if not skip_plots:
    ### Assign rough emission geometry parameters; keep 0, 0
   PA, incl = 0, 0

   ### Check that rescaling did what we expect
   export_vislist_rescaled=[]
   for i in data_params.keys():
      export_MS(data_params[i]['vis_avg_shift_rescaled'])
      export_vislist_rescaled.append(data_params[i]['vis_avg_shift_rescaled'].replace('.ms','.vis.npz'))

   plot_deprojected(export_vislist_rescaled,
                     fluxscale=[1.0]*len(export_vislist_rescaled), PA=PA, incl=incl, 
                     show_err=False)
   ### Make sure differences are no longer significant
   refdata='SB1'
   reference=prefix+'_'+refdata+'_initcont_shift.vis.npz'
   for i in data_params.keys():
      if i != refdata:
         estimate_flux_scale(reference=reference, 
                        comparison=prefix+'_'+i+'_initcont_shift_rescaled.vis.npz', 
                        incl=incl, PA=PA)

### Save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



###############################################################
################ SELF-CALIBRATION PREPARATION #################
###############################################################
selectedVis='vis_avg_rescaled'
#selectedVis='vis_avg_shift_rescaled'

### determine best reference antennas based on geometry and flagging
for i in data_params.keys():
   data_params[i]["refant"] = rank_refants(data_params[i][selectedVis])

'''Find reference antenna, pick 2 near array center'''
'''
if not skip_plots:
   for i in data_params.keys():
      if 'LB' in i:
         continue
      listobs(data_params[i]['vis'])
      plotants(data_params[i]['vis'])
      input("Press Enter key to advance to next MS/Caltable...")
'''

'''antenna name is DV/DA/PMXX'''
'''pad number is @AXXX '''
'''want antenna that is on the same pad if possible, list multiple in case one drops out'''
'''check listobs and fill in the SB_refant field '''
'''with the antenna name (DAXX, DVXX, or PMXX) @ pad number (AXXX)'''
'''so make a comma separated list like: DA43@A035,DV07@A011,...'''


#################### MANUALLY SET THIS ######################
#SB_refant   = 'DA43@A035,DV07@A011,DV05@A042' 

############### CHECK THESE, SHOULD BE FINE #################
SB_spwmap=[0,0,0,0,0,0,0]
SB_contspws = '' 


### Make a list of EBs to image
vislist=[]
for i in data_params.keys():
      if ('LB' in i): # skip over LB EBs if in SB-only mode
         continue
      vislist.append(data_params[i][selectedVis])


""" Set up a clean mask """

mask_ra  =  data_params[i]['common_dir'].split()[1].replace('h',':').replace('m',':').replace('s','')
mask_dec = data_params[i]['common_dir'].split()[2].replace('d','.').replace('m','.').replace('s','')
mask_pa  = 90.0 	# position angle of mask in degrees
mask_maj = 1.01	# semimajor axis of mask in arcsec
mask_min = 1.0 	# semiminor axis of mask in arcsec

common_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
              (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)
""" Define a noise annulus, measure the peak SNR in map """
noise_annulus = "annulus[[%s, %s],['%.2farcsec', '8.0arcsec']]" % \
                (mask_ra, mask_dec, 2.0*mask_maj) 


###############################################################
###################### SELF-CALIBRATION #######################
###############################################################

### Initial dirty map to assess DR
tclean_wrapper(vis=vislist, imagename=prefix+'_dirty', 
               scales=SB_scales, niter=0,parallel=parallel,cellsize='0.025arcsec',imsize=1600)
estimate_SNR(prefix+'_dirty.image.tt0', disk_mask=common_mask, 
             noise_mask=noise_annulus)


#IRS7B_SB-only_ap6.image.tt0
#Beam 0.272 arcsec x 0.205 arcsec (41.92 deg)
#Flux inside disk mask: 372.23 mJy
#Peak intensity of source: 122.69 mJy/beam
#rms: 1.92e-01 mJy/beam
#Peak SNR: 640.28


### Image produced by iter 0 has not selfcal applied, it's used to set the initial model
### only images >0 have self-calibration applied

### Run self-calibration command set
### 0. Split off corrected data from previous selfcal iteration (except iteration 0)
### 1. Image data to specified nsigma depth, set model column
### 2. Calculate self-cal gain solutions
### 3. Apply self-cal gain solutions to MS

############# USERS MAY NEED TO ADJUST NSIGMA AND SOLINT FOR EACH SELF-CALIBRATION ITERATION ##############
iteration=0
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=40.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)


### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

### Make note of key metrics of image in each round
#IRS7B_SB-only_p0.image.tt0
#Beam 0.271 arcsec x 0.204 arcsec (41.26 deg)
#Flux inside disk mask: 397.25 mJy
#Peak intensity of source: 118.70 mJy/beam
#rms: 4.67e-01 mJy/beam
#Peak SNR: 253.96

#IRS7B_SB-only_p0_post.image.tt0
#Beam 0.271 arcsec x 0.204 arcsec (41.26 deg)
#Flux inside disk mask: 401.28 mJy
#Peak intensity of source: 121.75 mJy/beam
#rms: 4.04e-01 mJy/beam
#Peak SNR: 301.49



iteration=1
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=20.0,solint='30s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#IRS7B_SB-only_p1.image.tt0
#Beam 0.271 arcsec x 0.204 arcsec (41.26 deg)
#Flux inside disk mask: 379.16 mJy
#Peak intensity of source: 119.83 mJy/beam
#rms: 2.14e-01 mJy/beam
#Peak SNR: 559.18

#IRS7B_SB-only_p1_post.image.tt0
#Beam 0.271 arcsec x 0.204 arcsec (41.26 deg)
#Flux inside disk mask: 379.17 mJy
#Peak intensity of source: 121.47 mJy/beam
#rms: 2.13e-01 mJy/beam
#Peak SNR: 570.58




iteration=2
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=5.0,solint='6s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#IRS7B_SB-only_p2.image.tt0
#Beam 0.271 arcsec x 0.204 arcsec (41.26 deg)
#Flux inside disk mask: 374.22 mJy
#Peak intensity of source: 121.34 mJy/beam
#rms: 1.92e-01 mJy/beam
#Peak SNR: 633.58

#IRS7B_SB-only_p2_post.image.tt0
#Beam 0.271 arcsec x 0.204 arcsec (41.26 deg)
#Flux inside disk mask: 374.87 mJy
#Peak intensity of source: 122.05 mJy/beam
#rms: 1.93e-01 mJy/beam
#Peak SNR: 632.71



iteration=3
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='int',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#IRS7B_SB-only_p3.image.tt0
#Beam 0.271 arcsec x 0.204 arcsec (41.26 deg)
#Flux inside disk mask: 373.91 mJy
#Peak intensity of source: 121.97 mJy/beam
#rms: 1.91e-01 mJy/beam
#Peak SNR: 639.18

#IRS7B_SB-only_p3_post.image.tt0
#Beam 0.271 arcsec x 0.204 arcsec (41.26 deg)
#Flux inside disk mask: 373.86 mJy
#Peak intensity of source: 122.06 mJy/beam
#rms: 1.91e-01 mJy/beam
#Peak SNR: 638.37



### Changing self-cal mode here to ap, see use of prevselfcalmode to ensure proper split

iteration=4
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',prevselfcalmode='p',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")


#IRS7B_SB-only_p4.image.tt0
#Beam 0.271 arcsec x 0.204 arcsec (41.26 deg)
#Flux inside disk mask: 373.86 mJy
#Peak intensity of source: 121.99 mJy/beam
#rms: 1.91e-01 mJy/beam
#Peak SNR: 638.78

#IRS7B_SB-only_p4_post.image.tt0
#Beam 0.272 arcsec x 0.205 arcsec (41.83 deg)
#Flux inside disk mask: 372.11 mJy
#Peak intensity of source: 122.62 mJy/beam
#rms: 1.92e-01 mJy/beam
#Peak SNR: 639.71

iteration=5
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='18s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key tto advance to next MS/Caltable...")

#IRS7B_SB-only_ap5.image.tt0
#Beam 0.272 arcsec x 0.205 arcsec (41.83 deg)
#Flux inside disk mask: 372.34 mJy
#Peak intensity of source: 122.55 mJy/beam
#rms: 1.91e-01 mJy/beam
#Peak SNR: 640.15

#IRS7B_SB-only_ap5_post.image.tt0
#Beam 0.272 arcsec x 0.205 arcsec (41.92 deg)
#Flux inside disk mask: 372.16 mJy
#Peak intensity of source: 122.75 mJy/beam
#rms: 1.92e-01 mJy/beam
#Peak SNR: 639.88

### Make the final image, will not run another self-calibration
iteration=6
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='18s',
               noisemasks=[common_mask,noise_annulus],SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,
               parallel=parallel,finalimageonly=True)



for i in data_params.keys():
   if 'SB' in i:
      data_params[i]['selfcal_spwmap_SB-only']=data_params[i]['selfcal_spwmap'].copy()
      data_params[i]['selfcal_tables_SB-only']=data_params[i]['selfcal_tables'].copy()
      data_params[i]['vis_avg_selfcal_SB-only']=(data_params[i]['vis_avg_selfcal']+'.')[:-1]  ## trick to copy the string

### Save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

###############################################################
################### SELF-CALIBRATION SB+LB ####################
###############################################################
LB_spwmap=[0,0,0,0,0,0,0]
LB_contspws = '' 


### Make a list of EBs to image
vislist=[]
for i in data_params.keys():
   vislist.append(data_params[i][selectedVis])

### Initial dirty map to assess DR
tclean_wrapper(vis=vislist, imagename=prefix+'_LB+SB_dirty', 
               scales=SB_scales, niter=0,parallel=parallel,cellsize='0.003arcsec',imsize=15000)
estimate_SNR(prefix+'_LB+SB_dirty.image.tt0', disk_mask=common_mask, 
             noise_mask=noise_annulus)


#IRS7B_LB+SB_dirty.image.tt0
#Beam 0.075 arcsec x 0.058 arcsec (82.08 deg)
#Flux inside disk mask: 836.78 mJy
#Peak intensity of source: 33.76 mJy/beam
#rms: 2.21e-01 mJy/beam
#Peak SNR: 153.01

### Image produced by iter 0 has not selfcal applied, it's used to set the initial model
### only images >0 have self-calibration applied

### Run self-calibration command set
### 0. Split off corrected data from previous selfcal iteration (except iteration 0)
### 1. Image data to specified nsigma depth, set model column
### 2. Calculate self-cal gain solutions
### 3. Apply self-cal gain solutions to MS

############# USERS MAY NEED TO ADJUST NSIGMA AND SOLINT FOR EACH SELF-CALIBRATION ITERATION ##############
############################ CONTINUE SELF-CALIBRATION ITERATIONS UNTIL ###################################
#################### THE S/N BEGINS TO DROP OR SOLINTS ARE AS LOW AS POSSIBLE #############################

#################### LOOK FOR ERRORS IN GAINCAL CLAIMING A FREQUENCY MISMATCH #############################
####### IF FOUND, CHANGE SOLINT, MAYBE TRY TO ALIGN WITH A CERTAIN NUMBER OF SCANS AND TRY AGAIN ##########
########################## IF ALL ELSE FAILS, SIMPLY START WITH solint='inf' ##############################

iteration=0
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=40.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,combine='spw,scan',parallel=parallel,smoothfactor=2.0,imsize=14400)


### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#IRS7B_LB+SB_p0.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 431.65 mJy
#Peak intensity of source: 26.60 mJy/beam
#rms: 6.97e-02 mJy/beam
#Peak SNR: 381.87

#IRS7B_LB+SB_p0_post.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 431.56 mJy
#Peak intensity of source: 27.08 mJy/beam
#rms: 6.09e-02 mJy/beam
#Peak SNR: 444.40


iteration=1
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=30.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,combine='spw',parallel=parallel,smoothfactor=2.0,imsize=14400)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#IRS7B_LB+SB_p1.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 407.10 mJy
#Peak intensity of source: 26.80 mJy/beam
#rms: 5.27e-02 mJy/beam
#Peak SNR: 508.79

#IRS7B_LB+SB_p1_post.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 406.86 mJy
#Peak intensity of source: 27.51 mJy/beam
#rms: 5.15e-02 mJy/beam
#Peak SNR: 534.03


iteration=2
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=20.0,solint='27.18s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,combine='spw',smoothfactor=2.0,imsize=14400)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#IRS7B_LB+SB_p2.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 392.68 mJy
#Peak intensity of source: 27.34 mJy/beam
#rms: 4.74e-02 mJy/beam
#Peak SNR: 576.20

#IRS7B_LB+SB_p2_post.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 392.47 mJy
#Peak intensity of source: 27.53 mJy/beam
#rms: 4.75e-02 mJy/beam
#Peak SNR: 579.77

iteration=3
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=10.0,solint='9.06',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,imsize=14400)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#IRS7B_LB+SB_p3.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 382.81 mJy
#Peak intensity of source: 27.45 mJy/beam
#rms: 4.55e-02 mJy/beam
#Peak SNR: 603.48

#IRS7B_LB+SB_p3_post.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 382.69 mJy
#Peak intensity of source: 27.60 mJy/beam
#rms: 4.56e-02 mJy/beam
#Peak SNR: 605.38

iteration=4
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=5.0,solint='int',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,imsize=14400)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#IRS7B_LB+SB_p4.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 377.62 mJy
#Peak intensity of source: 27.48 mJy/beam
#rms: 4.50e-02 mJy/beam
#Peak SNR: 610.68

#IRS7B_LB+SB_p4_post.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 377.58 mJy
#Peak intensity of source: 27.66 mJy/beam
#rms: 4.52e-02 mJy/beam
#Peak SNR: 611.54




iteration=5
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='int',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,imsize=14400)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")


#IRS7B_LB+SB_p5.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 374.57 mJy
#Peak intensity of source: 27.62 mJy/beam
#rms: 4.50e-02 mJy/beam
#Peak SNR: 613.42
#IRS7B_LB+SB_p5_post.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 374.54 mJy
#Peak intensity of source: 27.65 mJy/beam
#rms: 4.51e-02 mJy/beam
#Peak SNR: 613.80


iteration=6
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,
               LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,finalimageonly=True,imsize=14400)

#IRS7B_LB+SB_p6.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 374.42 mJy
#Peak intensity of source: 27.61 mJy/beam
#rms: 4.50e-02 mJy/beam
#Peak SNR: 613.37


#outcome was marginally worse with amplitude self-cal, so omitting

'''
### Changing self-cal mode here to ap, see use of prevselfcalmode to ensure proper split
iteration=6
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',prevselfcalmode='p',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,combine='spw',smoothfactor=2.0,imsize=14400)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")

#IRS7B_LB+SB_p6.image.tt0
#Beam 0.078 arcsec x 0.060 arcsec (80.42 deg)
#Flux inside disk mask: 374.42 mJy
#Peak intensity of source: 27.61 mJy/beam
#rms: 4.50e-02 mJy/beam
#Peak SNR: 613.37
#IRS7B_LB+SB_p6_post.image.tt0
#Beam 0.080 arcsec x 0.061 arcsec (80.32 deg)
#Flux inside disk mask: 374.55 mJy
#Peak intensity of source: 28.09 mJy/beam
#rms: 4.59e-02 mJy/beam
#Peak SNR: 612.60


iteration=6
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,
               LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,finalimageonly=True,imsize=14400)
#IRS7B_LB+SB_ap7.image.tt0
#Beam 0.080 arcsec x 0.061 arcsec (80.32 deg)
#Flux inside disk mask: 375.08 mJy
#Peak intensity of source: 28.18 mJy/beam
#rms: 4.59e-02 mJy/beam
#Peak SNR: 614.12

'''

###Backup gain table list for LB+SB runs
for i in data_params.keys():
   if 'SB' in i:
      data_params[i]['selfcal_spwmap']=data_params[i]['selfcal_spwmap_SB-only']+data_params[i]['selfcal_spwmap']
      data_params[i]['selfcal_tables']=data_params[i]['selfcal_tables_SB-only']+data_params[i]['selfcal_tables']

#save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



###############################################################
################# SPLIT OFF FINAL CONT DATA ###################
###############################################################

for i in data_params.keys():
   os.system('rm -rf '+prefix+'_'+i+'_continuum.ms '+prefix+'_'+i+'_continuum.ms.tgz')
   split(vis=data_params[i]['vis_avg_selfcal'], outputvis=prefix+'_'+i+'_continuum.ms',
      datacolumn='data')
   data_params[i]['vis_final']=prefix+'_'+i+'_continuum.ms'
   os.system('tar cvzf '+prefix+'_'+i+'_continuum.ms.tgz '+prefix+'_'+i+'_continuum.ms')

#save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


###############################################################
################## RUN A FINAL IMAGE SET ######################
###############################################################

### Generate a vislist
vislist=[]
for i in data_params.keys():
   vislist.append(data_params[i]['vis_final'])
scales = SB_scales
imsize=14400
cell='0.003arcsec'
for robust in [-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0]:
for robust in [2.0]:
    imagename=prefix+'_SBLB_continuum_robust_'+str(robust)
    os.system('rm -rf '+imagename+'*')

    sigma = get_sensitivity(data_params, specmode='mfs',imsize=imsize,robust=robust,cellsize=cell)
    # adjust sigma to corrector for the irregular noise in the images if needed
    # correction factor may vary or may not be needed at all depending on source
    #if robust == 2.0 or robust == 1.0:
    #   sigma=sigma*1.75
    tclean_wrapper(vis=vislist, imagename=imagename, sidelobethreshold=2.0, 
            smoothfactor=1.5, scales=scales, threshold=3.0*sigma, 
            noisethreshold=3.0, robust=robust, parallel=parallel, 
            cellsize=cell, imsize=imsize,phasecenter=data_params['SB1']['common_dir'].replace('J2000','ICRS'))

    imagename=imagename+'.image.tt0'
    exportfits(imagename=imagename, fitsimage=imagename+'.fits',overwrite=True,dropdeg=True)

###############################################################
########################### CLEANUP ###########################
###############################################################

### Remove extra image products
os.system('rm -rf *.residual* *.psf* *.model* *dirty* *.sumwt* *.gridwt* *.workdirectory')

### put selfcalibration intermediate images somewhere safe
os.system('rm -rf initial_images')
os.system('mkdir initial_images')
os.system('mv *initcont*.image *_p*.image* *_ap*.image* initial_images')
os.system('mv *initcont*.mask *_p*.mask *_ap*.mask initial_images')
os.system('rm -rf *_p*.alpha* *_p*.pb.tt0 *_ap*.alpha* *_ap*.pb.tt0')

### Remove intermediate selfcal MSfiles
os.system("rm -rf *p{0..99}.ms")
os.system("rm -rf *p{0..99}.ms.flagversions")
### Remove rescaled selfcal MSfiles
os.system('rm -rf *rescaled.ms')
os.system('rm -rf *rescaled.ms.flagversions')
### Remove rescaled selfcal MSfiles
os.system('rm -rf *initcont*.ms')
os.system('rm -rf *initcont*.ms.flagversions')






