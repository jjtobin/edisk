"""
eDisk data reduction script
This script was written for CASA 6.1.1/6.2
Originally derived from DSHARP reduction scripts


Datasets calibrated (in order of date observed):
SB1: 2013.1.00879.S
     (1 execution blocks)

LB1: 2017.1.00288.S
     Observed  (5 execution blocks)
     

reducer: J. Tobin
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
skip_plots = True	

### Add field names (corresponding to the field in the MS) here and prefix for 
### filenameing (can be different but try to keep same)
### Only make different if, for example, the field name has a space
field   = 'B335'
prefix  = 'B335' 

### always include trailing slashes!!
WD_path = '/lustre/cv/projects/edisk/B335/'
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
pl_data_params={'SB1': {'vis': SB_path+'uid___A002_X8a951a_X55b.ms.split.cal',
                        'spws': '0,1,2,3,4,5,6',
                        'field': 'B335'},
                'LB1': {'vis': LB_path+'uid___A002_Xc5802b_X5a5.ms.split.cal',
                        'spws': '0,1,2,3,4',
                        'field': 'B335'},
                'LB2': {'vis': LB_path+'uid___A002_Xc5e6d0_Xe87.ms.split.cal',
                        'spws': '0,1,2,3,4',
                        'field': 'B335'},
                'LB3': {'vis': LB_path+'uid___A002_Xc5f05e_X51e.ms.split.cal',
                        'spws': '0,1,2,3,4',
                        'field': 'B335'},
                'LB4': {'vis': LB_path+'uid___A002_Xc6141c_X2d94.ms.split.cal',
                        'spws': '0,1,2,3,4',
                        'field': 'B335'},
                'LB5': {'vis': LB_path+'uid___A002_Xc63024_X3e51.ms.split.cal',
                        'spws': '0,1,2,3,4',
                        'field': 'B335'},
               }

### Dictionary defining necessary metadata for each execution
### SiO at 217.10498e9 excluded because of non-detection
### Only bother specifying simple species that are likely present in all datasets
### Hot corino lines (or others) will get taken care of by using the cont.dat
data_params = {'SB1': {'vis' : WD_path+prefix+'_SB1.ms',
                       'name' : 'SB1',
                       'field': 'B335',
                       'line_spws': np.array([0,1,2,4,5]), # line SPWs, get from listobs
                       'line_freqs': np.array([219.56035410e9,219.94944200e9,220.39868420e9,230.538e9,231.32182830e9]), #restfreqs
                       'line_names': ['C18O','SO','13CO','12CO','N2D+'], #restfreqs
                       'flagrange': np.array([[-5.5,14.5],[-5.5,14.5],[-15.5,25.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5,6:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([30,30,30,4,60,30,4]), #n channels to average; approximately aiming for 30 MHz channels
                       #'cont_avg_width':  np.array([0,0,0,0,0,0]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2014/09/02/01:36:00~2014/09/02/02:23:00',
                       #'contdotdat' : 'LB/cont.dat'
                      }, 
               'LB1': {'vis' : WD_path+prefix+'_LB1.ms',
                       'name' : 'LB1',
                       'field': field,
                       'line_spws': np.array([1,2,3,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([230.538e9,217.104980e9,219.56035410e9,220.39868420e9]), #restfreqs
                       'line_names': ['12CO','SiO','C18O','13CO'], #restfreqs
                       'flagrange': np.array([[-15.5,24.5],[-15.5,24.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {0:0, 1:1, 2:2, 3:3, 4:4},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4]),  #spws to use for continuum
                       #'cont_avg_width':  np.array([0,0,0,0,0]), #n channels to average; approximately aiming for 30 MHz channels
                       'cont_avg_width':  np.array([4,30,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2017/10/08/00:49:00~2017/10/08/02:08:00',
                       #'contdotdat' : 'LB/cont.dat'
                      }, 
               'LB2': {'vis' : WD_path+prefix+'_LB2.ms',
                       'name' : 'LB2',
                       'field': field,
                       'line_spws': np.array([1,2,3,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([230.538e9,217.104980e9,219.56035410e9,220.39868420e9]), #restfreqs
                       'line_names': ['12CO','SiO','C18O','13CO'], #restfreqs
                       'flagrange': np.array([[-15.5,24.5],[-15.5,24.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {0:0, 1:1, 2:2, 3:3, 4:4},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4]),  #spws to use for continuum
                       #'cont_avg_width':  np.array([0,0,0,0,0]), #n channels to average; approximately aiming for 30 MHz channels
                       'cont_avg_width':  np.array([4,30,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2017/10/21/00:25:00~2017/10/21/01:44:00',
                       #'contdotdat' : 'LB/cont.dat'
                      }, 
               'LB3': {'vis' : WD_path+prefix+'_LB3.ms',
                       'name' : 'LB3',
                       'field': field,
                       'line_spws': np.array([1,2,3,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([230.538e9,217.104980e9,219.56035410e9,220.39868420e9]), #restfreqs
                       'line_names': ['12CO','SiO','C18O','13CO'], #restfreqs
                       'flagrange': np.array([[-15.5,24.5],[-15.5,24.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {0:0, 1:1, 2:2, 3:3, 4:4},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4]),  #spws to use for continuum
                       #'cont_avg_width':  np.array([0,0,0,0,0]), #n channels to average; approximately aiming for 30 MHz channels
                       'cont_avg_width':  np.array([4,30,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2017/10/22/00:32:00~2017/10/22/01:51:00',
                       #'contdotdat' : 'LB/cont.dat'
                      }, 
               'LB4': {'vis' : WD_path+prefix+'_LB4.ms',
                       'name' : 'LB4',
                       'field': field,
                       'line_spws': np.array([1,2,3,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([230.538e9,217.104980e9,219.56035410e9,220.39868420e9]), #restfreqs
                       'line_names': ['12CO','SiO','C18O','13CO'], #restfreqs
                       'flagrange': np.array([[-15.5,24.5],[-15.5,24.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {0:0, 1:1, 2:2, 3:3, 4:4},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4]),  #spws to use for continuum
                       #'cont_avg_width':  np.array([0,0,0,0,0]), #n channels to average; approximately aiming for 30 MHz channels
                       'cont_avg_width':  np.array([4,30,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2017/10/27/00:04:00~2017/10/27/01:21:00',
                       #'contdotdat' : 'LB/cont.dat'
                      }, 
               'LB5': {'vis' : WD_path+prefix+'_LB5.ms',
                       'name' : 'LB5',
                       'field': field,
                       'line_spws': np.array([1,2,3,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([230.538e9,217.104980e9,219.56035410e9,220.39868420e9]), #restfreqs
                       'line_names': ['12CO','SiO','C18O','13CO'], #restfreqs
                       'flagrange': np.array([[-15.5,24.5],[-15.5,24.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {0:0, 1:1, 2:2, 3:3, 4:4},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4]),  #spws to use for continuum
                       #'cont_avg_width':  np.array([0,0,0,0,0]), #n channels to average; approximately aiming for 30 MHz channels
                       'cont_avg_width':  np.array([4,30,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2017/10/29/23:14:00~2017/10/30/00:33:00',
                       #'contdotdat' : 'LB/cont.dat'
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
      split(vis=pl_data_params[i]['vis'],outputvis=prefix+'_'+i+'.ms',spw=pl_data_params[i]['spws'],field=pl_data_params[i]['field'],datacolumn='data') ### 'corrected' for newer data, 'data' for older data

### Backup the the flagging state at start of reduction
for i in data_params.keys():
    if not os.path.exists(data_params[i]['vis']+\
            ".flagversions/flags.starting_flags"):
       flagmanager(vis=data_params[i]['vis'], mode = 'save', versionname = 'starting_flags', comment = 'Flag states at start of reduction')

### Inspect data in each spw for each dataset
if not skip_plots:
    for i in data_params.keys():
        plotms(vis=data_params[i]['vis'], xaxis='frequency', yaxis='amplitude', 
               field=data_params[i]['field'], ydatacolumn='data', 
               avgtime='1e8', avgscan=True, avgbaseline=True, iteraxis='spw',
               transform=True,freqframe='LSRK')
        input("Press Enter key to advance to next MS/Caltable...")

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
       print('Imaging SB: ',i) 
       if 'LB' in i:
          image_each_obs(data_params[i], prefix, scales=LB_scales,  uvtaper=outertaper,
                   nsigma=5.0, sidelobethreshold=2.5, smoothfactor=1.5,interactive=False,parallel=parallel) 
       else:
          image_each_obs(data_params[i], prefix, scales=SB_scales, 
                   nsigma=5.0, sidelobethreshold=2.5, interactive=False,parallel=parallel)

       #check masks to ensure you are actually masking the image, lower sidelobethreshold if needed

""" Fit Gaussians to roughly estimate centers, inclinations, PAs """
""" Loops through each dataset specified """

for i in data_params.keys():
       print(i)
       data_params[i]['phasecenter']=fit_gaussian(prefix+'_'+i+'_initcont_exec0.image', region='',mask=prefix+'_'+i+'_initcont_exec0.mask')


### Check phase center fits in viewer, tf centers appear too shifted from the Gaussian fit, 
### manually set the phase center dictionary entry by eye


""" The emission centers are slightly misaligned.  So we split out the 
    individual executions, shift the peaks to the phase center, and reassign 
    the phase centers to a common direction. """

### Set common direction for each EB using one as reference (typically best looking LB image)

for i in data_params.keys():
       #################### MANUALLY SET THIS ######################
       data_params[i]['common_dir']='J2000 19h37m00.89941s +07d34m09.82s'

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
                                                     region='',mask=prefix+'_'+i+'_initcont_shift.mask')
      print('Phasecenter new: ',data_params[i]['phasecenter_new'])
      print('Phasecenter old: ',data_params[i]['phasecenter'])

### save updated data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



###############################################################
############### PLOT UV DATA TO CHECK SCALING #################
###############################################################
PA, incl = 0, 0

    ### Assign rough emission geometry parameters; keep 0, 0


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
refdata='LB3'

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

###############################################################
############### SCALE DATA RELATIVE TO ONE EB #################
###############################################################

os.system('rm -rf *_rescaled.ms')
for i in data_params.keys():
   rescale_flux(data_params[i]['vis_avg_shift'], [data_params[i]['gencal_scale']])
   data_params[i]['vis_avg_shift_rescaled']=data_params[i]['vis_avg_shift'].replace('.ms','_rescaled.ms')


###############################################################
############## PLOT UV DATA TO CHECK RE-SCALING ###############
###############################################################


    ### Assign rough emission geometry parameters; keep 0, 0
PA, incl = 0, 0

   ### Check that rescaling did what we expect
export_vislist_rescaled=[]
for i in data_params.keys():
      export_MS(data_params[i]['vis_avg_shift_rescaled'])
      export_vislist_rescaled.append(data_params[i]['vis_avg_shift_rescaled'].replace('.ms','.vis.npz'))
if not skip_plots:
   plot_deprojected(export_vislist_rescaled,
                     fluxscale=[1.0]*len(export_vislist_rescaled), PA=PA, incl=incl, 
                     show_err=False)
   ### Make sure differences are no longer significant
refdata='LB3'
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


### determine best reference antennas based on geometry and flagging
for i in data_params.keys():
   data_params[i]["refant"] = rank_refants(data_params[i]["vis_avg_shift_rescaled"])

############### CHECK THESE, SHOULD BE FINE #################
SB_spwmap=[0,0,0,0,0,0,0]
SB_contspws = '' 


### Make a list of EBs to image
vislist=[]
for i in data_params.keys():
      if ('LB' in i): # skip over LB EBs if in SB-only mode
         continue
      vislist.append(data_params[i]['vis_avg_shift_rescaled'])


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
               scales=SB_scales, niter=0,parallel=parallel,cellsize='0.025arcsec',imsize=1600,nterms=2)
estimate_SNR(prefix+'_dirty.image.tt0', disk_mask=common_mask, 
             noise_mask=noise_annulus)


#B335_dirty.image.tt0
#Beam 0.344 arcsec x 0.262 arcsec (54.46 deg)
#Flux inside disk mask: 43.99 mJy
#Peak intensity of source: 35.18 mJy/beam
#rms: 8.38e-01 mJy/beam
#Peak SNR: 41.97

#nterms=2
#B335_dirty.image.tt0
#Beam 0.340 arcsec x 0.260 arcsec (59.76 deg)
#Flux inside disk mask: 34.70 mJy
#Peak intensity of source: 31.92 mJy/beam
#rms: 6.32e-01 mJy/beam
#Peak SNR: 50.50


### Image produced by iter 0 has not selfcal applied, it's used to set the initial model
### only images >0 have self-calibration applied

### Run self-calibration command set
### 0. Split off corrected data from previous selfcal iteration (except iteration 0)
### 1. Image data to specified nsigma depth, set model column
### 2. Calculate self-cal gain solutions
### 3. Apply self-cal gain solutions to MS

############# USERS MAY NEED TO ADJUST NSIGMA AND SOLINT FOR EACH SELF-CALIBRATION ITERATION ##############
###initial nsigma ~ S/N /2


iteration=0
self_calibrate(prefix,data_params,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=20.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,nterms=2)


### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
       if 'LB' in i:
          continue
       plotms(vis=data_params[i]['vis_avg_shift_rescaled'].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_SB-only_p0.image.tt0
#Beam 0.340 arcsec x 0.260 arcsec (59.76 deg)
#Flux inside disk mask: 50.26 mJy
#Peak intensity of source: 31.92 mJy/beam
#rms: 3.55e-01 mJy/beam
#Peak SNR: 90.01


iteration=1
self_calibrate(prefix,data_params,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=10.0,solint='30s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,nterms=2)

if not skip_plots:
   for i in data_params.keys():
       if 'LB' in i:
          continue
       plotms(vis=data_params[i]['vis_avg_shift_rescaled'].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_SB-only_p1.image.tt0
#Beam 0.340 arcsec x 0.260 arcsec (59.76 deg)
#Flux inside disk mask: 63.25 mJy
#Peak intensity of source: 37.03 mJy/beam
#rms: 1.49e-01 mJy/beam
#Peak SNR: 247.94

iteration=2
self_calibrate(prefix,data_params,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=5.0,solint='18s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel,nterms=2)

if not skip_plots:
   for i in data_params.keys():
       if 'LB' in i:
          continue
       plotms(vis=data_params[i]['vis_avg_shift_rescaled'].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")


#B335_SB-only_p2.image.tt0
#Beam 0.340 arcsec x 0.260 arcsec (59.76 deg)
#Flux inside disk mask: 70.43 mJy
#Peak intensity of source: 41.47 mJy/beam
#rms: 1.09e-01 mJy/beam
#Peak SNR: 380.24

iteration=3
self_calibrate(prefix,data_params,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='int',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel,nterms=2)

if not skip_plots:
   for i in data_params.keys():
       if 'LB' in i:
          continue
       plotms(vis=data_params[i]['vis_avg_shift_rescaled'].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_SB-only_p3.image.tt0
#Beam 0.340 arcsec x 0.260 arcsec (59.76 deg)
#Flux inside disk mask: 75.98 mJy
#Peak intensity of source: 41.96 mJy/beam
#rms: 9.52e-02 mJy/beam
#Peak SNR: 440.69


### Changing self-cal mode here to ap, see use of prevselfcalmode to ensure proper split

iteration=4
self_calibrate(prefix,data_params,mode='SB-only',iteration=iteration,selfcalmode='ap',prevselfcalmode='p',nsigma=2.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel,nterms=2)

if not skip_plots:
   for i in data_params.keys():
       if 'LB' in i:
          continue
       plotms(vis=data_params[i]['vis_avg_shift_rescaled'].replace('.ms','_SB-only_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")

#B335_SB-only_p4.image.tt0
#Beam 0.340 arcsec x 0.260 arcsec (59.76 deg)
#Flux inside disk mask: 79.86 mJy
#Peak intensity of source: 42.31 mJy/beam
#rms: 8.79e-02 mJy/beam
#Peak SNR: 481.50

### Make the final image, will not run another self-calibration
iteration=5
self_calibrate(prefix,data_params,mode='SB-only',iteration=iteration,selfcalmode='ap',nsigma=2.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel,nterms=2,finalimageonly=True)

#B335_SB-only_ap5.image.tt0
#Beam 0.342 arcsec x 0.262 arcsec (59.97 deg)
#Flux inside disk mask: 77.50 mJy
#Peak intensity of source: 42.44 mJy/beam
#rms: 8.50e-02 mJy/beam
#Peak SNR: 498.98

###Backup gain table list for LB+SB runs
for i in data_params.keys():
   if 'SB' in i:
      data_params[i]['selfcal_spwmap_SB-only']=data_params[i]['selfcal_spwmap'].copy()
      data_params[i]['selfcal_tables_SB-only']=data_params[i]['selfcal_tables'].copy()
      data_params[i]['vis_avg_selfcal_SB-only']=(data_params[i]['vis_avg_selfcal']+'.')[:-1]  ## trick to copy the string
###############################################################
################### SELF-CALIBRATION SB+LB ####################
###############################################################
LB_spwmap=[0,0,0,0,0]
LB_contspws = '' 


### Make a list of EBs to image
vislist=[]
for i in data_params.keys():
   vislist.append(data_params[i]['vis_avg_shift_rescaled'])

### Initial dirty map to assess DR
tclean_wrapper(vis=vislist, imagename=prefix+'_LB+SB_dirty', 
               scales=SB_scales, niter=0,parallel=parallel,cellsize='0.003arcsec',imsize=5000,nterms=2, robust=0.0)
estimate_SNR(prefix+'_LB+SB_dirty.image.tt0', disk_mask=common_mask, 
             noise_mask=noise_annulus)

#B335_LB+SB_p0.image.tt0
#Beam 0.033 arcsec x 0.022 arcsec (-54.76 deg)
#Flux inside disk mask: 105.84 mJy
#Peak intensity of source: 4.85 mJy/beam
#rms: 1.30e-02 mJy/beam
#Peak SNR: 371.32

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
iteration=0
self_calibrate(prefix,data_params,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=40.0,solint='600s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,combine='spw,scan',parallel=parallel,smoothfactor=2.0,nterms=2)


### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i]['vis_avg_shift_rescaled'].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_LB+SB_p0.image.tt0
#Beam 0.033 arcsec x 0.022 arcsec (-54.76 deg)
#Flux inside disk mask: 105.03 mJy
#Peak intensity of source: 4.85 mJy/beam
#rms: 1.30e-02 mJy/beam
#Peak SNR: 371.52

iteration=1
self_calibrate(prefix,data_params,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=20.0,solint='240s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,combine='spw,scan',parallel=parallel,smoothfactor=2.0,nterms=2)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i]['vis_avg_shift_rescaled'].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_LB+SB_p1.image.tt0
#Beam 0.033 arcsec x 0.022 arcsec (-54.76 deg)
#Flux inside disk mask: 101.16 mJy
#Peak intensity of source: 5.27 mJy/beam
#rms: 1.21e-02 mJy/beam
#Peak SNR: 436.14

iteration=2
self_calibrate(prefix,data_params,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=5.0,solint='120s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,combine='spw,scan',smoothfactor=2.0,nterms=2)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i]['vis_avg_shift_rescaled'].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_LB+SB_p2.image.tt0
#Beam 0.033 arcsec x 0.022 arcsec (-54.76 deg)
#Flux inside disk mask: 91.56 mJy
#Peak intensity of source: 5.55 mJy/beam
#rms: 1.15e-02 mJy/beam
#Peak SNR: 483.08

iteration=3
self_calibrate(prefix,data_params,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,nterms=2)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i]['vis_avg_shift_rescaled'].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_LB+SB_p3.image.tt0
#Beam 0.033 arcsec x 0.022 arcsec (-54.76 deg)
#Flux inside disk mask: 87.40 mJy
#Peak intensity of source: 5.86 mJy/beam
#rms: 1.13e-02 mJy/beam
#Peak SNR: 518.69

iteration=4
self_calibrate(prefix,data_params,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='24.24s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,nterms=2)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i]['vis_avg_shift_rescaled'].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")


#B335_LB+SB_p4.image.tt0
#Beam 0.033 arcsec x 0.022 arcsec (-54.76 deg)
#Flux inside disk mask: 87.23 mJy
#Peak intensity of source: 6.14 mJy/beam
#rms: 1.12e-02 mJy/beam
#Peak SNR: 550.29

iteration=5
self_calibrate(prefix,data_params,mode='LB+SB',iteration=iteration,prevselfcalmode='p',selfcalmode='ap',nsigma=3.0,solint='300s',
               noisemasks=[common_mask,noise_annulus],SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,
               LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,nterms=2,combine='scan,spw')

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i]['vis_avg_shift_rescaled'].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")


#B335_LB+SB_p5.image.tt0
#Beam 0.033 arcsec x 0.022 arcsec (-54.76 deg)
#Flux inside disk mask: 86.66 mJy
#Peak intensity of source: 6.26 mJy/beam
#rms: 1.12e-02 mJy/beam
#Peak SNR: 558.04


iteration=6
self_calibrate(prefix,data_params,mode='LB+SB',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,nterms=2)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i]['vis_avg_shift_rescaled'].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_LB+SB_ap6.image.tt0
#Beam 0.032 arcsec x 0.023 arcsec (-54.27 deg)
#Flux inside disk mask: 85.74 mJy
#Peak intensity of source: 6.33 mJy/beam
#rms: 1.12e-02 mJy/beam
#Peak SNR: 566.77


#final image
iteration=7
self_calibrate(prefix,data_params,mode='LB+SB',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,
               parallel=parallel,smoothfactor=2.0,nterms=2,finalimageonly=True)


#B335_LB+SB_ap7.image.tt0
#Beam 0.032 arcsec x 0.022 arcsec (-54.50 deg)
#Flux inside disk mask: 86.81 mJy
#Peak intensity of source: 6.37 mJy/beam
#rms: 1.12e-02 mJy/beam
#Peak SNR: 566.23


###Combine gain table list for the SB runs 
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

for robust in [2.0,1.0,0.5,0.0,-0.5,-1.0,-2.0]:
    print('Generate Robust '+str(robust) + ' image')
    imagename=prefix+'_SBLB_continuum_robust_'+str(robust)
    os.system('rm -rf '+imagename+'*')

    sigma = get_sensitivity(data_params, specmode='mfs')
    
    if robust == 2.0 or robust == 1.0:
       uvrange='>60klambda'
    else:
       uvrange=''

    tclean_wrapper(vis=vislist, imagename=imagename, sidelobethreshold=2.0, 
            smoothfactor=1.5, scales=scales, threshold=2.0*sigma, 
            noisethreshold=3.0, robust=robust, parallel=parallel, 
            cellsize='0.003arcsec', imsize=8000,uvrange=uvrange,nterms=2)

    imagename=imagename+'.image.tt0'
    exportfits(imagename=imagename, fitsimage=imagename+'.fits',overwrite=True,dropdeg=True)

for taper in ['1000klambda','1500klambda','2000klambda','2500klambda','3000klambda']:
   for robust in [0.5]:
      print('Generate Robust '+str(robust) + ' taper '+taper+' image')
      imagename=prefix+'_SBLB_continuum_robust_'+str(robust)+'_taper_'+taper
      os.system('rm -rf '+imagename+'*')
      sigma = get_sensitivity(data_params, specmode='mfs')
      tclean_wrapper(vis=vislist, imagename=imagename, sidelobethreshold=2.0, 
            smoothfactor=1.5, scales=scales, threshold=3.0*sigma, 
            noisethreshold=3.0, robust=robust, parallel=parallel, 
            cellsize='0.003arcsec', imsize=8000,uvtaper=[taper],uvrange='>60klambda',nterms=2)
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
os.system('rm -rf *p*.alpha* *p*.pb.tt0')

### Remove intermediate selfcal MSfiles
os.system("rm -rf *p{0..99}.ms")
os.system("rm -rf *p{0..99}.ms.flagversions")
### Remove rescaled selfcal MSfiles
os.system('rm -rf *rescaled.ms')
os.system('rm -rf *rescaled.ms.flagversions')
### Remove rescaled selfcal MSfiles
os.system('rm -rf *initcont*.ms')
os.system('rm -rf *initcont*.ms.flagversions')






