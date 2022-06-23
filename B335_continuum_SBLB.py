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
skip_plots = True	

### Add field names (corresponding to the field in the MS) here and prefix for 
### filenameing (can be different but try to keep same)
### Only make different if, for example, the field name has a space
field   = {'SB':'B335', 'LB':'B335'}
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
                        'field': 'B335','column': 'data'},
                'LB1': {'vis': LB_path+'uid___A002_Xc5802b_X5a5.ms.split.cal',
                        'spws': '0,1,2,3,4',
                        'field': 'B335','column': 'data'},
                'LB2': {'vis': LB_path+'uid___A002_Xc5e6d0_Xe87.ms.split.cal',
                        'spws': '0,1,2,3,4',
                        'field': 'B335','column': 'data'},
                'LB3': {'vis': LB_path+'uid___A002_Xc5f05e_X51e.ms.split.cal',
                        'spws': '0,1,2,3,4',
                        'field': 'B335','column': 'data'},
                'LB4': {'vis': LB_path+'uid___A002_Xc6141c_X2d94.ms.split.cal',
                        'spws': '0,1,2,3,4',
                        'field': 'B335','column': 'data'},
                'LB5': {'vis': LB_path+'uid___A002_Xc63024_X3e51.ms.split.cal',
                        'spws': '0,1,2,3,4',
                        'field': 'B335','column': 'data'},
               }

### Dictionary defining necessary metadata for each execution
### SiO at 217.10498e9 excluded because of non-detection
### Only bother specifying simple species that are likely present in all datasets
### Hot corino lines (or others) will get taken care of by using the cont.dat
data_params = {'SB1': {'vis' : WD_path+prefix+'_SB1.ms',
                       'name' : 'SB1',
                       'field': field['SB'],
                       'line_spws': np.array([0,1,2,4,5]), # line SPWs, get from listobs
                       'line_freqs': np.array([219.56035410e9,219.94944200e9,220.39868420e9,230.538e9,231.32182830e9]), #restfreqs
                       'line_names': ['C18O','SO','13CO','12CO','N2D+'], #restfreqs
                       'flagrange': np.array([[-5.5,14.5],[-5.5,14.5],[-15.5,25.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {0:0, 1:1, 2:2, 3:3, 4:4, 5:5,6:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([240,240,240,1,480,240,1]), #n channels to average; approximately aiming for 30 MHz channels
                       #'cont_avg_width':  np.array([0,0,0,0,0,0]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2014/09/02/01:36:00~2014/09/02/02:23:00',
                       'contdotdat' : 'SB/cont.dat'
                      }, 
               'LB1': {'vis' : WD_path+prefix+'_LB1.ms',
                       'name' : 'LB1',
                       'field': field['LB'],
                       'line_spws': np.array([1,2,3,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([230.538e9,217.104980e9,219.56035410e9,220.39868420e9]), #restfreqs
                       'line_names': ['12CO','SiO','C18O','13CO'], #restfreqs
                       'flagrange': np.array([[-15.5,24.5],[-15.5,24.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {0:0, 1:1, 2:2, 3:3, 4:4},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4]),  #spws to use for continuum
                       #'cont_avg_width':  np.array([0,0,0,0,0]), #n channels to average; approximately aiming for 30 MHz channels
                       'cont_avg_width':  np.array([1,240,480,480,480]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2017/10/08/00:49:00~2017/10/08/02:08:00',
                       'contdotdat' : 'LB/cont.dat'
                      }, 
               'LB2': {'vis' : WD_path+prefix+'_LB2.ms',
                       'name' : 'LB2',
                       'field': field['LB'],
                       'line_spws': np.array([1,2,3,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([230.538e9,217.104980e9,219.56035410e9,220.39868420e9]), #restfreqs
                       'line_names': ['12CO','SiO','C18O','13CO'], #restfreqs
                       'flagrange': np.array([[-15.5,24.5],[-15.5,24.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {0:0, 1:1, 2:2, 3:3, 4:4},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4]),  #spws to use for continuum
                       #'cont_avg_width':  np.array([0,0,0,0,0]), #n channels to average; approximately aiming for 30 MHz channels
                       'cont_avg_width':  np.array([1,240,480,480,480]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2017/10/21/00:25:00~2017/10/21/01:44:00',
                       'contdotdat' : 'LB/cont.dat'
                      }, 
               'LB3': {'vis' : WD_path+prefix+'_LB3.ms',
                       'name' : 'LB3',
                       'field': field['LB'],
                       'line_spws': np.array([1,2,3,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([230.538e9,217.104980e9,219.56035410e9,220.39868420e9]), #restfreqs
                       'line_names': ['12CO','SiO','C18O','13CO'], #restfreqs
                       'flagrange': np.array([[-15.5,24.5],[-15.5,24.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {0:0, 1:1, 2:2, 3:3, 4:4},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4]),  #spws to use for continuum
                       #'cont_avg_width':  np.array([0,0,0,0,0]), #n channels to average; approximately aiming for 30 MHz channels
                       'cont_avg_width':  np.array([1,240,480,480,480]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2017/10/22/00:32:00~2017/10/22/01:51:00',
                       'contdotdat' : 'LB/cont.dat'
                      }, 
               'LB4': {'vis' : WD_path+prefix+'_LB4.ms',
                       'name' : 'LB4',
                       'field': field['LB'],
                       'line_spws': np.array([1,2,3,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([230.538e9,217.104980e9,219.56035410e9,220.39868420e9]), #restfreqs
                       'line_names': ['12CO','SiO','C18O','13CO'], #restfreqs
                       'flagrange': np.array([[-15.5,24.5],[-15.5,24.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {0:0, 1:1, 2:2, 3:3, 4:4},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4]),  #spws to use for continuum
                       #'cont_avg_width':  np.array([0,0,0,0,0]), #n channels to average; approximately aiming for 30 MHz channels
                       'cont_avg_width':  np.array([1,240,480,480,480]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2017/10/27/00:04:00~2017/10/27/01:21:00',
                       'contdotdat' : 'LB/cont.dat'
                      }, 
               'LB5': {'vis' : WD_path+prefix+'_LB5.ms',
                       'name' : 'LB5',
                       'field': field['LB'],
                       'line_spws': np.array([1,2,3,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([230.538e9,217.104980e9,219.56035410e9,220.39868420e9]), #restfreqs
                       'line_names': ['12CO','SiO','C18O','13CO'], #restfreqs
                       'flagrange': np.array([[-15.5,24.5],[-15.5,24.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {0:0, 1:1, 2:2, 3:3, 4:4},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4]),  #spws to use for continuum
                       #'cont_avg_width':  np.array([0,0,0,0,0]), #n channels to average; approximately aiming for 30 MHz channels
                       'cont_avg_width':  np.array([1,240,480,480,480]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2017/10/29/23:14:00~2017/10/30/00:33:00',
                       'contdotdat' : 'LB/cont.dat'
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
      split(vis=pl_data_params[i]['vis'], outputvis=prefix+'_'+i+'.ms', 
          spw=pl_data_params[i]['spws'], field=pl_data_params[i]['field'],
          datacolumn=pl_data_params[i]['column'])

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
      tclean_wrapper(vis=data_params[i]['vis_avg'], imagename=prefix+'_'+i+'_initial_cont', sidelobethreshold=2.0, 
            smoothfactor=1.5, scales=LB_scales, nsigma=5.0, robust=0.5, parallel=parallel, 
            uvtaper=[outertaper],nterms=1)
   else:
      tclean_wrapper(vis=data_params[i]['vis_avg'], imagename=prefix+'_'+i+'_initial_cont', sidelobethreshold=2.5, 
            smoothfactor=1.5, scales=LB_scales, nsigma=5.0, robust=0.5, parallel=parallel,nterms=1)

       #check masks to ensure you are actually masking the image, lower sidelobethreshold if needed

""" Fit Gaussians to roughly estimate centers, inclinations, PAs """
""" Loops through each dataset specified """
###default fit region is blank for an obvious single source
fit_region=''

###specify manual mask on brightest source if Gaussian fitting fails due to confusion
'''
mask_ra  =  '16h27m26.909s'.replace('h',':').replace('m',':').replace('s','')
mask_dec = '-24d40m50.848s'.replace('d','.').replace('m','.').replace('s','')
mask_pa  = 90.0 	# position angle of mask in degrees
mask_maj = 0.76	# semimajor axis of mask in arcsec
mask_min = 0.75 	# semiminor axis of mask in arcsec
fit_region = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
              (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)
'''
for i in data_params.keys():
       print(i)
       data_params[i]['phasecenter']=fit_gaussian(prefix+'_'+i+'_initial_cont.image.tt0', region=fit_region,mask=prefix+'_'+i+'_initial_cont.mask')


### Check phase center fits in viewer, tf centers appear too shifted from the Gaussian fit, 
### manually set the phase center dictionary entry by eye


""" The emission centers are slightly misaligned.  So we split out the 
    individual executions, shift the peaks to the phase center, and reassign 
    the phase centers to a common direction. """

### Set common direction for each EB using one as reference (typically best looking LB image)

for i in data_params.keys():
       #################### MANUALLY SET THIS ######################
       data_params[i]['common_dir']='J2000 19h37m00.8986s +07d34m09.52s'

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
   print('Imaging MS: ',i) 
   if 'LB' in i:
      tclean_wrapper(vis=data_params[i]['vis_avg_shift'], imagename=prefix+'_'+i+'_initial_cont_shifted', sidelobethreshold=2.0, 
            smoothfactor=1.5, scales=LB_scales, nsigma=5.0, robust=0.5, parallel=parallel, 
            uvtaper=[outertaper],nterms=1)
   else:
       tclean_wrapper(vis=data_params[i]['vis_avg_shift'], imagename=prefix+'_'+i+'_initial_cont_shifted', sidelobethreshold=2.5, 
            smoothfactor=1.5, scales=SB_scales, nsigma=5.0, robust=0.5, parallel=parallel,nterms=1)

for i in data_params.keys():
      print(i)     
      data_params[i]['phasecenter_new']=fit_gaussian(prefix+'_'+i+'_initial_cont_shifted.image.tt0',\
                                                     region=fit_region,mask=prefix+'_'+i+'_initial_cont_shifted.mask')
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

### Plot deprojected visibility profiles for all data together """
plot_deprojected(export_vislist,
                     fluxscale=[1.0]*len(export_vislist), PA=PA, incl=incl, 
                     show_err=False,outfile='amp-vs-uv-distance-pre-selfcal.png')

### Now inspect offsets by comparing against a reference 
### Set reference data using the dictionary key.
### Using SB1 as reference because it looks the nicest by far

#################### MANUALLY SET THIS ######################
refdata='LB2'

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

#################### MANUALLY SET THIS ######################
##DOES THE AMP VS. UV-DISTANCE LOOK VERY DISSIMILAR BETWEEN DIFFERENT EBS?
##IF SO, UNCOMMENT AND RUN THIS BLOCK TO REMOVE SCALING AND PROCEED WITH 2- METHOD PASS

for i in data_params.keys():
   data_params[i]['gencal_scale']=1.0


###############################################################
#################### MANUALLY SET THIS ########################
######### IF ON SECOND PASS WHERE SCALING IS KNOWN ############
################ PLACE SCALING COMMANDS HERE ##################
###############################################################

#if os.path.exists('gencal_scale.pickle'):
#   with open('gencal_scale.pickle', 'rb') as handle:
#      gencal_scale=pickle.load(handle)
#   for i in data_params.keys():
#      data_params[i]['gencal_scale']=gencal_scale[i]


data_params["SB1"]["gencal_scale"]=0.749
data_params["LB1"]["gencal_scale"]=1.002
data_params["LB2"]["gencal_scale"]=1.000
data_params["LB3"]["gencal_scale"]=1.035
data_params["LB4"]["gencal_scale"]=0.994
data_params["LB5"]["gencal_scale"]=1.005



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
#selectedVis='vis_avg_rescaled'
selectedVis='vis_avg_shift_rescaled'

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

### Initial map to assess DR
tclean_wrapper(vis=vislist, imagename=prefix+'_initial', 
               scales=SB_scales, sidelobethreshold=2.0, smoothfactor=1.5, nsigma=3.0, 
               noisethreshold=3.0, robust=0.5, parallel=parallel,cellsize='0.025arcsec',imsize=1600, 
               phasecenter=data_params['SB1']['common_dir'].replace('J2000','ICRS'))
initial_SNR,initial_RMS=estimate_SNR(prefix+'_initial.image.tt0', disk_mask=common_mask, 
                        noise_mask=noise_annulus)

listdict,scantimesdict,integrationsdict,integrationtimesdict,integrationtimes,n_spws,minspw,spwsarray=fetch_scan_times(vislist,[data_params['SB1']['field']])

solints,gaincal_combine=get_solints_simple(vislist,scantimesdict,integrationtimesdict)
print('Suggested Solints:')
print(solints)
print('Suggested Gaincal Combine params:')
print(gaincal_combine)
nsigma_init=np.max([initial_SNR/15.0,5.0]) # restricts initial nsigma to be at least 5
nsigma_per_solint=10**np.linspace(np.log10(nsigma_init),np.log10(3.0),len(solints))
print('Suggested nsigma per solint: ')
print(nsigma_per_solint)

#B335_initial.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 34.19 mJy
#Peak intensity of source: 16.76 mJy/beam
#rms: 1.58e-01 mJy/beam
#Peak SNR: 105.96
#Suggested Solints:
#['inf', 'inf', '187.49s', '90.72s', '42.34s', '18.14s', 'int']
#Suggested Gaincal Combine params:
#['spw,scan', 'spw', 'spw', 'spw', 'spw', 'spw', 'spw']
#Suggested nsigma per solint: 
#[7.06404726 6.12441868 5.30977538 4.60349235 3.99115599 3.46026993 3.        ]

#setup the same number of iterations as suggested solints




### Image produced by iter 0 has not selfcal applied, it's used to set the initial model
### only images >0 have self-calibration applied

### Run self-calibration command set
### 0. Split off corrected data from previous selfcal iteration (except iteration 0)
### 1. Image data to specified nsigma depth, set model column
### 2. Calculate self-cal gain solutions
### 3. Apply self-cal gain solutions to MS

############# USERS MAY NEED TO ADJUST NSIGMA AND SOLINT FOR EACH SELF-CALIBRATION ITERATION ##############
iteration=0
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',
               nsigma=7.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel,combine='spw,scan')


### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

### Make note of key metrics of image in each round

#B335_SB-only_p0.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 31.01 mJy
#Peak intensity of source: 17.02 mJy/beam
#rms: 1.78e-01 mJy/beam
#Peak SNR: 95.34

#B335_SB-only_p0_post.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 32.07 mJy
#Peak intensity of source: 18.31 mJy/beam
#rms: 1.28e-01 mJy/beam
#Peak SNR: 142.53


iteration=1
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=6.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_SB-only_p1.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 33.96 mJy
#Peak intensity of source: 18.44 mJy/beam
#rms: 1.21e-01 mJy/beam
#Peak SNR: 152.53

#B335_SB-only_p1_post.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 34.02 mJy
#Peak intensity of source: 19.67 mJy/beam
#rms: 1.08e-01 mJy/beam
#Peak SNR: 182.87



iteration=2
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=5.3,solint='90.72s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_SB-only_p2.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 35.56 mJy
#Peak intensity of source: 19.71 mJy/beam
#rms: 1.01e-01 mJy/beam
#Peak SNR: 195.69

#B335_SB-only_p2_post.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 35.87 mJy
#Peak intensity of source: 21.32 mJy/beam
#rms: 9.93e-02 mJy/beam
#Peak SNR: 214.83


iteration=3
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=4.2,solint='42.34s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_SB-only_p3.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 36.96 mJy
#Peak intensity of source: 21.31 mJy/beam
#rms: 9.19e-02 mJy/beam
#Peak SNR: 231.91


#B335_SB-only_p3_post.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 36.93 mJy
#Peak intensity of source: 21.82 mJy/beam
#rms: 9.22e-02 mJy/beam
#Peak SNR: 236.82


iteration=4
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=4.6,solint='18.41s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")


#B335_SB-only_p4.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 36.54 mJy
#Peak intensity of source: 21.78 mJy/beam
#rms: 9.29e-02 mJy/beam
#Peak SNR: 234.41

#B335_SB-only_p4_post.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 36.50 mJy
#Peak intensity of source: 22.06 mJy/beam
#rms: 9.27e-02 mJy/beam
#Peak SNR: 238.06


iteration=5
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=5.0,solint='int',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")
#B335_SB-only_p5.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 36.28 mJy
#Peak intensity of source: 22.04 mJy/beam
#rms: 9.36e-02 mJy/beam
#Peak SNR: 235.46
#B335_SB-only_p5_post.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 35.96 mJy
#Peak intensity of source: 22.15 mJy/beam
#rms: 9.30e-02 mJy/beam
#Peak SNR: 238.29

### Changing self-cal mode here to ap, see use of prevselfcalmode to ensure proper split

iteration=6
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',prevselfcalmode='p',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")

#B335_SB-only_p6.image.tt0
#Beam 0.341 arcsec x 0.262 arcsec (58.90 deg)
#Flux inside disk mask: 38.17 mJy
#Peak intensity of source: 22.23 mJy/beam
#rms: 8.62e-02 mJy/beam
#Peak SNR: 257.99

#B335_SB-only_p6_post.image.tt0
#Beam 0.347 arcsec x 0.266 arcsec (59.55 deg)
#Flux inside disk mask: 36.89 mJy
#Peak intensity of source: 22.43 mJy/beam
#rms: 8.48e-02 mJy/beam
#Peak SNR: 264.61


### Make the final image, will not run another self-calibration
iteration=7
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,
               parallel=parallel,finalimageonly=True)

#B335_SB-only_ap7.image.tt0
#Beam 0.347 arcsec x 0.266 arcsec (59.55 deg)
#Flux inside disk mask: 36.30 mJy
#Peak intensity of source: 22.44 mJy/beam
#rms: 8.54e-02 mJy/beam
#Peak SNR: 262.84


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
LB_spwmap=[0,0,0,0,0]
LB_contspws = '' 


### Make a list of EBs to image
vislist=[]
fieldlist=[]
for i in data_params.keys():
   vislist.append(data_params[i][selectedVis])
   fieldlist.append(data_params[i]['field'])

### Initial map to assess DR
tclean_wrapper(vis=vislist, imagename=prefix+'_initial_LB+SB',cellsize='0.003arcsec',imsize=6000, 
               scales=LB_scales, sidelobethreshold=2.0, smoothfactor=1.5, nsigma=3.0, 
               noisethreshold=3.0, robust=0.5, parallel=parallel, nterms=1,
               phasecenter=data_params['SB1']['common_dir'].replace('J2000','ICRS'))
initial_SNR,initial_RMS=estimate_SNR(prefix+'_initial_LB+SB.image.tt0', disk_mask=common_mask, 
                        noise_mask=noise_annulus)

listdict,scantimesdict,integrationsdict,integrationtimesdict,integrationtimes,n_spws,minspw,spwsarray=fetch_scan_times(vislist,fieldlist)

solints,gaincal_combine=get_solints_simple(vislist,scantimesdict,integrationtimesdict)
print('Suggested Solints:')
print(solints)
print('Suggested Gaincal Combine params:')
print(gaincal_combine)
nsigma_init=np.max([initial_SNR/15.0,5.0]) # restricts initial nsigma to be at least 5
nsigma_per_solint=10**np.linspace(np.log10(nsigma_init),np.log10(3.0),len(solints))
print('Suggested nsigma per solint: ')
print(nsigma_per_solint)
#B335_initial_LB+SB.image.tt0
#Beam 0.033 arcsec x 0.023 arcsec (-54.71 deg)
#Flux inside disk mask: 48.41 mJy
#Peak intensity of source: 4.57 mJy/beam
#rms: 2.01e-02 mJy/beam
#Peak SNR: 227.76

#Suggested Solints:
#['inf', 'inf', '18.14s', 'int']
#Suggested Gaincal Combine params:
#['spw,scan', 'spw', 'spw', 'spw']
#Suggested nsigma per solint: 
#[15.18370296  8.84352791  5.15078477  3.        ]



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
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=15.2,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,combine='spw,scan',parallel=parallel,smoothfactor=2.0)


### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_LB+SB_p0.image.tt0
#Beam 0.033 arcsec x 0.023 arcsec (-54.71 deg)
#Flux inside disk mask: 50.54 mJy
#Peak intensity of source: 4.66 mJy/beam
#rms: 2.02e-02 mJy/beam
#Peak SNR: 230.67
#B335_LB+SB_p0_post.image.tt0
#Beam 0.033 arcsec x 0.023 arcsec (-54.71 deg)
#Flux inside disk mask: 50.86 mJy
#Peak intensity of source: 4.91 mJy/beam
#rms: 2.01e-02 mJy/beam
#Peak SNR: 244.57


iteration=1
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=9.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,combine='spw',parallel=parallel,smoothfactor=2.0)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_LB+SB_p1.image.tt0
#Beam 0.033 arcsec x 0.023 arcsec (-54.71 deg)
#Flux inside disk mask: 50.18 mJy
#Peak intensity of source: 4.89 mJy/beam
#rms: 2.00e-02 mJy/beam
#Peak SNR: 244.33

#B335_LB+SB_p1_post.image.tt0
#Beam 0.033 arcsec x 0.023 arcsec (-54.71 deg)
#Flux inside disk mask: 50.18 mJy
#Peak intensity of source: 6.05 mJy/beam
#rms: 1.98e-02 mJy/beam
#Peak SNR: 306.08

iteration=2
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=5.0,solint='18.14s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,combine='spw',smoothfactor=2.0)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#B335_LB+SB_p2.image.tt0
#Beam 0.033 arcsec x 0.023 arcsec (-54.71 deg)
#Flux inside disk mask: 48.86 mJy
#Peak intensity of source: 5.96 mJy/beam
#rms: 1.95e-02 mJy/beam
#Peak SNR: 305.90

#B335_LB+SB_p2_post.image.tt0
#Beam 0.033 arcsec x 0.023 arcsec (-54.71 deg)
#Flux inside disk mask: 48.60 mJy
#Peak intensity of source: 6.26 mJy/beam
#rms: 1.95e-02 mJy/beam
#Peak SNR: 321.18

'''
### Changing self-cal mode here to ap, see use of prevselfcalmode to ensure proper split
iteration=3
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',prevselfcalmode='p',nsigma=3.0,solint='300s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,combine='scan,spw',smoothfactor=2.0)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")

#B335_LB+SB_p3.image.tt0
#Beam 0.033 arcsec x 0.023 arcsec (-54.71 deg)
#Flux inside disk mask: 47.75 mJy
#Peak intensity of source: 6.23 mJy/beam
#rms: 1.95e-02 mJy/beam
#Peak SNR: 320.04

#B335_LB+SB_p3_post.image.tt0
#Beam 0.033 arcsec x 0.023 arcsec (-54.61 deg)
#Flux inside disk mask: 42.73 mJy
#Peak intensity of source: 6.44 mJy/beam
#rms: 1.97e-02 mJy/beam
#Peak SNR: 326.35
'''

iteration=3
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,
               LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,finalimageonly=True)


#B335_LB+SB_ap4.image.tt0
#Beam 0.033 arcsec x 0.023 arcsec (-54.61 deg)
#Flux inside disk mask: 43.14 mJy
#Peak intensity of source: 6.46 mJy/beam
#rms: 1.97e-02 mJy/beam
#Peak SNR: 327.07




###Backup gain table list for LB+SB runs
for i in data_params.keys():
   if 'SB' in i:
      data_params[i]['selfcal_spwmap']=data_params[i]['selfcal_spwmap_SB-only']+data_params[i]['selfcal_spwmap']
      data_params[i]['selfcal_tables']=data_params[i]['selfcal_tables_SB-only']+data_params[i]['selfcal_tables']

#save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


###############################################################
############ SHIFT PHASE CENTERS TO CHECK SCALING #############
###############################################################

for i in data_params.keys():
   print(i)
   data_params[i]['vis_avg_shift_selfcal']=prefix+'_'+i+'_selfcal_cont_shift.ms'
   os.system('rm -rf '+data_params[i]['vis_avg_shift_selfcal'])
   fixvis(vis=data_params[i]['vis_avg_selfcal'], outputvis=data_params[i]['vis_avg_shift_selfcal'], 
       field=data_params[i]['field'], 
       phasecenter='J2000 '+data_params[i]['phasecenter'])
   ### fix planets may throw an error, usually safe to ignore
   fixplanets(vis=data_params[i]['vis_avg_shift_selfcal'], field=data_params[i]['field'], 
           direction=data_params[i]['common_dir'])

###############################################################
############### PLOT UV DATA TO CHECK SCALING #################
###############################################################

### Assign rough emission geometry parameters; keep 0, 0
PA, incl = 0, 0

### Export MS contents into Numpy save files 
export_vislist=[]
for i in data_params.keys():
   export_MS(data_params[i]['vis_avg_shift_selfcal'])
   export_vislist.append(data_params[i]['vis_avg_shift_selfcal'].replace('.ms','.vis.npz'))

### Plot deprojected visibility profiles for all data together """
plot_deprojected(export_vislist,
                     fluxscale=[1.0]*len(export_vislist), PA=PA, incl=incl, 
                     show_err=False,outfile='amp-vs-uv-distance-post-selfcal.png')

#################### MANUALLY SET THIS ######################
refdata='LB2'

reference=prefix+'_'+refdata+'_selfcal_cont_shift.vis.npz'
for i in data_params.keys():
   print(i)
   if i != refdata:
      data_params[i]['gencal_scale_selfcal']=estimate_flux_scale(reference=reference, 
                        comparison=prefix+'_'+i+'_selfcal_cont_shift.vis.npz', 
                        incl=incl, PA=PA)
   else:
      data_params[i]['gencal_scale_selfcal']=1.0
   print(' ')

gencal_scale={}

print('IF AND ONLY IF SCALING WAS NOT ALREADY SET IN SCRIPT')
print('COPY AND PLACE WHERE DESIGNATED FOR SCALING TOWARD TOP OF SCRIPT')
for i in data_params.keys():
   gencal_scale[i]=data_params[i]['gencal_scale_selfcal']
   print('data_params["{}"]["gencal_scale"]={:0.3f}'.format(i,data_params[i]['gencal_scale_selfcal']))

#WRITE OUT PICKLE FILE FOR SCALING IF MISSED
if not os.path.exists('gencal_scale.pickle'):
   with open('gencal_scale.pickle', 'wb') as handle:
      pickle.dump(gencal_scale, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
imsize=6000
cell='0.003arcsec'
for robust in [-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0]:
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

for taper in ['1000klambda', '2000klambda', '3000klambda']:
  for robust in [1.0, 2.0]:
    print('Generate Robust '+str(robust)+' taper '+taper+' image')
    imagename = prefix+'_SBLB_continuum_robust_'+str(robust)+'_taper_'+taper
    os.system('rm -rf '+imagename+'*')
    sigma = get_sensitivity(data_params, specmode='mfs',
                            imsize=imsize, robust=robust, cellsize=cell,uvtaper=taper)
    tclean_wrapper(vis=vislist, imagename=imagename, sidelobethreshold=2.0, 
                   smoothfactor=1.5, scales=scales, threshold=2.0*sigma, 
                   noisethreshold=3.0, robust=robust, parallel=parallel, 
                   cellsize=cell, imsize=imsize, nterms=1,
                   uvtaper=[taper], uvrange='',
                   phasecenter=data_params['SB1']['common_dir'].replace('J2000', 'ICRS'))
    imagename = imagename+'.image.tt0'
    exportfits(imagename=imagename, fitsimage=imagename+'.fits',
               overwrite=True, dropdeg=True)
if selectedVis=='vis_avg_shift_rescaled':
   tclean_wrapper(vis=data_params['LB1']['vis'].replace('.ms','_initcont.ms'), imagename='temporary.pbfix',
                   threshold=0.0,niter=0, scales=[0],
                   robust=0.5, parallel=parallel, 
                   cellsize=cell, imsize=imsize, nterms=1,
                   phasecenter=data_params['LB1']['common_dir'])
   pblist=glob.glob('*continuum*.pb.tt0') 
   os.system('mkdir orig_pbimages')
   for pbimage in pblist:
      os.system('mv '+pbimage+' orig_pbimages')
      os.system('cp -r temporary.pbfix.pb.tt0 '+pbimage)
   os.system('rm -rf temporary.pbfix.*')
###############################################################
########################### CLEANUP ###########################
###############################################################

### Remove extra image products
os.system('rm -rf *.residual* *.psf* *.model* *dirty* *.sumwt* *.gridwt* *.workdirectory')

### put selfcalibration intermediate images somewhere safe
os.system('rm -rf initial_images')
os.system('mkdir initial_images')
os.system('mv *initial*.image *_p*.image* *_ap*.image* initial_images')
os.system('mv *initial*.mask *_p*.mask *_ap*.mask initial_images')
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






