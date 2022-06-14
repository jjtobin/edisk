"""
eDisk data reduction script
This script was written for CASA 6.1.1/6.2
Originally derived from DSHARP reduction scripts


Datasets calibrated (in order of date observed):
SB1: uid___A002_Xebb7f0_X690.ms
SB2: uid___A002_Xebd1e8_X38be.ms

----
LB1: uid___A002_Xf1aa15_X91aa.ms
LB2: uid___A002_Xf1bb4a_X2738.ms
LB3: uid___A002_Xf1bb4a_X6092.ms    
LB4: uid___A002_Xf1bb4a_X101be.ms
LB5: uid___A002_Xf1bb4a_X144ad.ms

"""

"""
#To run from pickle file:
with open(prefix+'.pickle','rb') as handle:
	data_params = pickle.load(handle)
"""
	
"""

reducer: Sacha Gavino
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
execfile('../edisk/reduction_utils3.py', globals())


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
field   = {'SB':'BHR71_IRS2', 'LB':'BHR71_IRS2'}
prefix  = 'BHR71_IRS2' 

### always include trailing slashes!!
WD_path = '/lustre/cv/projects/edisk/BHR71_IRS2/'
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
pl_data_params={'SB1': {'vis': SB_path+'uid___A002_Xebb7f0_X690.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['SB'],
                        'column': 'corrected'},
                'SB2': {'vis': SB_path+'uid___A002_Xebd1e8_X38be.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['SB'],
                        'column': 'corrected'},
                'LB1': {'vis': LB_path+'uid___A002_Xf1aa15_X91aa.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['LB'],
                        'column': 'corrected'},
                'LB2': {'vis': LB_path+'uid___A002_Xf1bb4a_X2738.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['LB'],
                        'column': 'corrected'},
                'LB3': {'vis': LB_path+'uid___A002_Xf1bb4a_X6092.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['LB'],
                        'column': 'corrected'},
                'LB4': {'vis': LB_path+'uid___A002_Xf1bb4a_X101be.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['LB'],
                        'column': 'corrected'},
                'LB5': {'vis': LB_path+'uid___A002_Xf1bb4a_X144ad.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['LB'],
                        'column': 'corrected'},
               }

### Dictionary defining necessary metadata for each execution
### SiO at 217.10498e9 excluded because of non-detection
### Only bother specifying simple species that are likely present in all datasets
### Hot corino lines (or others) will get taken care of by using the cont.dat
data_params = {'SB1': {'vis' : WD_path+prefix+'_SB1.ms',
                       'name' : 'SB1',
                       'field': field['SB'],
                       'line_spws': np.array([0,1,2,3,4,6,4,4,4,4,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9,220.39868420e9,219.94944200e9,219.56035410e9,
                                               217.82215e9,230.538e9,217.94005e9,218.16044e9,217.2386e9,
                                               218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO','13CO','SO','C18O','c-C3H2','12CO','c-C3H2','c-C3H2','DCN','H2CO','H2CO'], #restfreqs
                       'flagrange': np.array([[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0]]),
                       'orig_spw_map': {25:0, 31:1, 29:2, 27:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/05/04/00:27:00~2021/05/04/01:33:00',
                       'contdotdat' : 'SB/cont-BHR71_IRS2.dat'
                      }, 
                'SB2': {'vis' : WD_path+prefix+'_SB2.ms',
                       'name' : 'SB2',
                       'field': field['SB'],
                       'line_spws': np.array([0,1,2,3,4,6,4,4,4,4,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9,220.39868420e9,219.94944200e9,219.56035410e9,
                                               217.82215e9,230.538e9,217.94005e9,218.16044e9,217.2386e9,
                                               218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO','13CO','SO','C18O','c-C3H2','12CO','c-C3H2','c-C3H2','DCN','H2CO','H2CO'], #restfreqs
                       'flagrange': np.array([[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/05/09/01:28:00~2021/05/09/02:34:00',
                       'contdotdat' : 'SB/cont-BHR71_IRS2.dat'
                      }, 
               'LB1': {'vis' : WD_path+prefix+'_LB1.ms',
                       'name' : 'LB1',
                       'field': field['LB'],
                       'line_spws': np.array([0,1,2,3,4,6,4,4,4,4,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9,220.39868420e9,219.94944200e9,219.56035410e9,
                                               217.82215e9,230.538e9,217.94005e9,218.16044e9,217.2386e9,
                                               218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO','13CO','SO','C18O','c-C3H2','12CO','c-C3H2','c-C3H2','DCN','H2CO','H2CO'], #restfreqs
                       'flagrange': np.array([[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/10/14/14:50:14~2021/10/14/16:06:23',
                       'contdotdat' : 'LB/cont-BHR71-IRS2.dat'
                      }, 
               'LB2': {'vis' : WD_path+prefix+'_LB2.ms',
                       'name' : 'LB2',
                       'field': field['LB'],
                       'line_spws': np.array([0,1,2,3,4,6,4,4,4,4,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9,220.39868420e9,219.94944200e9,219.56035410e9,
                                               217.82215e9,230.538e9,217.94005e9,218.16044e9,217.2386e9,
                                               218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO','13CO','SO','C18O','c-C3H2','12CO','c-C3H2','c-C3H2','DCN','H2CO','H2CO'], #restfreqs
                       'flagrange': np.array([[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/10/15/13:15:35~2021/10/15/14:37:49',
                       'contdotdat' : 'LB/cont-BHR71-IRS2.dat'
                      }, 
               'LB3': {'vis' : WD_path+prefix+'_LB3.ms',
                       'name' : 'LB3',
                       'field': field['LB'],
                       'line_spws': np.array([0,1,2,3,4,6,4,4,4,4,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9,220.39868420e9,219.94944200e9,219.56035410e9,
                                               217.82215e9,230.538e9,217.94005e9,218.16044e9,217.2386e9,
                                               218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO','13CO','SO','C18O','c-C3H2','12CO','c-C3H2','c-C3H2','DCN','H2CO','H2CO'], #restfreqs
                       'flagrange': np.array([[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/10/16/12:19:39~2021/10/16/13:40:37',
                       'contdotdat' : 'LB/cont-BHR71-IRS2.dat'
                      }, 
               'LB4': {'vis' : WD_path+prefix+'_LB4.ms',
                       'name' : 'LB4',
                       'field': field['LB'],
                       'line_spws': np.array([0,1,2,3,4,6,4,4,4,4,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9,220.39868420e9,219.94944200e9,219.56035410e9,
                                               217.82215e9,230.538e9,217.94005e9,218.16044e9,217.2386e9,
                                               218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO','13CO','SO','C18O','c-C3H2','12CO','c-C3H2','c-C3H2','DCN','H2CO','H2CO'], #restfreqs
                       'flagrange': np.array([[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/10/19/14:36:09~2021/10/19/15:58:25',
                       'contdotdat' : 'LB/cont-BHR71-IRS2.dat'
                      }, 
               'LB5': {'vis' : WD_path+prefix+'_LB5.ms',
                       'name' : 'LB5',
                       'field': field['LB'],
                       'line_spws': np.array([0,1,2,3,4,6,4,4,4,4,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9,220.39868420e9,219.94944200e9,219.56035410e9,
                                               217.82215e9,230.538e9,217.94005e9,218.16044e9,217.2386e9,
                                               218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO','13CO','SO','C18O','c-C3H2','12CO','c-C3H2','c-C3H2','DCN','H2CO','H2CO'], #restfreqs
                       'flagrange': np.array([[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0],[-10.0,0.0],
                                              [-10.0,0.0],[-10.0,0.0],[-10.0,0.0]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/10/20/13:59:02~2021/10/20/15:21:18',
                       'contdotdat' : 'LB/cont-BHR71-IRS2.dat'
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

#IRS1: ra 12:01:36.810 dec -65.08.49.220
#IRS2: ra 12:01:34.090 dec -65.08.47.360
###specify manual mask on brightest source if Gaussian fitting fails due to confusion
#mask_ra IRS2: 12h01m34.005s
#mask_dec IRS2: -65d08m48.101s
mask_ra  =  '12h01m34.005s'.replace('h',':').replace('m',':').replace('s','')
mask_dec = '-65d08m48.101s'.replace('d','.').replace('m','.').replace('s','')
mask_pa  = 90.0 	# position angle of mask in degrees
mask_maj = 0.76	# semimajor axis of mask in arcsec
mask_min = 0.75 	# semiminor axis of mask in arcsec
fit_region = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
              (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)

for i in data_params.keys():
       print(i)
       data_params[i]['phasecenter']=fit_gaussian(prefix+'_'+i+'_initial_cont.image.tt0', region=fit_region,mask=prefix+'_'+i+'_initial_cont.mask')

### Check phase center fits in viewer, if centers appear too shifted from the Gaussian fit, 
### manually set the phase center dictionary entry by eye


""" The emission centers are slightly misaligned.  So we split out the 
    individual executions, shift the peaks to the phase center, and reassign 
    the phase centers to a common direction. """

### Set common direction for each EB using one as reference (typically best looking LB image)

for i in data_params.keys():
       #################### MANUALLY SET THIS ######################
       data_params[i]['common_dir']='J2000 12h01m34.009044s -65d08m48.06040s'  #LB1

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
      tclean_wrapper(vis=data_params[i]['vis_avg_shift'], imagename=prefix+'_'+i+'_initial_cont_shift', sidelobethreshold=2.0, 
            smoothfactor=1.5, scales=LB_scales, nsigma=5.0, robust=0.5, parallel=parallel, 
            uvtaper=[outertaper],nterms=1)
   else:
       tclean_wrapper(vis=data_params[i]['vis_avg_shift'], imagename=prefix+'_'+i+'_initial_cont_shift', sidelobethreshold=2.5, 
            smoothfactor=1.5, scales=SB_scales, nsigma=5.0, robust=0.5, parallel=parallel,nterms=1)

for i in data_params.keys():
      print(i)     
      data_params[i]['phasecenter_new']=fit_gaussian(prefix+'_'+i+'_initial_cont_shift.image.tt0',\
                                                     region=fit_region,mask=prefix+'_'+i+'_initial_cont_shift.mask')
      print('Phasecenter new: ',data_params[i]['phasecenter_new'])
      print('Phasecenter old: ',data_params[i]['phasecenter'])

### save updated data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
SB1
12h01m34.009065s -65d08m48.05978s
#Peak of Gaussian component identified with imfit: J2000 12h01m34.009065s -65d08m48.05978s
#PA of Gaussian component: 29.90 deg
#Inclination of Gaussian component: 32.77 deg
#Pixel coordinates of peak: x = 799.984 y = 800.001
Phasecenter new:  12h01m34.009065s -65d08m48.05978s
Phasecenter old:  12h01m34.00893s -065d08m48.078537s
SB2
12h01m34.009137s -65d08m48.05953s
#Peak of Gaussian component identified with imfit: J2000 12h01m34.009137s -65d08m48.05953s
#PA of Gaussian component: 64.46 deg
#Inclination of Gaussian component: 41.91 deg
#Pixel coordinates of peak: x = 799.968 y = 800.009
Phasecenter new:  12h01m34.009137s -65d08m48.05953s
Phasecenter old:  12h01m34.01258s -065d08m48.054427s
LB1
12h01m34.009054s -65d08m48.06069s
#Peak of Gaussian component identified with imfit: J2000 12h01m34.009054s -65d08m48.06069s
#PA of Gaussian component: 33.17 deg
#Inclination of Gaussian component: 24.70 deg
#Pixel coordinates of peak: x = 2999.857 y = 2999.707
Phasecenter new:  12h01m34.009054s -65d08m48.06069s
Phasecenter old:  12h01m34.01098s -065d08m48.043737s
LB2
12h01m34.009042s -65d08m48.06042s
#Peak of Gaussian component identified with imfit: J2000 12h01m34.009042s -65d08m48.06042s
#PA of Gaussian component: 19.38 deg
#Inclination of Gaussian component: 26.65 deg
#Pixel coordinates of peak: x = 2999.884 y = 2999.795
Phasecenter new:  12h01m34.009042s -65d08m48.06042s
Phasecenter old:  12h01m34.01057s -065d08m48.069607s
LB3
12h01m34.009043s -65d08m48.06043s
#Peak of Gaussian component identified with imfit: J2000 12h01m34.009043s -65d08m48.06043s
#PA of Gaussian component: 12.72 deg
#Inclination of Gaussian component: 30.81 deg
#Pixel coordinates of peak: x = 2999.881 y = 2999.794
Phasecenter new:  12h01m34.009043s -65d08m48.06043s
Phasecenter old:  12h01m34.00970s -065d08m48.069617s
LB4
12h01m34.009044s -65d08m48.06041s
#Peak of Gaussian component identified with imfit: J2000 12h01m34.009044s -65d08m48.06041s
#PA of Gaussian component: 57.46 deg
#Inclination of Gaussian component: 45.41 deg
#Pixel coordinates of peak: x = 2999.879 y = 2999.800
Phasecenter new:  12h01m34.009044s -65d08m48.06041s
Phasecenter old:  12h01m34.00951s -065d08m48.073197s
LB5
12h01m34.009046s -65d08m48.06041s
#Peak of Gaussian component identified with imfit: J2000 12h01m34.009046s -65d08m48.06041s
#PA of Gaussian component: 98.86 deg
#Inclination of Gaussian component: 43.63 deg
#Pixel coordinates of peak: x = 2999.874 y = 2999.799
Phasecenter new:  12h01m34.009046s -65d08m48.06041s
Phasecenter old:  12h01m34.01024s -065d08m48.077467s
"""

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
### Using SB2 as reference because it looks the nicest

#################### MANUALLY SET THIS ######################
refdata='SB2'

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
#################### MANUALLY SET THIS ######################
##DOES THE AMP VS. UV-DISTANCE LOOK VERY DISSIMILAR BETWEEN DIFFERENT EBS?
##IF SO, UNCOMMENT AND RUN THIS BLOCK TO REMOVE SCALING AND PROCEED WITH 2- METHOD PASS

#for i in data_params.keys():
#   data_params[i]['gencal_scale']=1.0

"""
SB1
#The ratio of the fluxes of BHR71_IRS2_SB1_initcont_shift.vis.npz to BHR71_IRS2_SB2_initcont_shift.vis.npz is 0.79466
#The scaling factor for gencal is 0.891 for your comparison measurement
#The error on the weighted mean ratio is 1.848e-03, although it's likely that the weights in the measurement sets are off by some constant factor
 
SB2
 
LB1
#The ratio of the fluxes of BHR71_IRS2_LB1_initcont_shift.vis.npz to BHR71_IRS2_SB2_initcont_shift.vis.npz is 0.53250
#The scaling factor for gencal is 0.730 for your comparison measurement
#The error on the weighted mean ratio is 3.236e-03, although it's likely that the weights in the measurement sets are off by some constant factor
 
LB2
#The ratio of the fluxes of BHR71_IRS2_LB2_initcont_shift.vis.npz to BHR71_IRS2_SB2_initcont_shift.vis.npz is 0.85425
#The scaling factor for gencal is 0.924 for your comparison measurement
#The error on the weighted mean ratio is 2.920e-03, although it's likely that the weights in the measurement sets are off by some constant factor
 
LB3
#The ratio of the fluxes of BHR71_IRS2_LB3_initcont_shift.vis.npz to BHR71_IRS2_SB2_initcont_shift.vis.npz is 0.99755
#The scaling factor for gencal is 0.999 for your comparison measurement
#The error on the weighted mean ratio is 2.927e-03, although it's likely that the weights in the measurement sets are off by some constant factor
 
LB4
#The ratio of the fluxes of BHR71_IRS2_LB4_initcont_shift.vis.npz to BHR71_IRS2_SB2_initcont_shift.vis.npz is 0.89101
#The scaling factor for gencal is 0.944 for your comparison measurement
#The error on the weighted mean ratio is 3.217e-03, although it's likely that the weights in the measurement sets are off by some constant factor
 
LB5
#The ratio of the fluxes of BHR71_IRS2_LB5_initcont_shift.vis.npz to BHR71_IRS2_SB2_initcont_shift.vis.npz is 0.98350
#The scaling factor for gencal is 0.992 for your comparison measurement
#The error on the weighted mean ratio is 3.105e-03, although it's likely that the weights in the measurement sets are off by some constant factor
"""

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

data_params["SB1"]["gencal_scale"]=0.910
data_params["SB2"]["gencal_scale"]=1.000
data_params["LB1"]["gencal_scale"]=1.004
data_params["LB2"]["gencal_scale"]=1.049
data_params["LB3"]["gencal_scale"]=1.069
data_params["LB4"]["gencal_scale"]=1.068
data_params["LB5"]["gencal_scale"]=1.159


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
   refdata='SB2'
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
mask_maj = 4.01	# semimajor axis of mask in arcsec
mask_min = 4.0 	# semiminor axis of mask in arcsec

common_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
              (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)
""" Define a noise annulus, measure the peak SNR in map """
noise_annulus = "annulus[[%s, %s],['%.2farcsec', '10.0arcsec']]" % \
                (mask_ra, mask_dec, 2.0*mask_maj) 


###############################################################
###################### SELF-CALIBRATION #######################
###############################################################

### Initial dirty map to assess DR
tclean_wrapper(vis=vislist, imagename=prefix+'_initial', 
               scales=SB_scales, sidelobethreshold=2.0, smoothfactor=1.5, nsigma=3.0, 
               noisethreshold=3.0, robust=0.5, parallel=parallel,
               imsize = 3200, cellsize = '0.025arcsec', # newly added line
               phasecenter=data_params['SB1']['common_dir'].replace('J2000','ICRS'))
initial_SNR,initial_RMS=estimate_SNR(prefix+'_initial.image.tt0', disk_mask=common_mask, 
                        noise_mask=noise_annulus)


#BHR71_IRS2_initial.image.tt0
#Beam 0.277 arcsec x 0.200 arcsec (16.01 deg)
#Flux inside disk mask: 87.13 mJy
#Peak intensity of source: 16.17 mJy/beam
#rms: 1.41e-01 mJy/beam
#Peak SNR: 114.64


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
"""
Suggested Solints:
['inf', 'inf', '72.58s', '30.24s', '12.10s', 'int']
Suggested Gaincal Combine params:
['spw,scan', 'spw', 'spw', 'spw', 'spw', 'spw']
Suggested nsigma per solint: 
[7.64275917 6.33905312 5.25773396 4.36086681 3.61698776 3.        ]

"""



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
               nsigma=7.64275917,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel,combine='spw,scan')


### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#BHR71_IRS2_SB-only_p0.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 118.86 mJy
#Peak intensity of source: 16.07 mJy/beam
#rms: 1.56e-01 mJy/beam
#Peak SNR: 103.15

#BHR71_IRS2_SB-only_p0_post.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 108.45 mJy
#Peak intensity of source: 16.30 mJy/beam
#rms: 1.39e-01 mJy/beam
#Peak SNR: 117.38



iteration=1
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=6.33905312,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")


#BHR71_IRS2_SB-only_p1.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 105.63 mJy
#Peak intensity of source: 16.28 mJy/beam
#rms: 1.35e-01 mJy/beam
#Peak SNR: 120.90

#BHR71_IRS2_SB-only_p1_post.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 107.37 mJy
#Peak intensity of source: 17.06 mJy/beam
#rms: 1.18e-01 mJy/beam
#Peak SNR: 144.84

iteration=2
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=5.25773396,solint='72.58s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#BHR71_IRS2_SB-only_p2.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 103.41 mJy
#Peak intensity of source: 16.96 mJy/beam
#rms: 1.11e-01 mJy/beam
#Peak SNR: 152.53

#BHR71_IRS2_SB-only_p2_post.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 103.25 mJy
#Peak intensity of source: 17.22 mJy/beam
#rms: 1.10e-01 mJy/beam
#Peak SNR: 156.11



iteration=3
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=4.36086681,solint='30.24s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#BHR71_IRS2_SB-only_p3.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 100.87 mJy
#Peak intensity of source: 17.16 mJy/beam
#rms: 1.07e-01 mJy/beam
#Peak SNR: 160.90

#BHR71_IRS2_SB-only_p3_post.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 101.03 mJy
#Peak intensity of source: 17.31 mJy/beam
#rms: 1.07e-01 mJy/beam
#Peak SNR: 162.37


iteration=4
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=3.61698776,solint='12.10s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#BHR71_IRS2_SB-only_p4.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 99.03 mJy
#Peak intensity of source: 17.29 mJy/beam
#rms: 1.04e-01 mJy/beam
#Peak SNR: 166.75

#BHR71_IRS2_SB-only_p4_post.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 99.41 mJy
#Peak intensity of source: 17.42 mJy/beam
#rms: 1.04e-01 mJy/beam
#Peak SNR: 167.37


iteration=5
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='int',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")


#BHR71_IRS2_SB-only_p5.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 97.57 mJy
#Peak intensity of source: 17.39 mJy/beam
#rms: 1.02e-01 mJy/beam
#Peak SNR: 170.82

#BHR71_IRS2_SB-only_p5_post.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 97.98 mJy
#Peak intensity of source: 17.45 mJy/beam
#rms: 1.02e-01 mJy/beam
#Peak SNR: 171.21


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

#BHR71_IRS2_SB-only_p6.image.tt0
#Beam 0.262 arcsec x 0.199 arcsec (14.54 deg)
#Flux inside disk mask: 97.90 mJy
#Peak intensity of source: 17.43 mJy/beam
#rms: 1.02e-01 mJy/beam
#Peak SNR: 171.05

#BHR71_IRS2_SB-only_p6_post.image.tt0
#Beam 0.269 arcsec x 0.200 arcsec (15.93 deg)
#Flux inside disk mask: 91.17 mJy
#Peak intensity of source: 17.62 mJy/beam
#rms: 9.53e-02 mJy/beam
#Peak SNR: 184.88

iteration=7
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel,finalimageonly=True)

#BHR71_IRS2_SB-only_ap7.image.tt0
#Beam 0.269 arcsec x 0.200 arcsec (15.93 deg)
#Flux inside disk mask: 91.81 mJy
#Peak intensity of source: 17.66 mJy/beam
#rms: 9.58e-02 mJy/beam
#Peak SNR: 184.39

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
fieldlist=[]
for i in data_params.keys():
   vislist.append(data_params[i][selectedVis])
   fieldlist.append(data_params[i]['field'])

### Initial dirty map to assess DR
tclean_wrapper(vis=vislist, imagename=prefix+'_initial_LB+SB',imsize=7000,cellsize='0.006arcsec', 
               scales=LB_scales, sidelobethreshold=2.0, smoothfactor=1.5, nsigma=3.0, 
               noisethreshold=3.0, robust=0.5, parallel=parallel, nterms=1,
               phasecenter=data_params['SB1']['common_dir'].replace('J2000','ICRS'))
initial_SNR,initial_RMS=estimate_SNR(prefix+'_initial_LB+SB.image.tt0', disk_mask=common_mask, 
                        noise_mask=noise_annulus)

#BHR71_IRS2_initial_LB+SB.image.tt0
#Beam 0.065 arcsec x 0.049 arcsec (20.67 deg)
#Flux inside disk mask: 109.80 mJy
#Peak intensity of source: 4.41 mJy/beam
#rms: 2.02e-02 mJy/beam
#Peak SNR: 218.39
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


"""
Suggested Solints:
['inf', 'inf', '12.10s', 'int']
Suggested Gaincal Combine params:
['spw,scan', 'spw', 'spw', 'spw']
Suggested nsigma per solint: 
[13.54155502  8.19381418  4.95796758  3.        ]
"""



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
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=13.54155502,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,combine='spw,scan',parallel=parallel,smoothfactor=2.0,imsize=7000,cellsize='0.006arcsec')

### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#BHR71_IRS2_LB+SB_p0.image.tt0
#Beam 0.065 arcsec x 0.049 arcsec (20.66 deg)
#Flux inside disk mask: 120.26 mJy
#Peak intensity of source: 4.45 mJy/beam
#rms: 2.22e-02 mJy/beam
#Peak SNR: 200.66


#BHR71_IRS2_LB+SB_p0_post.image.tt0
#Beam 0.065 arcsec x 0.049 arcsec (20.66 deg)
#Flux inside disk mask: 116.71 mJy
#Peak intensity of source: 4.64 mJy/beam
#rms: 2.18e-02 mJy/beam
#Peak SNR: 213.03

iteration=1
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=8.19381418,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,combine='spw',parallel=parallel,smoothfactor=2.0,imsize=7000,cellsize='0.006arcsec')

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")


#BHR71_IRS2_LB+SB_p1.image.tt0
#Beam 0.065 arcsec x 0.049 arcsec (20.66 deg)
#Flux inside disk mask: 108.94 mJy
#Peak intensity of source: 4.60 mJy/beam
#rms: 2.14e-02 mJy/beam
#Peak SNR: 214.57

#BHR71_IRS2_LB+SB_p1_post.image.tt0
#Beam 0.065 arcsec x 0.049 arcsec (20.66 deg)
#Flux inside disk mask: 108.84 mJy
#Peak intensity of source: 7.02 mJy/beam
#rms: 1.79e-02 mJy/beam
#Peak SNR: 392.61

iteration=2
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=4.95796758,solint='12.00s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,combine='spw',smoothfactor=2.0,imsize=7000,cellsize='0.006arcsec')

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")


#BHR71_IRS2_LB+SB_p2.image.tt0
#Beam 0.065 arcsec x 0.049 arcsec (20.66 deg)
#Flux inside disk mask: 100.40 mJy
#Peak intensity of source: 7.12 mJy/beam
#rms: 1.73e-02 mJy/beam
#Peak SNR: 412.43

#BHR71_IRS2_LB+SB_p2_post.image.tt0
#Beam 0.065 arcsec x 0.049 arcsec (20.66 deg)
#Flux inside disk mask: 101.79 mJy
#Peak intensity of source: 8.46 mJy/beam
#rms: 1.66e-02 mJy/beam
#Peak SNR: 508.56

iteration=3
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='int',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,imsize=7000,cellsize='0.006arcsec')

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#BHR71_IRS2_LB+SB_p3.image.tt0
#Beam 0.065 arcsec x 0.049 arcsec (20.66 deg)
#Flux inside disk mask: 95.23 mJy
#Peak intensity of source: 8.50 mJy/beam
#rms: 1.63e-02 mJy/beam
#Peak SNR: 522.73

#BHR71_IRS2_LB+SB_p3_post.image.tt0
#Beam 0.065 arcsec x 0.049 arcsec (20.66 deg)
#Flux inside disk mask: 95.75 mJy
#Peak intensity of source: 8.75 mJy/beam
#rms: 1.62e-02 mJy/beam
#Peak SNR: 540.77


iteration=4
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',
               prevselfcalmode='p',nsigma=3.0,solint='600s',noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,
               parallel=parallel,combine='spw,scan',smoothfactor=2.0,imsize=7000,cellsize='0.006arcsec')

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")
#BHR71_IRS2_LB+SB_p4.image.tt0
#Beam 0.070 arcsec x 0.054 arcsec (19.38 deg)
#Flux inside disk mask: 102.42 mJy
#Peak intensity of source: 11.34 mJy/beam
#rms: 2.44e-02 mJy/beam
#Peak SNR: 464.08

#BHR71_IRS2_LB+SB_p4_post.image.tt0
#Beam 0.072 arcsec x 0.054 arcsec (21.33 deg)
#Flux inside disk mask: 98.67 mJy
#Peak intensity of source: 11.49 mJy/beam
#rms: 2.40e-02 mJy/beam
#Peak SNR: 478.13

iteration=5
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='300s',
               noisemasks=[common_mask,noise_annulus],SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,
               LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,combine='spw,scan',smoothfactor=2.0,
               imsize=7000,cellsize='0.006arcsec')

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")

#BHR71_IRS2_LB+SB_ap5.image.tt0
#Beam 0.072 arcsec x 0.054 arcsec (21.33 deg)
#Flux inside disk mask: 101.43 mJy
#Peak intensity of source: 11.52 mJy/beam
#rms: 2.41e-02 mJy/beam
#Peak SNR: 477.63

#BHR71_IRS2_LB+SB_ap5_post.image.tt0
#Beam 0.072 arcsec x 0.054 arcsec (21.37 deg)
#Flux inside disk mask: 100.89 mJy
#Peak intensity of source: 11.66 mJy/beam
#rms: 2.42e-02 mJy/beam
#Peak SNR: 481.28

iteration=6
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='300s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,
               combine='spw,scan',smoothfactor=2.0,imsize=7000,cellsize='0.006arcsec',finalimageonly=True)

#BHR71_IRS2_LB+SB_ap6.image.tt0
#Beam 0.072 arcsec x 0.054 arcsec (21.37 deg)
#Flux inside disk mask: 102.08 mJy
#Peak intensity of source: 11.66 mJy/beam
#rms: 2.43e-02 mJy/beam
#Peak SNR: 480.83


if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")



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
refdata='SB2'

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

"""
SB1
#The ratio of the fluxes of BHR71_IRS2_SB1_selfcal_cont_shift.vis.npz to BHR71_IRS2_SB2_selfcal_cont_shift.vis.npz is 1.04166
#The scaling factor for gencal is 1.021 for your comparison measurement
#The error on the weighted mean ratio is 2.275e-03, although it's likely that the weights in the measurement sets are off by some constant factor
../edisk/reduction_utils3.py:1306: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show(block = False)
 
SB2
 
LB1
#The ratio of the fluxes of BHR71_IRS2_LB1_selfcal_cont_shift.vis.npz to BHR71_IRS2_SB2_selfcal_cont_shift.vis.npz is 1.65871
#The scaling factor for gencal is 1.288 for your comparison measurement
#The error on the weighted mean ratio is 5.738e-03, although it's likely that the weights in the measurement sets are off by some constant factor
../edisk/reduction_utils3.py:1306: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show(block = False)

LB2
#The ratio of the fluxes of BHR71_IRS2_LB2_selfcal_cont_shift.vis.npz to BHR71_IRS2_SB2_selfcal_cont_shift.vis.npz is 1.25256
#The scaling factor for gencal is 1.119 for your comparison measurement
#The error on the weighted mean ratio is 3.637e-03, although it's likely that the weights in the measurement sets are off by some constant factor
../edisk/reduction_utils3.py:1306: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show(block = False)
 
LB3
#The ratio of the fluxes of BHR71_IRS2_LB3_selfcal_cont_shift.vis.npz to BHR71_IRS2_SB2_selfcal_cont_shift.vis.npz is 1.13206
#The scaling factor for gencal is 1.064 for your comparison measurement
#The error on the weighted mean ratio is 2.926e-03, although it's likely that the weights in the measurement sets are off by some constant factor
../edisk/reduction_utils3.py:1306: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show(block = False)
 
LB4
#The ratio of the fluxes of BHR71_IRS2_LB4_selfcal_cont_shift.vis.npz to BHR71_IRS2_SB2_selfcal_cont_shift.vis.npz is 1.28753
#The scaling factor for gencal is 1.135 for your comparison measurement
#The error on the weighted mean ratio is 3.797e-03, although it's likely that the weights in the measurement sets are off by some constant factor
../edisk/reduction_utils3.py:1306: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show(block = False)
 
LB5
#The ratio of the fluxes of BHR71_IRS2_LB5_selfcal_cont_shift.vis.npz to BHR71_IRS2_SB2_selfcal_cont_shift.vis.npz is 1.24615
#The scaling factor for gencal is 1.116 for your comparison measurement
#The error on the weighted mean ratio is 3.128e-03, although it's likely that the weights in the measurement sets are off by some constant factor
../edisk/reduction_utils3.py:1306: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
  plt.show(block = False)

IF AND ONLY IF SCALING WAS NOT ALREADY SET IN SCRIPT
COPY AND PLACE WHERE DESIGNATED FOR SCALING TOWARD TOP OF SCRIPT

data_params["SB1"]["gencal_scale"]=0.910
data_params["SB2"]["gencal_scale"]=1.000
data_params["LB1"]["gencal_scale"]=1.004
data_params["LB2"]["gencal_scale"]=1.049
data_params["LB3"]["gencal_scale"]=1.069
data_params["LB4"]["gencal_scale"]=1.068
data_params["LB5"]["gencal_scale"]=1.159

"""

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
    imagename=prefix+'_SBLB_continuum_robust_'+str(robust)
    os.system('rm -rf '+imagename+'*')

    sigma = get_sensitivity(data_params, specmode='mfs',imsize=imsize,robust=robust,cellsize=cell)
    # adjust sigma to corrector for the irregular noise in the images if needed
    # correction factor may vary or may not be needed at all depending on source
    if robust == 2.0 or robust == 1.0:
       sigma=sigma*1.75
    tclean_wrapper(vis=vislist, imagename=imagename, sidelobethreshold=2.0, 
            smoothfactor=1.5, scales=scales, threshold=3.0*sigma, 
            noisethreshold=3.0, robust=robust, parallel=parallel, 
            cellsize=cell, imsize=imsize,phasecenter=data_params['SB1']['common_dir'].replace('J2000','ICRS'))

    imagename=imagename+'.image.tt0'
    exportfits(imagename=imagename, fitsimage=imagename+'.fits',overwrite=True,dropdeg=True)

imsize=4300
cell='0.01arcsec'
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





