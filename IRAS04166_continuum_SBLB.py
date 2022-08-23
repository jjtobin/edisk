"""
eDisk data reduction script
This script was written for CASA 6.1.1/6.2
Originally derived from DSHARP reduction scripts


Datasets calibrated (in order of date observed):
SB1: 03-Jul-2022/13:17:00.7   to   03-Jul-2022/14:26:28.2
SB2: 03-Jul-2022/14:27:43.5
LB1: 24/10/2021
LB2: 18/10/2021
LB3: 01/10/2021
LB4: 30/09/2021
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
field   = {'SB':'IRAS04166+2706', 'LB':'IRAS04166+2706'}
prefix  = 'IRAS04166+2706' 

### always include trailing slashes!!
WD_path = '/lustre/cv/projects/edisk/IRAS04166/'
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
pl_data_params={'SB1': {'vis': SB_path+'uid___A002_Xfafcc0_X6944.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['SB'],
                        'column': 'corrected'}, # 03-Jul-2022/13:17:00.7   to   03-Jul-2022/14:26:28.2
                'SB2': {'vis': SB_path+'uid___A002_Xfafcc0_X6f3d.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['SB'],
                        'column': 'corrected'}, #  03-Jul-2022/14:27:43.5   to   03-Jul-2022/15:37:10.2
                'LB1': {'vis': LB_path+'uid___A002_Xf20692_X27ec.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['LB'],
                        'column': 'corrected'}, #24-Oct-2021/07:29:24.9   to   24-Oct-2021/08:50:16.1 (UTC)
                'LB2': {'vis': LB_path+'uid___A002_Xf1bb4a_Xb41c.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['LB'],
                        'column': 'corrected'}, #18-Oct-2021/04:44:31.5   to   18-Oct-2021/06:13:44.6 (UTC)
                'LB3': {'vis': LB_path+'uid___A002_Xf1479a_X141e.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['LB'],
                        'column': 'corrected'}, #01-Oct-2021/05:24:12.7   to   01-Oct-2021/06:54:11.2 (UTC)
                'LB4': {'vis': LB_path+'uid___A002_Xf138ff_X23de.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field['LB'],
                        'column': 'corrected'}, #30-Sep-2021/05:30:09.8   to   30-Sep-2021/06:50:32.4 (UTC)
               }
### Dictionary defining necessary metadata for each execution
### SiO at 217.10498e9 excluded because of non-detection
### Only bother specifying simple species that are likely present in all datasets
### Hot corino lines (or others) will get taken care of by using the cont.dat

data_params = { 'SB1': {'vis' : WD_path+prefix+'_SB1.ms',
                       'name' : 'SB1',
                       'field': field['SB'],
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
                       'timerange': '2022/07/03/13:17:00.7~2022/07/03/14:26:28.2',
                       'contdotdat' : SB_path+'cont.dat'
                      },
                'SB2': {'vis' : WD_path+prefix+'_SB2.ms',
                       'name' : 'SB2',
                       'field': field['SB'],
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
                       'timerange': '2022/07/03/14:27:43.5~2022/07/03/15:37:10.2',
                       'contdotdat' : SB_path+'cont.dat'
                      },
                
                'LB1': {'vis' : WD_path+prefix+'_LB1.ms',
                       'name' : 'LB1',
                       'field': field['LB'],
                       'line_spws': np.array([0,1,2,3,4,6,4,4,4,4,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9,220.39868420e9,219.94944200e9,219.56035410e9,
                                               217.82215e9,230.538e9,217.94005e9,218.16044e9,217.2386e9,
                                               218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO','13CO','SO','C18O','c-C3H2','12CO','c-C3H2','c-C3H2','DCN','H2CO','H2CO'], #restfreqs
                       'flagrange': np.array([[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],
                                              [-5.5,14.5],[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],
                                              [-5.5,14.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6}, #mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/10/24/07:29:24.9~2021/10/24/08:50:16.1',
                       'contdotdat' : LB_path+'cont.dat'},
               'LB2': {'vis' : WD_path+prefix+'_LB2.ms',
                       'name' : 'LB2',
                       'field': field['LB'],
                       'line_spws': np.array([0,1,2,3,4,6,4,4,4,4,4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9,220.39868420e9,219.94944200e9,219.56035410e9,
                                               217.82215e9,230.538e9,217.94005e9,218.16044e9,217.2386e9,
                                               218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO','13CO','SO','C18O','c-C3H2','12CO','c-C3H2','c-C3H2','DCN','H2CO','H2CO'], #restfreqs
                       'flagrange': np.array([[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],
                                              [-5.5,14.5],[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],
                                              [-5.5,14.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6}, #mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/10/18/04:44:31.5~2021/10/18/06:13:44.6',
                       'contdotdat' : LB_path+'cont.dat'},
                'LB3': {'vis' : WD_path+prefix+'_LB3.ms',
                       'name' : 'LB3',
                       'field': field['LB'],
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
                       'timerange': '2021/10/01/05:24:12.7~2021/10/01/06:54:11.2',
                       'contdotdat' : LB_path+'cont.dat'},
               'LB4': {'vis' : WD_path+prefix+'_LB4.ms',
                       'name' : 'LB4',
                       'field': field['LB'],
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
                       'timerange': '2021/09/30/05:30:09.8~2021/09/30/06:50:32.4',
                       'contdotdat' : LB_path+'cont.dat'},
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

mask_ra  =  '04h19m42.5s'.replace('h',':').replace('m',':').replace('s','')
mask_dec = '+27d13m36.0s'.replace('d','.').replace('m','.').replace('s','')
mask_pa  = 90.0 	# position angle of mask in degrees
mask_maj = 0.95	# semimajor axis of mask in arcsec
mask_min = 0.85 	# semiminor axis of mask in arcsec
fit_region = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
              (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)

for i in data_params.keys():
       print(i)
       data_params[i]['phasecenter']=fit_gaussian(prefix+'_'+i+'_initial_cont.image.tt0', region=fit_region, dooff=False) #mask=prefix+'_'+i+'_initial_cont.mask')

####SB1
#04h19m42.509353s +27d13m35.85890s
#Peak of Gaussian component identified with imfit: ICRS 04h19m42.509353s +27d13m35.85890s
##04h19m42.509353s +27d13m35.85890s
##Separation: radian = 7.35155e-08, degrees = 0.000004 = 4.21213e-06, arcsec = 0.015164 = 0.0151637
#Peak in J2000 coordinates: 04:19:42.50991, +027:13:35.845681
#PA of Gaussian component: 72.11 deg
#Inclination of Gaussian component: 43.52 deg
#Pixel coordinates of peak: x = 795.855 y = 795.281
####SB2
##04h19m42.509064s +27d13m35.82117s
#Peak of Gaussian component identified with imfit: ICRS 04h19m42.509064s +27d13m35.82117s
##04h19m42.509064s +27d13m35.82117s
##Separation: radian = 7.34796e-08, degrees = 0.000004 = 4.21007e-06, arcsec = 0.015156 = 0.0151563
#Peak in J2000 coordinates: 04:19:42.50962, +027:13:35.807952
#PA of Gaussian component: 152.34 deg
#Inclination of Gaussian component: 33.73 deg
#Pixel coordinates of peak: x = 795.983 y = 794.023
###LB1
#04h19m42.505524s +27d13m35.81580s
#Peak of Gaussian component identified with imfit: ICRS 04h19m42.505524s +27d13m35.81580s
#04h19m42.505524s +27d13m35.81580s
#Separation: radian = 7.34796e-08, degrees = 0.000004 = 4.21007e-06, arcsec = 0.015156 = 0.0151563
#Peak in J2000 coordinates: 04:19:42.50608, +027:13:35.802582
#PA of Gaussian component: 122.12 deg
#Inclination of Gaussian component: 39.99 deg
#Pixel coordinates of peak: x = 2975.570 y = 2938.441
#LB2
#04h19m42.505027s +27d13m35.84460s
#Peak of Gaussian component identified with imfit: ICRS 04h19m42.505027s +27d13m35.84460s
#04h19m42.505027s +27d13m35.84460s
#Separation: radian = 7.33849e-08, degrees = 0.000004 = 4.20464e-06, arcsec = 0.015137 = 0.0151367
#Peak in J2000 coordinates: 04:19:42.50558, +027:13:35.831382
#PA of Gaussian component: 22.37 deg
#Inclination of Gaussian component: 31.52 deg
#Pixel coordinates of peak: x = 2977.781 y = 2948.040
#LB3
#04h19m42.505839s +27d13m35.83305s
#Peak of Gaussian component identified with imfit: ICRS 04h19m42.505839s +27d13m35.83305s
#04h19m42.505839s +27d13m35.83305s
#Separation: radian = 7.36383e-08, degrees = 0.000004 = 4.21917e-06, arcsec = 0.015189 = 0.015189
#Peak in J2000 coordinates: 04:19:42.50640, +027:13:35.819832
#PA of Gaussian component: 58.20 deg
#Inclination of Gaussian component: 48.78 deg
#Pixel coordinates of peak: x = 2974.167 y = 2944.191
####LB4
#04h19m42.503261s +27d13m35.80493s
#Peak of Gaussian component identified with imfit: ICRS 04h19m42.503261s +27d13m35.80493s
#04h19m42.503261s +27d13m35.80493s
#Separation: radian = 7.35789e-08, degrees = 0.000004 = 4.21576e-06, arcsec = 0.015177 = 0.0151767
#Peak in J2000 coordinates: 04:19:42.50382, +027:13:35.791711
#PA of Gaussian component: 43.19 deg
#Inclination of Gaussian component: 63.45 deg
#Pixel coordinates of peak: x = 2985.631 y = 2934.817
######
### Check phase center fits in viewer, tf centers appear too shifted from the Gaussian fit, 
### manually set the phase center dictionary entry by eye


""" The emission centers are slightly misaligned.  So we split out the 
    individual executions, shift the peaks to the phase center, and reassign 
    the phase centers to a common direction. """

### Set common direction for each EB using one as reference (typically best looking LB image)

for i in data_params.keys():
       #################### MANUALLY SET THIS ######################
       data_params[i]['common_dir']='J2000 04h19m42.50554s +27d13m35.81580s'

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
   #fixplanets(vis=data_params[i]['vis_avg_shift'], field=data_params[i]['field'], 
   #        direction=data_params[i]['common_dir'])

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
                                                     region=fit_region) #,mask=prefix+'_'+i+'_initial_cont_shifted.mask')
      print('Phasecenter new: ',data_params[i]['phasecenter_new'])
      print('Phasecenter old: ',data_params[i]['phasecenter'])
####SB1
#04h19m42.509903s +27d13m35.84568s
#Peak of Gaussian component identified with imfit: J2000 04h19m42.509903s +27d13m35.84568s
#PA of Gaussian component: 72.21 deg
#Inclination of Gaussian component: 43.65 deg
#Pixel coordinates of peak: x = 800.012 y = 799.997
#Phasecenter new:  04h19m42.509903s +27d13m35.84568s
#Phasecenter old:  04h19m42.50991s +027d13m35.845681s
#SB2
#04h19m42.509620s +27d13m35.80795s
#Peak of Gaussian component identified with imfit: J2000 04h19m42.509620s +27d13m35.80795s
#PA of Gaussian component: 152.26 deg
#Inclination of Gaussian component: 33.75 deg
#Pixel coordinates of peak: x = 800.016 y = 799.977
#Phasecenter new:  04h19m42.509620s +27d13m35.80795s
#Phasecenter old:  04h19m42.50962s +027d13m35.807952s
#LB1
#04h19m42.506080s +27d13m35.80258s
#Peak of Gaussian component identified with imfit: J2000 04h19m42.506080s +27d13m35.80258s
#PA of Gaussian component: 122.11 deg
#Inclination of Gaussian component: 39.98 deg
#Pixel coordinates of peak: x = 2999.996 y = 3000.038
#Phasecenter new:  04h19m42.506080s +27d13m35.80258s
#Phasecenter old:  04h19m42.50608s +027d13m35.802582s
#LB2
#04h19m42.505585s +27d13m35.83140s
#Peak of Gaussian component identified with imfit: J2000 04h19m42.505585s +27d13m35.83140s
#PA of Gaussian component: 22.36 deg
#Inclination of Gaussian component: 31.56 deg
#Pixel coordinates of peak: x = 2999.754 y = 3000.018
#Phasecenter new:  04h19m42.505585s +27d13m35.83140s
#Phasecenter old:  04h19m42.50558s +027d13m35.831382s
#LB3
#04h19m42.506394s +27d13m35.81987s
#Peak of Gaussian component identified with imfit: J2000 04h19m42.506394s +27d13m35.81987s
#PA of Gaussian component: 58.21 deg
#Inclination of Gaussian component: 48.75 deg
#Pixel coordinates of peak: x = 2999.823 y = 3000.302
#Phasecenter new:  04h19m42.506394s +27d13m35.81987s
#Phasecenter old:  04h19m42.50640s +027d13m35.819832s
#LB4
#04h19m42.503813s +27d13m35.79169s
#Peak of Gaussian component identified with imfit: J2000 04h19m42.503813s +27d13m35.79169s
#PA of Gaussian component: 43.22 deg
#Inclination of Gaussian component: 63.47 deg
#Pixel coordinates of peak: x = 3000.296 y = 2999.848
#Phasecenter new:  04h19m42.503813s +27d13m35.79169s
#Phasecenter old:  04h19m42.50382s +027d13m35.791711s

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
##DOES THE AMP VS. UV-DISTANCE LOOK VERY DISSIMILAR BETWEEN DIFFERENT EBS?
##IF SO, UNCOMMENT AND RUN THIS BLOCK TO REMOVE SCALING AND PROCEED WITH 2- METHOD PASS
#################### MANUALLY SET THIS ######################
refdata='LB1'
reference=prefix+'_'+refdata+'_initcont_shift.vis.npz'
for i in data_params.keys():
   print(i)
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

#data_params["LB1"]["gencal_scale"]=1.000
#data_params["LB2"]["gencal_scale"]=0.977
#data_params["LB3"]["gencal_scale"]=0.923
#data_params["LB4"]["gencal_scale"]=0.932

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
                     show_err=False,outfile='amp-vs-uv-distance-pre-selfcal-2.png')
   ### Make sure differences are no longer significant
   refdata='LB1'
   reference=prefix+'_'+refdata+'_initcont_shift.vis.npz'
   for i in data_params.keys():
      if i != refdata:
         estimate_flux_scale(reference=reference, 
                        comparison=prefix+'_'+i+'_initcont_shift_rescaled.vis.npz', 
                        incl=incl, PA=PA)
# 


###

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

### Save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
mask_maj = 0.95	# semimajor axis of mask in arcsec
mask_min = 0.95 	# semiminor axis of mask in arcsec

common_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
              (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)
""" Define a noise annulus, measure the peak SNR in map """
noise_annulus = "annulus[[%s, %s],['%.2farcsec', '8.0arcsec']]" % \
                (mask_ra, mask_dec, 2.0*mask_maj) 


###############################################################
###################### Self-CALIBRATION #######################
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

#setup the same number of iterations as suggested solints
#IRAS04166+2706_initial.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 71.52 mJy
#Peak intensity of source: 44.44 mJy/beam
#rms: 1.60e-01 mJy/beam
#Peak SNR: 278.33
#Suggested Solints:
#['inf', 'inf', '60.48s', '30.24s', '12.10s', 'int']
#Suggested Gaincal Combine params:
#['spw,scan', 'spw', 'spw', 'spw', 'spw', 'spw']
#Suggested nsigma per solint: 
#[18.55555767 12.88853217  8.95226457  6.21816665  4.31908555  3.        ]



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
               nsigma=19.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel,combine='spw,scan',smoothfactor=2.0)


### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")
#IRAS04166+2706_SB-only_p0.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 70.77 mJy
#Peak intensity of source: 44.32 mJy/beam
#rms: 1.71e-01 mJy/beam
#Peak SNR: 258.71
#IRAS04166+2706_SB-only_p0_post.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 71.13 mJy
#Peak intensity of source: 45.43 mJy/beam
#rms: 1.49e-01 mJy/beam
#Peak SNR: 304.70

iteration=1
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=13.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,combine='spw',parallel=parallel,smoothfactor=2.0)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")
#IRAS04166+2706_SB-only_p1.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 71.66 mJy
#Peak intensity of source: 45.39 mJy/beam
#rms: 1.40e-01 mJy/beam
#Peak SNR: 323.17
#IRAS04166+2706_SB-only_p1_post.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 72.02 mJy
#Peak intensity of source: 53.28 mJy/beam
#rms: 6.65e-02 mJy/beam
#Peak SNR: 801.75

iteration=2
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=9.0,solint='60s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,combine='spw',parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")
#IRAS04166+2706_SB-only_p2.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 72.33 mJy
#Peak intensity of source: 53.22 mJy/beam
#rms: 5.03e-02 mJy/beam
#Peak SNR: 1058.42
#IRAS04166+2706_SB-only_p2_post.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 72.64 mJy
#Peak intensity of source: 55.03 mJy/beam
#rms: 4.71e-02 mJy/beam
#Peak SNR: 1168.72

iteration=3
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=6.0,solint='30s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,combine='spw',parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")
#IRAS04166+2706_SB-only_p3.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 72.72 mJy
#Peak intensity of source: 55.01 mJy/beam
#rms: 4.68e-02 mJy/beam
#Peak SNR: 1174.91
#IRAS04166+2706_SB-only_p3_post.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 73.04 mJy
#Peak intensity of source: 56.51 mJy/beam
#rms: 4.61e-02 mJy/beam
#Peak SNR: 1226.04

iteration=4
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=4.0,solint='12s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,combine='spw',parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")
#IRAS04166+2706_SB-only_p4.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 73.12 mJy
#Peak intensity of source: 56.46 mJy/beam
#rms: 4.58e-02 mJy/beam
#Peak SNR: 1231.81
#IRAS04166+2706_SB-only_p4_post.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 73.53 mJy
#Peak intensity of source: 57.41 mJy/beam
#rms: 4.58e-02 mJy/beam
#Peak SNR: 1254.76

iteration=5
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='int',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,combine='spw',parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")
#IRAS04166+2706_SB-only_p4.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 73.12 mJy
#Peak intensity of source: 56.46 mJy/beam
#rms: 4.58e-02 mJy/beam
#Peak SNR: 1231.81
#IRAS04166+2706_SB-only_p4_post.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 73.53 mJy
#Peak intensity of source: 57.41 mJy/beam
#rms: 4.58e-02 mJy/beam
#Peak SNR: 1254.76


### Changing self-cal mode here to ap, see use of prevselfcalmode to ensure proper split

iteration=6
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',prevselfcalmode='p',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,combine='spw',parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")


### Make the final image, will not run another self-calibration
iteration=7
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,
               combine='spw',parallel=parallel,finalimageonly=True)
#IRAS04166+2706_SB-only_p7.image.tt0
#Beam 0.367 arcsec x 0.250 arcsec (-4.34 deg)
#Flux inside disk mask: 73.58 mJy
#Peak intensity of source: 57.39 mJy/beam
#rms: 4.56e-02 mJy/beam
#Peak SNR: 1258.45


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

#IRAS04166+2706_initial_LB+SB.image.tt0
#Beam 0.068 arcsec x 0.049 arcsec (29.22 deg)
#Flux inside disk mask: 86.98 mJy
#Peak intensity of source: 7.13 mJy/beam
#rms: 3.35e-02 mJy/beam
#Peak SNR: 213.05
#Suggested Solints:
#['inf', 'inf', '18.14s', 'int']
#Suggested Gaincal Combine params:
#['spw,scan', 'spw', 'spw', 'spw']
#Suggested nsigma per solint: 
#[14.20362946  8.45875879  5.03748711  3.        ]

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
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,combine='spw,scan',parallel=parallel,smoothfactor=2.0,threshold='0.001Jy')
#IRAS04166+2706_LB+SB_p0.image.tt0
#Beam 0.068 arcsec x 0.049 arcsec (29.22 deg)
#Flux inside disk mask: 113.60 mJy
#Peak intensity of source: 8.11 mJy/beam
#rms: 3.28e-02 mJy/beam
#Peak SNR: 247.25
#IRAS04166+2706_LB+SB_p0_post.image.tt0
#Beam 0.068 arcsec x 0.049 arcsec (29.22 deg)
#Flux inside disk mask: 113.70 mJy
#Peak intensity of source: 9.07 mJy/beam
#rms: 2.88e-02 mJy/beam
#Peak SNR: 315.15

### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

iteration=1
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=14.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,combine='spw',
               parallel=parallel,smoothfactor=2.0,threshold='0.001Jy')
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180])
       input("Press Enter key to advance to next MS/Caltable...")
#IRAS04166+2706_LB+SB_p1.image.tt0
#Beam 0.068 arcsec x 0.049 arcsec (29.22 deg)
#Flux inside disk mask: 107.61 mJy
#Peak intensity of source: 8.75 mJy/beam
#rms: 2.84e-02 mJy/beam
#Peak SNR: 308.08
#IRAS04166+2706_LB+SB_p1_post.image.tt0
#Beam 0.068 arcsec x 0.049 arcsec (29.22 deg)
#Flux inside disk mask: 107.10 mJy
#Peak intensity of source: 13.84 mJy/beam
#rms: 2.04e-02 mJy/beam
#Peak SNR: 679.45

iteration=2
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=14.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,combine='spw',
               parallel=parallel,smoothfactor=2.0)
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180])
       input("Press Enter key to advance to next MS/Caltable...")

iteration=3
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=9.0,solint='18.4s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,combine='spw',smoothfactor=2.0)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#IRAS04166+2706_LB+SB_p2.image.tt0
#Beam 0.068 arcsec x 0.049 arcsec (29.22 deg)
#Flux inside disk mask: 88.58 mJy
#Peak intensity of source: 12.92 mJy/beam
#rms: 1.54e-02 mJy/beam
#Peak SNR: 839.10

#IRAS04166+2706_LB+SB_p2_post.image.tt0
#Beam 0.068 arcsec x 0.049 arcsec (29.22 deg)
#Flux inside disk mask: 88.61 mJy
#Peak intensity of source: 13.73 mJy/beam
#rms: 1.51e-02 mJy/beam
#Peak SNR: 910.33


iteration=4
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=5.0,solint='18.4s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,combine='spw',smoothfactor=2.0)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

iteration=5
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='int',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,combine='spw')

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")


iteration=6
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',prevselfcalmode='p',nsigma=3.0,solint='600s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,
               LB_spwmap=LB_spwmap,parallel=parallel,combine='spw,scan',smoothfactor=2.0)
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")

iteration=7
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='300s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,
               LB_spwmap=LB_spwmap,parallel=parallel,combine='spw,scan',smoothfactor=2.0)
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")

iteration=8
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,
               LB_spwmap=LB_spwmap,parallel=parallel,combine='spw',smoothfactor=2.0)
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")

iteration=9
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,
               LB_spwmap=LB_spwmap,parallel=parallel,combine='spw',smoothfactor=2.0,finalimageonly=True)


###Backup gain table list for LB+SB runs
for i in data_params.keys():
   if 'SB' in i:
      data_params[i]['selfcal_spwmap']=data_params[i]['selfcal_spwmap_SB-only']+data_params[i]['selfcal_spwmap']
      data_params[i]['selfcal_tables']=data_params[i]['selfcal_tables_SB-only']+data_params[i]['selfcal_tables']

#save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# check per EB images to ensure selfcal made them look ok
for i in data_params.keys():
   print('Imaging MS: ',i)
   if 'LB' in i:
      tclean_wrapper(vis=data_params[i]['vis_avg_selfcal'], imagename=prefix+'_'+i+'_perEB_cont_iteration_'+str(iteration), sidelobethreshold=2.0,
            smoothfactor=1.5, scales=LB_scales, nsigma=5.0, robust=0.5, parallel=parallel,
            nterms=2)
   else:
      tclean_wrapper(vis=data_params[i]['vis_avg_selfcal'], imagename=prefix+'_'+i+'_perEB_cont_iteration_'+str(iteration), sidelobethreshold=2.5,
            smoothfactor=1.5, scales=LB_scales, nsigma=5.0, robust=0.5, parallel=parallel,nterms=2)

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
refdata='LB1'
reference=data_params[refdata]['vis_avg_shift_selfcal'].replace('.ms','.vis.npz')
for i in data_params.keys():
   print(i)
   if i != refdata:
      data_params[i]['gencal_scale_selfcal']=estimate_flux_scale(reference=reference,
                        comparison=data_params[i]['vis_avg_shift_selfcal'].replace('.ms','.vis.npz'),
                        incl=incl, PA=PA)
   else:
      data_params[i]['gencal_scale_selfcal']=1.0
   print(' ')
#LB1
#LB2
#The ratio of the fluxes of IRAS04166+2706_LB2_selfcal_cont_shift.vis.npz to IRAS04166+2706_LB1_selfcal_cont_shift.vis.npz is 0.95543
#The scaling factor for gencal is 0.977 for your comparison measurement
#The error on the weighted mean ratio is 4.593e-04, although it's likely that the weights in the measurement sets are off by some constant factor
 
#LB3
#The ratio of the fluxes of IRAS04166+2706_LB3_selfcal_cont_shift.vis.npz to IRAS04166+2706_LB1_selfcal_cont_shift.vis.npz is 0.85264
#The scaling factor for gencal is 0.923 for your comparison measurement
#The error on the weighted mean ratio is 5.779e-04, although it's likely that the weights in the measurement sets are off by some constant factor
 
#LB4
#The ratio of the fluxes of IRAS04166+2706_LB4_selfcal_cont_shift.vis.npz to IRAS04166+2706_LB1_selfcal_cont_shift.vis.npz is 0.86828
#The scaling factor for gencal is 0.932 for your comparison measurement
#The error on the weighted mean ratio is 5.129e-04, although it's likely that the weights in the measurement sets are off by some constant factor


gencal_scale={}

print('IF AND ONLY IF SCALING WAS NOT ALREADY SET IN SCRIPT')
print('COPY AND PLACE WHERE DESIGNATED FOR SCALING TOWARD TOP OF SCRIPT')
for i in data_params.keys():
   gencal_scale[i]=data_params[i]['gencal_scale_selfcal']
   print('data_params["{}"]["gencal_scale"]={:0.3f}'.format(i,data_params[i]['gencal_scale_selfcal']))

#WRITE OUT PICKLE FILE FOR SCALING IF MISSED
#if not os.path.exists('gencal_scale.pickle'):
#   with open('gencal_scale.pickle', 'wb') as handle:
#      pickle.dump(gencal_scale, handle, protocol=pickle.HIGHEST_PROTOCOL)
#2nd pass
#data_params["LB1"]["gencal_scale"]=1.000
#data_params["LB2"]["gencal_scale"]=1.007
#data_params["LB3"]["gencal_scale"]=0.995
#data_params["LB4"]["gencal_scale"]=1.005


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





