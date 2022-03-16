"""
eDisk data reduction script
This script was written for CASA 6.1.1/6.2
Originally derived from DSHARP reduction scripts


Datasets calibrated (in order of date observed):
SB1: 2019.A.00034.S (uid___A002_Xf38bf9_X50b.ms; Dec. 3rd, 2021)

SB2: 2019.A.00034.S (uid___A002_Xf4073d_Xc890.ms; Dec. 16th, 2021)
 
LB1: 2019.1.00261.L (2021/08/20)
     
LB2: 2019.1.00261.L (2021/08/20)

reducer: Y. Yamato
"""

### Import statements
#sys.path.append('/home/yamato/Application/CASA/analysis_scripts/')
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
parallel = True

### if True, can run script non-interactively if later parameters properly set
skip_plots = True

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

### Add additional dictionary entries if need, i.e., SB2, SB3, LB1, LB2, etc. for each execution
### Note that C18O and 13CO have different spws in the DDT vis LP os the spw ordering
### is different for data that were originally part of the DDT than the LP
### DDT 2019.A.00034.S SB data need 'spws': '25,31,29,27,33,35,37'
### LP  2019.1.00261.L SB data need 'spws': '25,27,29,31,33,35,37'
pl_data_params={'SB1': {'vis': SB_path+'uid___A002_Xf38bf9_X50b.ms',
                        'spws': '25,31,29,27,33,35,37',
                        'field': field,
                        'column': 'corrected'},
				'SB2': {'vis': SB_path+'uid___A002_Xf4073d_Xc890.ms',
                        'spws': '25,31,29,27,33,35,37',
                        'field': field,
                        'column': 'corrected'},
                'LB1': {'vis': LB_path+'uid___A002_Xef6d27_X1023.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field,
                        'column': 'corrected'},
                'LB2': {'vis': LB_path+'uid___A002_Xef6d27_X168c.ms',
                        'spws': '25,27,29,31,33,35,37',
                        'field': field,
                        'column': 'corrected'}
               }

### Dictionary defining necessary metadata for each execution
### SiO at 217.10498e9 excluded because of non-detection
### Only bother specifying simple species that are likely present in all datasets
### Hot corino lines (or others) will get taken care of by using the cont.dat
data_params = {'SB1': {'vis': WD_path+prefix+'_SB1.ms',
                       'name': 'SB1',
                       'field': field,
                       'line_spws': np.array([0, 1, 2, 3, 4, 6, 4, 4, 4, 4, 4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9, 220.39868420e9, 219.94944200e9, 219.56035410e9,
                                               217.82215e9, 230.538e9, 217.94005e9, 218.16044e9,
                                               217.2386e9, 218.22219200e9, 218.47563200e9]), #restfreqs
                       'line_names': ['H2CO', '13CO', 'SO', 'C18O',
                                      'c-C3H2', '12CO', 'c-C3H2', 'c-C3H2',
                                      'DCN', 'H2CO', 'H2CO'], #restfreqs
                       'flagrange': np.array([[-7.0,20.0], [-7.0,20.0], [-7.0,20.0], [-7.0,20.0],
                                              [-7.0,20.0], [-7.0,20.0], [-7.0,20.0], [-7.0,20.0],
                                              [-7.0,20.0], [-7.0,20.0], [-7.0,20.0]]),
                       'orig_spw_map': {25:0, 31:1, 29:2, 27:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws': np.array([0, 1, 2, 3, 4, 5, 6]),  #spws to use for continuum
                       'cont_avg_width': np.array([480, 480, 480, 480, 60, 60, 60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/12/03/03:00:00~2021/12/03/04:20:00',
                       'contdotdat' : 'SB/cont-L1489IRS.dat'
                      }, 
			   'SB2': {'vis': WD_path+prefix+'_SB2.ms',
                       'name': 'SB2',
                       'field': field,
                       'line_spws': np.array([0, 1, 2, 3, 4, 6, 4, 4, 4, 4, 4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9, 220.39868420e9, 219.94944200e9, 219.56035410e9,
                                               217.82215e9, 230.538e9, 217.94005e9, 218.16044e9,
                                               217.2386e9, 218.22219200e9, 218.47563200e9]), #restfreqs
                       'line_names': ['H2CO', '13CO', 'SO', 'C18O',
                                      'c-C3H2', '12CO', 'c-C3H2', 'c-C3H2',
                                      'DCN', 'H2CO', 'H2CO'], #restfreqs
                       'flagrange': np.array([[-7.0,20.0], [-7.0,20.0], [-7.0,20.0], [-7.0,20.0],
                                              [-7.0,20.0], [-7.0,20.0], [-7.0,20.0], [-7.0,20.0],
                                              [-7.0,20.0], [-7.0,20.0], [-7.0,20.0]]),
                       'orig_spw_map': {25:0, 31:1, 29:2, 27:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws': np.array([0, 1, 2, 3, 4, 5, 6]),  #spws to use for continuum
                       'cont_avg_width': np.array([480, 480, 480, 480, 60, 60, 60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/12/16/03:30:00~2021/12/16/04:50:00',
                       'contdotdat' : 'SB/cont-L1489IRS.dat'
                      },
               'LB1': {'vis': WD_path+prefix+'_LB1.ms',
                       'name': 'LB1',
                       'field': field,
                       'line_spws': np.array([0, 1, 2, 3, 4, 6, 4, 4, 4, 4, 4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9, 220.39868420e9, 219.94944200e9, 219.56035410e9,
                                               217.82215e9, 230.538e9, 217.94005e9, 218.16044e9,
                                               217.2386e9, 218.22219200e9, 218.47563200e9]), #restfreqs
                       'line_names': ['H2CO', '13CO', 'SO', 'C18O',
                                      'c-C3H2', '12CO', 'c-C3H2', 'c-C3H2',
                                      'DCN', 'H2CO', 'H2CO'], #restfreqs
                       'flagrange': np.array([[-7.0,20.0], [-7.0,20.0], [-7.0,20.0], [-7.0,20.0],
                                              [-7.0,20.0], [-7.0,20.0], [-7.0,20.0], [-7.0,20.0],
                                              [-7.0,20.0], [-7.0,20.0], [-7.0,20.0]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0, 1, 2, 3, 4, 5, 6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480, 480, 480, 480, 60, 60, 60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/08/20/07:44:00~2021/08/20/09:16:00',
                       'contdotdat': 'LB/cont.dat'
                      }, 
               'LB2': {'vis': WD_path+prefix+'_LB2.ms',
                       'name': 'LB2',
                       'field': field,
                       'line_spws': np.array([0, 1, 2, 3, 4, 6, 4, 4, 4, 4, 4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9, 220.39868420e9, 219.94944200e9, 219.56035410e9,
                                               217.82215e9, 230.538e9, 217.94005e9, 218.16044e9,
                                               217.2386e9, 218.22219200e9,218.47563200e9]), #restfreqs
                       'line_names': ['H2CO', '13CO', 'SO', 'C18O',
                                      'c-C3H2', '12CO', 'c-C3H2', 'c-C3H2',
                                      'DCN', 'H2CO', 'H2CO'], #restfreqs
                       'flagrange': np.array([[-7.0,20.0], [-7.0,20.0], [-7.0,20.0], [-7.0,20.0],
                                              [-7.0,20.0], [-7.0,20.0], [-7.0,20.0], [-7.0,20.0],
                                              [-7.0,20.0], [-7.0,20.0], [-7.0,20.0]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws': np.array([0, 1, 2, 3, 4, 5, 6]),  #spws to use for continuum
                       'cont_avg_width': np.array([480, 480, 480, 480, 60, 60, 60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/08/20/09:30:00~2021/08/20/11:00:00',
                       'contdotdat': 'LB/cont.dat'
                      }, 
               }


### Flag range corresponds to velocity range in each spw that should be flagged. 
### Velocity range should correspond to 
### approximate width of the line contamination

#save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
#load data params from the pickle
# with open(prefix+'.pickle', 'rb') as handle:
#    data_params = pickle.load(handle)



###############################################################
#################### DATA PREPARATION #########################
###############################################################

### split out each pipeline-calibrated dataset into an MS only containing the target data 
for i in pl_data_params.keys():
  if os.path.exists(prefix+'_'+i+'.ms'):
    flagmanager(vis=prefix+'_'+i+'.ms', mode="restore",
                versionname="starting_flags")
  else:
    split(vis=pl_data_params[i]['vis'], outputvis=prefix+'_'+i+'.ms', 
          spw=pl_data_params[i]['spws'], field=pl_data_params[i]['field'],
          datacolumn=pl_data_params[i]['column'])

### Backup the the flagging state at start of reduction
for i in data_params.keys():
  if not os.path.exists(data_params[i]['vis']+".flagversions/flags.starting_flags"):
    flagmanager(vis=data_params[i]['vis'], mode='save', 
                versionname='starting_flags', 
                comment='Flag states at start of reduction')

### Inspect data in each spw for each dataset
#### OPTIONAL #####
if not skip_plots:
  for i in data_params.keys():
    plotms(vis=data_params[i]['vis'], xaxis='frequency', yaxis='amplitude', 
           field=data_params[i]['field'], ydatacolumn='data', 
           avgtime='1e8', avgscan=True, avgbaseline=True, iteraxis='spw',
           transform=True, freqframe='LSRK')
    input("Press Enter key to advance to next MS/Caltable...")
#### END OPTIONAL ###

### Flag spectral regions around lines and do spectral averaging to make a smaller continuum MS 
for i in data_params.keys():      
    flagchannels_string = get_flagchannels(data_params[i], prefix)
    s = ' '  # work around for Python 3 port of following string generating for loops
    print(i) 
    avg_cont(data_params[i], prefix, flagchannels=flagchannels_string, 
             contspws=s.join(str(elem) for elem in data_params[i]['cont_spws'].tolist()).replace(' ',','),
             width_array=data_params[i]['cont_avg_width'])
    data_params[i]['vis_avg'] = prefix+'_'+i+'_initcont.ms'
      


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
    data_params[i]['common_dir'] = 'J2000 04h04m43.080s 26d18m56.104s'

### save updated data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



###############################################################
#################### SHIFT PHASE CENTERS ######################
###############################################################
for i in data_params.keys():
   print(i)
   data_params[i]['vis_avg_shift'] = prefix+'_'+i+'_initcont_shift.ms'
   os.system('rm -rf '+data_params[i]['vis_avg_shift'])
   fixvis(vis=data_params[i]['vis_avg'],
          outputvis=data_params[i]['vis_avg_shift'], 
          field=data_params[i]['field'], 
          phasecenter='J2000 '+data_params[i]['phasecenter'])
   ### fix planets may throw an error, usually safe to ignore
   fixplanets(vis=data_params[i]['vis_avg_shift'],
              field=data_params[i]['field'], 
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
export_vislist = []
for i in data_params.keys():
    export_MS(data_params[i]['vis_avg_shift'])
    export_vislist.append(data_params[i]['vis_avg_shift'].replace('.ms', '.vis.npz'))

if not skip_plots:
    ### Plot deprojected visibility profiles for all data together """
    plot_deprojected(export_vislist, PA=PA, incl=incl, show_err=False,
                     fluxscale=[1.0]*len(export_vislist),outfile='amp-vs-uv-distance-pre-selfcal.png')

### Now inspect offsets by comparing against a reference 
### Set reference data using the dictionary key.
### Using SB1 as reference because it looks the nicest by far

#################### MANUALLY SET THIS ######################
refdata = 'SB1'

reference = prefix+'_'+refdata+'_initcont_shift.vis.npz'
for i in data_params.keys():
  print(i)
  if i != refdata:
    data_params[i]['gencal_scale'] = \
        estimate_flux_scale(reference=reference, incl=incl, PA=PA,
                            comparison=prefix+'_'+i+'_initcont_shift.vis.npz')
  else:
    data_params[i]['gencal_scale'] = 1.0
  print(' ')

#No rescaling here since just one dataset
#Go ahead with rescaling anyway to keep the flow of the script



###############################################################
############### SCALE DATA RELATIVE TO ONE EB #################
###############################################################

os.system('rm -rf *_rescaled.ms')
for i in data_params.keys():
    rescale_flux(data_params[i]['vis_avg_shift'],
                 [data_params[i]['gencal_scale']])
    rescale_flux(data_params[i]['vis_avg'],
                 [data_params[i]['gencal_scale']])
    data_params[i]['vis_avg_shift_rescaled'] \
        = data_params[i]['vis_avg_shift'].replace('.ms','_rescaled.ms')
    data_params[i]['vis_avg_rescaled'] \
        = data_params[i]['vis_avg'].replace('.ms','_rescaled.ms')



###############################################################
############## PLOT UV DATA TO CHECK RE-SCALING ###############
###############################################################

if not skip_plots:
  ### Assign rough emission geometry parameters; keep 0, 0
  PA, incl = 0, 0

  ### Check that rescaling did what we expect
  export_vislist_rescaled = []
  for i in data_params.keys():
    export_MS(data_params[i]['vis_avg_shift_rescaled'])
    export_vislist_rescaled.append(data_params[i]['vis_avg_shift_rescaled'].replace('.ms','.vis.npz'))

  plot_deprojected(export_vislist_rescaled, PA=PA, incl=incl, show_err=False,
                   fluxscale=[1.0]*len(export_vislist_rescaled))
  ### Make sure differences are no longer significant
  refdata = 'SB1'
  reference = prefix+'_'+refdata+'_initcont_shift.vis.npz'
  for i in data_params.keys():
    if i != refdata:
      estimate_flux_scale(reference=reference, incl=incl, PA=PA,               
                          comparison=prefix+'_'+i+'_initcont_shift_rescaled.vis.npz')

### Save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



###############################################################
################ SELF-CALIBRATION PREPARATION #################
###############################################################
selectedVis = 'vis_avg_rescaled'
#selectedVis='vis_avg_shift_rescaled'  # because the peak moved ~0.17" from SB to LB.

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
#SB_refant = 'DA43@A035,DV07@A011,DV05@A042' 

############### CHECK THESE, SHOULD BE FINE #################
SB_spwmap = [0, 0, 0, 0, 0, 0, 0]
SB_contspws = '' 

### Make a list of EBs to image
vislist = []
for i in data_params.keys():
  if ('LB' in i): # skip over LB EBs if in SB-only mode
    continue
  vislist.append(data_params[i][selectedVis])

""" Set up a clean mask """
mask_ra  = data_params[i]['common_dir'].split()[1].replace('h',':').replace('m',':').replace('s','')
mask_dec = data_params[i]['common_dir'].split()[2].replace('d','.').replace('m','.').replace('s','')
mask_pa  = 90.0 # position angle of mask in degrees
mask_maj = 4.0	# semimajor axis of mask in arcsec
mask_min = 4.0 	# semiminor axis of mask in arcsec

common_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]'\
              % (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)
""" Define a noise annulus, measure the peak SNR in map """
noise_annulus = "annulus[[%s, %s],['%.2farcsec', '10.0arcsec']]"\
                % (mask_ra, mask_dec, 2.0*mask_maj) 



###############################################################
###################### SELF-CALIBRATION #######################
###############################################################
### Initial dirty map to assess DR
tclean_wrapper(vis=vislist, imagename=prefix+'_initial', 
               scales=SB_scales, sidelobethreshold=2.0, smoothfactor=1.5, nsigma=3.0, 
               noisethreshold=3.0, robust=0.5, parallel=parallel, imsize=1600,cellsize='0.025arcsec',nterms=2,
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

#SUGGESTIONS APPLY TO THE PHASE-ONLY SELFCAL ITERATIONS
#AMPLITUDE SELFCAL ITERATIONS SHOULD ALWAYS USE NSIGMA=3 AND SOLINT = INF TO START


# nterms = 1
#L1489IRS_initial.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 55.49 mJy
#Peak intensity of source: 4.67 mJy/beam
#rms: 4.06e-02 mJy/beam
#Peak SNR: 114.95

# nterms = 2
#L1489IRS_initial.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 56.25 mJy
#Peak intensity of source: 4.66 mJy/beam
#rms: 4.03e-02 mJy/beam
#Peak SNR: 115.61

# Suggested Solints:
# ['inf', 'inf', '60.48s', '30.24s', '12.10s', 'int']
# Suggested Gaincal Combine params:
# ['spw,scan', 'spw', 'spw', 'spw', 'spw', 'spw']
# Suggested nsigma per solint:
# [7.70755405 6.3820105  5.28443365 4.37561784 3.62309999 3.        ]


### Image produced by iter 0 has not selfcal applied, it's used to set the initial model
### only images >0 have self-calibration applied

### Run self-calibration command set
### 0. Split off corrected data from previous selfcal iteration (except iteration 0)
### 1. Image data to specified nsigma depth, set model column
### 2. Calculate self-cal gain solutions
### 3. Apply self-cal gain solutions to MS

############# USERS MAY NEED TO ADJUST NSIGMA AND SOLINT FOR EACH SELF-CALIBRATION ITERATION ##############
### nterms = 1
iteration = 0
self_calibrate(prefix, data_params, selectedVis, mode='SB-only',
               iteration=iteration, selfcalmode='p', nsigma=7.66, solint='inf',
               noisemasks=[common_mask, noise_annulus],
               SB_contspws=SB_contspws, SB_spwmap=SB_spwmap,combine='scan,spw',
               parallel=parallel, nterms=1)

### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms', '_SB-only_p'+str(iteration)+'.g'),
              xaxis='time', yaxis='phase', gridrows=4, gridcols=1,
              iteraxis='antenna', xselfscale=True, plotrange=[0, 0, -180, 180]) 
       input("Press Enter key to advance to next MS/Caltable...")

### Make note of key metrics of image in each round
### nterms=1
#L1489IRS_SB-only_p0.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 50.11 mJy
#Peak intensity of source: 4.71 mJy/beam
#rms: 4.20e-02 mJy/beam
#Peak SNR: 112.03

#L1489IRS_SB-only_p0_post.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 52.65 mJy
#Peak intensity of source: 5.11 mJy/beam
#rms: 3.84e-02 mJy/beam
#Peak SNR: 132.96

### nterns=1
iteration = 1
self_calibrate(prefix, data_params, selectedVis, mode='SB-only',
               iteration=iteration, selfcalmode='p', nsigma=6.35, solint='inf',
               noisemasks=[common_mask, noise_annulus],
               SB_contspws=SB_contspws, SB_spwmap=SB_spwmap, combine='spw',
               parallel=parallel, nterms=1)

### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
     if 'SB' in i:
       plotms(vis=data_params[i][selectedVis].replace('.ms', '_SB-only_p'+str(iteration)+'.g'),
              xaxis='time', yaxis='phase', gridrows=4, gridcols=1,
              iteraxis='antenna', xselfscale=True, plotrange=[0, 0, -180, 180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#L1489IRS_SB-only_p1.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 54.20 mJy
#Peak intensity of source: 5.08 mJy/beam
#rms: 3.78e-02 mJy/beam
#Peak SNR: 134.50

#L1489IRS_SB-only_p1_post.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 56.42 mJy
#Peak intensity of source: 6.00 mJy/beam
#rms: 3.60e-02 mJy/beam
#Peak SNR: 166.50


# self-cal is stopped here because the peak intensity and the peak S/N decreased after iteration = 1 as recorded below.
iteration = 2
self_calibrate(prefix, data_params, selectedVis, mode='SB-only',
               iteration=iteration, selfcalmode='p', nsigma=5.27, solint='60.48s',
               noisemasks=[common_mask, noise_annulus],
               SB_contspws=SB_contspws, SB_spwmap=SB_spwmap, combine='spw',
               parallel=parallel, nterms=1)

#L1489IRS_SB-only_p2.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 58.08 mJy
#Peak intensity of source: 5.98 mJy/beam
#rms: 3.51e-02 mJy/beam
#Peak SNR: 170.24

#L1489IRS_SB-only_p2_post.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 58.14 mJy
#Peak intensity of source: 6.21 mJy/beam
#rms: 3.52e-02 mJy/beam
#Peak SNR: 176.36

### netrms = 1
iteration = 3
self_calibrate(prefix, data_params, selectedVis, mode='SB-only',
               iteration=iteration, selfcalmode='p', nsigma=4.37, solint='30.24s',
               noisemasks=[common_mask, noise_annulus],
               SB_contspws=SB_contspws, SB_spwmap=SB_spwmap, combine='spw',
               parallel=parallel, nterms=1)

#L1489IRS_SB-only_p3.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 59.28 mJy
#Peak intensity of source: 6.19 mJy/beam
#rms: 3.48e-02 mJy/beam
#Peak SNR: 177.80

#L1489IRS_SB-only_p3_post.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 59.33 mJy
#Peak intensity of source: 6.32 mJy/beam
#rms: 3.49e-02 mJy/beam
#Peak SNR: 181.11


### netrms = 1
iteration=4
self_calibrate(prefix, data_params, selectedVis, mode='SB-only',
               iteration=iteration, selfcalmode='p', nsigma=3.62, solint='12s',
               noisemasks=[common_mask, noise_annulus],
               SB_contspws=SB_contspws, SB_spwmap=SB_spwmap, combine='spw',
               parallel=parallel, nterms=1)

#L1489IRS_SB-only_p4.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 60.43 mJy
#Peak intensity of source: 6.32 mJy/beam
#rms: 3.45e-02 mJy/beam
#Peak SNR: 182.99

#L1489IRS_SB-only_p4_post.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 60.54 mJy
#Peak intensity of source: 6.37 mJy/beam
#rms: 3.45e-02 mJy/beam
#Peak SNR: 184.65



iteration=5
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',prevselfcalmode='p',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel, nterms=1)

#L1489IRS_SB-only_p5.image.tt0
#Beam 0.275 arcsec x 0.181 arcsec (-1.31 deg)
#Flux inside disk mask: 62.30 mJy
#Peak intensity of source: 6.34 mJy/beam
#rms: 3.40e-02 mJy/beam
#Peak SNR: 186.51

#L1489IRS_SB-only_p5_post.image.tt0
#Beam 0.282 arcsec x 0.187 arcsec (-1.69 deg)
#Flux inside disk mask: 51.73 mJy
#Peak intensity of source: 6.51 mJy/beam
#rms: 3.34e-02 mJy/beam
#Peak SNR: 194.77

iteration=6
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='30s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel, nterms=1)

#L1489IRS_SB-only_ap6.image.tt0
#Beam 0.283 arcsec x 0.188 arcsec (-1.95 deg)
#Flux inside disk mask: 49.97 mJy
#Peak intensity of source: 6.62 mJy/beam
#rms: 3.39e-02 mJy/beam
#Peak SNR: 195.65

#L1489IRS_SB-only_ap6_post.image.tt0
#Beam 0.281 arcsec x 0.186 arcsec (-1.80 deg)
#Flux inside disk mask: 50.67 mJy
#Peak intensity of source: 6.81 mJy/beam
#rms: 3.47e-02 mJy/beam
#Peak SNR: 196.49


# no improvement of SNR so stop at iteration 6
iteration=7
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='12s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel,nterms=1, finalimageonly=True)

#L1489IRS_SB-only_ap7.image.tt0
#Beam 0.281 arcsec x 0.186 arcsec (-1.80 deg)
#Flux inside disk mask: 50.76 mJy
#Peak intensity of source: 6.80 mJy/beam
#rms: 3.46e-02 mJy/beam
#Peak SNR: 196.37

### nterms=2
#L1489IRS_SB-only_ap7.image.tt0
#Beam 0.279 arcsec x 0.185 arcsec (-1.74 deg)
#Flux inside disk mask: 52.01 mJy
#Peak intensity of source: 6.77 mJy/beam
#rms: 3.44e-02 mJy/beam
#Peak SNR: 197.09

for i in data_params.keys():
  if 'SB' in i:
    data_params[i]['selfcal_spwmap_SB-only'] \
        = data_params[i]['selfcal_spwmap'].copy()
    data_params[i]['selfcal_tables_SB-only'] \
        = data_params[i]['selfcal_tables'].copy()
    data_params[i]['vis_avg_selfcal_SB-only'] \
        = (data_params[i]['vis_avg_selfcal']+'.')[:-1]  ## trick to copy the string

### Save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



###############################################################
################### SELF-CALIBRATION SB+LB ####################
###############################################################
LB_spwmap = [0, 0, 0, 0, 0, 0, 0]
LB_contspws = '' 

### Make a list of EBs to image

### Make a list of EBs to image
vislist=[]
fieldlist=[]
for i in data_params.keys():
   vislist.append(data_params[i][selectedVis])
   fieldlist.append(data_params[i]['field'])

### Initial dirty map to assess DR
tclean_wrapper(vis=vislist, imagename=prefix+'_initial_LB+SB',cellsize='0.003arcsec',imsize=6000, 
               scales=LB_scales, sidelobethreshold=2.0, smoothfactor=1.5, nsigma=3.0, 
               noisethreshold=3.0, robust=0.5, parallel=parallel, nterms=2,
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



#L1489IRS_initial_LB+SB.image.tt0
#Beam 0.070 arcsec x 0.040 arcsec (24.62 deg)
#Flux inside disk mask: 119.88 mJy
#Peak intensity of source: 3.40 mJy/beam
#rms: 1.66e-02 mJy/beam
#Peak SNR: 205.35

### nterms=2
#L1489IRS_initial_LB+SB.image.tt0
#Beam 0.070 arcsec x 0.040 arcsec (24.62 deg)
#Flux inside disk mask: 126.86 mJy
#Peak intensity of source: 3.43 mJy/beam
#rms: 1.72e-02 mJy/beam
#Peak SNR: 199.10

# Suggested Solints:

# ['inf', 'inf', '18.14s', 'int']

# Suggested Gaincal Combine params:

# ['spw,scan', 'spw', 'spw', 'spw']

# Suggested nsigma per solint:

# [13.27327118  8.08523033  4.9250067   3.        ]



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

iteration = 0
self_calibrate(prefix, data_params, selectedVis, mode='LB+SB',
               iteration=iteration, selfcalmode='p', nsigma=13.69, solint='inf',
               noisemasks=[common_mask, noise_annulus],
               SB_contspws=SB_contspws, SB_spwmap=SB_spwmap,
               LB_contspws=LB_contspws, LB_spwmap=LB_spwmap,
               combine='spw,scan', parallel=parallel, smoothfactor=2.0, nterms=1, sidelobethreshold=2.0)#, smoothfactor=1.5,
               #noisethreshold=3.0,
               #nterms=1)

### Plot gain corrections, loop through each
if not skip_plots:
  for i in data_params.keys():
    plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
           xaxis='time', yaxis='phase', gridrows=4, gridcols=1,
           iteraxis='antenna', xselfscale=True, plotrange=[0, 0, -180, 180]) 
    input("Press Enter key to advance to next MS/Caltable...")

#L1489IRS_LB+SB_p0.image.tt0
#Beam 0.071 arcsec x 0.040 arcsec (24.56 deg)
#Flux inside disk mask: 129.52 mJy
#Peak intensity of source: 3.60 mJy/beam
#rms: 1.68e-02 mJy/beam
#Peak SNR: 214.92

#L1489IRS_LB+SB_p0_post.image.tt0
#Beam 0.071 arcsec x 0.040 arcsec (24.56 deg)
#Flux inside disk mask: 129.35 mJy
#Peak intensity of source: 4.08 mJy/beam
#rms: 1.65e-02 mJy/beam
#Peak SNR: 247.37





iteration = 1
self_calibrate(prefix, data_params, selectedVis, mode='LB+SB',
               iteration=iteration, selfcalmode='p', nsigma=8.25, solint='inf',
               noisemasks=[common_mask, noise_annulus],
               SB_contspws=SB_contspws, SB_spwmap=SB_spwmap,
               LB_contspws=LB_contspws, LB_spwmap=LB_spwmap,
               combine='spw', parallel=parallel, smoothfactor=2.0, nterms=1, sidelobethreshold=2.0)


if not skip_plots:
  for i in data_params.keys():
    plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
           xaxis='time', yaxis='phase', gridrows=4, gridcols=1,
           iteraxis='antenna', xselfscale=True, plotrange=[0, 0, -180, 180]) 
    input("Press Enter key to advance to next MS/Caltable...")


#L1489IRS_LB+SB_p1.image.tt0
#Beam 0.071 arcsec x 0.040 arcsec (24.56 deg)
#Flux inside disk mask: 125.29 mJy
#Peak intensity of source: 4.06 mJy/beam
#rms: 1.64e-02 mJy/beam
#Peak SNR: 248.38

#L1489IRS_LB+SB_p1_post.image.tt0
#Beam 0.071 arcsec x 0.040 arcsec (24.56 deg)
#Flux inside disk mask: 125.22 mJy
#Peak intensity of source: 4.82 mJy/beam
#rms: 1.62e-02 mJy/beam
#Peak SNR: 297.12


### no improvement, stopped at iteration 2 ###

iteration = 2
self_calibrate(prefix, data_params, selectedVis, mode='LB+SB',
               iteration=iteration, selfcalmode='p', nsigma=4.98, solint='18.14s',
               noisemasks=[common_mask, noise_annulus],
               SB_contspws=SB_contspws, SB_spwmap=SB_spwmap,
               LB_contspws=LB_contspws, LB_spwmap=LB_spwmap,
               combine='spw', parallel=parallel, sidelobethreshold=2.0, smoothfactor=2.0,
               nterms=1, finalimageonly=True)

#L1489IRS_LB+SB_p2.image.tt0
#Beam 0.071 arcsec x 0.040 arcsec (24.56 deg)
#Flux inside disk mask: 116.48 mJy
#Peak intensity of source: 4.84 mJy/beam
#rms: 1.60e-02 mJy/beam
#Peak SNR: 301.71




###Backup gain table list for LB+SB runs
for i in data_params.keys():
  if 'SB' in i:
    data_params[i]['selfcal_spwmap'] \
        = data_params[i]['selfcal_spwmap_SB-only'] \
        + data_params[i]['selfcal_spwmap']
    data_params[i]['selfcal_tables'] \
        = data_params[i]['selfcal_tables_SB-only'] \
        + data_params[i]['selfcal_tables']

#save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



###############################################################
################# SPLIT OFF FINAL CONT DATA ###################
###############################################################
for i in data_params.keys():
    os.system('rm -rf '+prefix+'_'+i+'_continuum.ms '
              +prefix+'_'+i+'_continuum.ms.tgz')
    split(vis=data_params[i]['vis_avg_selfcal'],
          outputvis=prefix+'_'+i+'_continuum.ms', datacolumn='data')
    data_params[i]['vis_final'] = prefix+'_'+i+'_continuum.ms'
    os.system('tar cvzf '+prefix+'_'+i+'_continuum.ms.tgz '
              +prefix+'_'+i+'_continuum.ms')

#save data params to a pickle
with open(prefix+'.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol=pickle.HIGHEST_PROTOCOL)



###############################################################
################## RUN A FINAL IMAGE SET ######################
###############################################################

### Generate a vislist
vislist = []
for i in data_params.keys():
    vislist.append(data_params[i]['vis_final'])
scales = SB_scales
imsize = 6000
cell = '0.003arcsec'
# robust=2.0 needs lower sidelobethreshold
for robust in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
    print('Generate Robust '+str(robust)+' no taper image')
    imagename = prefix+'_SBLB_continuum_robust_'+str(robust)
    os.system('rm -rf '+imagename+'*')
    sigma = get_sensitivity(data_params, specmode='mfs',
                            imsize=imsize, robust=robust, cellsize=cell)
    if robust == 2.0:
       sidelobethreshold = 1.0 
    else:
       sidelobethreshold = 2.0
    tclean_wrapper(vis=vislist, imagename=imagename,
                   sidelobethreshold=sidelobethreshold, 
                   smoothfactor=1.5, scales=scales, threshold=2.0*sigma, 
                   noisethreshold=3.0, robust=robust, parallel=parallel, 
                   cellsize=cell, imsize=imsize, nterms=1,
                   phasecenter=data_params['SB1']['common_dir'].replace('J2000', 'ICRS'))
    imagename = imagename+'.image.tt0'
    exportfits(imagename=imagename, fitsimage=imagename+'.fits',
               overwrite=True, dropdeg=True)

for taper in ['1000klambda', '2000klambda', '3000klambda']:
  for robust in [1.0, 2.0]:
    print('Generate Robust '+str(robust)+' taper '+taper+' image')
    imagename = prefix+'_SBLB_continuum_robust_'+str(robust)+'_taper_'+taper
    os.system('rm -rf '+imagename+'*')
    sigma = get_sensitivity(data_params, specmode='mfs',
                            imsize=imsize, robust=robust, cellsize=cell)
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
                   robust=0.5, parallel=parallel,cellsize=cell, imsize=imsize, nterms=1,
                   phasecenter=data_params['LB1']['common_dir'])
   pblist=glob.glob('*continuum*.pb.tt0') 
   os.system('mkdir orig_pbimages')
   for pbimage in pblist:
      os.system('mv '+pbimage+' orig_pbimages/')
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



