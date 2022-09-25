"""
eDisk data reduction script
This script was written for CASA 6.1.1/6.2
Originally derived from DSHARP reduction scripts

Datasets calibrated (in order of date observed):
SB1: 2019.1.00261.L (2022/07/03)
SB2: 2019.1.00261.L (2022/07/03)
LB1: 2019.1.00261.L (2021/10/24)
LB2: 2019.1.00261.L (2021/09/30)
LB3: 2019.1.00261.L (2021/10/01)
LB4: 2019.1.00261.L (2021/10/18)

reducer: Ilseung Han
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
parallel = True

### if True, can run script non-interactively if later parameters properly set
skip_plots = True

### Add field names (corresponding to the field in the MS) here and prefix for 
### filenameing (can be different but try to keep same)
### Only make different if, for example, the field name has a space
field   = {'SB':'IRAS04169+2702', 'LB':'IRAS04169+2702'}
prefix  = 'IRAS04169' 

### always include trailing slashes!!
WD_path = '/lustre/cv/projects/edisk/IRAS04169/'
SB_path = WD_path + 'SB/'
LB_path = WD_path + 'LB/'

### scales for multi-scale clean
SB_scales = [0, 5]      # [0, 5, 10, 20]
LB_scales = [0, 5, 30]  # [0, 5, 30, 100, 200]

### Add additional dictionary entries if need, i.e., SB2, SB3, LB1, LB2, etc. for each execution
### Note that C18O and 13CO have different spws in the DDT vis LP os the spw ordering
### is different for data that were originally part of the DDT than the LP
### DDT 2019.A.00034.S SB data need 'spws': '25,31,29,27,33,35,37'
### LP  2019.1.00261.L SB data need 'spws': '25,27,29,31,33,35,37'
pl_data_params = {'SB1': {'vis': SB_path + 'uid___A002_Xfafcc0_X6944.ms',
                          'spws': '25, 27, 29, 31, 33, 35, 37',
                          'field': field['SB'],
                          'column': 'corrected'},
                  'SB2': {'vis': SB_path + 'uid___A002_Xfafcc0_X6f3d.ms',
                          'spws': '25, 27, 29, 31, 33, 35, 37',
                          'field': field['SB'],
                          'column': 'corrected'},
                  'LB1': {'vis': LB_path + 'uid___A002_Xf20692_X27ec.ms',
                          'spws': '25, 27, 29, 31, 33, 35, 37',
                          'field': field['LB'],
                          'column': 'corrected'},
                  'LB2': {'vis': LB_path + 'uid___A002_Xf138ff_X23de.ms',
                          'spws': '25, 27, 29, 31, 33, 35, 37',
                          'field': field['LB'],
                          'column': 'corrected'},
                  'LB3': {'vis': LB_path + 'uid___A002_Xf1479a_X141e.ms',
                          'spws': '25, 27, 29, 31, 33, 35, 37',
                          'field': field['LB'],
                          'column': 'corrected'},
                  'LB4': {'vis': LB_path + 'uid___A002_Xf1bb4a_Xb41c.ms',
                          'spws': '25, 27, 29, 31, 33, 35, 37',
                          'field': field['LB'],
                          'column': 'corrected'},
                 }

### Dictionary defining necessary metadata for each execution
### SiO at 217.10498e9 excluded because of non-detection
### Only bother specifying simple species that are likely present in all datasets
### Hot corino lines (or others) will get taken care of by using the cont.dat
data_params = {'SB1': {'vis' : WD_path + prefix + '_SB1.ms',
                       'name' : 'SB1',
                       'field': field['SB'],
                       'line_spws': np.array([0, 1, 2, 3, 4, 6, 4, 4, 4, 4, 4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9, 220.39868420e9, 219.94944200e9, 219.56035410e9,
                                               217.82215e9, 230.538e9, 217.94005e9, 218.16044e9,
                                               217.2386e9, 218.22219200e9, 218.47563200e9]), # restfreqs
                       'line_names': ['H2CO', '13CO', 'SO', 'C18O', 'c-C3H2', '12CO', 'c-C3H2', 'c-C3H2', 'DCN', 'H2CO', 'H2CO'], # restfreqs
                       'flagrange': np.array([[-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5],
                                              [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5],
                                              [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6}, # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws': np.array([0, 1, 2, 3, 4, 5, 6]), # spws to use for continuum
                       'cont_avg_width': np.array([480, 480, 480, 480, 60, 60, 60]), # n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2022/07/03/13:29:00~2022/07/03/14:26:00',
                       'contdotdat': 'SB/cont-IRAS04169.dat'
                      },
               'SB2': {'vis' : WD_path + prefix + '_SB2.ms',
                       'name' : 'SB2',
                       'field': field['SB'],
                       'line_spws': np.array([0, 1, 2, 3, 4, 6, 4, 4, 4, 4, 4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9, 220.39868420e9, 219.94944200e9, 219.56035410e9,
                                               217.82215e9, 230.538e9, 217.94005e9, 218.16044e9,
                                               217.2386e9, 218.22219200e9, 218.47563200e9]), # restfreqs
                       'line_names': ['H2CO', '13CO', 'SO', 'C18O', 'c-C3H2', '12CO', 'c-C3H2', 'c-C3H2', 'DCN', 'H2CO', 'H2CO'], # restfreqs
                       'flagrange': np.array([[-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5],
                                              [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5],
                                              [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6}, # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws': np.array([0, 1, 2, 3, 4, 5, 6]), # spws to use for continuum
                       'cont_avg_width': np.array([480, 480, 480, 480, 60, 60, 60]), # n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2022/07/03/14:39:00~2022/07/03/15:37:00',
                       'contdotdat': 'SB/cont-IRAS04169.dat'
                      }, 
               'LB1': {'vis': WD_path + prefix + '_LB1.ms',
                       'name': 'LB1',
                       'field': field['LB'],
                       'line_spws': np.array([0, 1, 2, 3, 4, 6, 4, 4, 4, 4, 4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9, 220.39868420e9, 219.94944200e9, 219.56035410e9,
                                               217.82215e9, 230.538e9, 217.94005e9, 218.16044e9,
                                               217.2386e9, 218.22219200e9, 218.47563200e9]), # restfreqs
                       'line_names': ['H2CO', '13CO', 'SO', 'C18O', 'c-C3H2', '12CO', 'c-C3H2', 'c-C3H2', 'DCN', 'H2CO', 'H2CO'], # restfreqs
                       'flagrange': np.array([[-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5],
                                              [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5],
                                              [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6}, # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws': np.array([0, 1, 2, 3, 4, 5, 6]), # spws to use for continuum
                       'cont_avg_width': np.array([480, 480, 480, 480, 60, 60, 60]), # n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/10/24/07:30:00~2021/10/24/08:50:00',
                       'contdotdat': 'LB/cont-IRAS04169.dat'
                      },
                'LB2': {'vis': WD_path + prefix + '_LB2.ms',
                       'name': 'LB2',
                       'field': field['LB'],
                       'line_spws': np.array([0, 1, 2, 3, 4, 6, 4, 4, 4, 4, 4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9, 220.39868420e9, 219.94944200e9, 219.56035410e9,
                                               217.82215e9, 230.538e9, 217.94005e9, 218.16044e9,
                                               217.2386e9, 218.22219200e9, 218.47563200e9]), # restfreqs
                       'line_names': ['H2CO', '13CO', 'SO', 'C18O', 'c-C3H2', '12CO', 'c-C3H2', 'c-C3H2', 'DCN', 'H2CO', 'H2CO'], # restfreqs
                       'flagrange': np.array([[-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5],
                                              [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5],
                                              [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6}, # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws': np.array([0, 1, 2, 3, 4, 5, 6]), # spws to use for continuum
                       'cont_avg_width': np.array([480, 480, 480, 480, 60, 60, 60]), # n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/09/30/05:30:00~2021/09/30/06:52:00',
                       'contdotdat': 'LB/cont-IRAS04169.dat'
                      },
               'LB3': {'vis': WD_path + prefix + '_LB3.ms',
                       'name': 'LB3',
                       'field': field['LB'],
                       'line_spws': np.array([0, 1, 2, 3, 4, 6, 4, 4, 4, 4, 4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9, 220.39868420e9, 219.94944200e9, 219.56035410e9,
                                               217.82215e9, 230.538e9, 217.94005e9, 218.16044e9,
                                               217.2386e9, 218.22219200e9, 218.47563200e9]), # restfreqs
                       'line_names': ['H2CO', '13CO', 'SO', 'C18O', 'c-C3H2', '12CO', 'c-C3H2', 'c-C3H2', 'DCN', 'H2CO', 'H2CO'], # restfreqs
                       'flagrange': np.array([[-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5],
                                              [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5],
                                              [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6}, # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws': np.array([0, 1, 2, 3, 4, 5, 6]), # spws to use for continuum
                       'cont_avg_width': np.array([480, 480, 480, 480, 60, 60, 60]), # n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/10/01/05:34:00~2021/10/01/06:54:00',
                       'contdotdat': 'LB/cont-IRAS04169.dat'
                      },
                'LB4': {'vis': WD_path + prefix + '_LB4.ms',
                       'name': 'LB4',
                       'field': field['LB'],
                       'line_spws': np.array([0, 1, 2, 3, 4, 6, 4, 4, 4, 4, 4]), # line SPWs, get from listobs
                       'line_freqs': np.array([218.76006600e9, 220.39868420e9, 219.94944200e9, 219.56035410e9,
                                               217.82215e9, 230.538e9, 217.94005e9, 218.16044e9,
                                               217.2386e9, 218.22219200e9, 218.47563200e9]), # restfreqs
                       'line_names': ['H2CO', '13CO', 'SO', 'C18O', 'c-C3H2', '12CO', 'c-C3H2', 'c-C3H2', 'DCN', 'H2CO', 'H2CO'], # restfreqs
                       'flagrange': np.array([[-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5],
                                              [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5],
                                              [-5.5, 14.5], [-5.5, 14.5], [-5.5, 14.5]]),
                       'orig_spw_map': {25:0, 27:1, 29:2, 31:3, 33:4, 35:5, 37:6}, # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws': np.array([0, 1, 2, 3, 4, 5, 6]), # spws to use for continuum
                       'cont_avg_width': np.array([480, 480, 480, 480, 60, 60, 60]), # n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2021/10/18/04:53:00~2021/10/18/06:12:00',
                       'contdotdat': 'LB/cont-IRAS04169.dat'
                      },
              }

### Flag range corresponds to velocity range in each spw that should be flagged. 
### Velocity range should correspond to 
### approximate width of the line contamination

# save data params to a pickle
with open(prefix + '.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

###############################################################
#################### DATA PREPARATION #########################
###############################################################

### split out each pipeline-calibrated dataset into an MS only containing the target data 
for i in pl_data_params.keys():
    if os.path.exists(prefix + '_' + i + '.ms'):
        flagmanager(vis = prefix + '_' + i + '.ms',
                    mode = 'restore',
                    versionname = 'starting_flags')
    else:
        split(vis = pl_data_params[i]['vis'],
              outputvis = prefix + '_' + i + '.ms', 
              spw = pl_data_params[i]['spws'],
              field = pl_data_params[i]['field'],
              datacolumn = pl_data_params[i]['column'])

### Backup the the flagging state at start of reduction
for i in data_params.keys():
    if not os.path.exists(data_params[i]['vis'] + ".flagversions/flags.starting_flags"):
        flagmanager(vis = data_params[i]['vis'],
                    mode = 'save',
                    versionname = 'starting_flags',
                    comment = 'Flag states at start of reduction')

flagdata(vis = data_params['SB1']['vis'], mode = 'manual', antenna = 'DA46&DV08')
flagdata(vis = data_params['SB2']['vis'], mode = 'manual', antenna = 'DA46&DV08')
flagdata(vis = data_params['LB1']['vis'], mode = 'manual', antenna = 'DA47&DV20')
flagdata(vis = data_params['LB2']['vis'], mode = 'manual', antenna = 'DA47&DV20; DA59&DV17; DA59&DV20; DA59&DV21; DV17&DV20')
flagdata(vis = data_params['LB3']['vis'], mode = 'manual', antenna = 'DA47&DV20; DA59&DV17; DA59&DV20; DA59&DV21; DV17&DV20')
flagdata(vis = data_params['LB4']['vis'], mode = 'manual', antenna = 'DA47&DV20')

# ### Inspect data in each spw for each dataset
# #### OPTIONAL #####
# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i]['vis'],
#                field = data_params[i]['field'],
#                xaxis = 'frequency',
#                yaxis = 'amplitude',
#                ydatacolumn = 'data',
#                yselfscale = True,
#                avgtime = '1e+08',
#                avgscan = True,
#                avgbaseline = True,
#                iteraxis = 'spw',
#                coloraxis = 'spw',
#                transform = True,
#                freqframe = 'LSRK')
#         # input("Press Enter key to advance to next MS/Caltable...")
# #### END OPTIONAL ###

### Flag spectral regions around lines and do spectral averaging to make a smaller continuum MS 
for i in data_params.keys():      
    os.system('rm -rf ' + prefix + '_' + i + '_initcont.ms.flagversions') 
    flagchannels_string = get_flagchannels(data_params[i], prefix)
    s=' ' # work around for Python 3 port of following string generating for loops
    print(i) 
    avg_cont(data_params[i],
             prefix,
             flagchannels = flagchannels_string,
             contspws = s.join(str(elem) for elem in data_params[i]['cont_spws'].tolist()).replace(' ', ','),
             width_array = data_params[i]['cont_avg_width'])
    data_params[i]['vis_avg'] = prefix + '_' + i + '_initcont.ms'

# Additional plotms for the averaged visibilities
# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i]['vis_avg'],
#                field = data_params[i]['field'],
#                xaxis = 'frequency',
#                yaxis = 'amplitude',
#                ydatacolumn = 'data',
#                yselfscale = True,
#                avgtime = '1e+08',
#                avgscan = True,
#                avgbaseline = True,
#                iteraxis = 'spw',
#                coloraxis = 'spw',
#                transform = True,
#                freqframe = 'LSRK')
#         # input("Press Enter key to advance to next MS/Caltable...")

###############################################################
############## INITIAL IMAGING FOR ALIGNMENT ##################
###############################################################

### Image each dataset individually to get source position in each image
### Images are saved in the format prefix+'_name_initcont_exec#.ms'
outertaper = '2000klambda' # taper if necessary to align using larger-scale uv data, small-scale may have subtle shifts from phase noise
for i in data_params.keys():      
    print('Imaging MS:', i) 
    if 'LB' in i:
        tclean_wrapper(vis = data_params[i]['vis_avg'],
                       imagename = prefix + '_' + i +'_initial_cont',
                       sidelobethreshold = 2.0, 
                       smoothfactor = 1.5,
                       scales = LB_scales,
                       nsigma = 5.0,
                       robust = 0.5,
                       parallel = parallel, 
                       uvtaper = [outertaper],
                       nterms = 1)
    else:
        tclean_wrapper(vis = data_params[i]['vis_avg'],
                       imagename = prefix + '_' + i + '_initial_cont',
                       sidelobethreshold = 2.5, 
                       smoothfactor = 1.5,
                       scales = LB_scales,
                       nsigma = 5.0,
                       robust = 0.5,
                       parallel = parallel,
                       nterms = 1)

       # check masks to ensure you are actually masking the image, lower sidelobethreshold if needed

""" Fit Gaussians to roughly estimate centers, inclinations, PAs """
""" Loops through each dataset specified """
### default fit region is blank for an obvious single source
fit_region = ''

### specify manual mask on brightest source if Gaussian fitting fails due to confusion
mask_ra  = '04h19m58.450s'.replace('h', ':').replace('m', ':').replace('s', '') # listobs
mask_dec = '27d09m57.100s'.replace('d', '.').replace('m', '.').replace('s', '') # listobs
mask_pa  = 0.0 # position angle of mask in degrees
mask_maj = 5.0 # semimajor axis of mask in arcsec
mask_min = 5.0 # semiminor axis of mask in arcsec
fit_region = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
              (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)

for i in data_params.keys():
    print(i)
    data_params[i]['phasecenter'] = fit_gaussian(prefix + '_' + i + '_initial_cont.image.tt0',
                                                 region = fit_region,
                                                 mask = prefix + '_' + i + '_initial_cont.mask')
    # SB1
    #   04h19m58.478311s +27d09m56.84964s
    #   #Peak of Gaussian component identified with imfit: ICRS 04h19m58.478311s +27d09m56.84964s
    #   04h19m58.478311s +27d09m56.84964s
    #   Separation: radian = 7.35253e-08, degrees = 0.000004 = 4.21269e-06, arcsec = 0.015166 = 0.0151657
    #   #Peak in J2000 coordinates: 04:19:58.47887, +027:09:56.836436
    #   #PA of Gaussian component: 86.56 deg
    #   #Inclination of Gaussian component: 34.88 deg
    #   #Pixel coordinates of peak: x = 787.378 y = 791.658

    # SB2
    #   04h19m58.478943s +27d09m56.83272s
    #   #Peak of Gaussian component identified with imfit: ICRS 04h19m58.478943s +27d09m56.83272s
    #   04h19m58.478943s +27d09m56.83272s
    #   Separation: radian = 7.34617e-08, degrees = 0.000004 = 4.20905e-06, arcsec = 0.015153 = 0.0151526
    #   #Peak in J2000 coordinates: 04:19:58.47950, +027:09:56.819516
    #   #PA of Gaussian component: 37.68 deg
    #   #Inclination of Gaussian component: 5.50 deg
    #   #Pixel coordinates of peak: x = 787.097 y = 791.094

    # LB1
    #   04h19m58.475250s +27d09m56.79892s
    #   #Peak of Gaussian component identified with imfit: ICRS 04h19m58.475250s +27d09m56.79892s
    #   04h19m58.475250s +27d09m56.79892s
    #   Separation: radian = 7.35571e-08, degrees = 0.000004 = 4.21451e-06, arcsec = 0.015172 = 0.0151722
    #   #Peak in J2000 coordinates: 04:19:58.47581, +027:09:56.785716
    #   #PA of Gaussian component: 142.22 deg
    #   #Inclination of Gaussian component: 39.44 deg
    #   #Pixel coordinates of peak: x = 2887.396 y = 2899.675

    # LB2
    #   04h19m58.473902s +27d09m56.80493s
    #   #Peak of Gaussian component identified with imfit: ICRS 04h19m58.473902s +27d09m56.80493s
    #   04h19m58.473902s +27d09m56.80493s
    #   Separation: radian = 7.34935e-08, degrees = 0.000004 = 4.21087e-06, arcsec = 0.015159 = 0.0151591
    #   #Peak in J2000 coordinates: 04:19:58.47446, +027:09:56.791726
    #   #PA of Gaussian component: 37.71 deg
    #   #Inclination of Gaussian component: 58.40 deg
    #   #Pixel coordinates of peak: x = 2893.390 y = 2901.680

    # LB3
    #   04h19m58.474805s +27d09m56.82936s
    #   #Peak of Gaussian component identified with imfit: ICRS 04h19m58.474805s +27d09m56.82936s
    #   04h19m58.474805s +27d09m56.82936s
    #   Separation: radian = 7.33983e-08, degrees = 0.000004 = 4.20541e-06, arcsec = 0.015139 = 0.0151395
    #   #Peak in J2000 coordinates: 04:19:58.47536, +027:09:56.816156
    #   #PA of Gaussian component: 47.81 deg
    #   #Inclination of Gaussian component: 41.59 deg
    #   #Pixel coordinates of peak: x = 2889.376 y = 2909.822

    # LB4
    #   04h19m58.474391s +27d09m56.83717s
    #   #Peak of Gaussian component identified with imfit: ICRS 04h19m58.474391s +27d09m56.83717s
    #   04h19m58.474391s +27d09m56.83717s
    #   Separation: radian = 7.35253e-08, degrees = 0.000004 = 4.21269e-06, arcsec = 0.015166 = 0.0151657
    #   #Peak in J2000 coordinates: 04:19:58.47495, +027:09:56.823966
    #   #PA of Gaussian component: 4.94 deg
    #   #Inclination of Gaussian component: 36.38 deg
    #   #Pixel coordinates of peak: x = 2891.214 y = 2912.427

### Check phase center fits in viewer, if centers appear too shifted from the Gaussian fit, 
### manually set the phase center dictionary entry by eye

""" The emission centers are slightly misaligned.  So we split out the 
    individual executions, shift the peaks to the phase center, and reassign 
    the phase centers to a common direction. """

### Set common direction for each EB using one as reference (typically best looking LB image)
for i in data_params.keys():
    #################### MANUALLY SET THIS ######################
    data_params[i]['common_dir'] = 'J2000 04h19m58.47887s 27d09m56.836436s'
        # Peak in J2000 coordinates (imfit)
        # SB1: data_params['SB1']['common_dir'] = 'J2000 04h19m58.47887s 27d09m56.836436s'
        # SB2: data_params['SB2']['common_dir'] = 'J2000 04h19m58.47950s 27d09m56.819516s'
        # LB1: data_params['LB1']['common_dir'] = 'J2000 04h19m58.47581s 27d09m56.785716s'
        # LB2: data_params['LB2']['common_dir'] = 'J2000 04h19m58.47446s 27d09m56.791726s'
        # LB3: data_params['LB3']['common_dir'] = 'J2000 04h19m58.47536s 27d09m56.816156s'
        # LB4: data_params['LB4']['common_dir'] = 'J2000 04h19m58.47495s 27d09m56.823966s'

### save updated data params to a pickle
with open(prefix + '.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

###############################################################
#################### SHIFT PHASE CENTERS ######################
###############################################################

for i in data_params.keys():
   print(i)
   data_params[i]['vis_avg_shift'] = prefix + '_' + i + '_initcont_shift.ms'
   os.system('rm -rf ' + data_params[i]['vis_avg_shift'])
   fixvis(vis = data_params[i]['vis_avg'],
          outputvis = data_params[i]['vis_avg_shift'], 
          field = data_params[i]['field'], 
          phasecenter = 'J2000 ' + data_params[i]['phasecenter'])
   ### fixplanets may throw an error, usually safe to ignore
   fixplanets(vis = data_params[i]['vis_avg_shift'],
              field = data_params[i]['field'], 
              direction = data_params[i]['common_dir'])

###############################################################
############### REIMAGING TO CHECK ALIGNMENT ##################
###############################################################

for i in data_params.keys():
    print('Imaging MS:', i) 
    if 'LB' in i:
        tclean_wrapper(vis = data_params[i]['vis_avg_shift'],
                       imagename = prefix + '_' + i + '_initial_cont_shifted',
                       sidelobethreshold = 2.0, 
                       smoothfactor = 1.5,
                       scales = LB_scales,
                       nsigma = 5.0,
                       robust = 0.5,
                       parallel = parallel, 
                       uvtaper = [outertaper],
                       nterms = 1)
    else:
        tclean_wrapper(vis = data_params[i]['vis_avg_shift'],
                       imagename = prefix + '_' + i + '_initial_cont_shifted',
                       sidelobethreshold = 2.5, 
                       smoothfactor = 1.5,
                       scales = SB_scales,
                       nsigma = 5.0,
                       robust = 0.5,
                       parallel = parallel,
                       nterms = 1)

for i in data_params.keys():
    print(i)     
    data_params[i]['phasecenter_new'] = fit_gaussian(prefix + '_' + i + '_initial_cont_shifted.image.tt0',
                                                     region = fit_region,
                                                     mask = prefix + '_' + i + '_initial_cont_shifted.mask')

    print('\nPhasecenter new:', data_params[i]['phasecenter_new'])
    print('Phasecenter old:', data_params[i]['phasecenter'])

    # SB1
    #   04h19m58.478866s +27d09m56.83640s
    #   #Peak of Gaussian component identified with imfit: J2000 04h19m58.478866s +27d09m56.83640s
    #   #PA of Gaussian component: 86.42 deg
    #   #Inclination of Gaussian component: 34.85 deg
    #   #Pixel coordinates of peak: x = 799.977 y = 800.018

    #   Phasecenter new: 04h19m58.478866s +27d09m56.83640s
    #   Phasecenter old: 04h19m58.47887s +027d09m56.836436s

    # SB2
    #   04h19m58.478867s +27d09m56.83641s
    #   #Peak of Gaussian component identified with imfit: J2000 04h19m58.478867s +27d09m56.83641s
    #   #PA of Gaussian component: 31.14 deg
    #   #Inclination of Gaussian component: 5.57 deg
    #   #Pixel coordinates of peak: x = 799.976 y = 800.018

    #   Phasecenter new: 04h19m58.478867s +27d09m56.83641s
    #   Phasecenter old: 04h19m58.47950s +027d09m56.819516s

    # LB1
    #   04h19m58.478866s +27d09m56.83644s
    #   #Peak of Gaussian component identified with imfit: J2000 04h19m58.478866s +27d09m56.83644s
    #   #PA of Gaussian component: 142.23 deg
    #   #Inclination of Gaussian component: 39.42 deg
    #   #Pixel coordinates of peak: x = 2999.768 y = 3000.190

    #   Phasecenter new: 04h19m58.478866s +27d09m56.83644s
    #   Phasecenter old: 04h19m58.47581s +027d09m56.785716s

    # LB2
    #   04h19m58.478876s +27d09m56.83660s
    #   #Peak of Gaussian component identified with imfit: J2000 04h19m58.478876s +27d09m56.83660s
    #   #PA of Gaussian component: 37.69 deg
    #   #Inclination of Gaussian component: 58.26 deg
    #   #Pixel coordinates of peak: x = 2999.724 y = 3000.241

    #   Phasecenter new: 04h19m58.478876s +27d09m56.83660s
    #   Phasecenter old: 04h19m58.47446s +027d09m56.791726s

    # LB3
    #   04h19m58.478872s +27d09m56.83648s
    #   #Peak of Gaussian component identified with imfit: J2000 04h19m58.478872s +27d09m56.83648s
    #   #PA of Gaussian component: 47.85 deg
    #   #Inclination of Gaussian component: 41.59 deg
    #   #Pixel coordinates of peak: x = 2999.739 y = 3000.203

    #   Phasecenter new: 04h19m58.478872s +27d09m56.83648s
    #   Phasecenter old: 04h19m58.47536s +027d09m56.816156s

    # LB4
    #   04h19m58.478867s +27d09m56.83648s
    #   #Peak of Gaussian component identified with imfit: J2000 04h19m58.478867s +27d09m56.83648s
    #   #PA of Gaussian component: 4.99 deg
    #   #Inclination of Gaussian component: 36.40 deg
    #   #Pixel coordinates of peak: x = 2999.764 y = 3000.204

    #   Phasecenter new: 04h19m58.478867s +27d09m56.83648s
    #   Phasecenter old: 04h19m58.47495s +027d09m56.823966s

### save updated data params to a pickle
with open(prefix + '.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

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

### Plot deprojected visibility profiles for all data together
plot_deprojected(export_vislist,
                 fluxscale = [1.0] * len(export_vislist),
                 PA = PA,
                 incl = incl, 
                 show_err = False,
                 outfile = 'amp-vs-uv-distance-pre-selfcal.png')

### Now inspect offsets by comparing against a reference 
### Set reference data using the dictionary key.
### Using SB1 as reference because it looks the nicest by far

#################### MANUALLY SET THIS ######################
refdata = 'SB1'

reference = prefix + '_' + refdata + '_initcont_shift.vis.npz'
for i in data_params.keys():
    print(i)
    if i != refdata:
        data_params[i]['gencal_scale'] = estimate_flux_scale(reference = reference, 
                                                             comparison = prefix + '_' + i + '_initcont_shift.vis.npz', 
                                                             incl = incl,
                                                             PA = PA)
    else:
        data_params[i]['gencal_scale'] = 1.0
    print('')

#################### MANUALLY SET THIS ######################
##DOES THE AMP VS. UV-DISTANCE LOOK VERY DISSIMILAR BETWEEN DIFFERENT EBS?
##IF SO, UNCOMMENT AND RUN THIS BLOCK TO REMOVE SCALING AND PROCEED WITH 2- METHOD PASS

# for i in data_params.keys():
#     data_params[i]['gencal_scale'] = 1.0

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

data_params['SB1']['gencal_scale'] = 1.000
data_params['SB2']['gencal_scale'] = 0.993
data_params['LB1']['gencal_scale'] = 0.978
data_params['LB2']['gencal_scale'] = 0.926
data_params['LB3']['gencal_scale'] = 0.948
data_params['LB4']['gencal_scale'] = 0.966

###############################################################
############### SCALE DATA RELATIVE TO ONE EB #################
###############################################################

os.system('rm -rf *_rescaled.ms')
for i in data_params.keys():
    rescale_flux(data_params[i]['vis_avg_shift'], [data_params[i]['gencal_scale']])
    rescale_flux(data_params[i]['vis_avg'], [data_params[i]['gencal_scale']])
    data_params[i]['vis_avg_shift_rescaled'] = data_params[i]['vis_avg_shift'].replace('.ms', '_rescaled.ms')
    data_params[i]['vis_avg_rescaled'] = data_params[i]['vis_avg'].replace('.ms', '_rescaled.ms')

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
        export_vislist_rescaled.append(data_params[i]['vis_avg_shift_rescaled'].replace('.ms', '.vis.npz'))

    plot_deprojected(export_vislist_rescaled,
                     fluxscale = [1.0] * len(export_vislist_rescaled),
                     PA = PA,
                     incl = incl, 
                     show_err = False)

    ### Make sure differences are no longer significant
    refdata = 'SB1'
    reference = prefix + '_' + refdata + '_initcont_shift.vis.npz'
    for i in data_params.keys():
        if i != refdata:
            estimate_flux_scale(reference = reference, 
                                comparison = prefix + '_' + i + '_initcont_shift_rescaled.vis.npz', 
                                incl = incl,
                                PA = PA)

### Save data params to a pickle
with open(prefix + '.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

###############################################################
################ SELF-CALIBRATION PREPARATION #################
###############################################################

selectedVis = 'vis_avg_rescaled'
# selectedVis = 'vis_avg_shift_rescaled'

### determine best reference antennas based on geometry and flagging
for i in data_params.keys():
    data_params[i]['refant'] = rank_refants(data_params[i][selectedVis])

    # plotms(vis = data_params[i][selectedVis],
    #        field = data_params[i]['field'],
    #        xaxis = 'UVwave',
    #        yaxis = 'amp',
    #        xselfscale = True,
    #        yselfscale = True,
    #        avgchannel = '1e+08',
    #        avgtime = '1e+08',
    #        avgscan = True,
    #        iteraxis = 'antenna',
    #        coloraxis = 'spw',
    #        gridrows = 8,
    #        gridcols = 6)

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
# SB_refant   = 'DA43@A035,DV07@A011,DV05@A042' 

############### CHECK THESE, SHOULD BE FINE #################
SB_spwmap = [0, 0, 0, 0, 0, 0, 0]
SB_contspws = '' 

### Make a list of EBs to image
vislist = []
for i in data_params.keys():
    if 'LB' in i: # skip over LB EBs if in SB-only mode
        continue
    vislist.append(data_params[i][selectedVis])

""" Set up a clean mask """
mask_ra  = data_params[i]['common_dir'].split()[1].replace('h', ':').replace('m', ':').replace('s', '')
mask_dec = data_params[i]['common_dir'].split()[2].replace('d', '.').replace('m', '.').replace('s', '')
mask_pa  = 0.0  # position angle of mask in degrees
mask_maj = 2.0  # semimajor axis of mask in arcsec
mask_min = 2.0  # semiminor axis of mask in arcsec

common_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
              (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)

""" Define a noise annulus, measure the peak SNR in map """
noise_annulus = 'annulus[[%s, %s],[%.2farcsec, 8.0arcsec]]' % \
                (mask_ra, mask_dec, 2.0 * mask_maj)

###############################################################
###################### SELF-CALIBRATION #######################
###############################################################

fieldlist = []
for i in data_params.keys():
    if ('LB' in i): # skip over LB EBs if in SB-only mode
        continue
    fieldlist.append(data_params[i]['field'])

### Initial cleaned map to assess DR
tclean_wrapper(vis = vislist,
               imagename = prefix + '_initial',
               cellsize = '0.025arcsec',
               imsize = 1600, 
               scales = SB_scales,
               sidelobethreshold = 2.0,
               smoothfactor = 1.5,
               nsigma = 3.0, 
               noisethreshold = 3.0,
               robust = 0.5,
               parallel = parallel,
               phasecenter = data_params['SB1']['common_dir'].replace('J2000', 'ICRS'),
               nterms = 1)

initial_SNR, initial_RMS = estimate_SNR(prefix + '_initial.image.tt0',
                                        disk_mask = common_mask, 
                                        noise_mask = noise_annulus)

# IRAS04169_initial.image.tt0
# Beam 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mask: 91.11 mJy
# Peak intensity of source: 56.42 mJy/beam
# rms: 1.83e-01 mJy/beam
# Peak SNR: 308.31

listdict, scantimesdict, integrationsdict, integrationtimesdict, integrationtimes, n_spws, minspw, spwsarray = fetch_scan_times(vislist, fieldlist)
solints, gaincal_combine = get_solints_simple(vislist, scantimesdict, integrationtimesdict)

print('Suggested Solints:')
print(solints)
print('Suggested Gaincal Combine params:')
print(gaincal_combine)

nsigma_init = np.max([initial_SNR / 15.0, 5.0]) # restricts initial nsigma to be at least 5
nsigma_per_solint = 10**np.linspace(np.log10(nsigma_init), np.log10(3.0), len(solints))
nsigma_per_solint = nsigma_per_solint.round(2)

print('Suggested nsigma per solint: ')
print(nsigma_per_solint)

# Suggested Solints:
# ['inf', 'inf', '72.58s', '36.29s', '12.10s', 'int']
# Suggested Gaincal Combine params:
# ['spw,scan', 'spw', 'spw', 'spw', 'spw', 'spw']
# Suggested nsigma per solint: 
# [20.55 13.99 9.52 6.48 4.41 3.  ]

### Image produced by iter 0 has not selfcal applied, it's used to set the initial model
### only images >0 have self-calibration applied

### Run self-calibration command set
### 0. Split off corrected data from previous selfcal iteration (except iteration 0)
### 1. Image data to specified nsigma depth, set model column
### 2. Calculate self-cal gain solutions
### 3. Apply self-cal gain solutions to MS
### 4. Check S/N before after

############# USERS MAY NEED TO ADJUST NSIGMA AND SOLINT FOR EACH SELF-CALIBRATION ITERATION ##############
############################ CONTINUE SELF-CALIBRATION ITERATIONS UNTIL ###################################
#################### THE S/N BEGINS TO DROP OR SOLINTS ARE AS LOW AS POSSIBLE #############################

#################### LOOK FOR ERRORS IN GAINCAL CLAIMING A FREQUENCY MISMATCH #############################
####### IF FOUND, CHANGE SOLINT, MAYBE TRY TO ALIGN WITH A CERTAIN NUMBER OF SCANS AND TRY AGAIN ##########
########################## IF ALL ELSE FAILS, SIMPLY START WITH solint='inf' ##############################

iteration = 0
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'SB-only',
               iteration = iteration,
               selfcalmode = 'p',
               nsigma = nsigma_per_solint[0],
               solint = solints[0],
               combine = gaincal_combine[0],
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
                # LB_contspws = LB_contspws,
                # LB_spwmap = LB_spwmap,
               parallel = parallel,
                # smoothfactor = 2.0,
                # imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -2.0},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

# ### Plot gain corrections, loop through each
# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_p' + str(iteration) + '.g'),
#                xaxis = 'time',
#                yaxis = 'phase',
#                gridrows = 8,
#                gridcols = 6,
#                iteraxis = 'antenna',
#                xselfscale = True,
#                plotrange = [0, 0, -180, 180])
#         # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_SB-only_p0.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 86.99 mJy
# Peak intensity of source: 56.15 mJy/beam
# rms: 1.83e-01 mJy/beam
# Peak SNR: 307.38

# IRAS04169_SB-only_p0_post.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 87.43 mJy
# Peak intensity of source: 57.40 mJy/beam
# rms: 1.57e-01 mJy/beam
# Peak SNR: 366.36

iteration = 1
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'SB-only',
               iteration = iteration,
               selfcalmode = 'p',
               nsigma = nsigma_per_solint[1],
               solint = solints[1],
               combine = gaincal_combine[1],
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
                # LB_contspws = LB_contspws,
                # LB_spwmap = LB_spwmap,
               parallel = parallel,
                # smoothfactor = 2.0,
                # imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -2.0},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_p' + str(iteration) + '.g'),
#                xaxis = 'time',
#                yaxis = 'phase',
#                gridrows = 8,
#                gridcols = 6,
#                iteraxis = 'antenna',
#                xselfscale = True,
#                plotrange = [0, 0, -180, 180])
#        # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_SB-only_p1.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 89.26 mJy
# Peak intensity of source: 57.38 mJy/beam
# rms: 1.55e-01 mJy/beam
# Peak SNR: 371.18

# IRAS04169_SB-only_p1_post.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 89.61 mJy
# Peak intensity of source: 64.94 mJy/beam
# rms: 5.76e-02 mJy/beam
# Peak SNR: 1126.86

iteration = 2
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'SB-only',
               iteration = iteration,
               selfcalmode = 'p',
               nsigma = nsigma_per_solint[2],
               solint = solints[2],
               combine = gaincal_combine[2],
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
                # LB_contspws = LB_contspws,
                # LB_spwmap = LB_spwmap,
               parallel = parallel,
                # smoothfactor = 2.0,
                # imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -3.5},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_p' + str(iteration) + '.g'),
#                xaxis = 'time',
#                yaxis = 'phase',
#                gridrows = 8,
#                gridcols = 6,
#                iteraxis = 'antenna',
#                xselfscale = True,
#                plotrange=[0, 0, -180, 180]) 
#         # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_SB-only_p2.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 92.25 mJy
# Peak intensity of source: 64.90 mJy/beam
# rms: 3.98e-02 mJy/beam
# Peak SNR: 1630.14

# IRAS04169_SB-only_p2_post.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 92.37 mJy
# Peak intensity of source: 67.32 mJy/beam
# rms: 3.83e-02 mJy/beam
# Peak SNR: 1756.14

iteration = 3
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'SB-only',
               iteration = iteration,
               selfcalmode = 'p',
               nsigma = nsigma_per_solint[3],
               solint = solints[3],
               combine = gaincal_combine[3],
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
                # LB_contspws = LB_contspws,
                # LB_spwmap = LB_spwmap,
               parallel = parallel,
                # smoothfactor = 2.0,
                # imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -3.7},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_p' + str(iteration) + '.g'),
#                xaxis = 'time',
#                yaxis = 'phase',
#                gridrows = 8,
#                gridcols = 6,
#                iteraxis = 'antenna',
#                xselfscale = True,
#                plotrange = [0, 0, -180, 180]) 
#         # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_SB-only_p3.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 92.76 mJy
# Peak intensity of source: 67.19 mJy/beam
# rms: 3.49e-02 mJy/beam
# Peak SNR: 1924.74

# IRAS04169_SB-only_p3_post.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 92.86 mJy
# Peak intensity of source: 68.24 mJy/beam
# rms: 3.40e-02 mJy/beam
# Peak SNR: 2008.77

iteration = 4
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'SB-only',
               iteration = iteration,
               selfcalmode = 'p',
               nsigma = nsigma_per_solint[4],
                # solint = solints[4],
               solint = '12s',
               combine = gaincal_combine[4],
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
                # LB_contspws = LB_contspws,
                # LB_spwmap = LB_spwmap,
               parallel = parallel,
                # smoothfactor = 2.0,
                # imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -3.7},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_p' + str(iteration) + '.g'),
#                xaxis = 'time',
#                yaxis = 'phase',
#                gridrows = 8,
#                gridcols = 6,
#                iteraxis = 'antenna',
#                xselfscale = True,
#                plotrange = [0, 0, -180, 180]) 
#         # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_SB-only_p4.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 93.26 mJy
# Peak intensity of source: 68.13 mJy/beam
# rms: 3.26e-02 mJy/beam
# Peak SNR: 2087.58

# IRAS04169_SB-only_p4_post.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 93.49 mJy
# Peak intensity of source: 69.43 mJy/beam
# rms: 3.32e-02 mJy/beam
# Peak SNR: 2091.51

iteration = 5
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'SB-only',
               iteration = iteration,
               selfcalmode = 'p',
               nsigma = nsigma_per_solint[5],
               solint = solints[5],
               combine = gaincal_combine[5],
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
                # LB_contspws = LB_contspws,
                # LB_spwmap = LB_spwmap,
               parallel = parallel,
                # smoothfactor = 2.0,
                # imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -3.7},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_p' + str(iteration) + '.g'),
#                xaxis = 'time',
#                yaxis = 'phase',
#                gridrows = 8,
#                gridcols = 6,
#                iteraxis = 'antenna',
#                xselfscale = True,
#                plotrange = [0, 0, -180, 180]) 
#         # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_SB-only_p5.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 94.05 mJy
# Peak intensity of source: 69.34 mJy/beam
# rms: 3.19e-02 mJy/beam
# Peak SNR: 2172.61

# IRAS04169_SB-only_p5_post.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 94.29 mJy
# Peak intensity of source: 69.92 mJy/beam
# rms: 3.21e-02 mJy/beam
# Peak SNR: 2180.36

### Changing self-cal mode here to ap, see use of prevselfcalmode to ensure proper split
iteration = 6
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'SB-only',
               iteration = iteration,
               selfcalmode = 'ap',
               prevselfcalmode = 'p',
               nsigma = 3.0,
                # solint = '600s',
                # solint = '500s',
               solint = '300s',
               combine = 'spw, scan',
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
                # LB_contspws = LB_contspws,
                # LB_spwmap = LB_spwmap,
               parallel = parallel,
                # smoothfactor = 2.0,
                # imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -3.7},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

# if not skip_plots:
#    for i in data_params.keys():
#        plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_ap' + str(iteration) + '.g'),
#               xaxis = 'time',
#               yaxis = 'amp',
#               gridrows = 8,
#               gridcols = 6,
#               iteraxis = 'antenna',
#               xselfscale = True,
#               plotrange = [0, 0, 0, 2])
#        # input("Press Enter key to advance to next MS/Caltable...")
#        # *Check the "phase vs. time" plots, too!

# IRAS04169_SB-only_p6.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.36 deg)
# Flux inside disk mass: 94.43 mJy
# Peak intensity of source: 69.85 mJy/beam
# rms: 3.18e-02 mJy/beam
# Peak SNR: 2194.76

# IRAS04169_SB-only_p6_post.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.17 deg)
# Flux inside disk mass: 94.32 mJy
# Peak intensity of source: 69.95 mJy/beam
# rms: 3.11e-02 mJy/beam
# Peak SNR: 2246.69

iteration = 7
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'SB-only',
               iteration = iteration,
               selfcalmode = 'ap',
               nsigma = 3.0,
                # solint = '150s',
                # combine = gaincal_combine[0],
               solint = 'inf',
               combine = 'spw',
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
                # LB_contspws = LB_contspws,
                # LB_spwmap = LB_spwmap,
               parallel = parallel,
                # smoothfactor = 2.0,
                # imsize = 6000,
               nterms = 2,
               finalimageonly = True)

# imview(raster = [{'file': 'IRAS04169_SB-only_ap%s.image.tt0' % iteration, 'scaling': -3.7},
#                  {'file': 'IRAS04169_SB-only_ap%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_ap%s.mask' % iteration})

# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_ap' + str(iteration) + '.g'),
#                xaxis = 'time',
#                yaxis = 'amp',
#                gridrows = 8,
#                gridcols = 6,
#                iteraxis = 'antenna',
#                xselfscale = True,
#                plotrange = [0, 0, 0, 2])
#         # input("Press Enter key to advance to next MS/Caltable...")
#         # *Check the "phase vs. time" plots, too!

# IRAS04169_SB-only_ap7.image.tt0
# Beam: 0.372 arcsec x 0.252 arcsec (-4.17 deg)
# Flux inside disk mass: 94.28 mJy
# Peak intensity of source: 69.87 mJy/beam
# rms: 3.10e-02 mJy/beam
# Peak SNR: 2250.39

for i in data_params.keys():
    if 'SB' in i:
        data_params[i]['selfcal_spwmap_SB-only'] = data_params[i]['selfcal_spwmap'].copy()
        data_params[i]['selfcal_tables_SB-only'] = data_params[i]['selfcal_tables'].copy()
        data_params[i]['vis_avg_selfcal_SB-only'] = (data_params[i]['vis_avg_selfcal'] + '.')[:-1]  ## trick to copy the string

### save data params to a pickle
with open(prefix + '.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

###############################################################
################### SELF-CALIBRATION SB+LB ####################
###############################################################

LB_spwmap = [0, 0, 0, 0, 0, 0, 0]
LB_contspws = '' 

### Make a list of EBs to image
vislist = []
fieldlist = []
for i in data_params.keys():
    vislist.append(data_params[i][selectedVis])
    fieldlist.append(data_params[i]['field'])

### Initial cleaned map to assess DR
tclean_wrapper(vis = vislist,
               imagename = prefix + '_initial_LB+SB',
               cellsize = '0.003arcsec',
               imsize = 6000, 
               scales = LB_scales,
               sidelobethreshold = 2.0,
               smoothfactor = 1.5,
               nsigma = 3.0, 
               noisethreshold = 3.0,
               robust = 0.5,
               parallel = parallel,
               phasecenter = data_params['LB4']['common_dir'].replace('J2000', 'ICRS'),
               nterms = 1)

# IRAS04169_initial_LB+SB.image.tt0
# Beam 0.074 arcsec x 0.054 arcsec (28.58 deg)
# Flux inside disk mask: 100.50 mJy
# Peak intensity of source: 5.82 mJy/beam
# rms: 3.53e-02 mJy/beam
# Peak SNR: 165.06

initial_SNR, initial_RMS = estimate_SNR(prefix + '_initial_LB+SB.image.tt0',
                                        disk_mask = common_mask, 
                                        noise_mask = noise_annulus)

listdict, scantimesdict, integrationsdict, integrationtimesdict, integrationtimes, n_spws, minspw, spwsarray = fetch_scan_times(vislist, fieldlist)
solints, gaincal_combine = get_solints_simple(vislist, scantimesdict, integrationtimesdict)

print('Suggested Solints:')
print(solints)
print('Suggested Gaincal Combine params:')
print(gaincal_combine)

nsigma_init = np.max([initial_SNR / 15.0, 5.0]) # restricts initial nsigma to be at least 5
nsigma_per_solint = 10**np.linspace(np.log10(nsigma_init), np.log10(3.0), len(solints))
nsigma_per_solint = nsigma_per_solint.round(2)

print('Suggested nsigma per solint: ')
print(nsigma_per_solint)

# Suggested Solints:
# ['inf', 'inf', '12.10s', 'int']
# Suggested Gaincal Combine params:
# ['spw,scan', 'spw', 'spw', 'spw']
# Suggested nsigma per solint: 
# [11. 7.14 4.63 3.  ]

### Image produced by iter 0 has not selfcal applied, it's used to set the initial model
### only images >0 have self-calibration applied

### Run self-calibration command set
### 0. Split off corrected data from previous selfcal iteration (except iteration 0)
### 1. Image data to specified nsigma depth, set model column
### 2. Calculate self-cal gain solutions
### 3. Apply self-cal gain solutions to MS
### 4. Check S/N before after

############# USERS MAY NEED TO ADJUST NSIGMA AND SOLINT FOR EACH SELF-CALIBRATION ITERATION ##############
############################ CONTINUE SELF-CALIBRATION ITERATIONS UNTIL ###################################
#################### THE S/N BEGINS TO DROP OR SOLINTS ARE AS LOW AS POSSIBLE #############################

#################### LOOK FOR ERRORS IN GAINCAL CLAIMING A FREQUENCY MISMATCH #############################
####### IF FOUND, CHANGE SOLINT, MAYBE TRY TO ALIGN WITH A CERTAIN NUMBER OF SCANS AND TRY AGAIN ##########
########################## IF ALL ELSE FAILS, SIMPLY START WITH solint='inf' ##############################

iteration = 0
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'LB+SB',
               iteration = iteration,
               selfcalmode = 'p',
               nsigma = nsigma_per_solint[0],
               solint = solints[0],
               combine = gaincal_combine[0],
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
               LB_contspws = LB_contspws,
               LB_spwmap = LB_spwmap,
               parallel = parallel,
               smoothfactor = 2.0,
               imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_LB+SB_p%s.image.tt0' % iteration, 'scaling': -1.0},
#                  {'file': 'IRAS04169_LB+SB_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_LB+SB_p%s.mask' % iteration})

# ### Plot gain corrections, loop through each
# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i][selectedVis].replace('.ms', '_LB+SB_p' + str(iteration) + '.g'),
#                xaxis = 'time',
#                yaxis = 'phase',
#                gridrows = 8,
#                gridcols = 6,
#                iteraxis = 'antenna',
#                xselfscale = True,
#                plotrange = [0, 0, -180, 180])
#         # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_LB+SB_p0.image.tt0
# Beam: 0.074 arcsec x 0.054 arcsec (28.58 deg)
# Flux inside disk mass: 109.49 mJy
# Peak intensity of source: 6.61 mJy/beam
# rms: 3.20e-02 mJy/beam
# Peak SNR: 206.35

# IRAS04169_LB+SB_p0_post.image.tt0
# Beam: 0.074 arcsec x 0.054 arcsec (28.58 deg)
# Flux inside disk mass: 108.90 mJy
# Peak intensity of source: 7.49 mJy/beam
# rms: 2.66e-02 mJy/beam
# Peak SNR: 281.14

iteration = 1
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'LB+SB',
               iteration = iteration,
               selfcalmode = 'p',
               nsigma = nsigma_per_solint[1],
               solint = solints[1],
               combine = gaincal_combine[1],
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
               LB_contspws = LB_contspws,
               LB_spwmap = LB_spwmap,
               parallel = parallel,
               smoothfactor = 2.0,
               imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_LB+SB_p%s.image.tt0' % iteration, 'scaling': -1.5},
#                  {'file': 'IRAS04169_LB+SB_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_LB+SB_p%s.mask' % iteration})

# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i][selectedVis].replace('.ms', '_LB+SB_p' + str(iteration) + '.g'),
#                xaxis = 'time',
#                yaxis = 'phase',
#                gridrows = 8,
#                gridcols = 6,
#                iteraxis = 'antenna',
#                xselfscale = True,
#                plotrange = [0, 0, -180, 180])
#        # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_LB+SB_p1.image.tt0
# Beam: 0.074 arcsec x 0.054 arcsec (28.58 deg)
# Flux inside disk mass: 104.12 mJy
# Peak intensity of source: 7.07 mJy/beam
# rms: 2.62e-02 mJy/beam
# Peak SNR: 269.43

# IRAS04169_LB+SB_p1_post.image.tt0
# Beam: 0.074 arcsec x 0.054 arcsec (28.58 deg)
# Flux inside disk mass: 103.30 mJy
# Peak intensity of source: 9.61 mJy/beam
# rms: 1.85e-02 mJy/beam
# Peak SNR: 518.40

iteration = 2
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'LB+SB',
               iteration = iteration,
               selfcalmode = 'p',
               nsigma = nsigma_per_solint[1],
               solint = solints[1],
               combine = gaincal_combine[1],
                # nsigma, solint, and combine in p2 are the same as in p1.
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
               LB_contspws = LB_contspws,
               LB_spwmap = LB_spwmap,
               parallel = parallel,
               smoothfactor = 2.0,
               imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_LB+SB_p%s.image.tt0' % iteration, 'scaling': -2.5},
#                  {'file': 'IRAS04169_LB+SB_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_LB+SB_p%s.mask' % iteration})

# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i][selectedVis].replace('.ms', '_LB+SB_p' + str(iteration) + '.g'),
#                xaxis = 'time',
#                yaxis = 'phase',
#                gridrows = 8,
#                gridcols = 6,
#                iteraxis = 'antenna',
#                xselfscale = True,
#                plotrange=[0, 0, -180, 180]) 
#         # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_LB+SB_p2.image.tt0
# Beam: 0.074 arcsec x 0.054 arcsec (28.58 deg)
# Flux inside disk mass: 101.21 mJy
# Peak intensity of source: 8.69 mJy/beam
# rms: 1.53e-02 mJy/beam
# Peak SNR: 566.74

# IRAS04169_LB+SB_p2_post.image.tt0
# Beam: 0.074 arcsec x 0.054 arcsec (28.58 deg)
# Flux inside disk mass: 101.22 mJy
# Peak intensity of source: 8.70 mJy/beam
# rms: 1.52e-02 mJy/beam
# Peak SNR: 574.02

iteration = 3
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'LB+SB',
               iteration = iteration,
               selfcalmode = 'p',
               nsigma = nsigma_per_solint[2],
                # solint = solints[2],
               solint = '12s',
               combine = gaincal_combine[2],
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
               LB_contspws = LB_contspws,
               LB_spwmap = LB_spwmap,
               parallel = parallel,
               smoothfactor = 2.0,
               imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_LB+SB_p%s.image.tt0' % iteration, 'scaling': -2.0},
#                  {'file': 'IRAS04169_LB+SB_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_LB+SB_p%s.mask' % iteration})

# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i][selectedVis].replace('.ms', '_LB+SB_p' + str(iteration) + '.g'),
#                xaxis = 'time',
#                yaxis = 'phase',
#                gridrows = 8,
#                gridcols = 6,
#                iteraxis = 'antenna',
#                xselfscale = True,
#                plotrange=[0, 0, -180, 180]) 
#         # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_LB+SB_p3.image.tt0
# Beam: 0.074 arcsec x 0.054 arcsec (28.58 deg)
# Flux inside disk mass: 100.78 mJy
# Peak intensity of source: 8.69 mJy/beam
# rms: 1.52e-02 mJy/beam
# Peak SNR: 573.72

# IRAS04169_LB+SB_p3_post.image.tt0
# Beam: 0.074 arcsec x 0.054 arcsec (28.58 deg)
# Flux inside disk mass: 100.61 mJy
# Peak intensity of source: 9.20 mJy/beam
# rms: 1.51e-02 mJy/beam
# Peak SNR: 608.59

iteration = 4
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'LB+SB',
               iteration = iteration,
               selfcalmode = 'p',
               nsigma = nsigma_per_solint[3],
               solint = solints[3],
               combine = gaincal_combine[3],
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
               LB_contspws = LB_contspws,
               LB_spwmap = LB_spwmap,
               parallel = parallel,
               smoothfactor = 2.0,
               imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_LB+SB_p%s.image.tt0' % iteration, 'scaling': -2.5},
#                  {'file': 'IRAS04169_LB+SB_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_LB+SB_p%s.mask' % iteration})

# if not skip_plots:
#     for i in data_params.keys():
#         plotms(vis = data_params[i][selectedVis].replace('.ms', '_LB+SB_p' + str(iteration) + '.g'),
#                xaxis = 'time',
#                yaxis = 'phase',
#                gridrows = 8,
#                gridcols = 6,
#                iteraxis = 'antenna',
#                xselfscale = True,
#                plotrange=[0, 0, -180, 180]) 
#         # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_LB+SB_p4.image.tt0
# Beam: 0.074 arcsec x 0.054 arcsec (28.58 deg)
# Flux inside disk mass: 99.02 mJy
# Peak intensity of source: 9.02 mJy/beam
# rms: 1.50e-02 mJy/beam
# Peak SNR: 602.88

# IRAS04169_LB+SB_p4_post.image.tt0
# Beam: 0.074 arcsec x 0.054 arcsec (28.58 deg)
# Flux inside disk mass: 99.01 mJy
# Peak intensity of source: 9.02 mJy/beam
# rms: 1.50e-02 mJy/beam
# Peak SNR: 600.67

### Changing self-cal mode here to ap, see use of prevselfcalmode to ensure proper split
iteration = 5
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'LB+SB',
               iteration = iteration,
               selfcalmode = 'ap',
               prevselfcalmode = 'p',
               nsigma = 3.0,
               solint = '450s',
               combine = 'scan, spw',
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
               LB_contspws = LB_contspws,
               LB_spwmap = LB_spwmap,
               parallel = parallel,
               smoothfactor = 2.0,
               imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_LB+SB_p%s.image.tt0' % iteration, 'scaling': -2.5},
#                  {'file': 'IRAS04169_LB+SB_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_LB+SB_p%s.mask' % iteration})

# if not skip_plots:
#    for i in data_params.keys():
#        plotms(vis = data_params[i][selectedVis].replace('.ms', '_LB+SB_ap' + str(iteration) + '.g'),
#               xaxis = 'time',
#               yaxis = 'amp',
#               gridrows = 8,
#               gridcols = 6,
#               iteraxis = 'antenna',
#               xselfscale = True,
#               plotrange = [0, 0, 0, 2])
#        # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_LB+SB_p5.image.tt0
# Beam: 0.074 arcsec x 0.054 arcsec (28.58 deg)
# Flux inside disk mass: 98.90 mJy
# Peak intensity of source: 9.04 mJy/beam
# rms: 1.50e-02 mJy/beam
# Peak SNR: 601.77

# IRAS04169_LB+SB_p5_post.image.tt0
# Beam: 0.075 arcsec x 0.056 arcsec (28.46 deg)
# Flux inside disk mass: 96.30 mJy
# Peak intensity of source: 9.27 mJy/beam
# rms: 1.48e-02 mJy/beam
# Peak SNR: 626.39

iteration = 6
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'LB+SB',
               iteration = iteration,
               selfcalmode = 'ap',
               nsigma = 3.0,
               solint = '300s',
               combine = 'scan, spw',
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
               LB_contspws = LB_contspws,
               LB_spwmap = LB_spwmap,
               parallel = parallel,
               smoothfactor = 2.0,
               imsize = 6000,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_LB+SB_ap%s.image.tt0' % iteration, 'scaling': -2.5},
#                  {'file': 'IRAS04169_LB+SB_ap%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_LB+SB_ap%s.mask' % iteration})

# if not skip_plots:
#    for i in data_params.keys():
#        plotms(vis = data_params[i][selectedVis].replace('.ms', '_LB+SB_ap' + str(iteration) + '.g'),
#               xaxis = 'time',
#               yaxis = 'amp',
#               gridrows = 8,
#               gridcols = 6,
#               iteraxis = 'antenna',
#               xselfscale = True,
#               plotrange = [0, 0, 0, 2])
#        # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_LB+SB_ap6.image.tt0
# Beam: 0.075 arcsec x 0.056 arcsec (28.46 deg)
# Flux inside disk mass: 97.88 mJy
# Peak intensity of source: 9.35 mJy/beam
# rms: 1.48e-02 mJy/beam
# Peak SNR: 631.79

# IRAS04169_LB+SB_ap6_post.image.tt0
# Beam: 0.075 arcsec x 0.056 arcsec (28.21 deg)
# Flux inside disk mass: 97.81 mJy
# Peak intensity of source: 9.42 mJy/beam
# rms: 1.49e-02 mJy/beam
# Peak SNR: 633.28

iteration = 7
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'LB+SB',
               iteration = iteration,
               selfcalmode = 'ap',
               nsigma = 3.0,
               solint = '150s',
               combine = 'scan, spw',
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
               LB_contspws = LB_contspws,
               LB_spwmap = LB_spwmap,
               parallel = parallel,
               smoothfactor = 2.0,
               imsize = 6000,
               nterms = 2,
               finalimageonly = True)

# imview(raster = [{'file': 'IRAS04169_LB+SB_ap%s.image.tt0' % iteration, 'scaling': -2.5},
#                  {'file': 'IRAS04169_LB+SB_ap%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_LB+SB_ap%s.mask' % iteration})

# if not skip_plots:
#    for i in data_params.keys():
#        plotms(vis = data_params[i][selectedVis].replace('.ms', '_LB+SB_ap' + str(iteration) + '.g'),
#               xaxis = 'time',
#               yaxis = 'amp',
#               gridrows = 8,
#               gridcols = 6,
#               iteraxis = 'antenna',
#               xselfscale = True,
#               plotrange = [0, 0, 0, 2])
#        # input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_LB+SB_ap7.image.tt0
# Beam: 0.075 arcsec x 0.056 arcsec (28.21 deg)
# Flux inside disk mass: 97.66 mJy
# Peak intensity of source: 9.41 mJy/beam
# rms: 1.49e-02 mJy/beam
# Peak SNR: 633.20

### Backup gain table list for LB+SB runs
for i in data_params.keys():
    if 'SB' in i:
        data_params[i]['selfcal_spwmap'] = data_params[i]['selfcal_spwmap_SB-only'] + data_params[i]['selfcal_spwmap']
        data_params[i]['selfcal_tables'] = data_params[i]['selfcal_tables_SB-only'] + data_params[i]['selfcal_tables']

### save data params to a pickle
with open(prefix + '.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

###############################################################
############ SHIFT PHASE CENTERS TO CHECK SCALING #############
###############################################################

for i in data_params.keys():
    print(i)
    data_params[i]['vis_avg_shift_selfcal'] = prefix + '_' + i + '_selfcal_cont_shift.ms'
    os.system('rm -rf ' + data_params[i]['vis_avg_shift_selfcal'])
    fixvis(vis = data_params[i]['vis_avg_selfcal'],
           outputvis = data_params[i]['vis_avg_shift_selfcal'], 
           field = data_params[i]['field'], 
           phasecenter = 'J2000 ' + data_params[i]['phasecenter'])
    ### fix planets may throw an error, usually safe to ignore
    fixplanets(vis = data_params[i]['vis_avg_shift_selfcal'],
               field = data_params[i]['field'], 
               direction = data_params[i]['common_dir'])

###############################################################
############### PLOT UV DATA TO CHECK SCALING #################
###############################################################

### Assign rough emission geometry parameters; keep 0, 0
PA, incl = 0, 0

### Export MS contents into Numpy save files 
export_vislist = []
for i in data_params.keys():
   export_MS(data_params[i]['vis_avg_shift_selfcal'])
   export_vislist.append(data_params[i]['vis_avg_shift_selfcal'].replace('.ms', '.vis.npz'))

if not skip_plots:
    ### Plot deprojected visibility profiles for all data together """
    plot_deprojected(export_vislist,
                     fluxscale = [1.0] * len(export_vislist),
                     PA = PA,
                     incl = incl, 
                     show_err = False,
                     outfile = 'amp-vs-uv-distance-post-selfcal.png')

#################### MANUALLY SET THIS ######################
refdata = 'SB1'

reference = prefix + '_' + refdata + '_selfcal_cont_shift.vis.npz'
for i in data_params.keys():
    print(i)
    if i != refdata:
        data_params[i]['gencal_scale_selfcal'] = estimate_flux_scale(reference = reference, 
                                                                     comparison = prefix + '_' + i + '_selfcal_cont_shift.vis.npz', 
                                                                     incl = incl,
                                                                     PA = PA)
    else:
        data_params[i]['gencal_scale_selfcal'] = 1.0
    print(' ')

gencal_scale = {}

print('IF AND ONLY IF SCALING WAS NOT ALREADY SET IN SCRIPT')
print('COPY AND PLACE WHERE DESIGNATED FOR SCALING TOWARD TOP OF SCRIPT')
for i in data_params.keys():
    gencal_scale[i] = data_params[i]['gencal_scale_selfcal']
    print('data_params["{}"]["gencal_scale"] = {:0.3f}'.format(i, data_params[i]['gencal_scale_selfcal']))

    # Before scaling,
    # data_params['SB1']['gencal_scale'] = 1.000
    # data_params['SB2']['gencal_scale'] = 0.993
    # data_params['LB1']['gencal_scale'] = 0.978
    # data_params['LB2']['gencal_scale'] = 0.926
    # data_params['LB3']['gencal_scale'] = 0.948
    # data_params['LB4']['gencal_scale'] = 0.966

    # After scaling,
    # data_params['SB1']['gencal_scale'] = 1.000
    # data_params['SB2']['gencal_scale'] = 1.000
    # data_params['LB1']['gencal_scale'] = 0.993
    # data_params['LB2']['gencal_scale'] = 0.994
    # data_params['LB3']['gencal_scale'] = 0.996
    # data_params['LB4']['gencal_scale'] = 0.992

# WRITE OUT PICKLE FILE FOR SCALING IF MISSED
if not os.path.exists('gencal_scale.pickle'):
    with open('gencal_scale.pickle', 'wb') as handle:
        pickle.dump(gencal_scale, handle, protocol = pickle.HIGHEST_PROTOCOL)

###############################################################
################# SPLIT OFF FINAL CONT DATA ###################
###############################################################

for i in data_params.keys():
    os.system('rm -rf ' + prefix + '_' + i + '_continuum.ms ' + prefix + '_' + i + '_continuum.ms.tgz')
    split(vis = data_params[i]['vis_avg_selfcal'],
          outputvis = prefix + '_' + i + '_continuum.ms',
          datacolumn = 'data')
    data_params[i]['vis_final'] = prefix + '_' + i + '_continuum.ms'
    os.system('tar cvzf ' + prefix + '_' + i + '_continuum.ms.tgz ' + prefix + '_' + i + '_continuum.ms')

# save data params to a pickle
with open(prefix + '.pickle', 'wb') as handle:
    pickle.dump(data_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

###############################################################
################## RUN A FINAL IMAGE SET ######################
###############################################################

### Generate a vislist
vislist = []
for i in data_params.keys():
    vislist.append(data_params[i]['vis_final'])

scales = SB_scales
imsize = 6000
cellsize = '0.003arcsec'

for robust in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
    imagename = prefix + '_SBLB_continuum_robust_' + str(robust)
    os.system('rm -rf ' + imagename + '*')

    sigma = get_sensitivity(data_params,
                            specmode = 'mfs',
                            imsize = imsize,
                            robust = robust,
                            cellsize = cellsize)

    # adjust sigma to corrector for the irregular noise in the images if needed
    # correction factor may vary or may not be needed at all depending on source
    # if robust == 2.0 or robust == 1.0:
    #     sigma = sigma * 1.75
    tclean_wrapper(vis = vislist,
                   imagename = imagename,
                   sidelobethreshold = 2.0, 
                   smoothfactor = 1.5,
                   scales = scales,
                   threshold = 3.0 * sigma, 
                   noisethreshold = 3.0,
                   robust = robust,
                   parallel = parallel, 
                   cellsize = cellsize,
                   imsize = imsize,
                   phasecenter = data_params['SB1']['common_dir'].replace('J2000', 'ICRS'),
                    # nterms = 1,
                   nterms = 2)
    imagename = imagename + '.image.tt0'
    exportfits(imagename = imagename,
               fitsimage = imagename + '.fits',
               overwrite = True,
               dropdeg = True)

for taper in ['1000klambda', '2000klambda', '3000klambda']:
    for robust in [1.0, 2.0]:
        print('Generate Robust ' + str(robust) + ' taper ' + taper + ' image')
        imagename = prefix + '_SBLB_continuum_robust_' + str(robust) + '_taper_' + taper
        os.system('rm -rf ' + imagename + '*')
        sigma = get_sensitivity(data_params,
                                specmode = 'mfs',
                                imsize = imsize,
                                robust = robust,
                                cellsize = cellsize,
                                uvtaper = taper)
        tclean_wrapper(vis = vislist,
                       imagename = imagename,
                       sidelobethreshold = 2.0, 
                       smoothfactor = 1.5,
                       scales = scales,
                       threshold = 2.0 * sigma, 
                       noisethreshold = 3.0,
                       robust = robust,
                       parallel = parallel, 
                       cellsize = cellsize,
                       imsize = imsize,
                        # nterms = 1,
                       nterms = 2,
                       uvtaper = [taper],
                       uvrange = '',
                       phasecenter = data_params['SB1']['common_dir'].replace('J2000', 'ICRS'))
        imagename = imagename + '.image.tt0'
        exportfits(imagename = imagename,
                   fitsimage = imagename + '.fits',
                   overwrite = True,
                   dropdeg = True)

    # if selectedVis == 'vis_avg_shift_rescaled':
    # tclean_wrapper(vis = data_params['LB1']['vis'].replace('.ms', '_initcont.ms'),
    #                imagename = 'temporary.pbfix',
    #                threshold = 0.0,
    #                niter = 0,
    #                scales = [0],
    #                robust = 0.5,
    #                parallel = parallel, 
    #                cellsize = cellsize,
    #                imsize = imsize,
    #                nterms = 1,
    #                phasecenter = data_params['LB1']['common_dir'])

    pblist = glob.glob('*continuum*.pb.tt0') 
    os.system('mkdir orig_pbimages')
    for pbimage in pblist:
        os.system('mv ' + pbimage + ' orig_pbimages')
    #     os.system('cp -r temporary.pbfix.pb.tt0 ' + pbimage)
    # os.system('rm -rf temporary.pbfix.*')

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
