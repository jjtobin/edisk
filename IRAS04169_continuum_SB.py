"""
eDisk data reduction script
This script was written for CASA 6.1.1/6.2
Originally derived from DSHARP reduction scripts

Datasets calibrated (in order of date observed):
SB1: 2019.1.00261.L (2022/07/03)
SB2: 2019.1.00261.L (2022/07/03)

reducer: Ilseung Han
"""

# ### Import statements
# #sys.path.append('/home/casa/contrib/AIV/science/analysis_scripts/')
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
                       'contdotdat': SB_path+'cont-IRAS04169.dat'
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
                       'contdotdat': SB_path+'cont-IRAS04169.dat'
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

for i in data_params.keys():
   if 'SB' in i:
      flagdata(vis=data_params[i]['vis'],mode='manual',antenna='DA46&DV08')

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

# *Additional plotms for the averaged visibility
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
    #   04h19m58.478312s +27d09m56.84964s
    #   #Peak of Gaussian component identified with imfit: ICRS 04h19m58.478312s +27d09m56.84964s
    #   04h19m58.478312s +27d09m56.84964s
    #   Separation: radian = 7.34935e-08, degrees = 0.000004 = 4.21087e-06, arcsec = 0.015159 = 0.0151591
    #   #Peak in J2000 coordinates: 04:19:58.47887, +027:09:56.836436
    #   #PA of Gaussian component: 86.32 deg
    #   #Inclination of Gaussian component: 34.76 deg
    #   #Pixel coordinates of peak: x = 787.377 y = 791.658

    # SB2
    #   04h19m58.478942s +27d09m56.83269s
    #   #Peak of Gaussian component identified with imfit: ICRS 04h19m58.478942s +27d09m56.83269s
    #   04h19m58.478942s +27d09m56.83269s
    #   Separation: radian = 7.34935e-08, degrees = 0.000004 = 4.21087e-06, arcsec = 0.015159 = 0.0151591
    #   #Peak in J2000 coordinates: 04:19:58.47950, +027:09:56.819486
    #   #PA of Gaussian component: 36.92 deg
    #   #Inclination of Gaussian component: 5.97 deg
    #   #Pixel coordinates of peak: x = 787.097 y = 791.093

### Check phase center fits in viewer, if centers appear too shifted from the Gaussian fit, 
### manually set the phase center dictionary entry by eye

""" The emission centers are slightly misaligned.  So we split out the 
    individual executions, shift the peaks to the phase center, and reassign 
    the phase centers to a common direction. """

### Set common direction for each EB using one as reference (typically best looking LB image)
for i in data_params.keys():
    #################### MANUALLY SET THIS ######################
    data_params[i]['common_dir'] = 'J2000 04h19m58.47887s 27d09m56.836436s'
        # SB1: data_params['SB1']['common_dir'] = 'J2000 04h19m58.47887s 27d09m56.836436s'
        # SB2: data_params['SB1']['common_dir'] = 'J2000 04h19m58.47950s 27d09m56.819486s'

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
    #   04h19m58.478865s +27d09m56.83635s
    #   #Peak of Gaussian component identified with imfit: J2000 04h19m58.478865s +27d09m56.83635s
    #   #PA of Gaussian component: 86.28 deg
    #   #Inclination of Gaussian component: 34.65 deg
    #   #Pixel coordinates of peak: x = 799.977 y = 800.016

    #   Phasecenter new: 04h19m58.478865s +27d09m56.83635s
    #   Phasecenter old: 04h19m58.47887s +027d09m56.836436s

    # SB2
    #   04h19m58.478867s +27d09m56.83643s
    #   #Peak of Gaussian component identified with imfit: J2000 04h19m58.478867s +27d09m56.83643s
    #   #PA of Gaussian component: 31.80 deg
    #   #Inclination of Gaussian component: 6.25 deg
    #   #Pixel coordinates of peak: x = 799.976 y = 800.019

    #   Phasecenter new: 04h19m58.478867s +27d09m56.83643s
    #   Phasecenter old: 04h19m58.47950s +027d09m56.819486s

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
data_params['SB2']['gencal_scale'] = 0.994

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
    if ('LB' in i): # skip over LBs if in SB-only mode
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
# Beam 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mask: 91.10 mJy
# Peak intensity of source: 56.19 mJy/beam
# rms: 1.82e-02 mJy/beam
# Peak SNR: 308.24

listdict, scantimesdict, integrationsdict, integrationtimesdict, integrationtimes, n_spws, minspw, spwsarray = fetch_scan_times(vislist, [data_params['SB1']['field']])
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
               parallel = parallel,

               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -2.0},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

### Plot gain corrections, loop through each
if not skip_plots:
    for i in data_params.keys():
        plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_p' + str(iteration) + '.g'),
               xaxis = 'time',
               yaxis = 'phase',
               gridrows = 8,
               gridcols = 6,
               iteraxis = 'antenna',
               xselfscale = True,
               plotrange = [0, 0, -180, 180])
        input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_SB-only_p0.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 86.95 mJy
# Peak intensity of source: 55.96 mJy/beam
# rms: 1.82e-01 mJy/beam
# Peak SNR: 306.81

# IRAS04169_SB-only_p0_post.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 87.39 mJy
# Peak intensity of source: 57.19 mJy/beam
# rms: 1.57e-01 mJy/beam
# Peak SNR: 365.35

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
               parallel = parallel,

               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -2.5},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

if not skip_plots:
    for i in data_params.keys():
        plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_p' + str(iteration) + '.g'),
               xaxis = 'time',
               yaxis = 'phase',
               gridrows = 8,
               gridcols = 6,
               iteraxis = 'antenna',
               xselfscale = True,
               plotrange = [0, 0, -180, 180])
        input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_SB-only_p0.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 89.21 mJy
# Peak intensity of source: 57.17 mJy/beam
# rms: 1.54e-01 mJy/beam
# Peak SNR: 370.26

# IRAS04169_SB-only_p0_post.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 89.56 mJy
# Peak intensity of source: 64.70 mJy/beam
# rms: 5.72e-02 mJy/beam
# Peak SNR: 1131.17

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
               parallel = parallel,

               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -3.5},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

if not skip_plots:
    for i in data_params.keys():
        plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_p' + str(iteration) + '.g'),
               xaxis = 'time',
               yaxis = 'phase',
               gridrows = 8,
               gridcols = 6,
               iteraxis = 'antenna',
               xselfscale = True,
               plotrange=[0, 0, -180, 180]) 
        input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_SB-only_p2.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 92.19 mJy
# Peak intensity of source: 64.65 mJy/beam
# rms: 3.99e-02 mJy/beam
# Peak SNR: 1618.50

# IRAS04169_SB-only_p2_post.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 92.31 mJy
# Peak intensity of source: 67.09 mJy/beam
# rms: 3.85e-02 mJy/beam
# Peak SNR: 1743.99

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
               parallel = parallel,

               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -3.7},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

if not skip_plots:
    for i in data_params.keys():
        plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_p' + str(iteration) + '.g'),
               xaxis = 'time',
               yaxis = 'phase',
               gridrows = 8,
               gridcols = 6,
               iteraxis = 'antenna',
               xselfscale = True,
               plotrange = [0, 0, -180, 180]) 
        input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_SB-only_p3.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 92.72 mJy
# Peak intensity of source: 66.97 mJy/beam
# rms: 3.51e-02 mJy/beam
# Peak SNR: 1909.06

# IRAS04169_SB-only_p3_post.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 92.82 mJy
# Peak intensity of source: 68.01 mJy/beam
# rms: 3.41e-02 mJy/beam
# Peak SNR: 1992.63

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
               parallel = parallel,
               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -3.7},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

if not skip_plots:
    for i in data_params.keys():
        plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_p' + str(iteration) + '.g'),
               xaxis = 'time',
               yaxis = 'phase',
               gridrows = 8,
               gridcols = 6,
               iteraxis = 'antenna',
               xselfscale = True,
               plotrange = [0, 0, -180, 180]) 
        input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_SB-only_p4.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 93.17 mJy
# Peak intensity of source: 67.90 mJy/beam
# rms: 3.27e-02 mJy/beam
# Peak SNR: 2077.27

# IRAS04169_SB-only_p4_post.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 93.40 mJy
# Peak intensity of source: 69.19 mJy/beam
# rms: 3.32e-02 mJy/beam
# Peak SNR: 2082.34

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
               parallel = parallel,

               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -3.7},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

if not skip_plots:
    for i in data_params.keys():
        plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_p' + str(iteration) + '.g'),
               xaxis = 'time',
               yaxis = 'phase',
               gridrows = 8,
               gridcols = 6,
               iteraxis = 'antenna',
               xselfscale = True,
               plotrange = [0, 0, -180, 180]) 
        input("Press Enter key to advance to next MS/Caltable...")

# IRAS04169_SB-only_p5.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 93.93 mJy
# Peak intensity of source: 69.10 mJy/beam
# rms: 3.19e-02 mJy/beam
# Peak SNR: 2165.00

# IRAS04169_SB-only_p5_post.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 94.16 mJy
# Peak intensity of source: 69.68 mJy/beam
# rms: 3.21e-02 mJy/beam
# Peak SNR: 2173.02

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
               solint = '500s',
               combine = 'spw, scan',
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
               parallel = parallel,

               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_p%s.image.tt0' % iteration, 'scaling': -3.7},
#                  {'file': 'IRAS04169_SB-only_p%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_p%s.mask' % iteration})

if not skip_plots:
   for i in data_params.keys():
       plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_ap' + str(iteration) + '.g'),
              xaxis = 'time',
              yaxis = 'amp',
              gridrows = 8,
              gridcols = 6,
              iteraxis = 'antenna',
              xselfscale = True,
              plotrange = [0, 0, 0, 2])
       input("Press Enter key to advance to next MS/Caltable...")
       # *Check the "phase vs. time" plots, too!

# IRAS04169_SB-only_p6.image.tt0
# Beam: 0.369 arcsec x 0.252 arcsec (-4.41 deg)
# Flux inside disk mass: 94.30 mJy
# Peak intensity of source: 69.61 mJy/beam
# rms: 3.18e-02 mJy/beam
# Peak SNR: 2186.68

# IRAS04169_SB-only_p6_post.image.tt0
# Beam: 0.370 arcsec x 0.253 arcsec (-4.23 deg)
# Flux inside disk mass: 94.16 mJy
# Peak intensity of source: 69.72 mJy/beam
# rms: 3.11e-02 mJy/beam
# Peak SNR: 2239.51

iteration = 7
self_calibrate(prefix,
               data_params,
               selectedVis,
               mode = 'SB-only',
               iteration = iteration,
               selfcalmode = 'ap',
               nsigma = 3.0,
               solint = '300s',
               combine = 'spw, scan',
               noisemasks = [common_mask, noise_annulus],
               SB_contspws = SB_contspws,
               SB_spwmap = SB_spwmap,
               parallel = parallel,

               nterms = 2)

# imview(raster = [{'file': 'IRAS04169_SB-only_ap%s.image.tt0' % iteration, 'scaling': -3.7},
#                  {'file': 'IRAS04169_SB-only_ap%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_ap%s.mask' % iteration})

if not skip_plots:
   for i in data_params.keys():
       plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_ap' + str(iteration) + '.g'),
              xaxis = 'time',
              yaxis = 'amp',
              gridrows = 8,
              gridcols = 6,
              iteraxis = 'antenna',
              xselfscale = True,
              plotrange = [0, 0, 0, 2])
       input("Press Enter key to advance to next MS/Caltable...")
       # *Check the "phase vs. time" plots, too!

# IRAS04169_SB-only_ap7.image.tt0
# Beam: 0.370 arcsec x 0.253 arcsec (-4.23 deg)
# Flux inside disk mass: 94.13 mJy
# Peak intensity of source: 69.65 mJy/beam
# rms: 3.11e-02 mJy/beam
# Peak SNR: 2239.91

# IRAS04169_SB-only_ap7_post.image.tt0
# Beam: 0.371 arcsec x 0.253 arcsec (-4.14 deg)
# Flux inside disk mass: 94.00 mJy
# Peak intensity of source: 69.75 mJy/beam
# rms: 3.11e-02 mJy/beam
# Peak SNR: 2240.38

iteration = 8
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
               parallel = parallel,

               nterms = 2,
               finalimageonly = True)

# imview(raster = [{'file': 'IRAS04169_SB-only_ap%s.image.tt0' % iteration, 'scaling': -3.7},
#                  {'file': 'IRAS04169_SB-only_ap%s.residual.tt0' % iteration, 'scaling': 0.0}],
#        contour = {'file': 'IRAS04169_SB-only_ap%s.mask' % iteration})

if not skip_plots:
    for i in data_params.keys():
        plotms(vis = data_params[i][selectedVis].replace('.ms', '_SB-only_ap' + str(iteration) + '.g'),
               xaxis = 'time',
               yaxis = 'amp',
               gridrows = 8,
               gridcols = 6,
               iteraxis = 'antenna',
               xselfscale = True,
               plotrange = [0, 0, 0, 2])
        input("Press Enter key to advance to next MS/Caltable...")
        # *Check the "phase vs. time" plots, too!

# IRAS04169_SB-only_ap8.image.tt0
# Beam: 0.371 arcsec x 0.253 arcsec (-4.14 deg)
# Flux inside disk mass: 93.96 mJy
# Peak intensity of source: 69.67 mJy/beam
# rms: 3.11e-02 mJy/beam
# Peak SNR: 2241.91

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

    # After scaling,
    # data_params["SB1"]["gencal_scale"] = 1.000
    # data_params["SB2"]["gencal_scale"] = 1.000

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
imsize = 1600
cellsize = '0.025arcsec'

for robust in [-2.0, -1.0, -0.5, +0.0, +0.5, +1.0, +2.0]:
    imagename = prefix + '_SB_continuum_robust_' + str(robust)
    os.system('rm -rf ' + imagename + '*')

    sigma = get_sensitivity(data_params,
                            specmode = 'mfs',
                            imsize = imsize,
                            robust = robust,
                            cellsize = cellsize)

    if robust == 2.0 or robust == 1.0:
        sigma = sigma * 1.75
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

    # pblist = glob.glob('*continuum*.pb.tt0') 
    # os.system('mkdir orig_pbimages')
    # for pbimage in pblist:
    #    os.system('mv ' + pbimage + ' orig_pbimages')
    #    os.system('cp -r temporary.pbfix.pb.tt0 ' + pbimage)
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
