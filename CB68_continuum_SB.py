"""
eDisk data reduction script
This script was written for CASA 6.1.1/6.2
Originally derived from DSHARP reduction scripts


Datasets calibrated (in order of date observed):
SB1: 2019.A.00034.S
vis=uid___A002_Xfa2f45_X14241.ms
     
reducer: Miyu Kido
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
field   = 'CB68'
prefix  = 'CB68' 

### always include trailing slashes!!
WD_path = '/lustre/cv/projects/edisk/CB68/'
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
pl_data_params={'SB1': {'vis': SB_path+'uid___A002_Xfa2f45_X14241.ms', # M:DDT data
                        'spws': '25,31,29,27,33,35,37',
                        'field': field,
                        'column': 'corrected'},
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
                       'timerange': '2022/06/14/03:12:41.5~2022/06/14/04:13:27.6',
                       'contdotdat' : 'SB/cont.dat'
                      }, 
               }

'''    
No LB yet                 
               'LB1': {'vis' : LB1_path,
                       'name' : 'LB1',
                       'field' : 'Ced110IRS4',
                       'line_spws': np.array([]), # CO SPWs 
                       'line_freqs': np.array([]),
                       'flagrange': np.array([]), 
                       'cont_spws':  np.array([]), # CO SPWs 
                       'cont_avg_width':  np.array([]), 
                       'phasecenter': ' ',
                       'timerange': '2015/11/01/00:00:00~2015/11/02/00:00:00',
                      }

               }

'''
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

mask_ra  = '16h57m19.644s'.replace('h',':').replace('m',':').replace('s','')
mask_dec = '-16d09m23.986s'.replace('d','.').replace('m','.').replace('s','')
mask_pa  = 81.0 	# position angle of mask in degrees
mask_maj = 0.70	# semimajor axis of mask in arcsec
mask_min = 0.50 	# semiminor axis of mask in arcsec
fit_region = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
              (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)

for i in data_params.keys():
       print(i)
       data_params[i]['phasecenter']=fit_gaussian(prefix+'_'+i+'_initial_cont.image.tt0', region=fit_region,mask=prefix+'_'+i+'_initial_cont.mask')

'''
SB1
16h57m19.644179s -16d09m23.99378s
#Peak of Gaussian component identified with imfit: ICRS 16h57m19.644179s -16d09m23.99378s
16h57m19.644179s -16d09m23.99378s
Separation: radian = 7.25918e-08, degrees = 0.000004 = 4.15921e-06, arcsec = 0.014973 = 0.0149731
#Peak in J2000 coordinates: 16:57:19.64488, -016:09:23.982726
#PA of Gaussian component: 47.91 deg
#Inclination of Gaussian component: 60.75 deg
#Pixel coordinates of peak: x = 797.965 y = 796.842
'''
### Check phase center fits in viewer, tf centers appear too shifted from the Gaussian fit, 
### manually set the phase center dictionary entry by eye

""" The emission centers are slightly misaligned.  So we split out the 
    individual executions, shift the peaks to the phase center, and reassign 
    the phase centers to a common direction. """

### Set common direction for each EB using one as reference (typically best looking LB image)

for i in data_params.keys():
       #################### MANUALLY SET THIS ######################
       data_params[i]['common_dir']='J2000 16h57m19.644s -16d09m23.994s'

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
'''
SB1
16h57m19.643995s -16d09m23.99398s
#Peak of Gaussian component identified with imfit: J2000 16h57m19.643995s -16d09m23.99398s
#PA of Gaussian component: 47.92 deg
#Inclination of Gaussian component: 60.74 deg
#Pixel coordinates of peak: x = 800.034 y = 799.998
Phasecenter new:  16h57m19.643995s -16d09m23.99398s
Phasecenter old:  16h57m19.64488s -016d09m23.982726s
''' 
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
refdata='SB1'

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

#for i in data_params.keys():
#   data_params[i]['gencal_scale']=1.0


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

#data_params["SB1"]["gencal_scale"]=0.956
#data_params["LB1"]["gencal_scale"]=1.011
#data_params["LB2"]["gencal_scale"]=1.003
#data_params["LB3"]["gencal_scale"]=1.000



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
                     show_err=False,outfile='scaling_amp-vs-uv-distance-pre-selfcal.png')
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
mask_pa  = 81.0 	# position angle of mask in degrees
mask_maj = 0.7	# semimajor axis of mask in arcsec
mask_min = 0.5 	# semiminor axis of mask in arcsec

common_mask = 'ellipse[[%s, %s], [%.1farcsec, %.1farcsec], %.1fdeg]' % \
              (mask_ra, mask_dec, mask_maj, mask_min, mask_pa)
""" Define a noise annulus, measure the peak SNR in map """
noise_annulus = "annulus[[%s, %s],['%.2farcsec', '8.0arcsec']]" % \
                (mask_ra, mask_dec, 2.0*mask_maj) 


###############################################################
###################### SELF-CALIBRATION #######################
###############################################################
fieldlist=[]
for i in data_params.keys():
      if ('LB' in i): # skip over LB EBs if in SB-only mode
         continue
      fieldlist.append(data_params[i]['field'])


### Initial map to assess DR
tclean_wrapper(vis=vislist, imagename=prefix+'_initial', 
               scales=SB_scales, sidelobethreshold=2.0, smoothfactor=1.5, nsigma=3.0, 
               noisethreshold=3.0, robust=0.5, parallel=parallel, imsize=1600,cellsize='0.025arcsec',nterms=1,
               phasecenter=data_params['SB1']['common_dir'].replace('J2000','ICRS'))
initial_SNR,initial_RMS=estimate_SNR(prefix+'_initial.image.tt0', disk_mask=common_mask, 
                        noise_mask=noise_annulus)
'''
#CB68_initial.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 50.05 mJy
#Peak intensity of source: 39.31 mJy/beam
#rms: 1.45e-01 mJy/beam
#Peak SNR: 271.48
'''
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
'''
Suggested Solints:
['inf', 'inf', '90.72s', '42.34s', '18.14s', 'int']
Suggested Gaincal Combine params:
['spw,scan', 'spw', 'spw', 'spw', 'spw', 'spw']
Suggested nsigma per solint: 
[18.09834166 12.63383722  8.81925239  6.15642036  4.29758782  3.        ]
'''

### Image produced by iter 0 has not selfcal applied, it's used to set the initial model
### only images >0 have self-calibration applied

### Run self-calibration command set
### 0. Split off corrected data from previous selfcal iteration (except iteration 0)
### 1. Image data to specified nsigma depth, set model column
### 2. Calculate self-cal gain solutions
### 3. Apply self-cal gain solutions to MS
### 4. Check S/N before and after

############# USERS MAY NEED TO ADJUST NSIGMA AND SOLINT FOR EACH SELF-CALIBRATION ITERATION ##############
iteration=0
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=18.1,solint='inf',
               noisemasks=[common_mask,noise_annulus],combine='spw,scan',
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)
#CB68_SB-only_p0.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 51.08 mJy
#Peak intensity of source: 39.48 mJy/beam
#rms: 1.57e-01 mJy/beam
#Peak SNR: 250.90

#CB68_SB-only_p0_post.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 52.77 mJy
#Peak intensity of source: 40.83 mJy/beam
#rms: 6.50e-02 mJy/beam
#Peak SNR: 628.14


### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

iteration=1
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=12.6,solint='inf',
               noisemasks=[common_mask,noise_annulus],combine='spw',
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")
       
#CB68_SB-only_p1.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 52.40 mJy
#Peak intensity of source: 40.98 mJy/beam
#rms: 5.47e-02 mJy/beam
#Peak SNR: 748.52

#CB68_SB-only_p1_post.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 52.46 mJy
#Peak intensity of source: 41.69 mJy/beam
#rms: 3.93e-02 mJy/beam
#Peak SNR: 1061.15

iteration=2
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=8.8,solint='90.72s',
               noisemasks=[common_mask,noise_annulus],combine='spw',
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#CB68_SB-only_p2.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 52.34 mJy
#Peak intensity of source: 41.68 mJy/beam
#rms: 3.78e-02 mJy/beam
#Peak SNR: 1102.70

#CB68_SB-only_p2_post.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 52.34 mJy
#Peak intensity of source: 41.89 mJy/beam
#rms: 3.77e-02 mJy/beam
#Peak SNR: 1110.59


iteration=3
#self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=6.2,solint='42.34s',
#               noisemasks=[common_mask,noise_annulus],combine='spw',
#               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=6.2,solint='41.34s',
               noisemasks=[common_mask,noise_annulus],combine='spw',
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)               
#CB68_SB-only_p3.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)                                                                                                                   
#Flux inside disk mask: 52.28 mJy
#Peak intensity of source: 41.86 mJy/beam
#rms: 3.72e-02 mJy/beam
#Peak SNR: 1126.38

#CB68_SB-only_p3_post.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 52.29 mJy
#Peak intensity of source: 42.08 mJy/beam
#rms: 3.72e-02 mJy/beam
#Peak SNR: 1131.02

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")



iteration=4
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=4.3,solint='18.14s',
               noisemasks=[common_mask,noise_annulus],combine='spw',
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")
       
#CB68_SB-only_p4.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 52.24 mJy
#Peak intensity of source: 42.04 mJy/beam
#rms: 3.67e-02 mJy/beam
#Peak SNR: 1145.53

#CB68_SB-only_p4_post.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 52.28 mJy
#Peak intensity of source: 42.21 mJy/beam
#rms: 3.69e-02 mJy/beam
#Peak SNR: 1143.00

iteration=5
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='int',
               noisemasks=[common_mask,noise_annulus],combine='spw',
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")
#CB68_SB-only_p5.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 52.28 mJy
#Peak intensity of source: 42.17 mJy/beam
#rms: 3.63e-02 mJy/beam
#Peak SNR: 1161.89

#CB68_SB-only_p5_post.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 52.36 mJy
#Peak intensity of source: 42.32 mJy/beam
#rms: 3.67e-02 mJy/beam
#Peak SNR: 1152.40
  
### Changing self-cal mode here to ap, see use of prevselfcalmode to ensure proper split

iteration=6
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',prevselfcalmode='p',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")

#CB68_SB-only_p6.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 52.35 mJy
#Peak intensity of source: 42.27 mJy/beam
#rms: 3.64e-02 mJy/beam
#Peak SNR: 1160.24

#CB68_SB-only_p6_post.image.tt0
#Beam 0.374 arcsec x 0.274 arcsec (82.17 deg)
#Flux inside disk mask: 51.74 mJy
#Peak intensity of source: 42.54 mJy/beam
#rms: 3.65e-02 mJy/beam
#Peak SNR: 1166.73

#SNR hardly improved, so stop here
### Make the final image, will not run another self-calibration
iteration=7
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,
               parallel=parallel,finalimageonly=True)

#CB68_SB-only_p6.image.tt0
#Beam 0.371 arcsec x 0.272 arcsec (82.95 deg)
#Flux inside disk mask: 52.35 mJy
#Peak intensity of source: 42.27 mJy/beam
#rms: 3.64e-02 mJy/beam
#Peak SNR: 1160.24

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

if not skip_plots:
    ### Plot deprojected visibility profiles for all data together """
    plot_deprojected(export_vislist,
                     fluxscale=[1.0]*len(export_vislist), PA=PA, incl=incl, 
                     show_err=False,outfile='amp-vs-uv-distance-post-selfcal.png')

#################### MANUALLY SET THIS ######################
refdata='SB1'

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

cellsize='0.025arcsec'
imsize=1600
for robust in [2.0,1.0,0.5,0.0,-0.5,-1.0,-2.0]:
    imagename=prefix+'_SB_continuum_robust_'+str(robust)
    os.system('rm -rf '+imagename+'*')

    sigma = get_sensitivity(data_params, specmode='mfs',robust=robust,cellsize=cellsize, imsize=imsize)
    
    if robust == 2.0 or robust == 1.0:
       sigma=sigma*1.75
    tclean_wrapper(vis=vislist, imagename=imagename, sidelobethreshold=2.0, 
            smoothfactor=1.5, scales=scales, threshold=3.0*sigma, 
            noisethreshold=3.0, robust=robust, parallel=parallel, 
            cellsize=cellsize, imsize=imsize,phasecenter=data_params['SB1']['common_dir'].replace('J2000','ICRS'))

    imagename=imagename+'.image.tt0'
    exportfits(imagename=imagename, fitsimage=imagename+'.fits',overwrite=True,dropdeg=True)
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





