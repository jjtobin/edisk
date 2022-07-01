"""
eDisk data reduction script
This script was written for CASA 6.1.1/6.2
Originally derived from DSHARP reduction scripts

SCRIPT REQUIRES CASA >6.4 TO WORK AROUND GAINCAL BUG

Datasets calibrated (in order of date observed):
SB1: 2019.A.00034.S uid___A002_Xfa2f45_X14835
 
SB2: 2019.A.00034.S uid___A002_Xfa5545_Xa6f

LB1: 2015.1.01512.S uid___A002_Xac5575_X8ed9
     
LB2: 2015.1.01512.S uid___A002_Xac9e3c_X8bb

LB3: 2015.1.01512.S uid___A002_Xc49eba_Xa51

reducer: John Tobin
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
field   = {'SB':'OphIRS63', 'LB':'IRS_63'}
prefix  = 'OphIRS63' 

### always include trailing slashes!!
WD_path = '/lustre/cv/projects/edisk/OphIRS63/'
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
pl_data_params={'SB1': {'vis': SB_path+'uid___A002_Xfa2f45_X14835.ms',
                        'spws': '25,31,29,27,33,35,37',
                        'field': field['SB'],
                        'column': 'corrected'},
                'SB2': {'vis': SB_path+'uid___A002_Xfa5545_Xa6f.ms',
                        'spws': '25,31,29,27,33,35,37',
                        'field': field['SB'],
                        'column': 'corrected'},
                'LB1': {'vis': LB_path+'uid___A002_Xac5575_X8ed9.ms.split.cal',
                        'spws': '17,19,21,23',
                        'field': field['LB'],
                        'column': 'data'},
                'LB2': {'vis': LB_path+'uid___A002_Xac9e3c_X8bb.ms.split.cal',
                        'spws': '17,19,21,23',
                        'field': field['LB'],
                        'column': 'data'},
                'LB3': {'vis': LB_path+'uid___A002_Xc49eba_Xa51.ms.split.cal',
                        'spws': '17,19,21,23',
                        'field': field['LB'],
                        'column': 'data'},
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
                       'flagrange': np.array([[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],
                                              [-5.5,14.5],[-5.5,14.5],[-5.5,14.5],[-5.5,14.5],
                                              [-5.5,14.5],[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {25:0, 27:3, 29:2, 31:1, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2022/06/14/04:15:00~2022/06/14/05:37:00',
                       'contdotdat' : 'SB/cont-OphIRS63.dat'
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
                       'orig_spw_map': {25:0, 27:3, 29:2, 31:1, 33:4, 35:5, 37:6},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3,4,5,6]),  #spws to use for continuum
                       'cont_avg_width':  np.array([480,480,480,480,60,60,60]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2022/06/15/00:46:00~2022/06/14/02:10:00',
                       'contdotdat' : 'SB/cont-OphIRS63.dat'
                      }, 
               'LB1': {'vis' : WD_path+prefix+'_LB1.ms',
                       'name' : 'LB1',
                       'field': field['LB'],
                       'line_spws': np.array([0,0]), # line SPWs, get from listobs
                       'line_freqs': np.array([0,0]), #restfreqs
                       'line_names': ['none','none'], #restfreqs
                       'flagrange': np.array([[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {17:0, 19:1, 21:2, 23:3},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3]),  #spws to use for continuum
                       'cont_avg_width':  np.array([8,8,8,8]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2015/11/07/18:54:29~2015/11/01/20:26:13',
                       #'contdotdat' : 'LB/cont.dat'
                      }, 
               'LB2': {'vis' : WD_path+prefix+'_LB2.ms',
                       'name' : 'LB2',
                       'field': field['LB'],
                       'line_spws': np.array([0,0]), # line SPWs, get from listobs
                       'line_freqs': np.array([0,0]), #restfreqs
                       'line_names': ['none','none'], #restfreqs
                       'flagrange': np.array([[-5.5,14.5],[-5.5,14.5]]),
                       'orig_spw_map': {17:0, 19:1, 21:2, 23:3},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3]),  #spws to use for continuum
                       'cont_avg_width':  np.array([8,8,8,8]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2015/11/08/14:14:25~2015/11/08/15:40:04',
                       #'contdotdat' : 'LB/cont.dat'
                      }, 
               'LB3': {'vis' : WD_path+prefix+'_LB3.ms',
                       'name' : 'LB3',
                       'field': field['LB'],
                       'line_spws': np.array([0,0]), # line SPWs, get from listobs
                       'line_freqs': np.array([0,0]), #restfreqs
                       'line_names': ['none','none'], #restfreqs
                       'flagrange': np.array([[-5.5,-5.5],[-5.5,-5.5]]),
                       'orig_spw_map': {17:0, 19:1, 21:2, 23:3},  # mapping of old spws to new spws (needed for cont.dat to work)
                       'cont_spws':  np.array([0,1,2,3]),  #spws to use for continuum
                       'cont_avg_width':  np.array([8,8,8,8]), #n channels to average; approximately aiming for 30 MHz channels
                       'phasecenter': '',
                       'timerange': '2017/09/18/23:46:54.0~2017/09/19/01:11:28.0',
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
      split(vis=pl_data_params[i]['vis'], outputvis=prefix+'_'+i+'.ms', 
          spw=pl_data_params[i]['spws'], field=pl_data_params[i]['field'],
          datacolumn=pl_data_params[i]['column'],intent='*TARGET*')

### Backup the the flagging state at start of reduction
for i in data_params.keys():
    if not os.path.exists(data_params[i]['vis']+\
            ".flagversions/flags.starting_flags"):
       flagmanager(vis=data_params[i]['vis'], mode = 'save', versionname = 'starting_flags', comment = 'Flag states at start of reduction')

#flag problematic baseline
for i in ['LB1','LB2']:
   flagdata(vis=data_params[i]['vis'],mode='manual',antenna='DA45&DV03')
flagdata(vis=data_params['LB1']['vis'],mode='manual',antenna='DA45&DV10')
#flagdata(vis=data_params['LB3']['vis'],mode='manual',antenna='DA61&DV09')

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
       #data_params[i]['common_dir']='J2000 16h31m35.65742s -24d01m29.946880s'
       #use current position from short baseline
       data_params[i]['common_dir']='J2000 16h31m35.65479s -24d01m30.061490s'
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
      tclean_wrapper(vis=data_params[i]['vis_avg_shift'], imagename=prefix+'_'+i+'_initcont_shifted', sidelobethreshold=2.0, 
            smoothfactor=1.5, scales=LB_scales, nsigma=5.0, robust=0.5, parallel=parallel, 
            uvtaper=[outertaper],nterms=1)
   else:
       tclean_wrapper(vis=data_params[i]['vis_avg_shift'], imagename=prefix+'_'+i+'_initcont_shifted', sidelobethreshold=2.5, 
            smoothfactor=1.5, scales=SB_scales, nsigma=5.0, robust=0.5, parallel=parallel,nterms=1)

for i in data_params.keys():
      print(i)     
      data_params[i]['phasecenter_new']=fit_gaussian(prefix+'_'+i+'_initcont_shifted.image.tt0',\
                                                     region=fit_region,mask=prefix+'_'+i+'_initcont_shifted.mask')
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


data_params["SB1"]["gencal_scale"]=1.000
data_params["SB2"]["gencal_scale"]=1.001
data_params["LB1"]["gencal_scale"]=0.988
data_params["LB2"]["gencal_scale"]=1.023
data_params["LB3"]["gencal_scale"]=1.024

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
   refdata='LB2'
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
#Suggested Solints:
#['inf', 'inf', '30.24s', '12.10s', 'int']
#Suggested Gaincal Combine params:
#['spw,scan', 'spw', 'spw', 'spw', 'spw']
#Suggested nsigma per solint: 
#[27.18272059 15.66751086  9.03040208  5.20492135  3.        ]


#OphIRS63_initial.image.tt0
#Beam 0.376 arcsec x 0.271 arcsec (81.06 deg)
#Flux inside disk mask: 324.40 mJy
#Peak intensity of source: 115.30 mJy/beam
#rms: 2.83e-01 mJy/beam
#Peak SNR: 407.74


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
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=27.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel,combine='scan,spw')


### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

### Make note of key metrics of image in each round
#OphIRS63_SB-only_p0.image.tt0
#Beam 0.376 arcsec x 0.271 arcsec (81.06 deg)
#Flux inside disk mask: 317.69 mJy
#Peak intensity of source: 112.14 mJy/beam
#rms: 2.58e-01 mJy/beam
#Peak SNR: 434.13

#OphIRS63_SB-only_p0_post.image.tt0
#Beam 0.376 arcsec x 0.271 arcsec (81.06 deg)
#Flux inside disk mask: 322.64 mJy
#Peak intensity of source: 114.38 mJy/beam
#rms: 1.25e-01 mJy/beam
#Peak SNR: 912.32

iteration=1
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=16.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

iteration=2
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=9.0,solint='30.24s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#OphIRS63_SB-only_p2.image.tt0
#Beam 0.376 arcsec x 0.271 arcsec (81.06 deg)
#Flux inside disk mask: 316.51 mJy
#Peak intensity of source: 115.63 mJy/beam
#rms: 3.76e-02 mJy/beam
#Peak SNR: 3072.30

#OphIRS63_SB-only_p2_post.image.tt0
#Beam 0.376 arcsec x 0.271 arcsec (81.06 deg)
#Flux inside disk mask: 316.62 mJy
#Peak intensity of source: 116.07 mJy/beam
#rms: 3.67e-02 mJy/beam
#Peak SNR: 3160.24


iteration=3
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=5.0,solint='12.0s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")
#OphIRS63_SB-only_p3.image.tt0
#Beam 0.376 arcsec x 0.271 arcsec (81.06 deg)
#Flux inside disk mask: 316.40 mJy
#Peak intensity of source: 115.98 mJy/beam
#rms: 3.57e-02 mJy/beam
#Peak SNR: 3252.26

#OphIRS63_SB-only_p3_post.image.tt0
#Beam 0.376 arcsec x 0.271 arcsec (81.06 deg)
#Flux inside disk mask: 316.57 mJy
#Peak intensity of source: 116.25 mJy/beam
#rms: 3.54e-02 mJy/beam
#Peak SNR: 3282.84

iteration=4
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='int',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")
#OphIRS63_SB-only_p4.image.tt0
#Beam 0.376 arcsec x 0.271 arcsec (81.06 deg)
#Flux inside disk mask: 316.48 mJy
#Peak intensity of source: 116.15 mJy/beam
#rms: 3.47e-02 mJy/beam
#Peak SNR: 3350.37

#OphIRS63_SB-only_p4_post.image.tt0
#Beam 0.376 arcsec x 0.271 arcsec (81.06 deg)
#Flux inside disk mask: 316.63 mJy
#Peak intensity of source: 116.29 mJy/beam
#rms: 3.45e-02 mJy/beam
#Peak SNR: 3365.93


### Changing self-cal mode here to ap, see use of prevselfcalmode to ensure proper split

iteration=5
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',prevselfcalmode='p',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")
#OphIRS63_SB-only_p5.image.tt0
#Beam 0.376 arcsec x 0.271 arcsec (81.06 deg)
#Flux inside disk mask: 316.61 mJy
#Peak intensity of source: 116.24 mJy/beam
#rms: 3.46e-02 mJy/beam
#Peak SNR: 3362.84

#OphIRS63_SB-only_p5_post.image.tt0
#Beam 0.378 arcsec x 0.272 arcsec (80.61 deg)
#Flux inside disk mask: 315.61 mJy
#Peak intensity of source: 116.61 mJy/beam
#rms: 3.44e-02 mJy/beam
#Peak SNR: 3392.80

iteration=6
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='12.0s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,parallel=parallel)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_SB-only_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key tto advance to next MS/Caltable...")


#OphIRS63_SB-only_ap6.image.tt0
#Beam 0.378 arcsec x 0.272 arcsec (80.61 deg)
#Flux inside disk mask: 315.72 mJy
#Peak intensity of source: 116.55 mJy/beam
#rms: 3.43e-02 mJy/beam
#Peak SNR: 3394.41


#OphIRS63_SB-only_ap6_post.image.tt0
#Beam 0.379 arcsec x 0.272 arcsec (80.26 deg)
#Flux inside disk mask: 314.99 mJy
#Peak intensity of source: 116.84 mJy/beam
#rms: 3.45e-02 mJy/beam
#Peak SNR: 3387.38


### Make the final image, will not run another self-calibration
iteration=7
self_calibrate(prefix,data_params,selectedVis,mode='SB-only',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='12.1s',
               noisemasks=[common_mask,noise_annulus],SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,
               parallel=parallel,finalimageonly=True)

#OphIRS63_SB-only_ap7.image.tt0
#Beam 0.379 arcsec x 0.272 arcsec (80.26 deg)
#Flux inside disk mask: 315.06 mJy
#Peak intensity of source: 116.77 mJy/beam
#rms: 3.44e-02 mJy/beam
#Peak SNR: 3390.45

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
LB_spwmap=[0,0,0,0]
LB_contspws = '' 


### Make a list of EBs to image
vislist=[]
fieldlist=[]
for i in data_params.keys():
   vislist.append(data_params[i][selectedVis])
   fieldlist.append(data_params[i]['field'])

### Initial map to assess DR
tclean_wrapper(vis=vislist, imagename=prefix+'_initial_LB+SB',cellsize='0.003arcsec',imsize=3000, 
               scales=LB_scales, sidelobethreshold=2.0, smoothfactor=1.5, nsigma=3.0, 
               noisethreshold=3.0, robust=0.5, parallel=parallel, nterms=1,
               phasecenter=data_params['LB1']['common_dir'].replace('J2000','ICRS'))
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

#OphIRS63_initial_LB+SB.image.tt0
#Beam 0.048 arcsec x 0.028 arcsec (52.83 deg)
#Flux inside disk mask: 365.19 mJy
#Peak intensity of source: 5.64 mJy/beam
#rms: 2.51e-02 mJy/beam
#Peak SNR: 224.86

#Suggested Solints:
#['inf', 'inf', '12.10s', 'int']
#Suggested Gaincal Combine params:
#['spw,scan', 'spw', 'spw', 'spw']
#Suggested nsigma per solint: 
#[14.99076183  8.76845117  5.12887449  3.        ]


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
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=15.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,
               combine='spw,scan',parallel=parallel,smoothfactor=2.0,imsize=3000)


### Plot gain corrections, loop through each
if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#OphIRS63_LB+SB_p0.image.tt0
#Beam 0.048 arcsec x 0.028 arcsec (52.81 deg)
#Flux inside disk mask: 479.00 mJy
#Peak intensity of source: 5.85 mJy/beam
#rms: 2.35e-02 mJy/beam
#Peak SNR: 248.59

#OphIRS63_LB+SB_p0_post.image.tt0
#Beam 0.048 arcsec x 0.028 arcsec (52.81 deg)
#Flux inside disk mask: 478.84 mJy
#Peak intensity of source: 5.78 mJy/beam
#rms: 2.30e-02 mJy/beam
#Peak SNR: 251.82




iteration=1
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=9.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,combine='spw',
               parallel=parallel,smoothfactor=2.0,imsize=3000)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True, plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#OphIRS63_LB+SB_p1.image.tt0
#Beam 0.048 arcsec x 0.028 arcsec (52.81 deg)
#Flux inside disk mask: 419.55 mJy
#Peak intensity of source: 5.67 mJy/beam
#rms: 2.22e-02 mJy/beam
#Peak SNR: 254.95

#OphIRS63_LB+SB_p1_post.image.tt0
#Beam 0.048 arcsec x 0.028 arcsec (52.81 deg)
#Flux inside disk mask: 418.93 mJy
#Peak intensity of source: 7.23 mJy/beam
#rms: 1.61e-02 mJy/beam
#Peak SNR: 449.37


iteration=2
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=5.1,solint='12.0s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,
               combine='spw',smoothfactor=2.0,imsize=3000)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")


#OphIRS63_LB+SB_p2.image.tt0
#Beam 0.048 arcsec x 0.028 arcsec (52.81 deg)
#Flux inside disk mask: 368.88 mJy
#Peak intensity of source: 6.96 mJy/beam
#rms: 1.49e-02 mJy/beam
#Peak SNR: 465.95

#OphIRS63_LB+SB_p2_post.image.tt0
#Beam 0.048 arcsec x 0.028 arcsec (52.81 deg)
#Flux inside disk mask: 368.55 mJy
#Peak intensity of source: 7.70 mJy/beam
#rms: 1.32e-02 mJy/beam
#Peak SNR: 581.80

iteration=3
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='p',nsigma=3.0,solint='int',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,imsize=3000)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_p'+str(iteration)+'.g'),
               xaxis='time', yaxis='phase',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,-180,180]) 
       input("Press Enter key to advance to next MS/Caltable...")

#OphIRS63_LB+SB_p3.image.tt0
#Beam 0.048 arcsec x 0.028 arcsec (52.81 deg)
#Flux inside disk mask: 349.79 mJy
#Peak intensity of source: 7.57 mJy/beam
#rms: 1.29e-02 mJy/beam
#Peak SNR: 588.38

#OphIRS63_LB+SB_p3_post.image.tt0
#Beam 0.048 arcsec x 0.028 arcsec (52.81 deg)
#Flux inside disk mask: 349.00 mJy
#Peak intensity of source: 7.93 mJy/beam
#rms: 1.30e-02 mJy/beam
#Peak SNR: 611.59

### Changing self-cal mode here to ap, see use of prevselfcalmode to ensure proper split
iteration=4
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',prevselfcalmode='p',nsigma=3.0,solint='300s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,combine='scan,spw',smoothfactor=2.0,imsize=3000)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")

#OphIRS63_LB+SB_p4.image.tt0
#Beam 0.048 arcsec x 0.028 arcsec (52.81 deg)
#Flux inside disk mask: 348.27 mJy
#Peak intensity of source: 7.93 mJy/beam
#rms: 1.29e-02 mJy/beam
#Peak SNR: 613.53

#OphIRS63_LB+SB_p4_post.image.tt0
#Beam 0.049 arcsec x 0.029 arcsec (54.05 deg)
#Flux inside disk mask: 339.69 mJy
#Peak intensity of source: 8.21 mJy/beam
#rms: 1.28e-02 mJy/beam
#Peak SNR: 638.72


#OphIRS63_LB+SB_p6.image.tt0
#Beam 0.041 arcsec x 0.025 arcsec (54.00 deg)
#Flux inside disk mask: 314.86 mJy
#Peak intensity of source: 6.76 mJy/beam
#rms: 1.31e-02 mJy/beam
#Peak SNR: 515.36

#OphIRS63_LB+SB_p6_post.image.tt0
#Beam 0.041 arcsec x 0.025 arcsec (55.82 deg)
#Flux inside disk mask: 306.54 mJy
#Peak intensity of source: 7.02 mJy/beam
#rms: 1.33e-02 mJy/beam
#Peak SNR: 528.87

iteration=5
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='150s',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,combine='scan,spw',smoothfactor=2.0,imsize=3000)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")


#OphIRS63_LB+SB_ap5.image.tt0
#Beam 0.049 arcsec x 0.029 arcsec (54.05 deg)
#Flux inside disk mask: 342.59 mJy
#Peak intensity of source: 8.19 mJy/beam
#rms: 1.28e-02 mJy/beam
#Peak SNR: 639.41

#OphIRS63_LB+SB_ap5_post.image.tt0
#Beam 0.049 arcsec x 0.029 arcsec (53.81 deg)
#Flux inside disk mask: 338.45 mJy
#Peak intensity of source: 8.57 mJy/beam
#rms: 1.29e-02 mJy/beam
#Peak SNR: 665.90


#OphIRS63_LB+SB_ap7.image.tt0
#Beam 0.041 arcsec x 0.025 arcsec (55.82 deg)
#Flux inside disk mask: 300.15 mJy
#Peak intensity of source: 7.00 mJy/beam
#rms: 1.33e-02 mJy/beam
#Peak SNR: 524.34

#OphIRS63_LB+SB_ap7_post.image.tt0
#Beam 0.041 arcsec x 0.025 arcsec (55.49 deg)
#Flux inside disk mask: 299.21 mJy
#Peak intensity of source: 7.12 mJy/beam
#rms: 1.35e-02 mJy/beam
#Peak SNR: 526.92

iteration=6
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],
               SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,combine='scan,spw',smoothfactor=2.0,imsize=3000)

if not skip_plots:
   for i in data_params.keys():
       plotms(vis=data_params[i][selectedVis].replace('.ms','_LB+SB_ap'+str(iteration)+'.g'), xaxis='time',
              yaxis='amp',gridrows=4,gridcols=1,iteraxis='antenna', xselfscale=True,plotrange=[0,0,0,2])
       input("Press Enter key to advance to next MS/Caltable...")
#OphIRS63_LB+SB_ap6.image.tt0
#Beam 0.049 arcsec x 0.029 arcsec (53.81 deg)
#Flux inside disk mask: 340.38 mJy
#Peak intensity of source: 8.54 mJy/beam
#rms: 1.28e-02 mJy/beam
#Peak SNR: 665.08

#OphIRS63_LB+SB_ap6_post.image.tt0
#Beam 0.047 arcsec x 0.031 arcsec (61.84 deg)
#Flux inside disk mask: 335.21 mJy
#Peak intensity of source: 8.79 mJy/beam
#rms: 1.28e-02 mJy/beam
#Peak SNR: 685.09

iteration=7
self_calibrate(prefix,data_params,selectedVis,mode='LB+SB',iteration=iteration,selfcalmode='ap',nsigma=3.0,solint='inf',
               noisemasks=[common_mask,noise_annulus],SB_contspws=SB_contspws,SB_spwmap=SB_spwmap,
               LB_contspws=LB_contspws,LB_spwmap=LB_spwmap,parallel=parallel,smoothfactor=2.0,finalimageonly=True,imsize=3000)


#OphIRS63_LB+SB_ap7.image.tt0
#Beam 0.047 arcsec x 0.031 arcsec (61.84 deg)
#Flux inside disk mask: 340.79 mJy
#Peak intensity of source: 8.86 mJy/beam
#rms: 1.28e-02 mJy/beam
#Peak SNR: 690.17


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
   export_MS(data_params[i]['vis_avg_selfcal'])
   export_vislist.append(data_params[i]['vis_avg_selfcal'].replace('.ms','.vis.npz'))


### Plot deprojected visibility profiles for all data together """
plot_deprojected(export_vislist,
                     fluxscale=[1.0]*len(export_vislist), PA=PA, incl=incl, 
                     show_err=False,outfile='amp-vs-uv-distance-post-selfcal.png')

#################### MANUALLY SET THIS ######################
refdata='SB1'

reference=prefix+'_'+refdata+'_initcont_shift_rescaled_LB+SB_ap7.vis.npz'
for i in data_params.keys():
   print(i)
   if i != refdata:
      data_params[i]['gencal_scale_selfcal']=estimate_flux_scale(reference=reference, 
                        comparison=prefix+'_'+i+'_initcont_shift_rescaled_LB+SB_ap7.vis.npz', 
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
            cellsize=cell, imsize=imsize,phasecenter=data_params['LB1']['common_dir'].replace('J2000','ICRS'))

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
                   phasecenter=data_params['LB1']['common_dir'].replace('J2000', 'ICRS'))
    imagename = imagename+'.image.tt0'
    exportfits(imagename=imagename, fitsimage=imagename+'.fits',
               overwrite=True, dropdeg=True)

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
###############################################################
################ CLEANUP AND FITS CONVERSION ##################
###############################################################
selectedVis='vis_shift'

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






