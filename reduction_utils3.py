"""
Functions useful for data reduction
Derived from DSHARP reduction utils
modified for eDisk project

"""
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy
from casatools import table as tbtool
from casatools import ms as mstool
def LSRKvel_to_chan(msfile, field, spw, restfreq, LSRKvelocity):
    """
    Identifies the channel(s) corresponding to input LSRK velocities. 
    Useful for choosing which channels to split out or flag if a line is expected to be present

    Parameters
    ==========
    msfile: Name of measurement set (string)
    spw: Spectral window number (int)
    obsid: Observation ID corresponding to the selected spectral window 
    restfreq: Rest frequency in Hz (float)
    LSRKvelocity: input velocity in LSRK frame in km/s (float or array of floats)

    Returns
    =======
    Channel number most closely corresponding to input LSRK velocity 
    """
    cc = 299792458. #speed of light in m/s

    tb.open(msfile)
    spw_col = tb.getcol('DATA_DESC_ID')
    obs_col = tb.getcol('OBSERVATION_ID')

    tb.close()
    obsid = np.unique(obs_col[np.where(spw_col==spw)]) 
    
    tb.open(msfile+'/SPECTRAL_WINDOW')
    chanfreqs = tb.getcol('CHAN_FREQ', startrow = spw, nrow = 1)
    tb.close()
    tb.open(msfile+'/FIELD')
    fieldnames = tb.getcol('NAME')
    tb.close()
    tb.open(msfile+'/OBSERVATION')
    obstime = np.squeeze(tb.getcol('TIME_RANGE', startrow = obsid, nrow = 1))[0]
    tb.close()
    nchan = len(chanfreqs)
    ms.open(msfile)
    
    lsrkfreqs = ms.cvelfreqs(spwids = [spw], fieldids = int(np.where(fieldnames==field)[0][0]), mode = 'channel', nchan = nchan, obstime = str(obstime)+'s', start = 0, outframe = 'LSRK')
    chanvelocities = (restfreq-lsrkfreqs)/restfreq*cc/1.e3 #converted to LSRK velocities in km/s
    ms.close()
    if type(LSRKvelocity)==np.ndarray:
        outchans = np.zeros_like(LSRKvelocity)
        for i in range(len(LSRKvelocity)):
            outchans[i] = np.argmin(np.abs(chanvelocities - LSRKvelocity[i]))
        return outchans
    else:
        return np.argmin(np.abs(chanvelocities - LSRKvelocity))

def LSRKfreq_to_chan(msfile, field, spw, LSRKfreq):
    """
    Identifies the channel(s) corresponding to input LSRK frequencies. 
    Useful for choosing which channels to split out or flag if a line has been identified by the pipeline.

    Parameters
    ==========
    msfile: Name of measurement set (string)
    spw: Spectral window number (int)
    obsid: Observation ID corresponding to the selected spectral window 
    restfreq: Rest frequency in Hz (float)
    LSRKvelocity: input velocity in LSRK frame in km/s (float or array of floats)

    Returns
    =======
    Channel number most closely corresponding to input LSRK frequency.
    """
    tb.open(msfile)
    spw_col = tb.getcol('DATA_DESC_ID')
    obs_col = tb.getcol('OBSERVATION_ID')

    tb.close()
    obsid = np.unique(obs_col[np.where(spw_col==spw)]) 
    
    tb.open(msfile+'/SPECTRAL_WINDOW')
    chanfreqs = tb.getcol('CHAN_FREQ', startrow = spw, nrow = 1)
    tb.close()
    tb.open(msfile+'/FIELD')
    fieldnames = tb.getcol('NAME')
    tb.close()
    tb.open(msfile+'/OBSERVATION')
    obstime = np.squeeze(tb.getcol('TIME_RANGE', startrow = obsid, nrow = 1))[0]
    tb.close()
    nchan = len(chanfreqs)
    ms.open(msfile)
    
    lsrkfreqs = ms.cvelfreqs(spwids = [spw], fieldids = int(np.where(fieldnames==field)[0][0]), mode = 'channel', nchan = nchan, \
            obstime = str(obstime)+'s', start = 0, outframe = 'LSRK') / 1e9
    ms.close()

    if type(LSRKfreq)==np.ndarray:
        outchans = np.zeros_like(LSRKfreq)
        for i in range(len(LSRKfreq)):
            outchans[i] = np.argmin(np.abs(lsrkfreqs - LSRKfreq[i]))
        return outchans
    else:
        return np.argmin(np.abs(lsrkfreqs - LSRKfreq))

def parse_contdotdat(contdotdat_file, orig_spw_map):
    """
    Parses the cont.dat file that includes line emission automatically identified by the ALMA pipeline.

    Parameters
    ==========
    msfile: Name of the cont.dat file (string)

    Returns
    =======
    Dictionary with the boundaries of the frequency range including line emission. The dictionary keys correspond to the spectral windows identified 
    in the cont.dat file, and the entries include numpy arrays with shape (nline, 2), with the 2 corresponding to min and max frequencies identified.
    """
    f = open(contdotdat_file,'r')
    lines = f.readlines()
    f.close()

    while '\n' in lines:
        lines.remove('\n')

    contdotdat = {}
    for i, line in enumerate(lines):
        if 'Field' in line:
            continue
        elif 'SpectralWindow' in line:
            spw = int(line.split()[-1])
            contdotdat[orig_spw_map[spw]] = []
        else:
            contdotdat[orig_spw_map[spw]] += [line.split()[0].split("G")[0].split("~")]

    for spw in contdotdat:
        contdotdat[spw] = np.array(contdotdat[spw], dtype=float)

    return contdotdat

def flagchannels_from_contdotdat(ms_dict):
    """
    Generates a string with the list of lines identified by the cont.dat file from the ALMA pipeline, that need to be flagged.

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set

    Returns
    =======
    String of channels to be flagged, in a format that can be passed to the spw parameter in CASA's flagdata task. 
    """
    contdotdat = parse_contdotdat(ms_dict['contdotdat'], ms_dict['orig_spw_map'])

    flagchannels_string = ''
    for j,spw in enumerate(contdotdat):
        flagchannels_string += '%d:' % (spw)

        tb.open(ms_dict['vis']+'/SPECTRAL_WINDOW')
        nchan = tb.getcol('CHAN_FREQ', startrow = spw, nrow = 1).size
        tb.close()

        chans = np.array([])
        for k in range(contdotdat[spw].shape[0]):
            print(spw, contdotdat[spw][k])

            chans = np.concatenate((LSRKfreq_to_chan(ms_dict['vis'], ms_dict['field'], spw, contdotdat[spw][k]),chans))

            """
            if flagchannels_string == '':
                flagchannels_string+='%d:%d~%d' % (spw, np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
            else:
                flagchannels_string+=', %d:%d~%d' % (spw, np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
            """

        chans = np.sort(chans)

        flagchannels_string += '0~%d;' % (chans[0])
        for i in range(1,chans.size-1,2):
            flagchannels_string += '%d~%d;' % (chans[i], chans[i+1])
        flagchannels_string += '%d~%d, ' % (chans[-1], nchan-1)

    print("# Flagchannels input string for %s from cont.dat file: \'%s\'" % (ms_dict['name'], flagchannels_string))

    return flagchannels_string



def get_flagchannels(ms_dict, output_prefix):
    """
    Identify channels to flag based on provided velocity range of the line emission

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set
    output_prefix: Prefix for all output file names (string)
    velocity_range: Velocity range (in km/s) over which line emission has been identified, in the format np.array([min_velocity, max_velocity]) 
    velocity_range: turned into part of the ms dictionary to specify individual ranges around particular lines
    Returns
    =======
    String of channels to be flagged, in a format that can be passed to the spw parameter in CASA's flagdata task. 
    """
    flagchannels_string = ''
    for spw in np.unique(ms_dict['line_spws']):
        flagchannels_string += '%d:' % (spw)

        for count, j in enumerate(np.where(ms_dict['line_spws'] == spw)[0]):
            print(ms_dict['flagrange'][j],j)
            if count==0:
                chans = LSRKvel_to_chan(ms_dict['vis'], ms_dict['field'], spw, ms_dict['line_freqs'][j] , \
                        ms_dict['flagrange'][j]).reshape((1,ms_dict['flagrange'][j].size))
                #flagchannels_string+='%d~%d' % (np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
            else:
                chans = np.concatenate((LSRKvel_to_chan(ms_dict['vis'], ms_dict['field'], spw, ms_dict['line_freqs'][j] , \
                        ms_dict['flagrange'][j]).reshape((1,ms_dict['flagrange'][j].size)), chans), axis=0)
                #flagchannels_string+=';%d~%d' % (np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))

        # Check that there's no overlap in their ranges.

        chans.sort(axis=1)

        chans = list(chans[chans[:,0].argsort(),:].flatten())

        i = 1
        while True:
            if i >= len(chans)-1 or len(chans) == 2:
                break

            if chans[i] > chans[i+1]:
                chans.pop(i + 1)
                if chans[i+1] > chans[i]:
                    chans.pop(i)
                else:
                    chans.pop(i+1)
            else:
                i += 2

        # Now add these ranges to the string.

        for i in range(0,len(chans),2):
            if i == 0:
                flagchannels_string+='%d~%d' % (chans[i], chans[i+1])
            else:
                flagchannels_string+=';%d~%d' % (chans[i], chans[i+1])
        flagchannels_string += ", "

    if 'contdotdat' in ms_dict:
        print("# cont.dat file found, adding lines identified by the ALMA pipeline.")
        contdot_dat_flagchannels_string = flagchannels_from_contdotdat(ms_dict)

        # Merge the two strings together properly. First create (N,2) arrays of (start flagging, stop flagging) channels and store them in
        # dictionaries where the keys are the SPWs.

        basic_dict = dict([line.split(":") for line in flagchannels_string.split()])
        for spw in basic_dict: 
            basic_dict[spw] = basic_dict[spw].split(',')[0].split(';')
            for i in range(len(basic_dict[spw])): 
                basic_dict[spw][i] = basic_dict[spw][i].split('~') 
            basic_dict[spw] = numpy.array(basic_dict[spw], dtype=int) 

        cont_dict = dict([line.split(":") for line in contdot_dat_flagchannels_string.split()])
        for spw in cont_dict:
            cont_dict[spw] = cont_dict[spw].split(',')[0].split(';')
            for i in range(len(cont_dict[spw])):
                cont_dict[spw][i] = cont_dict[spw][i].split('~')
            cont_dict[spw] = numpy.array(cont_dict[spw], dtype=int)

        # Now merge those two dictionaries into one.

        keys = np.unique(np.concatenate((np.array(list(basic_dict.keys())), np.array(list(cont_dict.keys())))))
        merged_dict = {}
        for key in keys:
            if key in basic_dict and key not in cont_dict:
                merged_dict[key] = basic_dict[key].flatten()
            elif key in cont_dict and key not in basic_dict:
                merged_dict[key] = cont_dict[key].flatten()
            else:
                merged_dict[key] = numpy.concatenate((basic_dict[key],cont_dict[key]), axis=0)

                # Now make sure we remove any overlap in (start, stop) channels. E.g. if we have (0, 10) and (6, 11) we want to replace that with 
                # just (0, 11).

                chans = list(merged_dict[key][merged_dict[key][:,0].argsort(),:].flatten())

                i = 1
                while True:
                    if i >= len(chans)-1 or len(chans) == 2:
                        break

                    if chans[i] > chans[i+1]:
                        chans.pop(i + 1)
                        if chans[i+1] > chans[i]:
                            chans.pop(i)
                        else:
                            chans.pop(i+1)
                    else:
                        i += 2

                merged_dict[key] = chans

        # Finally, recreate the string.

        flagchannels_string = ''
        for key in merged_dict.keys():
            flagchannels_string += key+":"

            for i in range(0,len(merged_dict[key]),2):
                if i > 0:
                    flagchannels_string += ';'
                flagchannels_string += '%d~%d' % (merged_dict[key][i], merged_dict[key][i+1])

            flagchannels_string += ", "
    flagchannels_string=flagchannels_string[:-2]
    print("# Flagchannels input string for %s: \'%s\'" % (ms_dict['name'], flagchannels_string))

    return flagchannels_string

def get_flagchannels_withcont(ms_dict, output_prefix):
    """
    Identify channels to flag based on provided velocity range of the line emission

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set
    output_prefix: Prefix for all output file names (string)
    velocity_range: Velocity range (in km/s) over which line emission has been identified, in the format np.array([min_velocity, max_velocity]) 
    velocity_range: turned into part of the ms dictionary to specify individual ranges around particular lines
    Returns
    =======
    String of channels to be flagged, in a format that can be passed to the spw parameter in CASA's flagdata task. 
    """
    flagchannels_string = ''
    for j,spw in enumerate(ms_dict['line_spws']):
        print(ms_dict['flagrange'][j],j)
        chans = LSRKvel_to_chan(ms_dict['vis'], ms_dict['field'], spw, ms_dict['line_freqs'][j] , ms_dict['flagrange'][j]) 
        if j==0:
            flagchannels_string+='%d:%d~%d' % (spw, np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
        else:
            flagchannels_string+=', %d:%d~%d' % (spw, np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
    for j,spw in enumerate(ms_dict['cont_spws_orig']):
        if flagchannels_string != '':
           flagchannels_string+=', %d:%d~%d' % (spw, 0,0)
        else:
           flagchannels_string+='%d:%d~%d' % (spw, 0,0)

    print("# Flagchannels input string for %s: \'%s\'" % (ms_dict['name'], flagchannels_string))
    return flagchannels_string

def get_flagchannels_concat(ms_dict, output_prefix):
    """
    Identify channels to flag based on provided velocity range of the line emission

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set
    output_prefix: Prefix for all output file names (string)
    velocity_range: Velocity range (in km/s) over which line emission has been identified, in the format np.array([min_velocity, max_velocity]) 
    velocity_range: turned into part of the ms dictionary to specify individual ranges around particular lines
    Returns
    =======
    String of channels to be flagged, in a format that can be passed to the spw parameter in CASA's flagdata task. 
    """
    flagchannels_string = ''
    for j,spw in enumerate(ms_dict['contsub_spws_concat']):
        print(ms_dict['flagrange'][j],j)
        chans = LSRKvel_to_chan(ms_dict['vis'], ms_dict['field'], ms_dict['line_spws'][j], ms_dict['line_freqs'][j] , ms_dict['flagrange'][j]) 
        if j==0:
            flagchannels_string+='%d:%d~%d' % (spw, np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
        else:
            flagchannels_string+=', %d:%d~%d' % (spw, np.min([chans[0], chans[1]]), np.max([chans[0], chans[1]]))
    for j,spw in enumerate(ms_dict['cont_spws']):
        if flagchannels_string != '':
           flagchannels_string+=', %d:%d~%d' % (spw, 0,0)
        else:
           flagchannels_string+='%d:%d~%d' % (spw, 0,0)
    print("# Flagchannels input string for %s: \'%s\'" % (ms_dict['name'], flagchannels_string))
    return flagchannels_string

def get_contsub_spws(ms_dict, output_prefix):
    """
    Identify channels to flag based on provided velocity range of the line emission

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set
    output_prefix: Prefix for all output file names (string)
    velocity_range: Velocity range (in km/s) over which line emission has been identified, in the format np.array([min_velocity, max_velocity]) 
    velocity_range: turned into part of the ms dictionary to specify individual ranges around particular lines
    Returns
    =======
    String of channels to be flagged, in a format that can be passed to the spw parameter in CASA's flagdata task. 
    """
    flagchannels_string = ''
    for j,spw in enumerate(ms_dict['contsub_spws_concat']):
        print(ms_dict['flagrange'][j],j)
        chans = LSRKvel_to_chan(ms_dict['vis'], ms_dict['field'], ms_dict['line_spws'][j], ms_dict['line_freqs'][j] , ms_dict['flagrange'][j]) 
        if j==0:
            flagchannels_string+='%d' % (spw)
        else:
            flagchannels_string+=', %d' % (spw)
    for j,spw in enumerate(ms_dict['cont_spws']):
        if flagchannels_string != '':
           flagchannels_string+=', %d' % (spw)
           #flagchannels_string+=', %d:%d~%d' % (spw, 0,0)
        else:
           flagchannels_string+='%d' % (spw)
    print("# Contsub spw selection for %s: \'%s\'" % (ms_dict['name'], flagchannels_string))
    return flagchannels_string


def get_contsub_spws_indivdual_ms(ms_dict, output_prefix,only_cont_spws=True):
    """
    Identify channels to flag based on provided velocity range of the line emission

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set
    output_prefix: Prefix for all output file names (string)
    velocity_range: Velocity range (in km/s) over which line emission has been identified, in the format np.array([min_velocity, max_velocity]) 
    velocity_range: turned into part of the ms dictionary to specify individual ranges around particular lines
    Returns
    =======
    String of channels to be flagged, in a format that can be passed to the spw parameter in CASA's flagdata task. 
    """
    flagchannels_string = ''
    if only_cont_spws == False:

       for j,spw in enumerate(ms_dict['line_spws']):
           print(ms_dict['flagrange'][j],j)
           chans = LSRKvel_to_chan(ms_dict['vis'], ms_dict['field'], ms_dict['line_spws'][j], ms_dict['line_freqs'][j] , ms_dict['flagrange'][j]) 
           if j==0:
               flagchannels_string+='%d' % (spw)
           else:
               flagchannels_string+=', %d' % (spw)
    for j,spw in enumerate(ms_dict['cont_spws']):
        if flagchannels_string != '':
           flagchannels_string+=', %d' % (spw)
           #flagchannels_string+=', %d:%d~%d' % (spw, 0,0)
        else:
           flagchannels_string+='%d' % (spw)
    print("# Contsub spw selection for %s: \'%s\'" % (ms_dict['name'], flagchannels_string))
    return flagchannels_string

def avg_cont(ms_dict, output_prefix, flagchannels = '', maxchanwidth = 125, datacolumn = 'data', contspws = None, width_array = None): 
    """
    Produce spectrally averaged continuum measurement sets 

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set
    output_prefix: Prefix for all output file names (string)
    flagchannels: Argument to be passed for flagchannels parameter in flagdata task
    maxchanwidth: Maximum width of channel (MHz). This is the value recommended by ALMA for Band 6 to avoid bandwidth smearing
    datacolumn: Column to pull from for continuum averaging (usually will be 'data', but may sometimes be 'corrected' if there was flux rescaling applied)
    contspws: Argument to be passed to CASA for the spw parameter in split. If not set, all SPWs will be selected by default. (string)
    width_array: Argument to be passed to CASA for the width parameter in split. If not set, all SPWs will be selected by default. (array)
    """
    msfile = ms_dict['vis']
    tb.open(msfile+'/SPECTRAL_WINDOW')
    total_bw = tb.getcol('TOTAL_BANDWIDTH')
    num_chan = tb.getcol('NUM_CHAN')
    tb.close()
    if width_array is None and contspws is None:
        width_array = (num_chan/np.ceil(total_bw/(1.e6*maxchanwidth))).astype('int').tolist() #array of number of channels to average to form an output channel (to be passed to mstransform)
        contspws = '%d~%d' % (0, len(total_bw)-1)#by default select all SPWs
    elif (width_array is not None and contspws is None) or (width_array is None and contspws is not None):
        print("If either contspws or width_array is set to a value, the other parameter has to be manually set as well")
        return
    #not doing any time binning for now.
    if 'LB' in ms_dict['name']:
        timebin = '0s'
    else:
        timebin = '0s' #default in CASA

    
    #start of CASA commands

    if len(flagchannels)==0:
        outputvis = output_prefix+'_'+ms_dict['name']+'_initcont.ms'
        os.system('rm -rf '+outputvis)
        split(vis=msfile,
              field = ms_dict['field'],
              spw = contspws,      
              outputvis = outputvis,
              width = width_array,
              timebin = timebin,
              datacolumn=datacolumn,
              intent = 'OBSERVE_TARGET#ON_SOURCE',
              keepflags = True)
    else:
        if os.path.isdir(msfile+'.flagversions/flags.before_cont_flags'):
            flagmanager(vis = msfile, mode = 'delete', versionname = 'before_cont_flags') # clear out old versions of the flags 
        if os.path.isdir(msfile+'.flagversions/flags.cont_flags'):
            flagmanager(vis = msfile, mode = 'delete', versionname = 'cont_flags') # clear out old versions of the flags 

        flagmanager(vis = msfile, mode = 'save', versionname = 'before_cont_flags', comment = 'Flag states before spectral lines are flagged') #save flag state before flagging spectral lines
        flagdata(vis=msfile, mode='manual', spw=flagchannels, flagbackup=False, field = ms_dict['field']) #flag spectral lines 
        flagmanager(vis = msfile, mode = 'save', versionname = 'cont_flags', comment = 'Flag state after line flagging')
        outputvis = output_prefix+'_'+ms_dict['name']+'_initcont.ms'
        os.system('rm -rf '+outputvis)
        split(vis=msfile,
              field = ms_dict['field'],
              spw = contspws,      
              outputvis = outputvis,
              width = width_array,
              timebin = timebin,
              datacolumn=datacolumn,
              intent = 'OBSERVE_TARGET#ON_SOURCE',
              keepflags = True)

        flagmanager(vis = msfile, mode = 'restore', versionname = 'before_cont_flags') #restore flagged spectral line channels       
    
    print("#Averaged continuum dataset saved to %s" % outputvis)
def split_spec(ms_dict, output_prefix, datacolumn = 'data'): 
    """
    Produce spectrally averaged continuum measurement sets 

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set
    output_prefix: Prefix for all output file names (string)
    flagchannels: Argument to be passed for flagchannels parameter in flagdata task
    maxchanwidth: Maximum width of channel (MHz). This is the value recommended by ALMA for Band 6 to avoid bandwidth smearing
    datacolumn: Column to pull from for continuum averaging (usually will be 'data', but may sometimes be 'corrected' if there was flux rescaling applied)
    contspws: Argument to be passed to CASA for the spw parameter in split. If not set, all SPWs will be selected by default. (string)
    width_array: Argument to be passed to CASA for the width parameter in split. If not set, all SPWs will be selected by default. (array)
    """
    msfile = ms_dict['vis']
    tb.open(msfile+'/SPECTRAL_WINDOW')
    total_bw = tb.getcol('TOTAL_BANDWIDTH')
    num_chan = tb.getcol('NUM_CHAN')
    tb.close()


    if 'LB' in ms_dict['name']:
        timebin = '6s'
    else:
        timebin = '0s' #default in CASA


    split(vis=msfile,field = ms_dict['field'],outputvis=prefix+'_initspec.ms')


def contsub(msfile, output_prefix, spw='',flagchannels = '', datacolumn = 'data', excludechans = True,combine=''): 
    """
    Produce spectrally averaged continuum measurement sets 

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set
    output_prefix: Prefix for all output file names (string)
    flagchannels: Argument to be passed for flagchannels parameter in flagdata task
    maxchanwidth: Maximum width of channel (MHz). This is the value recommended by ALMA for Band 6 to avoid bandwidth smearing
    datacolumn: Column to pull from for continuum averaging (usually will be 'data', but may sometimes be 'corrected' if there was flux rescaling applied)
    contspws: Argument to be passed to CASA for the spw parameter in split. If not set, all SPWs will be selected by default. (string)
    width_array: Argument to be passed to CASA for the width parameter in split. If not set, all SPWs will be selected by default. (array)
    """

    #start of CASA commands
    uvcontsub(vis=msfile, fitspw=flagchannels, spw=spw,field = '',fitorder=1,solint='int',combine=combine,excludechans=excludechans) #flag spectral lines 

    print("#Continuum subtracted dataset saved to %s" % msfile+'.contsub') 

def tclean_wrapper(vis, imagename, scales, smallscalebias = 0.6, mask = '', nsigma=5.0, imsize = None, cellsize = None, interactive = False, robust = 0.5, gain = 0.1, niter = 50000, cycleniter = 300, uvtaper = [], savemodel = 'none', sidelobethreshold=3.0,smoothfactor=1.0,noisethreshold=5.0,lownoisethreshold=1.5,parallel=False,nterms=2,
cyclefactor=3,uvrange='',threshold='0.0Jy',phasecenter='',startmodel='',pblimit=0.1,pbmask=0.1):
    """
    Wrapper for tclean with keywords set to values desired for the Large Program imaging
    See the CASA 6.1.1 documentation for tclean to get the definitions of all the parameters
    """
    if 'LB' in vis and sidelobethreshold>3.0:
       sidelobethreshold=2.0  
    if mask == '':
       usemask='auto-multithresh'
    else:
       usemask='user'
    if threshold != '0.0Jy':
       nsigma=0.0

    if imsize is None:
        if 'LB' in vis or 'combined' in vis or 'continuum' in vis:
            imsize = 6000
        elif 'SB' in vis:
            imsize = 1600
        else:
            print("Error: need to set imsize manually")

    if cellsize is None:
        if 'LB' in vis or 'combined' in vis or 'continuum' in vis:
            cellsize = '.003arcsec'
        elif 'SB' in vis:
            cellsize = '.03arcsec'
        else:
            print("Error: need to set cellsize manually")

    for ext in ['.image*', '.mask', '.model*', '.pb*', '.psf*', '.residual*', '.sumwt*','.gridwt*']:
        os.system('rm -rf '+ imagename + ext)
    tclean(vis= vis, 
           imagename = imagename, 
           specmode = 'mfs', 
           deconvolver = 'mtmfs',
           scales = scales, 
           weighting='briggs', 
           robust = robust,
           gain = gain,
           imsize = imsize,
           cell = cellsize, 
           smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
           niter = niter, #we want to end on the threshold
           interactive = interactive,
           nsigma=nsigma,    
           cycleniter = cycleniter,
           cyclefactor = cyclefactor, 
           uvtaper = uvtaper, 
           savemodel = 'none',
           mask=mask,
           usemask=usemask,
           sidelobethreshold=sidelobethreshold,
           smoothfactor=smoothfactor,
           pbmask=pbmask,
           pblimit=pblimit,
           nterms = nterms,
           uvrange=uvrange,
           threshold=threshold,
           parallel=parallel,
           phasecenter=phasecenter,
           startmodel=startmodel)
     #this step is a workaround a bug in tclean that doesn't always save the model during multiscale clean. See the "Known Issues" section for CASA 5.1.1 on NRAO's website
    if savemodel=='modelcolumn':
          print("")
          print("Running tclean a second time to save the model...")
          tclean(vis= vis, 
                 imagename = imagename, 
                 specmode = 'mfs', 
                 deconvolver = 'mtmfs',
                 scales = scales, 
                 weighting='briggs', 
                 robust = robust,
                 gain = gain,
                 imsize = imsize,
                 cell = cellsize, 
                 smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
                 niter = 0, 
                 interactive = False,
                 nsigma=nsigma, 
                 cycleniter = cycleniter,
                 cyclefactor = cyclefactor, 
                 uvtaper = uvtaper, 
                 usemask='auto-multithresh',
                 savemodel = savemodel,
                 sidelobethreshold=sidelobethreshold,
                 smoothfactor=smoothfactor,
                 pbmask=pbmask,
                 pblimit=pblimit,
                 calcres = False,
                 calcpsf = False,
                 nterms = nterms,
                 uvrange=uvrange,
                 threshold=threshold,
                 parallel=False,
                 phasecenter=phasecenter)
    
def tclean_spectral_line_wrapper(vis, imagename, start, width, nchan, restfreq, spw, scales, smallscalebias = 0.6, mask = '', nsigma=5.0, imsize = None, cellsize = None, interactive = False, robust = 0.5, gain = 0.1, niter = 50000, cycleniter = 300, uvtaper = [], savemodel = 'none', sidelobethreshold=3.0,smoothfactor=1.0,noisethreshold=5.0,lownoisethreshold=1.5,
parallel=False,cyclefactor=3,threshold='0.0Jy',uvrange='',weighting='briggsbwtaper',phasecenter=''):

    """
    Wrapper for tclean with keywords set to values desired for the Large Program imaging
    See the CASA 6.1.1 documentation for tclean to get the definitions of all the parameters
    """
    if 'LB' in vis and sidelobethreshold>3.0:
       sidelobethreshold=2.0  
    if mask == '':
       usemask='auto-multithresh'
    else:
       usemask='user'
    if threshold != '0.0Jy':
       nsigma=0.0


    if imsize is None:
        if 'LB' in vis or 'combined' in vis or 'continuum' in vis:
            imsize = 3000
        elif 'SB' in vis:
            imsize = 1600
        else:
            print("Error: need to set imsize manually")

    if cellsize is None:
        if 'LB' in vis or 'combined' in vis or 'continuum' in vis:
            cellsize = '.003arcsec'
        elif 'SB' in vis:
            cellsize = '.03arcsec'
        else:
            print("Error: need to set cellsize manually")

    for ext in ['.image*', '.mask', '.model*', '.pb*', '.psf*', '.residual*', '.sumwt*', '.workdirectory*']:
        os.system('rm -rf '+ imagename + ext)
    tclean(vis= vis, 
           imagename = imagename, 
           specmode = 'cube', 
           deconvolver = 'multiscale',
           scales = scales, 
           weighting=weighting, 
           robust = robust,
           gain = gain,
           start=start,
           width=width,
           spw=spw,
           nchan=nchan,
           veltype='radio',
           outframe='LSRK',
           imsize = imsize,
           cell = cellsize, 
           smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
           niter = niter, #we want to end on the threshold
           interactive = interactive,
           nsigma=nsigma,    
           cycleniter = cycleniter,
           cyclefactor = cyclefactor, 
           uvtaper = uvtaper, 
           savemodel = 'none',
           mask=mask,
           usemask=usemask,
           sidelobethreshold=sidelobethreshold,
           lownoisethreshold=lownoisethreshold,
           noisethreshold=noisethreshold,
           smoothfactor=smoothfactor,
           pbmask=0.1,
           pblimit=0.1,
           restoringbeam='common',
           threshold=threshold,
           uvrange=uvrange,
           restfreq=restfreq,
           parallel=parallel,
           phasecenter=phasecenter)

def image_each_obs(ms_dict, prefix, scales, smallscalebias = 0.6, mask = '', nsigma=5.0, imsize = None, cellsize = None, interactive = False, robust = 0.5, gain = 0.3, niter = 50000, cycleniter = 300,sidelobethreshold=3.0,smoothfactor=1.0,noisethreshold=5.0,uvtaper='',parallel=False):
    """
    Wrapper for tclean that will loop through all the observations in a measurement set and image them individual

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set
    prefix: Prefix for all output file names (string)
    
    See the CASA 5.1.1 documentation for tclean to get the definitions of all other parameters
    """
    msfile = prefix+'_'+ms_dict['name']+'_initcont.ms'
    tb.open(msfile+'/OBSERVATION')
    num_observations = (tb.getcol('TIME_RANGE')).shape[1] #picked an arbitrary column to count the number of observations
    tb.close()

    if imsize is None:
        if 'LB' in ms_dict['name']:
            imsize = 3000
        else:
            imsize = 1024

    if cellsize is None:
        if 'LB' in ms_dict['name']:
            cellsize = '0.003arcsec'
        else:
            imsize = 900
            cellsize = '0.03arcsec'
    if 'LB' in ms_dict['name'] and sidelobethreshold>3.0:
       sidelobethreshold=2.0       
    #start of CASA commands
    for i in range(num_observations):
        observation = '%d' % i
        imagename = prefix+'_'+ms_dict['name']+'_initcont_exec%s' % observation
        for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']:
            os.system('rm -rf '+ imagename + ext)
        tclean(vis= msfile, 
               imagename = imagename, 
               observation = observation,
               specmode = 'mfs', 
               deconvolver = 'multiscale',
               scales = scales, 
               weighting='briggs', 
               robust = robust,
               gain = gain,
               imsize = imsize,
               cell = cellsize, 
               smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
               niter = niter, #we want to end on the threshold
               interactive = interactive,
               nsigma=nsigma,
               cycleniter = cycleniter,
               cyclefactor = 1, 
               usemask='auto-multithresh',
               sidelobethreshold=sidelobethreshold,
               smoothfactor=smoothfactor,
               noisethreshold=noisethreshold,
               nterms = 1,
               uvtaper=[uvtaper],parallel=parallel)

    print("Each observation saved in the format %sOBSERVATIONNUMBER.image" % (prefix+'_'+ms_dict['name']+'_initcont_exec',))

def image_each_obs_shift(msfile, prefix, scales, smallscalebias = 0.6, mask = '', nsigma=5.0, imsize = None, cellsize = None, interactive = False, robust = 0.5, gain = 0.3, niter = 50000, cycleniter = 300,sidelobethreshold=3.0,smoothfactor=1.0,uvtaper='',parallel=False):
    """
    Wrapper for tclean that will loop through all the observations in a measurement set and image them individual

    Parameters
    ==========
    ms_dict: Dictionary of information about measurement set
    prefix: Prefix for all output file names (string)
    
    See the CASA 5.1.1 documentation for tclean to get the definitions of all other parameters
    """
    baselinename=''
    if 'SB1' in msfile:
       baseline_name='SB1'
    elif 'SB2' in msfile:
       baseline_name='SB2'
    elif 'SB3' in msfile:
       baseline_name='SB3'
    elif 'LB1' in msfile:
       baseline_name='LB1'
    elif 'LB2' in msfile:
       baseline_name='LB2'
    elif 'LB3' in msfile:
       baseline_name='LB3'
    elif 'LB4' in msfile:
       baseline_name='LB4'
    elif 'LB5' in msfile:
       baseline_name='LB5'
    if imsize is None:
        if 'LB' in msfile:
            imsize = 3000
        else:
            imsize = 1024

    if cellsize is None:
        if 'LB' in msfile:
            cellsize = '0.003arcsec'
        else:
            imsize = 900
            cellsize = '0.03arcsec'
    if 'LB' in msfile and sidelobethreshold>3.0:
       sidelobethreshold=2.0       
    #start of CASA commands
    imagename = prefix+'_'+baseline_name+'_initcont_shift'
    for ext in ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']:
       os.system('rm -rf '+ imagename + ext)
    tclean(vis= msfile, 
               imagename = imagename, 
               specmode = 'mfs', 
               deconvolver = 'multiscale',
               scales = scales, 
               weighting='briggs', 
               robust = robust,
               gain = gain,
               imsize = imsize,
               cell = cellsize, 
               smallscalebias = smallscalebias, #set to CASA's default of 0.6 unless manually changed
               niter = niter, #we want to end on the threshold
               interactive = interactive,
               nsigma=nsigma,
               cycleniter = cycleniter,
               cyclefactor = 1, 
               usemask='auto-multithresh',
               sidelobethreshold=sidelobethreshold,
               smoothfactor=smoothfactor,
               nterms = 1,
               uvtaper=[uvtaper],parallel=parallel)

    print("Each observation saved in the format "+imagename )

    
    
def fit_gaussian(imagename, region, dooff = False, mask=''):
    """
    Wrapper for imfit in CASA to fit a single Gaussian component to a selected region of the image
    Parameters
    ==========
    imagename: Name of CASA image (ending in .image) (string)
    region: CASA region format, e.g., 'circle[[200pix, 200pix], 3arcsec]' (string)
    dooff: boolean option to allow for fitting a zero-level offset 
    """
    imfitdict = imfit(imagename = imagename, region = region, dooff = dooff,mask=mask)
    # Check if the source was resolved
    was_resolved = not imfitdict['deconvolved']['component0']['ispoint']
    # Get the coordinate system
    coordsystem = imfitdict['deconvolved']['component0']['shape']['direction']['refer']
    # Get the parameters
    headerlist = imhead(imagename)
    phasecenter_ra, phasecenter_dec = headerlist['refval'][:2]
    peak_ra = imfitdict['deconvolved']['component0']['shape']['direction']['m0']['value']
    peak_dec = imfitdict['deconvolved']['component0']['shape']['direction']['m1']['value']
    xcen, ycen = headerlist['refpix'][:2]
    deltax, deltay = headerlist['incr'][:2]
    peak_x = xcen+np.unwrap(np.array([0, peak_ra-phasecenter_ra]))[1]/deltax*np.cos(phasecenter_dec)
    peak_y = ycen+(peak_dec-phasecenter_dec)/deltay
    # Print
    coords=''
    if coordsystem=='J2000':
        coords=au.rad2radec(imfitdict = imfitdict, hmsdms = True, delimiter = ' ')
        print('#Peak of Gaussian component identified with imfit: J2000 %s' % coords)

    elif coordsystem=='ICRS':
        coords=au.rad2radec(imfitdict = imfitdict, hmsdms = True, delimiter = ' ')
        print('#Peak of Gaussian component identified with imfit: ICRS %s' % coords)
        J2000coords = au.ICRSToJ2000(au.rad2radec(imfitdict = imfitdict, hmsdms = True, delimiter = ' '))
        print('#Peak in J2000 coordinates: %s' % J2000coords)
    else:
       print("#If the coordinates aren't in ICRS or J2000, then something weird is going on")
    # If the object was resolved, print the inclination, PA, major and minor axis
    if was_resolved:    
        PA = imfitdict['deconvolved']['component0']['shape']['positionangle']['value']
        majoraxis = imfitdict['deconvolved']['component0']['shape']['majoraxis']['value']
        minoraxis = imfitdict['deconvolved']['component0']['shape']['minoraxis']['value']

        print('#PA of Gaussian component: %.2f deg' % PA)
        print('#Inclination of Gaussian component: %.2f deg' % (np.arccos(minoraxis/majoraxis)*180/np.pi,))
    print('#Pixel coordinates of peak: x = %.3f y = %.3f' % (peak_x, peak_y))
    if coordsystem =='J2000':
        return coords
    elif coordsystem =='ICRS':
        return au.convertColonDelimitersToHMSDMS(J2000coords).replace(',','')

def fit_gaussian_pc(imagename, region, dooff = False, mask=''):
    """
    Wrapper for imfit in CASA to fit a single Gaussian component to a selected region of the image
    Parameters
    ==========
    imagename: Name of CASA image (ending in .image) (string)
    region: CASA region format, e.g., 'circle[[200pix, 200pix], 3arcsec]' (string)
    dooff: boolean option to allow for fitting a zero-level offset 
    """
    imfitdict = imfit(imagename = imagename, region = region, dooff = dooff,mask=mask)
    # Check if the source was resolved
    was_resolved = not imfitdict['deconvolved']['component0']['ispoint']
    # Get the coordinate system
    coordsystem = imfitdict['deconvolved']['component0']['shape']['direction']['refer']
    # Get the parameters
    headerlist = imhead(imagename)
    phasecenter_ra, phasecenter_dec = headerlist['refval'][:2]
    phasecenter_ra_deg=phasecenter_ra*180.0/3.141592653589793
    phasecenter_dec_deg=phasecenter_dec*180.0/3.141592653589793
    if phasecenter_ra_deg < 0.0:
       phasecenter_ra_deg = phasecenter_ra_deg+360.0
    peak_ra = imfitdict['deconvolved']['component0']['shape']['direction']['m0']['value']
    peak_dec = imfitdict['deconvolved']['component0']['shape']['direction']['m1']['value']
    peak_ra_deg=peak_ra*180.0/3.141592653589793
    peak_dec_deg=peak_dec*180.0/3.141592653589793
    if peak_ra_deg < 0.0:
       peak_ra_deg = peak_ra_deg+360.0

    xcen, ycen = headerlist['refpix'][:2]
    deltax, deltay = headerlist['incr'][:2]
    peak_x = xcen+np.unwrap(np.array([0, peak_ra-phasecenter_ra]))[1]/deltax*np.cos(phasecenter_dec)
    peak_y = ycen+(peak_dec-phasecenter_dec)/deltay
    # Print
    coords=''
    if coordsystem=='J2000':
        coords=au.rad2radec(imfitdict = imfitdict, hmsdms = True, delimiter = ' ')
        print('#Peak of Gaussian component identified with imfit: J2000 %s' % coords)

    elif coordsystem=='ICRS':
        coords=au.rad2radec(imfitdict = imfitdict, hmsdms = True, delimiter = ' ')
        print('#Peak of Gaussian component identified with imfit: ICRS %s' % coords)
        J2000coords = au.ICRSToJ2000(au.rad2radec(imfitdict = imfitdict, hmsdms = True, delimiter = ' '))
        print('#Peak in J2000 coordinates: %s' % J2000coords)
    else:
       print("#If the coordinates aren't in ICRS or J2000, then something weird is going on")
    # If the object was resolved, print the inclination, PA, major and minor axis
    if was_resolved:    
        PA = imfitdict['deconvolved']['component0']['shape']['positionangle']['value']
        majoraxis = imfitdict['deconvolved']['component0']['shape']['majoraxis']['value']
        minoraxis = imfitdict['deconvolved']['component0']['shape']['minoraxis']['value']

        print('#PA of Gaussian component: %.2f deg' % PA)
        print('#Inclination of Gaussian component: %.2f deg' % (np.arccos(minoraxis/majoraxis)*180/np.pi,))
    print('#Pixel coordinates of peak: x = %.3f y = %.3f' % (peak_x, peak_y))

    if coordsystem =='J2000':
        return coords,peak_ra_deg,peak_dec_deg,phasecenter_ra_deg,phasecenter_dec_deg
    elif coordsystem =='ICRS':
        return au.convertColonDelimitersToHMSDMS(J2000coords).replace(',',''),peak_ra_deg,peak_dec_deg,phasecenter_ra_deg,phasecenter_dec_deg

def split_all_obs(msfile, nametemplate):
    """

    Split out individual observations in a measurement set 

    Parameters
    ==========
    msfile: Name of measurement set, ending in '.ms' (string)
    nametemplate: Template name of output measurement sets for individual observations (string)
    """
    tb.open(msfile)
    spw_col = tb.getcol('DATA_DESC_ID')
    obs_col = tb.getcol('OBSERVATION_ID')
    field_col = tb.getcol('FIELD_ID') 
    tb.close()

    obs_ids = np.unique(obs_col)


    #yes, it would be more logical to split out by observation id, but splitting out by observation id in practice leads to some issues with the metadata
    for i in obs_ids:
        spws = np.unique(spw_col[np.where(obs_col==i)])
        fields = np.unique(field_col[np.where(obs_col==i)]) #sometimes the MS secretly has multiple field IDs lurking even if listobs only shows one field
        if len(spws)==1:
            spw = str(spws[0])
        else:
            spw = "%d~%d" % (spws[0], spws[-1])

        if len(fields)==1:
            field = str(fields[0])
        else:
            field = "%d~%d" % (fields[0], fields[-1])
        #start of CASA commands
        outputvis = nametemplate+'%d.ms' % i
        os.system('rm -rf '+outputvis)
        print("#Saving observation %d of %s to %s" % (i, msfile, outputvis))
        split(vis=msfile,
              spw = spw, 
              field = field, 
              outputvis = outputvis,
              datacolumn='data')
    
def export_MS(msfile):
    """
    Spectrally averages visibilities to a single channel per SPW and exports to .npz file

    msfile: Name of CASA measurement set, ending in '.ms' (string)
    """
    filename = msfile
    if filename[-3:]!='.ms':
        print("MS name must end in '.ms'")
        return
    # strip off the '.ms'
    MS_filename = filename.replace('.ms', '')

    # get information about spectral windows

    tb.open(MS_filename+'.ms/SPECTRAL_WINDOW')
    num_chan = tb.getcol('NUM_CHAN').tolist()
    tb.close()

    # spectral averaging (1 channel per SPW)

    os.system('rm -rf %s' % MS_filename+'_spavg.ms')
    split(vis=MS_filename+'.ms', width=num_chan, datacolumn='data',
    outputvis=MS_filename+'_spavg.ms')

    # get the data tables
    tb.open(MS_filename+'_spavg.ms')
    data   = np.squeeze(tb.getcol("DATA"))
    flag   = np.squeeze(tb.getcol("FLAG"))
    uvw    = tb.getcol("UVW")
    weight = tb.getcol("WEIGHT")
    spwid  = tb.getcol("DATA_DESC_ID")
    tb.close()

    # get frequency information
    tb.open(MS_filename+'_spavg.ms/SPECTRAL_WINDOW')
    freqlist = np.squeeze(tb.getcol("CHAN_FREQ"))
    tb.close()

    # get rid of any flagged columns
    good   = np.squeeze(np.any(flag, axis=0)==False)
    data   = data[:,good]
    weight = weight[:,good]
    uvw    = uvw[:,good]
    spwid = spwid[good]
    
    # compute spatial frequencies in lambda units
    get_freq = lambda ispw: freqlist[ispw]
    freqs = get_freq(spwid) #get spectral frequency corresponding to each datapoint
    u = uvw[0,:] * freqs / 2.9979e8
    v = uvw[1,:] * freqs / 2.9979e8

    #average the polarizations
    Re  = np.sum(data.real*weight, axis=0) / np.sum(weight, axis=0)
    Im  = np.sum(data.imag*weight, axis=0) / np.sum(weight, axis=0)
    Vis = Re + 1j*Im
    Wgt = np.sum(weight, axis=0)

    #output to npz file and delete intermediate measurement set
    os.system('rm -rf %s' % MS_filename+'_spavg.ms')
    os.system('rm -rf '+MS_filename+'.vis.npz')
    np.savez(MS_filename+'.vis', u=u, v=v, Vis=Vis, Wgt=Wgt)
    print("#Measurement set exported to %s" % (MS_filename+'.vis.npz',))


def deproject_vis(data, bins=np.array([0.]), incl=0., PA=0., offx=0., offy=0., 
                  errtype='mean'):
    """
    Deprojects and azimuthally averages visibilities 

    Parameters
    ==========
    data: Length-4 tuple of u,v, visibilities, and weight arrays 
    bins: 1-D array of uv distance bins (kilolambda)
    incl: Inclination of disk (degrees)
    PA: Position angle of disk (degrees)
    offx: Horizontal offset of disk center from phase center (arcseconds)
    offy: Vertical offset of disk center from phase center (arcseconds) 

    Returns
    =======
    uv distance bins (1D array), visibilities (1D array), errors on averaged visibilities (1D array) 
    """

    # - read in, parse data
    u, v, vis, wgt = data
    # - convert keywords into relevant units
    inclr = np.radians(incl)
    PAr = 0.5*np.pi-np.radians(PA)
    offx *= -np.pi/(180.*3600.)
    offy *= -np.pi/(180.*3600.)

    # - change to a deprojected, rotated coordinate system
    uprime = (u*np.cos(PAr) + v*np.sin(PAr)) 
    vprime = (-u*np.sin(PAr) + v*np.cos(PAr)) * np.cos(inclr)
    rhop = np.sqrt(uprime**2 + vprime**2)

    # - phase shifts to account for offsets
    shifts = np.exp(-2.*np.pi*1.0j*(u*-offx + v*-offy))
    visp = vis*shifts
    realp = visp.real
    imagp = visp.imag

    # - if requested, return a binned (averaged) representation
    if (bins.size > 1.):
        avbins = 1e3*bins	# scale to lambda units (input in klambda)
        bwid = 0.5*(avbins[1]-avbins[0])
        bvis = np.zeros_like(avbins, dtype='complex')
        berr = np.zeros_like(avbins, dtype='complex')
        for ib in np.arange(len(avbins)):
            inb = np.where((rhop >= avbins[ib]-bwid) & (rhop < avbins[ib]+bwid))
            if (len(inb[0]) >= 5):
                bRe, eRemu = np.average(realp[inb], weights=wgt[inb], 
                                        returned=True)
                eRese = np.std(realp[inb])
                bIm, eImmu = np.average(imagp[inb], weights=wgt[inb], 
                                        returned=True)
                eImse = np.std(imagp[inb])
                bvis[ib] = bRe+1j*bIm
                if (errtype == 'scat'):
                    berr[ib] = eRese+1j*eImse
                else: berr[ib] = 1./np.sqrt(eRemu)+1j/np.sqrt(eImmu)
            else:
                bvis[ib] = 0+1j*0
                berr[ib] = 0+1j*0
        parser = np.where(berr.real != 0)
        output = avbins[parser], bvis[parser], berr[parser]
        return output       
        
    # - if not, returned the unbinned representation
    output = rhop, realp+1j*imagp, 1./np.sqrt(wgt)

    return output

def plot_deprojected(filelist, incl = 0, PA = 0, offx = 0, offy = 0, fluxscale = None, uvbins = None, show_err = True,outfile=''):
    """
    Plots real and imaginary deprojected visibilities from a list of .npz files

    Parameters
    ==========
    filelist: List of names of .npz files storing visibility data
    incl: Inclination of disk (degrees)
    PA: Position angle of disk (degrees)
    offx: Horizontal offset of disk center from phase center (arcseconds)
    offy: Vertical offset of disk center from phase center (arcseconds) 
    fluxscale: List of scaling factors to multiply the visibility values by before plotting. Default value is set to all ones. 
    uvbins: Array of bins at which to plot the visibility values, in lambda. By default, the range plotted will be from 10 to 1000 kilolambda
    show_err: If True, plot error bars. 
    """
    if fluxscale is None:
        fluxscale = np.ones(len(filelist))
    assert len(filelist)==len(fluxscale)

    if uvbins is None:
        uvbins = 10.+10.*np.arange(100)

    minvis = np.zeros(len(filelist))
    maxvis = np.zeros(len(filelist))
    fig, ax = plt.subplots(2,1,sharex  = True)
    for i, filename in enumerate(filelist):

        # read in the data
        inpf = np.load(filename)
        u    = inpf['u']
        v    = inpf['v']
        vis  = fluxscale[i]*inpf['Vis']
        wgt  = inpf['Wgt']

        # deproject the visibilities and do the annular averaging
        vp   = deproject_vis([u, v, vis, wgt], bins=uvbins, incl=incl, PA=PA, 
                         offx=offx, offy=offy)
        vp_rho, vp_vis, vp_sig = vp

        # calculate min, max of deprojected, averaged reals (for visualization)
        minvis[i] = np.min(vp_vis.real)
        maxvis[i] = np.max(vp_vis.real)

        # plot the profile
        if show_err:
            ax[0].errorbar(1e-3*vp_rho, vp_vis.real, yerr = vp_sig.real, label = filename, fmt = '.')
            ax[1].errorbar(1e-3*vp_rho, vp_vis.imag, yerr = vp_sig.imag, label = filename, fmt = '.')
        else:
            ax[0].plot(1e-3*vp_rho, vp_vis.real, 'o', markersize=2.8, label = filename)
            ax[1].plot(1e-3*vp_rho, vp_vis.imag, 'o', markersize=2.8, label = filename)

    allmaxvis = np.max(maxvis)
    allminvis = np.min(minvis)
    if ((allminvis < 0) or (allminvis-0.1*allmaxvis < 0)):
        ax[0].axis([0, np.max(uvbins), allminvis-0.1*allmaxvis, 1.1*allmaxvis])
        ax[1].axis([0, np.max(uvbins), allminvis-0.1*allmaxvis, 1.1*allmaxvis])
    else: 
        ax[0].axis([0, np.max(uvbins), 0., 1.1*allmaxvis])
        ax[1].axis([0, np.max(uvbins), 0., 1.1*allmaxvis])

    ax[0].plot([0, np.max(uvbins)], [0, 0], '--k')
    ax[1].plot([0, np.max(uvbins)], [0, 0], '--k')
    plt.xlabel('deprojected baseline length [kilo$\lambda$]')
    ax[0].set_ylabel('average real [Jy]')
    ax[1].set_ylabel('average imag [Jy]')
    ax[0].legend(fontsize=4)
    if outfile !='':
       plt.savefig(outfile,dpi=200.0)
    plt.show(block = False)
    

def estimate_flux_scale(reference, comparison, incl = 0, PA = 0, uvbins = None, offx = 0, offy = 0):
    """
    Calculates the weighted average of the flux ratio between two observations of a source 
    The minimum baseline compared is the longer of the minimum baselines in the individual datasets
    The longest baseline compared is either the shorter of the longest baselines in the individual datasets, or 800 kilolambda
    
    Parameters
    ==========
    reference: Name of .npz file holding the reference dataset (with the "correct" flux")
    comparison: Name of .npz file holding the comparison dataset (with the flux ratio being checked)
    filelist: List of names of .npz files storing visibility data
    incl: Inclination of disk (degrees)
    PA: Position angle of disk (degrees)
    offx: Horizontal offset of disk center from phase center (arcseconds)
    offy: Vertical offset of disk center from phase center (arcseconds) 
    uvbins: Array of bins at which to compare the visibility values, in lambda. 
            By default, the minimum baseline compared is the longer of the minimum baselines in the individual datasets. 
            The longest baseline compared is either the shorter of the longest baselines in the individual datasets, or 800 kilolambda, whichever comes first.
    """

    inpf = np.load(reference)
    u_ref    = inpf['u']
    v_ref    = inpf['v']
    vis_ref  = inpf['Vis']
    wgt_ref  = inpf['Wgt']

    inpf = np.load(comparison)
    u_comp    = inpf['u']
    v_comp    = inpf['v']
    vis_comp  = inpf['Vis']
    wgt_comp  = inpf['Wgt']

    uvdist_ref = np.sqrt(u_ref**2+v_ref**2)
    uvdist_comp = np.sqrt(u_comp**2+v_comp**2)

    mindist = np.max(np.array([np.min(uvdist_ref), np.min(uvdist_comp)]))
    maxdist = np.min(np.array([np.max(uvdist_ref), np.max(uvdist_ref), 8e5])) #the maximum baseline we want to compare is the longest shared baseline or 800 kilolambda, whichever comes first (we don't want to go out to a baseline that's too long because phase decorrelation becomes a bigger issue at longer baselines.

    if uvbins is None:
        uvbins = mindist/1.e3+10.*np.arange(np.floor((maxdist-mindist)/1.e4))


    # deproject the visibilities and do the annular averaging
    vp   = deproject_vis([u_ref, v_ref, vis_ref, wgt_ref], bins=uvbins, incl=incl, PA=PA, 
                         offx=offx, offy=offy)
    ref_rho, ref_vis, ref_sig = vp

    # deproject the visibilities and do the annular averaging
    vp   = deproject_vis([u_comp, v_comp, vis_comp, wgt_comp], bins=uvbins, incl=incl, PA=PA, 
                         offx=offx, offy=offy)
    comp_rho, comp_vis, comp_sig = vp

    maxlen = np.min(np.array([len(comp_rho), len(ref_rho)]))

    rho_intersection = np.intersect1d(ref_rho, comp_rho) #we only want to compare overlapping baseline intervals

    comp_sig_intersection = comp_sig[np.where(np.in1d(comp_rho, rho_intersection))].real #they're the same for the real and imaginary components
    comp_vis_intersection = comp_vis[np.where(np.in1d(comp_rho, rho_intersection))]
    ref_sig_intersection = ref_sig[np.where(np.in1d(ref_rho, rho_intersection))].real 
    ref_vis_intersection = ref_vis[np.where(np.in1d(ref_rho, rho_intersection))]

    ratio = np.abs(comp_vis_intersection)/np.abs(ref_vis_intersection)
    err = ratio*np.sqrt((comp_sig_intersection/np.abs(comp_vis_intersection))**2+(ref_sig_intersection/np.abs(ref_vis_intersection))**2)

    w = 1/err**2
    ratio_avg =  np.sum(w*ratio)/np.sum(w)
    print( "#The ratio of the fluxes of %s to %s is %.5f" % (comparison, reference, ratio_avg))
    print("#The scaling factor for gencal is %.3f for your comparison measurement" % (np.sqrt(ratio_avg)))
    print("#The error on the weighted mean ratio is %.3e, although it's likely that the weights in the measurement sets are off by some constant factor" % (1/np.sqrt(np.sum(w)),))
    plt.figure()
    plt.errorbar(1e-3*rho_intersection, ratio, yerr = err, fmt = '.', label = 'Binned ratios')
    plt.plot(1e-3*rho_intersection, np.ones_like(ratio)*ratio_avg, label = 'weighted average')
    plt.ylabel('Visibility amplitude ratios')
    plt.xlabel('UV distance (kilolambda)')
    plt.legend()
    plt.show(block = False)
    dataset_strings=comparison.split('_')
    dataset=dataset_strings[1]
    reference_strings=reference.split('_')
    reference_data=reference_strings[1]
    plt.savefig('estimate_fluxscale_'+dataset+'_ref_'+reference_data+'.png',dpi=200)
    return np.sqrt(ratio_avg)

def rescale_flux(vis, gencalparameter):
    """
    Rescale visibility fluxes using gencal, then split into a new measurement set
 
    Parameters:
    vis: Measurement set name, ending in ms (string)
    gencalparameter: List of flux rescaling parameters to be passed to 'parameter' for gencal task
    """
    caltable = 'scale_'+vis.replace('.ms', '.gencal')
    os.system('rm -rf '+caltable)
    gencal(vis = vis, caltable = caltable, caltype = 'amp', parameter = gencalparameter)
    applycal(vis = vis, gaintable = caltable, calwt = True, flagbackup = True)
    vis_rescaled = vis.replace('.ms', '_rescaled.ms')
    print("#Splitting out rescaled values into new MS: %s" % (vis_rescaled,))
    os.system('rm -rf '+vis_rescaled+'*')
    split(vis=vis, outputvis=vis_rescaled, datacolumn='corrected')

def estimate_SNR(imagename, disk_mask, noise_mask):
    """
    Estimate peak SNR of source

    Parameters:
    imagename: Image name ending in '.image' (string)
    disk_mask: , in the CASa region format, e.g. 
    noise_mask: Annulus to measure image rms, in the CASA region format, e.g. 'annulus[[500pix, 500pix],["1arcsec", "2arcsec"]]' (string)
    """
    headerlist = imhead(imagename, mode = 'list')
    beammajor = headerlist['beammajor']['value']
    beamminor = headerlist['beamminor']['value']
    beampa = headerlist['beampa']['value']
    print("#%s" % imagename)
    print("#Beam %.3f arcsec x %.3f arcsec (%.2f deg)" % (beammajor, beamminor, beampa))
    disk_stats = imstat(imagename = imagename, region = disk_mask)
    disk_flux = disk_stats['flux'][0]
    print("#Flux inside disk mask: %.2f mJy" % (disk_flux*1000,))
    peak_intensity = disk_stats['max'][0]
    print("#Peak intensity of source: %.2f mJy/beam" % (peak_intensity*1000,))
    rms = imstat(imagename = imagename, region = noise_mask)['rms'][0]
    print("#rms: %.2e mJy/beam" % (rms*1000,))
    SNR = peak_intensity/rms
    print("#Peak SNR: %.2f" % (SNR,))
    return SNR,rms

def get_station_numbers(msfile, antenna_name):
    """
    Get the station numbers for all observations in which the given antenna appears

    Parameters
    ==========
    msfile: Name of measurement set (string)
    antenna_name: Name of antenna (e.g. "DA48")
    """
    tb.open(msfile+'/ANTENNA')
    ant_names = tb.getcol('NAME')
    ant_stations = tb.getcol('STATION')
    tb.close()

    ant_numbers = np.where(ant_names == antenna_name)[0]

    tb.open(msfile)
    antenna1 = tb.getcol('ANTENNA1')
    obsid = tb.getcol('OBSERVATION_ID')
    tb.close()

    for i in ant_numbers:
        matching_obs = np.unique(obsid[np.where(antenna1==i)])
        for j in matching_obs:
            print("#Observation ID %d: %s@%s" % (j, antenna_name, ant_stations[i]))


def self_calibrate(prefix,data_params,selectedVis='vis_avg_shift_rescaled',mode='SB-only',iteration=0,selfcalmode='p',
                  prevselfcalmode=None,nsigma=50.0,solint='inf',
                  noisemasks=['',''],parallel=False,SB_contspws='',
                  SB_spwmap=[0,0,0,0,0,0,0],LB_contspws='',LB_spwmap=[0,0,0,0,0,0,0],
                  cellsize=None,imsize=None,scales=None,finalimageonly=False,remove_all_following_iterations=True,         
                  sidelobethreshold=2.5,noisethreshold=5.0,lownoisethreshold=1.5,smoothfactor=1.0,combine='spw',skipImage=False,nterms=2,
                  evalImage=True,robust=0.5,minsnr=2.0,uvrange='',minsnr_ap=4.0,uvtaper='',refantmode='flex',threshold='0.0Jy'):
   ### Use skipImage with care
   contspws=''
   spwmap=[0,0,0,0,0]
   if remove_all_following_iterations:   ### Remove all following selfcal files after the iteration being run
      if not skipImage:   # don't erase the current iteration if you want to skip imaging and re-run with current model
         os.system("rm -rf "+"*_"+mode+"*p{{{0:d}..99}}*".format(iteration+1))
      else:
         os.system("rm -rf "+"*_"+mode+"*p{{{0:d}..99}}*".format(iteration))
   if prevselfcalmode==None:
      prevselfcalmode=selfcalmode

   if (mode == 'LB+SB') and (scales==None):
      scales=[0,5,30]
   else:
      scales=[0,5]
   
   if (mode == 'LB+SB') and (imsize is None):
      imsize=6000
   elif imsize is None:
      imsize=1600


   if cellsize is None:
      if (mode == 'LB+SB'):
         cellsize='0.003arcsec'
      else:
         cellsize='0.025arcsec'
  
   #set up lists of MSes to image depending on SB-only or SB and LB
   vislist=[]
   for i in data_params.keys():
      if (mode =='SB-only') and ('LB' in i): # skip over LB EBs if in SB-only mode
         continue
      if (mode =='LB-only') and ('LB' in i): # skip over SB EBs if in LB-only mode
         continue
      if (mode =='LB+SB') and ('SB' in i) and (iteration == 0): # slightly different flow for the SB data in iter0
         if not skipImage:
            os.system('rm -rf '+data_params[i][selectedVis].replace('.ms','_'+mode+'_p0.ms'))
            os.system('cp -r '+data_params[i]['vis_avg_selfcal_SB-only']+' '+data_params[i][selectedVis].replace('.ms','_'+mode+'_p0.ms'))
            data_params[i]['selfcal_spwmap']=[]
            data_params[i]['selfcal_tables']=[]
            data_params[i]['vis_avg_selfcal']=data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms')
         vislist.append(data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms'))
         continue
      if iteration == 0:
         if not skipImage:
            os.system('rm -rf '+data_params[i][selectedVis].replace('.ms','_'+mode+'_p0.ms'))
            os.system('cp -r '+data_params[i][selectedVis]+' '+data_params[i][selectedVis].replace('.ms','_'+mode+'_p0.ms'))
            data_params[i]['selfcal_tables']=[]
            data_params[i]['selfcal_spwmap']=[]
            data_params[i]['vis_avg_selfcal']=data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms')
      elif iteration > 0:
         if len(data_params[i]['selfcal_tables']) > iteration:  ### remove selfcal table entries when redoing self-cal stages
            data_params[i]['selfcal_tables'] = data_params[i]['selfcal_tables'][0:iteration]
            data_params[i]['selfcal_spwmap'] = data_params[i]['selfcal_spwmap'][0:iteration]
         if not skipImage:
            print('Input MS: ',data_params[i][selectedVis].replace('.ms','_'+mode+'_'+prevselfcalmode+str(iteration-1)+'.ms'))
            os.system('rm -rf '+data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms'))
            split(vis=data_params[i][selectedVis].replace('.ms','_'+mode+'_'+prevselfcalmode+str(iteration-1)+'.ms'),
                  outputvis=data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms'))
            data_params[i]['vis_avg_selfcal']=data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms')
            print('Output MS: ',data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms'))
      vislist.append(data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms'))



   if not skipImage:
      print(vislist)
      #create single image using all visibilities; model column always writes with parallel=False
      os.system('rm -rf '+prefix+'_'+mode+'_'+prevselfcalmode+str(iteration)+'*')
      tclean_wrapper(vis=vislist, imagename=prefix+'_'+mode+'_'+prevselfcalmode+str(iteration),scales=scales, nsigma=nsigma,
                  savemodel='modelcolumn',parallel=parallel,cellsize=cellsize,imsize=imsize,sidelobethreshold=sidelobethreshold,
                  noisethreshold=noisethreshold,lownoisethreshold=lownoisethreshold,smoothfactor=smoothfactor,nterms=nterms,
                  robust=robust,uvtaper=[uvtaper],threshold=threshold)


   if finalimageonly==True: # break out of function if we just wanted final image and no more gain solutions
      estimate_SNR(prefix+'_'+mode+'_'+prevselfcalmode+str(iteration)+'.image.tt0', disk_mask=noisemasks[0], 
             noise_mask=noisemasks[1])
      return

   for i in data_params.keys():
      if (mode =='SB-only') and ('LB' in i): # skip over LB EBs if in SB-only mode
         continue
      if (mode =='LB-only') and ('LB' in i): # skip over SB EBs if in LB-only mode
         continue
      os.system('rm -rf '+data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.g'))
      if 'LB' in i:
         contspws=LB_contspws
      if 'SB' in i:
         contspws=SB_contspws
      if selfcalmode=='p':
         gaincal(vis=data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms'), 
                    caltable=data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.g'), 
                    gaintype='T', spw=contspws,refant=data_params[i]["refant"], calmode=selfcalmode, solint=solint, 
                    minsnr=minsnr, minblperant=4,combine=combine,uvrange=uvrange,refantmode=refantmode)
         plot_ants_flagging_colored('plot_ants_'+data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.png'),\
                                    data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms'),\
                                    data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.g'))
      elif selfcalmode=='ap':
         gaincal(vis=data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms'), 
                    caltable=data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.g'), 
                    gaintype='T', spw=contspws,refant=data_params[i]["refant"], calmode=selfcalmode, solint=solint, 
                    minsnr=minsnr_ap, minblperant=4,combine=combine,solnorm=True,uvrange=uvrange)
         flagdata(vis=data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.g'),mode='clip',
                  clipoutside=True,clipminmax =[0.5,1.5],datacolumn='CPARAM')
         plot_ants_flagging_colored('plot_ants_'+data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.png'),\
                                    data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms'),\
                                    data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.g'))
      # Add caltable to the dictionary for use later
      data_params[i]['selfcal_tables'].append(data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.g'))

   #apply the solutions
   for i in data_params.keys():
      if (mode =='SB-only') and ('LB' in i): # skip over LB EBs if in SB-only mode
         continue
      if (mode =='LB-only') and ('LB' in i): # skip over SB EBs if in LB-only mode
         continue
      if 'LB' in i:
         spwmap=LB_spwmap
      if 'SB' in i:
         spwmap=SB_spwmap
      applycal(vis=data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.ms'),
               gaintable=data_params[i][selectedVis].replace('.ms','_'+mode+'_'+selfcalmode+str(iteration)+'.g'),
               interp='linearPD', calwt=True,spwmap=spwmap,applymode='calonly')
      data_params[i]['selfcal_spwmap'].append(spwmap) # Add spw map to the dictionary for use later

   if evalImage:
      print(vislist)
      estimate_SNR(prefix+'_'+mode+'_'+prevselfcalmode+str(iteration)+'.image.tt0', disk_mask=noisemasks[0], 
             noise_mask=noisemasks[1])
      #create single image using all visibilities; model column always writes with parallel=False
      os.system('rm -rf '+prefix+'_'+mode+'_'+prevselfcalmode+str(iteration)+'_post*')
      if nterms==1:
         startmodel=[prefix+'_'+mode+'_'+prevselfcalmode+str(iteration)+'.model.tt0']
      elif nterms==2:
         startmodel=[prefix+'_'+mode+'_'+prevselfcalmode+str(iteration)+'.model.tt0',prefix+'_'+mode+'_'+prevselfcalmode+str(iteration)+'.model.tt1']
      if (nterms == 2) or (nterms ==1):
         tclean_wrapper(vis=vislist, imagename=prefix+'_'+mode+'_'+prevselfcalmode+str(iteration)+'_post',scales=scales, nsigma=nsigma,
                  savemodel='none',parallel=parallel,cellsize=cellsize,imsize=imsize,sidelobethreshold=sidelobethreshold,
                  noisethreshold=noisethreshold,lownoisethreshold=lownoisethreshold,smoothfactor=smoothfactor,nterms=nterms,
                  niter=0,startmodel=startmodel,robust=robust,uvtaper=[uvtaper])
         estimate_SNR(prefix+'_'+mode+'_'+prevselfcalmode+str(iteration)+'_post'+'.image.tt0', disk_mask=noisemasks[0], 
             noise_mask=noisemasks[1])



def rank_refants(vis):
     # Get the antenna names and offsets.

     msmd = casatools.msmetadata()
     tb = casatools.table()

     msmd.open(vis)
     names = msmd.antennanames()
     offset = [msmd.antennaoffset(name) for name in names]
     msmd.close()

     # Calculate the mean longitude and latitude.

     mean_longitude = numpy.mean([offset[i]["longitude offset"]\
             ['value'] for i in range(len(names))])
     mean_latitude = numpy.mean([offset[i]["latitude offset"]\
             ['value'] for i in range(len(names))])

     # Calculate the offsets from the center.

     offsets = [numpy.sqrt((offset[i]["longitude offset"]['value'] -\
             mean_longitude)**2 + (offset[i]["latitude offset"]\
             ['value'] - mean_latitude)**2) for i in \
             range(len(names))]

     # Calculate the number of flags for each antenna.

     nflags = [tb.calc('[select from '+vis+' where ANTENNA1=='+\
             str(i)+' giving  [ntrue(FLAG)]]')['0'].sum() for i in \
             range(len(names))]

     # Calculate a score based on those two.

     score = [offsets[i] / max(offsets) + nflags[i] / max(nflags) \
             for i in range(len(names))]

     # Print out the antenna scores.

     print("Scores for "+vis)
     for i in numpy.argsort(score):
         print(names[i], score[i])

     # Return the antenna names sorted by score.

     return ','.join(numpy.array(names)[numpy.argsort(score)])

def get_sensitivity(data_params,specmode='mfs',spw=[],chan=0,cellsize='0.025arcsec',imsize=1600,robust=0.5,uvtaper=''):
   sensitivities=np.zeros(len(data_params.keys()))
   counter=0
   scalefactor=1.0
   spwstring=''
   for key in data_params.keys():
      if specmode=='mfs':
         spw=data_params[key]['cont_spws'].tolist()
         vis=data_params[key]['vis_final']
         scalefactor=2.5
      if (type(spw[0]) == str):
         spwstring=''.join(str(spwnum)+',' for spwnum in spw)[:-1]
      elif (type(spw[0]) == list):
          spwstring=spw[0][counter]
      if specmode=='cube':
         print('Using Cube Mode')
         vis=data_params[key]['vis_contsub']
         spwstring=spwstring+':'+str(chan)+'~'+str(chan)
         print(spwstring)
      im.open(vis)
      im.selectvis(field='',spw=spwstring)
      im.defineimage(mode=specmode,stokes='I',spw=spw,cellx=cellsize,celly=cellsize,nx=imsize,ny=imsize)  
      im.weight(type='briggs',robust=robust)  
      if uvtaper != '':
         if 'klambda' in uvtaper:
            uvtaper=uvtaper.replace('klambda','')
            uvtaperflt=float(uvtaper)
            bmaj=str(206.0/uvtaperflt)+'arcsec'
            bmin=bmaj
            bpa='0.0deg'
         if 'arcsec' in uvtaper:
            bmaj=uvtaper
            bmin=uvtaper
            bpa='0.0deg'
         print('uvtaper: '+bmaj+' '+bmin+' '+bpa)
         im.filter(type='gaussian', bmaj=bmaj, bmin=bmin, bpa=bpa)
      try:
          sens=im.apparentsens()
      except:
          print('#')
          print('# Sensisitivity Calculation failed for '+vis)
          print('# Continuing to next MS') 
          print('# Data in this spw/MS may be flagged')
          print('#')
          continue
      print(vis,'Briggs Sensitivity = ', sens[1])
      print(vis,'Relative to Natural Weighting = ', sens[2])  
      sensitivities[counter]=sens[1]*scalefactor
      counter+=1
   estsens=np.sum(sensitivities)/float(counter)/(float(counter))**0.5
   return estsens

def get_solints_simple(vislist,scantimesdict,integrationtimes):
   all_integrations=np.array([])
   for vis in vislist:
      targets=integrationtimes[vis].keys()
      for target in targets:
         all_integrations=np.append(all_integrations,integrationtimes[vis][target])
   integration_time=np.max(all_integrations) # use the longest integration time from all MS files

   allscantimes=np.array([])
   for vis in vislist: # use if we put all scan times from all MSes into single array
      targets=scantimesdict[vis].keys()
      for target in targets:
         allscantimes=np.append(allscantimes,scantimesdict[vis][target])
      #mix of short and long baseline data could have differing integration times and hence solints

   max_scantime=np.median(allscantimes)
   median_scantime=np.max(allscantimes)
   min_scantime=np.min(allscantimes)
   
   solints=np.array([])
   solint=max_scantime/2.0
   n_scans=len(allscantimes)
   while solint > 1.90*integration_time:      #1.1*integration_time will ensure that a single int will not be returned such that solint='int' can be appended to the final list.
      ints_per_solint=solint/integration_time
      if ints_per_solint.is_integer():
         solint=solint
      else:
         remainder=ints_per_solint-float(int(ints_per_solint))     # calculate delta_T greater than an a fixed multile of integrations
         solint=solint-remainder*integration_time # add remainder to make solint a fixed number of integrations

      ints_per_solint=float(int(ints_per_solint))
      #print('Checking solint = ',ints_per_solint*integration_time)
      delta=test_truncated_scans(ints_per_solint, allscantimes,integration_time) 
      solint=(ints_per_solint+delta)*integration_time
      
      solints=np.append(solints,[solint])                       # add solint to list of solints now that it is an integer number of integrations
      solint = solint/2.0  
      #print('Next solint: ',solint)                                        #divide solint by 2.0 for next solint

   solints_list=[]
   for solint in solints:
      solint_string='{:0.2f}s'.format(solint)
      solints_list.append(solint_string)
   solints_list.insert(0,'inf')
   solints_list.append('int')
   gaincal_combine=['spw']*len(solints_list)
   #insert another solint, but with combine='scan,spw' for long baseline data to align average phases of each EB
   solints_list.insert(0,'inf')
   gaincal_combine.insert(0,'spw,scan')
   return solints_list,gaincal_combine

def fetch_scan_times(vislist,targets):
   listdict={}
   scantimesdict={}
   integrationsdict={}
   integrationtimesdict={}
   integrationtime=np.array([])
   n_spws=np.array([])
   min_spws=np.array([])
   spwslist=np.array([])
   for vis in vislist:
      listdict[vis]=listobs(vis)
      scantimesdict[vis]={}
      integrationsdict[vis]={}
      integrationtimesdict[vis]={}
      keylist=list(listdict[vis].keys())  
      for target in targets:
         countscans=0
         scantimes=np.array([])
         integrations=np.array([])
         for key in keylist:
            if 'scan' in key and listdict[vis][key]['0']['FieldName']==target:
               countscans+=1
               scantime=(listdict[vis][key]['0']['EndTime']- listdict[vis][key]['0']['BeginTime'])*86400.0
               ints_per_scan=np.round(scantime/listdict[vis][key]['0']['IntegrationTime'])
               integrationtime=np.append(integrationtime,np.array([listdict[vis][key]['0']['IntegrationTime']]))
               #print('Key:', key,scantime)
               scantimes=np.append(scantimes,np.array([scantime]))
               integrations=np.append(integrations,np.array([ints_per_scan]))
               n_spws=np.append(len(listdict[vis][key]['0']['SpwIds']),n_spws)
               min_spws=np.append(np.min(listdict[vis][key]['0']['SpwIds']),min_spws)
               spwslist=np.append(listdict[vis][key]['0']['SpwIds'],spwslist)
               #print(scantimes)

         scantimesdict[vis][target]=scantimes.copy()
         #assume each band only has a single integration time
         integrationtimesdict[vis][target]=np.median(integrationtime)
         integrationsdict[vis][target]=integrations.copy()
   if np.mean(n_spws) != np.max(n_spws):
      print('WARNING, INCONSISTENT NUMBER OF SPWS IN SCANS/MSes')
   if np.max(min_spws) != np.min(min_spws):
      print('WARNING, INCONSISTENT MINIMUM SPW IN SCANS/MSes')
   spwslist=np.unique(spwslist).astype(int)
   return listdict,scantimesdict,integrationsdict,integrationtimesdict, integrationtime,np.max(n_spws),np.min(min_spws),spwslist

def test_truncated_scans(ints_per_solint, allscantimes,integration_time ):
   delta_ints_per_solint=[0 , -1, 1,-2,2]
   n_truncated_scans=np.zeros(len(delta_ints_per_solint))
   n_remaining_ints=np.zeros(len(delta_ints_per_solint))
   min_index=0
   for i in range(len(delta_ints_per_solint)):
      diff_ints_per_scan=((allscantimes-((ints_per_solint+delta_ints_per_solint[i])*integration_time))/integration_time)+0.5
      diff_ints_per_scan=diff_ints_per_scan.astype(int)
      trimmed_scans=( (diff_ints_per_scan > 0.0)  & (diff_ints_per_scan < ints_per_solint+delta_ints_per_solint[i])).nonzero()
      if len(trimmed_scans[0]) >0:
         n_remaining_ints[i]=np.max(diff_ints_per_scan[trimmed_scans[0]])
      else:
         n_remaining_ints[i]=0.0
      #print((ints_per_solint+delta_ints_per_solint[i])*integration_time,ints_per_solint+delta_ints_per_solint[i],  diff_ints_per_scan)
      
      #print('Max ints remaining: ', n_remaining_ints[i])
      #print('N truncated scans: ', len(trimmed_scans[0]))
      n_truncated_scans[i]=len(trimmed_scans[0])
      # check if there are fewer truncated scans in the current trial and if
      # if one trial has more scans left off or fewer. Favor more left off, such that remainder might be able to 
      # find a solution
      # if ((i > 0) and (n_truncated_scans[i] <= n_truncated_scans[min_index]):   # if we don't care about the amount of 
      #if ((i > 0) and (n_truncated_scans[i] <= n_truncated_scans[min_index]) and (n_remaining_ints[i] > n_remaining_ints[min_index])):
      if ((i > 0) and (n_truncated_scans[i] <= n_truncated_scans[min_index]) and (n_remaining_ints[i] < n_remaining_ints[min_index])):
         min_index=i
      #print(delta_ints_per_solint[min_index])
   return delta_ints_per_solint[min_index]


def plot_ants_flagging_colored(filename,vis,gaintable):
   names, offset_x,offset_y, offsets, nflags, nunflagged,fracflagged=get_flagged_solns_per_ant(gaintable,vis)
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   ants_zero_flagging=np.where(fracflagged == 0.0)
   ants_lt10pct_flagging=((fracflagged <= 0.1) & (fracflagged > 0.0)).nonzero()
   ants_lt25pct_flagging=((fracflagged <= 0.25) & (fracflagged > 0.10)).nonzero()
   ants_lt50pct_flagging=((fracflagged <= 0.5) & (fracflagged > 0.25)).nonzero()
   ants_lt75pct_flagging=((fracflagged <= 0.75) & (fracflagged > 0.5)).nonzero()
   ants_gt75pct_flagging=np.where(fracflagged > 0.75)
   fig, ax = plt.subplots(1,1,figsize=(12, 12))
   ax.scatter(offset_x[ants_zero_flagging[0]],offset_y[ants_zero_flagging[0]],marker='o',color='green',label='No Flagging',s=120)
   ax.scatter(offset_x[ants_lt10pct_flagging[0]],offset_y[ants_lt10pct_flagging[0]],marker='o',color='blue',label='<10% Flagging',s=120)
   ax.scatter(offset_x[ants_lt25pct_flagging[0]],offset_y[ants_lt25pct_flagging[0]],marker='o',color='yellow',label='<25% Flagging',s=120)
   ax.scatter(offset_x[ants_lt50pct_flagging[0]],offset_y[ants_lt50pct_flagging[0]],marker='o',color='magenta',label='<50% Flagging',s=120)
   ax.scatter(offset_x[ants_lt75pct_flagging[0]],offset_y[ants_lt75pct_flagging[0]],marker='o',color='cyan',label='<75% Flagging',s=120)
   ax.scatter(offset_x[ants_gt75pct_flagging[0]],offset_y[ants_gt75pct_flagging[0]],marker='o',color='black',label='>75% Flagging',s=120)
   ax.legend(fontsize=20)
   for i in range(len(names)):
      ax.text(offset_x[i],offset_y[i],names[i])
   ax.set_xlabel('Latitude Offset (m)',fontsize=20)
   ax.set_ylabel('Longitude Offset (m)',fontsize=20)
   ax.set_title('Antenna Positions colorized by Selfcal Flagging',fontsize=20)
   plt.savefig(filename,dpi=200.0)
   plt.close()

def plot_image(filename,outname,min=None,max=None):
   header=imhead(filename)
   size=np.max(header['shape'])
   if min == None:
      imview(raster={'file': filename, 'scaling': -1},\
          zoom={'blc': [int(size/4),int(size/4)],\
                'trc': [int(size-size/4),int(size-size/4)]},\
          out={'file': outname, 'orient': 'landscape'})
   else:
      imview(raster={'file': filename, 'scaling': -1, 'range': [min,max]},\
          zoom={'blc': [int(size/4),int(size/4)],\
                'trc': [int(size-size/4),int(size-size/4)]},\
          out={'file': outname, 'orient': 'landscape'})
   #make image square since imview makes it a strange dimension
   im = Image.open(outname)
   width, height = im.size
   if height > width:
      remainder=height-width
      trim_amount=int(remainder/2.0)
      im1=im.crop((0,trim_amount,width-1,height-trim_amount-1))
   else:
      remainder=width-height
      trim_amount=int(remainder/2.0)
      im1=im.crop((trim_amount,height-1,width,width-trim_amount-1,0))
   im1.save(outname)

def get_flagged_solns_per_ant(gaintable,vis):
     # Get the antenna names and offsets.

     msmd = casatools.msmetadata()
     tb = casatools.table()

     msmd.open(vis)
     names = msmd.antennanames()
     offset = [msmd.antennaoffset(name) for name in names]
     msmd.close()

     # Calculate the mean longitude and latitude.

     mean_longitude = numpy.mean([offset[i]["longitude offset"]\
             ['value'] for i in range(len(names))])
     mean_latitude = numpy.mean([offset[i]["latitude offset"]\
             ['value'] for i in range(len(names))])

     # Calculate the offsets from the center.

     offsets = [numpy.sqrt((offset[i]["longitude offset"]['value'] -\
             mean_longitude)**2 + (offset[i]["latitude offset"]\
             ['value'] - mean_latitude)**2) for i in \
             range(len(names))]
     offset_y=[(offset[i]["latitude offset"]['value']) for i in \
             range(len(names))]
     offset_x=[(offset[i]["longitude offset"]['value']) for i in \
             range(len(names))]
     # Calculate the number of flags for each antenna.

     nflags = [tb.calc('[select from '+gaintable+' where ANTENNA1=='+\
             str(i)+' giving  [ntrue(FLAG)]]')['0'].sum() for i in \
             range(len(names))]
     nunflagged = [tb.calc('[select from '+gaintable+' where ANTENNA1=='+\
             str(i)+' giving  [nfalse(FLAG)]]')['0'].sum() for i in \
             range(len(names))]

     fracflagged=np.array(nflags)/(np.array(nflags)+np.array(nunflagged))
     # Calculate a score based on those two.
     return names, np.array(offset_x),np.array(offset_y),offsets, nflags, nunflagged,fracflagged



