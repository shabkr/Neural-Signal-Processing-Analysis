import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.fft import fft, ifft
import mne
from mne.viz import plot_topomap

def plot_simEEG(*args):
    """
    plot_simEEG - plot function for MXC's course on neural time series analysis
    """
    if not args:
        raise ValueError('No inputs!')
    elif len(args) == 1:
        EEG = args[0]
        chan, fignum = 0, 0
    elif len(args) == 2:
        EEG, chan = args
        fignum = 0
    elif len(args) == 3:
        EEG, chan, fignum = args

    plt.figure(fignum, figsize=(16,10))
    plt.clf()

    # ERP
    plt.subplot(211)
    plt.plot(EEG.times, np.squeeze(EEG.data[chan,:,:]), linewidth=0.5, color=[.75, .75, .75])
    plt.plot(EEG.times, np.squeeze(np.mean(EEG.data[chan,:,:], axis=1)), 'k', linewidth=3)
    plt.xlabel('Time (s)')
    plt.ylabel('Activity')
    plt.title(f'ERP from channel {chan}')

    # static power spectrum
    hz = np.linspace(0, EEG.srate, EEG.pnts)
    if len(EEG.data.shape) == 3:
        pw = np.mean((2 * np.abs(fft(EEG.data[chan,:,:], axis=0) / EEG.pnts))**2, axis=1)
    else:
        pw = (2 * np.abs(fft(EEG.data[chan,:], axis=0) / EEG.pnts))**2

    plt.subplot(223)
    plt.plot(hz, pw, linewidth=2)
    plt.xlim([0, 40])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Static power spectrum')
    
    # time-frequency analysis
    frex = np.linspace(2, 30, 40)  # frequencies in Hz (hard-coded to 2 to 30 in 40 steps)
    waves = 2 * (np.linspace(3, 10, len(frex)) / (2 * np.pi * frex))**2  # number of wavelet cycles (hard-coded to 3 to 10)

    # setup wavelet and convolution parameters
    wavet = np.arange(-2, 2, 1/EEG.srate)
    halfw = len(wavet) // 2
    nConv = EEG.pnts * EEG.trials + len(wavet) - 1

    # initialize time-frequency matrix
    tf = np.zeros((len(frex), EEG.pnts))

    # spectrum of data
    dataX = fft(np.reshape(EEG.data[chan,:,:], -1, order='F'), n=nConv)
    # loop over frequencies
    for fi in range(len(frex)):
        # create wavelet
        waveX = fft(np.exp(2 * 1j * np.pi * frex[fi] * wavet) * np.exp(-wavet**2 / waves[fi]), n=nConv)
        waveX = waveX / np.max(waveX) # normalize
        
        # convolve
        as_ = ifft(waveX * dataX)
        # trim and reshape
        as_ = np.reshape(as_[halfw:len(as_)-halfw+1], [EEG.pnts, EEG.trials], order='F')

        # power
        tf[fi, :] = np.mean(np.abs(as_), axis=1)

    # show a map of the time-frequency power
    plt.subplot(224)
    plt.contourf(EEG.times, frex, tf, 40, cmap='jet')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.title('Time-frequency plot')

    plt.show()
    
def plot_simEEG_mne(*args):
  """
  plot_simEEG - plot function for MXC's course on neural time series analysis
  """
  if not args:
      raise ValueError('No inputs!')
  elif len(args) == 1:
      EEG = args[0]
      chan, fignum = 0, 0
  elif len(args) == 2:
      EEG, chan = args
      fignum = 0
  elif len(args) == 3:
      EEG, chan, fignum = args

  plt.figure(fignum, figsize=(16,10))
  plt.clf()

  # some convenience variables from EEG
  srate = EEG.info['sfreq']
  pnts = len(EEG.times)
  data = EEG.get_data() # epochs,channels,times
  data = np.transpose(data, axes = (1,2,0)) #channels, times, epochs
  trials = len(EEG.events)

  # ERP
  plt.subplot(211)
  plt.plot(EEG.times, np.squeeze(data[chan,:,:]), linewidth=0.5, color=[.75, .75, .75])
  plt.plot(EEG.times, np.squeeze(np.mean(data[chan,:,:], axis=1)), 'k', linewidth=3)
  plt.xlabel('Time (s)')
  plt.ylabel('Activity')
  plt.title(f'ERP from channel {chan}')

  # static power spectrum
  hz = np.linspace(0, srate, pnts)
  if len(data.shape) == 3:
      pw = np.mean((2 * np.abs(fft(data[chan,:,:], axis=0) / pnts))**2, axis=1)
  else:
      pw = (2 * np.abs(fft(data[chan,:], axis=0) / pnts))**2

  plt.subplot(223)
  plt.plot(hz, pw, linewidth=2)
  plt.xlim([0, 40])
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Power')
  plt.title('Static power spectrum')

  # time-frequency analysis
  frex = np.linspace(2, 30, 40)  # frequencies in Hz (hard-coded to 2 to 30 in 40 steps)
  waves = 2 * (np.linspace(3, 10, len(frex)) / (2 * np.pi * frex))**2  # number of wavelet cycles (hard-coded to 3 to 10)

  # setup wavelet and convolution parameters
  wavet = np.arange(-2, 2, 1/srate)
  halfw = len(wavet) // 2
  nConv = pnts * trials + len(wavet) - 1

  # initialize time-frequency matrix
  tf = np.zeros((len(frex), pnts))

  # spectrum of data
  dataX = fft(np.reshape(data[chan,:,:], -1, order='F'), n=nConv)
  # loop over frequencies
  for fi in range(len(frex)):
      # create wavelet
      waveX = fft(np.exp(2 * 1j * np.pi * frex[fi] * wavet) * np.exp(-wavet**2 / waves[fi]), n=nConv)
      waveX = waveX / np.max(waveX) # normalize
      
      # convolve
      as_ = ifft(waveX * dataX)
      # trim and reshape
      as_ = np.reshape(as_[halfw:len(as_)-halfw+1], [pnts, trials], order='F')

      # power
      tf[fi, :] = np.mean(np.abs(as_), axis=1)

  # show a map of the time-frequency power
  plt.subplot(224)
  plt.contourf(EEG.times, frex, tf, 40, cmap='jet')
  plt.xlabel('Time')
  plt.ylabel('Frequency (Hz)')
  plt.title('Time-frequency plot')

  plt.show()


def topoPlotIndie(eeg, values, title='Topoplot', vlim=(None, None), cmap='RdBu_r', contours=6):
    '''
        eeg = eeg mat file loaded
        values = volt values to plot
    '''
    def pol2cart(theta, rho):
        theta_rad = np.deg2rad(theta)
        x = rho * np.cos(theta_rad)
        y = rho * np.sin(theta_rad)
        return x, y


    head_rad = 0.095
    plot_rad = 0.51
    squeezefac = head_rad/plot_rad

    eeg_chanlocs = []
    for i in range(64):
        local_chanloc = []
        x = list(eeg['chanlocs'][0][0][0][i])
        th = x[1][0][0]
        rd = x[2][0][0]

        th, rd = pol2cart(th,rd)
        eeg_chanlocs.append([rd * squeezefac,th*squeezefac])

    eeg_chanlocs = np.array(eeg_chanlocs)
    
    fig, ax = plt.subplots(figsize=(8,8))
    im, _ = plot_topomap(values, eeg_chanlocs, axes=ax, show=False, 
                         cmap=cmap, ch_type='eeg', size = 200,
                        contours=contours, vlim=vlim)
    plt.colorbar(im)

    # title
    plt.title(title)
    plt.show()
    
def time_to_id(times_arr, time2plot):
    # convert time in ms to time in indices
    return np.argmin(np.abs(times_arr - time2plot))

def read_emptyEEG(mat_file = "../data/emptyEEG.mat",times=None,trials=None):
    '''
    reads emptyEEG.mat file and returns it in an MNE friendly format
        Parameters:
            mat_file (str): string path to data. Default assumes the file is in the data folder
            times (int): integer number of timepoints. Default (none), uses the number of times in the data
            trials (int): integer number of trials. Default (none), uses the number of trials in the data
        Returns:
            mne_data (Epoch): the mne.Epoch formatted version of the data
            lead_field: Whatever the leadfield data is
    '''
    eeg_data = scipy.io.loadmat(mat_file) #loads the .mat file as a dictionary
    eeg = eeg_data['EEG'] #access the actual EEG data within the loaded matlab file
    lf = eeg_data["lf"] #access the leadfield data within the loaded matlab file

    # information from the Matlab array for easy checking
    n_channels = int(eeg['nbchan'][0])
    if trials is None:
        trials = int(eeg['trials'][0])
    time_points = int(eeg['pnts'][0])
    sample_rate = int(eeg['srate'][0])
    xmin = float(eeg['xmin'][0])
    xmax = float(eeg['xmax'][0])
    if times is None:
        times = eeg['times']
    channel_names = [str(item[0]) for item in eeg['chanlocs'][0][0][0]['labels']]

    # note, in Matlab X is front back, but in MNE Y is front back
    # note in MATLAB Y is left right, but in MNE X is left right
    # note in MATLAB the left/right sign is reversed from MNE
    channel_x = [-float(item[0])/1000 for item in eeg['chanlocs'][0][0][0]['Y']] #left right
    channel_y = [float(item[0])/1000 for item in eeg['chanlocs'][0][0][0]['X']] #front back
    channel_z = [float(item[0])/1000 for item in eeg['chanlocs'][0][0][0]['Z']] # up down
    coords = [(channel_x[i],channel_y[i],channel_z[i]) for i in range(len(channel_x))]
    data_array = np.zeros(shape=(trials,n_channels,time_points)) # in matlab, the data was channels, timespoints, trials/epochs, but it  needs to be trials, channels, timepoints

    # Build the MNE version of the EEG data
    mne_info = mne.create_info(
        ch_names = channel_names,
        sfreq = sample_rate,
        ch_types = "eeg",
    )    

    mne_data = mne.EpochsArray(info=mne_info,data=data_array, tmin=xmin)
    mne_data = mne_data.set_montage(mne.channels.make_dig_montage(dict(zip(channel_names, coords)), coord_frame="head"))

    #now, prepare the leadfield data
    # lf in matlab contains the following
    # MEGMethod (empty string)
    # EEGMethod (string)
    # Gain (64 x 3 x 2004 double) <- forward solution?
    # Comment (string)
    # HeadModelType (string)
    # ***** GridLoc (2004 x 3 double) dipoles by xyz coords
    # GridOrient (2004 x 3 double)
    # SurfaceFile
    # InputSurfaces (empty array)
    # Param (empty array)
    # History (1 x 3 cell)
    lead_field = {"GridLoc": lf["GridLoc"][0][0],
                  "Gain": lf["Gain"][0][0]}

    coord_order = np.array([1,0,2])   
    lead_field["GridLoc"] = lead_field["GridLoc"][:,coord_order]/1000
    lead_field["Gain"] = lead_field["Gain"][:,coord_order,:]

    return mne_data, lead_field