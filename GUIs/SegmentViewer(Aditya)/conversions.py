import pickle
import numpy as np
import mne

def get_pkl_array(path):
  with open(path, 'rb') as file:
    data = pickle.load(file)
    metadata = {}
    if len(data) > 1 and isinstance(data[1], dict):
      metadata = data[1]
    elif len(data) > 1:
      print(f"PKL metadata (non-dict): {data[1]}")
    data = np.array(data[0])
  return data, metadata

def get_gdf_array(path):
  raw = mne.io.read_raw_gdf(path, preload=True)
  metadata = {
    'sfreq': raw.info['sfreq'],
    'ch_names': raw.info['ch_names'],
  }
  return raw.get_data(), metadata

def get_xdf_array(file_path):
    """
    Load XDF file and return EEG data array and metadata.
    
    Args:
        file_path: Path to.xdf file
        
    Returns:
        data: numpy array of shape (channels, samples) - EEG data
        metadata: dict with keys:
            - 'sfreq': sampling rate (Hz)
            - 'ch_names': list of channel names
            - 'markers': list of tuples (timestamp, marker_label)
            - 'start_time': recording start timestamp
    """
    try:
        import pyxdf
    except ImportError:
        raise ImportError("pyxdf not installed. Install with: pip install pyxdf")
    
    import numpy as np
    
    # Load XDF file
    streams, header = pyxdf.load_xdf(file_path)
    
    if len(streams) == 0:
        raise ValueError("No streams found in XDF file")
    
    # Find EEG stream (type='EEG')
    eeg_stream = None
    marker_stream = None
    
    for stream in streams:
        stream_type = stream['info']['type'][0]
        if stream_type == 'EEG':
            eeg_stream = stream
        elif stream_type == 'Markers':
            marker_stream = stream
    
    if eeg_stream is None:
        raise ValueError("No EEG stream found in XDF file")
    
    # Extract EEG data (convert from samples x channels to channels x samples)
    eeg_data = eeg_stream['time_series'].T  # Shape: (channels, samples)
    
    # Extract metadata
    sfreq = float(eeg_stream['info']['nominal_srate'][0])
    
    # Extract channel names
    ch_names = []
    if 'desc' in eeg_stream['info']:
        desc = eeg_stream['info']['desc'][0]
        if 'channels' in desc and 'channel' in desc['channels']:
            for ch in desc['channels']['channel']:
                if 'label' in ch:
                    ch_names.append(ch['label'][0])
    
    # If no channel names found, generate default names
    if not ch_names:
        num_channels = int(eeg_stream['info']['channel_count'][0])
        ch_names = [f"Ch{i+1}" for i in range(num_channels)]
    
    # Extract markers if present
    markers = []
    if marker_stream is not None:
        marker_data = marker_stream['time_series']
        marker_timestamps = marker_stream['time_stamps']
        
        for i in range(len(marker_timestamps)):
            # marker_data is array of shape (samples, 1)
            marker_label = marker_data[i][0] if len(marker_data[i]) > 0 else ''
            # Convert timestamp to sample index relative to EEG start
            eeg_start_time = eeg_stream['time_stamps'][0]
            relative_time = marker_timestamps[i] - eeg_start_time
            markers.append((relative_time, marker_label))
    
    metadata = {
        'sfreq': sfreq,
        'ch_names': ch_names,
        'markers': markers,
        'start_time': eeg_stream['time_stamps'][0]
    }
    
    return eeg_data, metadata
