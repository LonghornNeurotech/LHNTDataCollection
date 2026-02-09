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
