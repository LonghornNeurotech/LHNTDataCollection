import pickle
import numpy as np
import mne

def get_pkl_array(path):
  with open(path, 'rb') as file:
    data = pickle.load(file)
    print(data[1])
    data = np.array(data[0])
  return data

def get_gdf_array(path):
  raw = mne.io.read_raw_gdf(path)
  return raw.get_data()