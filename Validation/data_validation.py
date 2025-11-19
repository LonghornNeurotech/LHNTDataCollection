
import numpy as np

def detect_nan_inf(data):
  """
  Detect NaN or Inf values in EEG data. 
  INPUT :
    data : np . ndarray , shape ( channels , samples )
    Raw EEG data from board
  
  OUTPUT :
  is_valid : bool
    True if no NaN / Inf found
  nan_channels : list
    Indices of channels containing NaN / Inf
  
  EXAMPLE :
      >>> data = np.array([[1.0 , 2.0 , np.nan] , [3.0 , 4.0 , 5.0]])
      >>> is_valid , bad_ch = detect_nan_inf(data)
      >>> print(is_valid) # False
      >>> print(bad_ch) # [0]
  """
  # Vectorized detection of NaN/Inf across all channels
  # Check for NaN or Inf in each channel (returns boolean array per channel)
  nan_mask = np.isnan(data).any(axis=1)  # True if channel has any NaN
  inf_mask = np.isinf(data).any(axis=1)  # True if channel has any Inf
  
  # Combine both conditions - channels with NaN OR Inf
  bad_channels_mask = nan_mask | inf_mask
  
  # Get indices of bad channels
  nan_channels = np.where(bad_channels_mask)[0].tolist()

  is_valid = len(nan_channels) == 0
  return is_valid, nan_channels

def detect_flatline(data, threshold=0.1):
  """
  Detect flatline channels (constant/near-constant signal).
  INPUT :
    data : np.ndarray, shape (channels, samples)
      EEG data segment
    threshold : float
      Standard deviation threshold in micro volts
      Default : 0.1 uV

  OUTPUT :
    flatline_channels : list
      Indices of flatline channels
    channel_stds : np.ndarray
      Standard deviation of each channel
  REASONING :
    Healthy EEG has std > 1 uV typically
    Flatline means electrode disconnected or broken
  """
  # YOUR IMPLEMENTATION HERE
  # Hint : Calculate std per channel, compare to threshold
  channel_stds = np.std(data, axis=1)
  flatline_channels = np.where(channel_stds < threshold)[0].tolist()
  return flatline_channels, channel_stds


def detect_extreme_noise(data, iqr_threshold=5.0):
  """
  Detect extreme noise / spike artifacts.
  INPUT :
    data : np.ndarray, shape (channels, samples)
    iqr_threshold : float
      Q3-outlier threshold for spike detection
      Default : 1.5
  OUTPUT :
    has_spikes : bool
      True if spikes detected in any channel
    spike_channels : list
      Channels containing spikes
    spike_percentage : float
      Percentage of spiked channels that are spikes
  """
  # YOUR IMPLEMENTATION HERE

  channel_q1 = np.quantile(data, 0.25, axis=1, keepdims=True)
  channel_q3 = np.quantile(data, 0.75, axis=1, keepdims=True)
  iqrs = channel_q3 - channel_q1
  
  spike_mask = (data > (channel_q3 + iqr_threshold*iqrs)) | (data < (channel_q1 - iqr_threshold*iqrs))

  # Count spikes per channel
  spikes_per_channel = np.sum(spike_mask, axis=1)
  
  # Find channels with > 1% spikes
  spike_percentage_per_channel = spikes_per_channel / data.shape[1]
  spike_channels = np.where(spike_percentage_per_channel > 0.01)[0].tolist()
  
  has_spikes = len(spike_channels) > 0
  
  # Average spike percentage across spike channels
  if len(spike_channels) > 0:
    spike_percentage = np.mean(spike_percentage_per_channel[spike_channels]) * 100
  else:
    spike_percentage = 0
  
  return has_spikes, spike_channels, spike_percentage

def detect_channel_duplication(data, correlation_threshold=0.99):
  correlation_matrix = np.corrcoef(data)
  
  # Find high correlations in upper triangle only
  high_corr_mask = np.triu((correlation_matrix > correlation_threshold), k=1)
  
  # Get indices of high correlations
  i_indices, j_indices = np.where(high_corr_mask)
  
  # Convert to list of tuples
  duplicate_pairs = list(zip(i_indices.tolist(), j_indices.tolist()))

  return duplicate_pairs, correlation_matrix

class ValidationPackage:
  def __init__(self, data, timestamp):
    self.data = data
    self.timestamp = timestamp
    self.is_valid = True
    self.quality_metrics = {}

