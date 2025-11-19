import numpy as np
from data_validation import *

def test_nan_detection():
  data = np.random.randn(8, 100)
  data[2, 50] = np.nan
  is_valid, bad_ch = detect_nan_inf(data)
  assert not is_valid, "Should detect NaN"
  assert 2 in bad_ch, "Should identify channel 2"
  print("     NaN detection works")

def test_flatline():
  data = np.random.randn(8, 100)
  data[5, :] = 0.0
  flat_ch, stds = detect_flatline(data)
  assert 5 in flat_ch, "Should detect flatline in channel 5"
  print("     Flatline detection works")

def test_noise():
  data = np.random.randn(8, 100)
  data[3, 10:20] = 50.0
  has_spike, spike_ch, pct = detect_extreme_noise(data)
  assert has_spike, "Should detect spike"
  assert len(spike_ch) == 1, f"One channel must be detected: {spike_ch}"
  print("     Noise detection works")


test_nan_detection()
test_flatline()
test_noise()
print("All Tests Passed!")