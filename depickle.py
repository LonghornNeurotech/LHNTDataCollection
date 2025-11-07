import numpy as np
import matplotlib.pyplot as plt
from conversions import get_pkl_array, get_gdf_array
'''
Testing script for pkl files to get an idea of how data is represented
'''
  
# data = get_pkl_array('nandini_senthilkumar_Session1/left_1.pkl')
data = get_gdf_array('BCICIV_2a_gdf/A01T.gdf')
print(data.shape)
t = np.linspace(0, 14, data.shape[1])

plt.figure(figsize=(12, 20))

for i in range(data.shape[0]):
  plt.subplot(data.shape[0], 1, i + 1)
  plt.plot(t, data[i])
  plt.ylabel(f'Channel {i + 1}')
  channel_q1 = np.quantile(data[i], 0.25, keepdims=True)
  channel_q3 = np.quantile(data[i], 0.75, keepdims=True)
  iqr = channel_q3 - channel_q1
  plt.ylim(channel_q1 - iqr * 10, channel_q3 + iqr * 10)
  if i == data.shape[0] - 1:
    plt.xlabel('Time (s)')
  plt.legend([f'Channel {i+1}'])

plt.tight_layout()
plt.show()


  # fft_out=fft.fft(data[15])
  # freqs = fft.fftfreq(data.shape[1], d=(t[1]-t[0]))

  # reconstruct = fft.ifft(fft_out)
  # plt.subplot(2, 1, 1)
  # plt.plot(t, reconstruct)
  # plt.subplot(2, 1, 2)
  # plt.plot(t, data[15])

  # print(np.dot(data[15], data[15]) - np.dot(data[15], reconstruct))
  # plt.show()
