import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft

with open('right_6.pkl', 'rb') as file:
  data = pickle.load(file)
  print(data[1])
  data = np.array(data[0])
  print(data[1].shape)
  
  t = np.linspace(0, 14, data.shape[1])
  print(len(t))
  
  plt.figure(figsize=(12, 20))
  
  for i in range(16):
    plt.subplot(16, 1, i + 1)
    plt.plot(t, data[i])
    plt.ylabel(f'Channel {i + 1}')
    if i == 15:
      plt.xlabel('Time (s)')
    plt.legend(['Right 6'])
  
  plt.tight_layout()
  plt.show()

  fft_out=fft.fft(data[15])
  freqs = fft.fftfreq(data.shape[1], d=(t[1]-t[0]))

  reconstruct = fft.ifft(fft_out)
  plt.subplot(2, 1, 1)
  plt.plot(t, reconstruct)
  plt.subplot(2, 1, 2)
  plt.plot(t, data[15])

  print(np.dot(data[15], data[15]) - np.dot(data[15], reconstruct))
  plt.show()
