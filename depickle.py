import pickle
import numpy as np

with open('right_6.pkl', 'rb') as file:
  data = pickle.load(file)
  print(data[1])
  data = np.array(data[0])
  print(data.shape)