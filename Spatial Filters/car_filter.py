# Andy Teng

import numpy as np
import mne


class CARFilter:

    def __init__(self, edfpath):
        self.edfpath = edfpath
        self.raw = mne.io.read_raw_edf(self.edfpath, preload=True)
        self.data, self.times = self.raw[:, :]
        self.n_channels = len(self.raw.ch_names)
        self.car_data = None

    def apply_car(self):
        self.car_data = self.data - np.mean(self.data, axis=0, keepdims=True)
        return self.car_data

    def get_data(self):
        if self.car_data is None:
            raise RuntimeError("CAR filter has not been applied yet.")
        return self.car_data


