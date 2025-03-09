import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from scipy.signal import butter, lfilter, iirnotch
import platform
import serial
import serial.tools.list_ports
import torch

def find_serial_port():
    """
    Automatically find the correct serial port for the device across different operating systems.
    
    Returns:
        str: The path of the detected serial port, or None if not found.
    """
    system = platform.system()
    ports = list(serial.tools.list_ports.comports())
    
    for port in ports:
        if system == "Darwin":  # macOS
            if any(identifier in port.device.lower() for identifier in ["usbserial", "cu.usbmodem", "tty.usbserial"]):
                return port.device
        elif system == "Windows":
            if "com" in port.device.lower():
                return port.device
        elif system == "Linux":
            if "ttyUSB" in port.device or "ttyACM" in port.device:
                return port.device
    
    return None

def partition_trial_data(trial_data, min_boundary = 125, window_ratio = 0.1):
            # Get total number of time points
            total_length = trial_data.shape[2]  # Use dim=2 for time axis (1,16,total_length)

            # Define window size (10% of total time points)
            window_size = max(int(window_ratio * total_length), 1)

            # Ensure there's enough room for at least 125 points in beginning and end
            max_start_idx = total_length - window_size - min_boundary

            if max_start_idx <= min_boundary:
                print("Warning: Not enough data points to ensure a valid partition with 125-point buffers.")
                # Handle the situation
                trial_beginning = trial_data[:, :, :min_boundary]  # Use the first 125 points
                test_window = trial_data[:, :, min_boundary:min_boundary + window_size]  # Just after the boundary
                trial_end = trial_data[:, :, min_boundary + window_size:]  # The rest
            else:
                # Randomly select start index ensuring the window is not too close to the edges
                window_start = np.random.randint(min_boundary, max_start_idx)

                # Partition along the time axis (dim=2)
                trial_beginning = trial_data[:, :, :window_start]  # Before the window
                test_window = trial_data[:, :, window_start:window_start + window_size]  # Random window
                trial_end = trial_data[:, :, window_start + window_size:]  # After the window

            return trial_beginning, test_window, trial_end


class EEGProcessor:
    def __init__(self):
        # Initialize BrainFlow
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()

        # UNCOMMENT THE FOLLOWING 3 LINES FOR REAL BOARD
        #serial_port = find_serial_port()
        #params.serial_port = serial_port
        #self.board_id = BoardIds.CYTON_DAISY_BOARD.value

        # COMMENT OUT THE FOLLOWING LINE FOR REAL BOARD
        self.board_id = BoardIds.SYNTHETIC_BOARD.value

        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()
        self.board.start_stream()
        print("BrainFlow streaming started...")

        # Sampling rate and window size
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.window_size_sec = 14  # seconds
        self.window_size_samples = int(self.window_size_sec * self.sampling_rate)

        # we set raw window size to 17 seconds
        self.window_size_raw = int(17 * self.sampling_rate)
        self.lowcut = 1.0
        self.highcut = 50.0
        self.notch = 60.0

        # Get EEG channels
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)

        # Initialize buffers
        self.raw_data_buffer = np.empty((len(self.eeg_channels), 0))
        self.processed_data_buffer = np.empty((len(self.eeg_channels), 0))

    def stop(self):
        # Stop the data stream and release the session
        self.board.stop_stream()
        self.board.release_session()
        print("BrainFlow streaming stopped.")

    def get_recent_data(self, duration=14):
        """
        Returns the most recent n seconds of processed EEG data.

        The data is bandpass filtered, notch filtered, and z-scored.
        Each data point is filtered only once.
        """
        data = self.board.get_board_data() 
        if data.shape[1] == 0:
            # No new data
            pass
        else:
        
            # Append new raw data to the raw_data_buffer
            eeg_data = data[self.eeg_channels, :]
            self.raw_data_buffer = np.hstack((self.raw_data_buffer, eeg_data))

            # Process new data
            new_processed_data = np.empty(self.raw_data_buffer.shape)
            # It is important to process each channel separately (why?)
            for i in range(len(self.eeg_channels)):

                # it is important to use the whole buffer for filtering (why?)
                # Get the channel data
                channel_data = self.raw_data_buffer[i, :].copy()

                # Bandpass filter
                b, a = butter(2, [self.lowcut, self.highcut], btype='band', fs=self.sampling_rate)
                channel_data = lfilter(b, a, channel_data)
                
                # Notch filter
                b, a = iirnotch(self.notch, 30, fs=self.sampling_rate)
                channel_data = lfilter(b, a, channel_data)

                # add channel dimension to channel_data
                new_processed_data[i, :] =  channel_data

            
            self.processed_data_buffer = np.hstack((self.processed_data_buffer, new_processed_data))

            max_buffer_size = self.window_size_samples * 2
            if self.raw_data_buffer.shape[1] > self.window_size_raw:
                self.raw_data_buffer = self.raw_data_buffer[:, -self.window_size_raw:]
            if self.processed_data_buffer.shape[1] > max_buffer_size:
                self.processed_data_buffer = self.processed_data_buffer[:, -max_buffer_size:]

        if self.processed_data_buffer.shape[1] >= self.window_size_samples:
            recent_data = self.processed_data_buffer[:, -self.window_size_samples:]
        else:
            recent_data = self.processed_data_buffer
        
        # only as long as duration
        model_input =  recent_data[:, -int(duration * self.sampling_rate):]
        # add dimension for model input
        model_input = np.array([model_input])
        # convert to torch tensor
        model_input = torch.from_numpy(model_input)
        # change dtype to float32
        model_input = model_input.float()

        return model_input