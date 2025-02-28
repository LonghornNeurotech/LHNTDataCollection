import pygame
import sys
import time
import pickle
import numpy as np
from scipy.signal import butter, lfilter, iirnotch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from checkbox import Checkbox
import platform
import serial
import serial.tools.list_ports
import datetime
from datetime import timedelta
import pandas as pd
from boxsdk import Client, OAuth2
import zipfile
import os

# test change
def authenticate():
    """
    Returns client authorization for Box uploading, must be called before download_file, upload_file, or update_file can work.

    """
    client_id = 'bq09tmdv7v99bcivrw6z5z6hdgny907i'
    client_secret = 'secret'
    # dev token HAS to be refreshed during every session for now, it only lasts an hour
    developer_token = 'token'
    auth = OAuth2(
        client_id=client_id,
        client_secret=client_secret,
        access_token=developer_token
    )
    return Client(auth)

def download_file(client, file_id, download_dir):
    """
    Downloads a file from Box specified by file_id into the path specified by download_dir.
    Exclusively used for downloading the user table
    file_id can be found at the end of the URL of the specific file you want, i.e. https://utexas.app.box.com/file/file_id
    
    Returns the path to the downloaded file as a string
    """
    file = client.file(file_id).get()
    download_path = os.path.join(download_dir, file.name)
    with open(download_path, 'wb') as open_file:
        file.download_to(open_file)
    print(f'{file.name} has been downloaded to {download_path}')
    return download_path

def upload_file(client, folder_id, local_file_path):
    """
    Uploads a file specified by local_file_path to the Box folder specified by folder_id.  
    folder_id can be found at the end of the URL of the specific file you want, i.e. https://utexas.app.box.com/folder/folder_id
    
    Returns the ID of the new file created 
    """
    file_name = os.path.basename(local_file_path)
    uploaded_file = client.folder(folder_id).upload(local_file_path, file_name)
    print(f'File {file_name} uploaded to Box folder {folder_id} with ID {uploaded_file.id}')
    return uploaded_file.id

def update_file(client, file_id, new_file_path):
    """
    Overwrites an existing file in Box by uploading a new version.
    
    Parameters:
    client (Client): The authenticated Box client object.
    file_id (str): The ID of the file to be updated.
    new_file_path (str): Path to the new file that will replace the existing one.
    
    Returns:
    str: The name of the file that was updated.
    """
    if not os.path.exists(new_file_path):
        raise FileNotFoundError(f"The file {new_file_path} does not exist.")
    
    try:
        # Get the file object from Box by its ID
        box_file = client.file(file_id).get()
        
        # Use update_contents_with_stream to handle the file update
        with open(new_file_path, 'rb') as new_file:
            updated_file = box_file.update_contents_with_stream(new_file)
            
        return updated_file.name
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def zip_directory(dir_name, zip_name):
    """
    Zips the directory specified by zip_name.

    Assumes the the directory to be zipped is located within the script directory
    
    Writes the zipped file back into the script directory

    """

    # Establish paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sub_dir_path = os.path.join(script_dir, dir_name)
    zip_file_path = os.path.join(script_dir, zip_name)

     # Create the zip file
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the subdirectory and add each file to the zip file
        for root, dirs, files in os.walk(sub_dir_path):
            for file in files:
                # Get the full file path
                full_path = os.path.join(root, file)
                # Add file to the zip file, keeping the directory structure
                zipf.write(full_path, os.path.relpath(full_path, sub_dir_path))
    
    # Return the path to the created zip file
    return zip_file_path

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

# Function to check if a user is in the table and last survey time
def check_user_table(table, eid):

    """
    Not currently in use.

    Function for checking if a user needs to finish the metadata portion of the survey.
    Decides based on whether the user is present and if they recorded a session within the last 12 hours.

    Returns 0 or 1,
        0 = Don't do the second half of the survey
        1 = Do the second half of the survey
    """
    # Check all rows for the name
    for i in range(table.shape[0]):
        if table.loc[i, 'EID'] == eid:
            table_time = datetime.datetime.strptime(table.loc[i, 'LastTime'], '%Y-%m-%d %H:%M:%S.%f')
            # Update 'SessionNum' using loc to avoid the chained assignment warning
            table.loc[i, 'SessionNum'] += 1
            delta = datetime.datetime.now() - table_time
            seconds = delta.total_seconds()
            if seconds < 43200:  # 12 hours = 12*60*60 = 43200
                # Don't do survey
                return 0
            break
    # Do survey
    return 1

def track_user(table, first, last, eid, caffeine_mg, meal_size, meal_desc, exercised_TF,
               exercise_desc, stim_use_TF = 0, hair_product= '', other_hair= ''):
    """
    Updates the user table with responses collected from the survey.

    Returns: The modified user table and the individual row of metadata for the specified user (both as pandas dataframes). Handles session number tracking internally 
    by either initializing SessionNum to 1 (new user) or incrementing by 1 (existing user).
    """
    present = 0
    index = 0
    # Check if the user is already in the table
    for i in range(table.shape[0]):
        if table['First'][i] == first and table['Last'][i] == last and table['EID'][i] == eid:
            present = 1
            index = i
            break

    # If they're there, update their values with new responses
    if present == 1:
        table.at[index, 'StimulantUse'] = stim_use_TF
        table.at[index, 'CaffeineMg'] = caffeine_mg
        table.at[index, 'MealSize'] = meal_size
        table.at[index, 'MealDesc'] = meal_desc
        table.at[index, 'Exercised'] = exercised_TF
        table.at[index, 'ExerciseDesc'] = exercise_desc
        table.at[index, 'HairProduct'] = hair_product
        table.at[index, 'OtherHair'] = other_hair
        table.at[index, 'LastTime'] = str(datetime.datetime.now())
        table.at[index, 'SessionNum'] = table.at[index, 'SessionNum'] + 1

    # If they are not there, create a new row with their responses
    else:
        id = table.shape[0] + 1
        new_row = pd.DataFrame({
            'ID': [id],
            'First': [first],
            'Last': [last],
            'EID': [eid],
            'StimulantUse': [stim_use_TF],
            'CaffeineMg': [caffeine_mg],
            'MealSize': [meal_size],
            'MealDesc': [meal_desc],
            'Exercised': [exercised_TF],
            'ExerciseDesc': [exercise_desc],
            'HairProduct': [hair_product],
            'OtherHair': [other_hair],
            'LastTime': [str(datetime.datetime.now())],
            'SessionNum': [1]  # New user, start at session 1
        })
        table = pd.concat([table, new_row], ignore_index=True)
        index = table.shape[0] - 1  # Update index to the new row

    # Return the updated table and the row as a DataFrame
    return table, table.iloc[[index]]

def get_user_data(table, eid):
    """
    Provides an immutable method for accessing metadata rows in the table. 
    Primarily used when you know a user is present. Will return an empty dataframe if absent.

    Parameters: Table to search through, eid string to search for
    Returns: Row of metadata. 
    """
    row = table.loc[table['EID'] == eid]
    return row

def create_user_directory(first_name, last_name, session_num):
    """
    Creates a new folder in the current script directory.
    Directory Name = {first_name}_{last_name}_Session{session_num}
    Returns the name of the directory for later manipulation of the directory.
    """
    dir_name = first_name + '_' + last_name + '_' + 'Session' + str(session_num)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    new_dir_path = os.path.join(script_dir, dir_name)
    os.mkdir(new_dir_path)
    return dir_name

class EEGProcessor:
    def __init__(self):
        # Initialize BrainFlow
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        #serial_port = find_serial_port()
        #params.serial_port = serial_port
        #self.board_id = BoardIds.CYTON_DAISY_BOARD.value
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
        Returns the most recent 14 seconds of processed EEG data.

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
        return recent_data[:, -int(duration * self.sampling_rate):]
    
# Save last 14 seconds of signal and metadata to its own .npy file in the session directory
def save_data(eeg_processor, duration, direction, trial_num, directory):
    sig = eeg_processor.get_recent_data(duration) 
    print(f"Duration of trial: {duration:.2f}s, Data shape: {sig.shape}")
    # Checks for nan's or if there are any channels that have a standard deviation of 1
    if np.isnan(sig).any():
        return None
    for i in range(sig.shape[0]):
        if np.std(sig[i, :]) == 0:
            return None
    #Establish a filename - I think maybe we could do [Direction]_[Number].npy but maybe we could just work that out
    filename = f"{direction}_{trial_num}.npy"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    intermediate = os.path.join(script_dir, directory)
    filepath = os.path.join(intermediate, filename)

    # Save the signal into a .npy file
    np.save(filepath, sig)
    print(f"Signal saved to: {filepath}")
    

def main():
    eeg_processor = EEGProcessor()
    
    # Load table from Box
    client = authenticate()
    file_id = '1679766376012'
    download_dir = os.path.dirname(os.path.abspath(__file__))
    table_path = download_file(client, file_id, download_dir)
    print("Table saved to " + table_path)
    user_table = pd.read_csv(table_path)

    # Initialize Pygame
    pygame.init()
    infoObject = pygame.display.Info()
    screen = pygame.display.set_mode((infoObject.current_w, infoObject.current_h), pygame.FULLSCREEN)
    pygame.display.set_caption("Motor Imagery Task")

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    # Fonts
    large_font = pygame.font.SysFont(None, 200)
    medium_font = pygame.font.SysFont(None, 100)
    small_font = pygame.font.SysFont(None, 50)

    # Control Variables
    running = True
    in_menu = True
    in_input = False
    in_trial_menu = False
    in_questionaire_subject = False
    in_questionaire_physiological = False
    in_buffer_screen = False
    in_after_session_menu = False
    trial_number = 1
    total_trials = 1 # Default number of trials
    time_between_sessions = 180 # number of seconds to wait between sessions of data collection
    start_enable_time = time.time() # the time at/after which the start button is enabled
    saved_questionnaire_data = False
    uploaded_session = False

    # Bar Settings
    green_bar_width = 20
    green_bar_height = 200
    loading_bar_thickness = 30
    arrow_offset_y = 100  # Move arrow up by 100 pixels

    # Positions for Green Bars
    left_green_bar_pos = (100, infoObject.current_h // 2 - green_bar_height // 2)
    right_green_bar_pos = (infoObject.current_w - 100 - green_bar_width, infoObject.current_h // 2 - green_bar_height // 2)

    # Center Position
    center_pos = (infoObject.current_w // 2, infoObject.current_h // 2)

    # Clock
    clock = pygame.time.Clock()

    # Input Variables
    input_text = ""
    input_error = False

    # Questionaire data
    identity_index = 0
    free_response_index = 0
    identity_answers = ["", "", ""]
    free_response_answers = ["", ""]
    button_answers = [-1, -1, -1]

    # Questionaire Button Positioning
    height_delta = infoObject.current_h // 11
    width_delta = infoObject.current_w // 11

    stim_button = Checkbox(screen, width_delta, height_delta * 2, 0, caption='0 mg', font_color=(255, 255, 255))
    stim_button2 = Checkbox(screen, width_delta * 3, height_delta * 2, 1, caption='1 - 49 mg', font_color=(255, 255, 255))
    stim_button3 = Checkbox(screen, width_delta * 5, height_delta * 2, 2, caption='50 - 99 mg', font_color=(255, 255, 255))
    stim_button4 = Checkbox(screen, width_delta * 7, height_delta * 2, 3, caption='100 - 150 mg', font_color=(255, 255, 255))
    stim_button5 = Checkbox(screen, width_delta * 9, height_delta * 2, 4, caption='> 150 mg', font_color=(255, 255, 255))

    stimulant_boxes = []
    stimulant_boxes.append(stim_button)
    stimulant_boxes.append(stim_button2)
    stimulant_boxes.append(stim_button3)
    stimulant_boxes.append(stim_button4)
    stimulant_boxes.append(stim_button5)

    meal_button = Checkbox(screen, width_delta, height_delta * 4, 5, caption='No meal', font_color=(255, 255, 255))
    meal_button2 = Checkbox(screen, width_delta * 3, height_delta * 4, 6, caption='Light meal', font_color=(255, 255, 255))
    meal_button3 = Checkbox(screen, width_delta * 5, height_delta * 4, 7, caption='Medium meal', font_color=(255, 255, 255))
    meal_button4 = Checkbox(screen, width_delta * 7, height_delta * 4, 8, caption='Heavy meal', font_color=(255, 255, 255))
    meal_button5 = Checkbox(screen, width_delta * 9, height_delta * 4, 9, caption='Not sure', font_color=(255, 255, 255))

    meal_boxes = []
    meal_boxes.append(meal_button)
    meal_boxes.append(meal_button2)
    meal_boxes.append(meal_button3)
    meal_boxes.append(meal_button4)
    meal_boxes.append(meal_button5)

    yes_exercise = Checkbox(screen, width_delta * 5, height_delta * 8, 10, caption='yes', font_color=(255, 255, 255))
    no_exercise = Checkbox(screen, width_delta * 6, height_delta * 8, 11, caption='no', font_color=(255, 255, 255))

    exercise_bool_boxes = []
    exercise_bool_boxes.append(yes_exercise)
    exercise_bool_boxes.append(no_exercise)
    
    direction = 'left'  # Start with 'left' and alternate

    while running:

        if in_menu:
            # Display Main Menu
            screen.fill(BLACK)
            title_text = large_font.render("EEG Motor Imagery", True, WHITE)
            start_text = medium_font.render("Press S to Start", True, GREEN)
            set_text = medium_font.render("Press N to Set Number", True, WHITE)
            quit_text = medium_font.render("Press Q to Quit", True, RED)
            trials_text = small_font.render(f"Total Trials: {total_trials}", True, WHITE)
            wait_text = small_font.render(f"You have to wait {round(start_enable_time - time.time())} seconds before starting!", True, WHITE)

            # Positioning Text
            title_rect = title_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 4))
            start_rect = start_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 - 50))
            set_rect = set_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 + 50))
            quit_rect = quit_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 + 150))
            trials_rect = trials_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 - 150))
            wait_rect = wait_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 + 250))

            # Blit Text to Screen
            screen.blit(title_text, title_rect)
            screen.blit(start_text, start_rect)
            screen.blit(set_text, set_rect)
            screen.blit(quit_text, quit_rect)
            screen.blit(trials_text, trials_rect)
            if (start_enable_time > time.time()): # If the start button is currently disabled
                screen.blit(wait_text, wait_rect)
            pygame.display.flip()

            # Processing Input at the Main Menu
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s: 
                        if time.time() >= start_enable_time:
                            in_menu = False
                            in_questionaire_subject = True
                    elif event.key == pygame.K_n:
                        in_input = True
                        in_menu = False
                        input_text = ""
                        input_error = False
                    elif event.key == pygame.K_q:
                        running = False


        elif in_questionaire_subject:
            # Displays the questions about the subject 
            screen.fill(BLACK)

            # Form questions from questionaire
            first_name_text = medium_font.render("Enter first name", True, WHITE)
            last_name_text = medium_font.render("Enter last name", True, WHITE)
            eid_text = medium_font.render("Enter eid", True, WHITE)

            # Position for questions
            height_delta = infoObject.current_h // 6
            first_name_rect = first_name_text.get_rect(center=(infoObject.current_w // 2, height_delta))
            last_name_rect = last_name_text.get_rect(center=(infoObject.current_w // 2, height_delta * 3))
            eid_rect = eid_text.get_rect(center=(infoObject.current_w // 2, height_delta * 5))

            # Form answers for questionaire
            first_name_response = medium_font.render(identity_answers[0], True, WHITE)
            last_name_response = medium_font.render(identity_answers[1], True, WHITE)
            eid_response = medium_font.render(identity_answers[2], True, WHITE)

            # Position for answer
            first_name_response_rect = first_name_response.get_rect(center=(infoObject.current_w // 2, height_delta * 2))
            last_name_response_rect = last_name_response.get_rect(center=(infoObject.current_w // 2, height_delta * 4))
            eid_response_rect = eid_response.get_rect(center=(infoObject.current_w // 2, height_delta * 6))

            # Blit Text to Screen
            screen.blit(first_name_text, first_name_rect)
            screen.blit(last_name_text, last_name_rect)
            screen.blit(eid_text, eid_rect)
                        
            screen.blit(first_name_response, first_name_response_rect)
            screen.blit(last_name_response, last_name_response_rect)
            screen.blit(eid_response, eid_response_rect)
            pygame.display.flip()

            # Subject info page handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    # Change question selection
                    if event.key == pygame.K_DOWN and identity_index < 2:
                        identity_index += 1
                    elif event.key == pygame.K_UP and identity_index > 0:
                        identity_index -= 1
                    elif event.key == pygame.K_BACKSPACE:
                        identity_answers[identity_index] = identity_answers[identity_index] [:-1]
                    elif event.key == pygame.K_RETURN:
                        in_questionaire_subject = False 
                        in_questionaire_physiological = True
                        #TODO: Check for prev subject name to skip over questionaire
                    else:
                        identity_answers[identity_index] += event.unicode


        elif in_questionaire_physiological:
            # Display the questions about the subject's physiological condition
            screen.fill(BLACK)

            # Multiple Choice Questions
            stimulant_text = small_font.render("How much stimulant (e.g. caffiene) have you consumed in the past 12 hours?", True, WHITE)
            meal_text = small_font.render("Have you consumed a light, medium, or heavy meal in the past 12 hours?", True, WHITE)
            exercise_text = small_font.render("Have you exercised in the past 12 hours?", True, WHITE)

            # Free Response Questions
            food_description_text = small_font.render("Describe what you ate in detail to the best of your ability, include portion size if possible", True, WHITE)
            exercise_type_text = small_font.render("If you have exercised, please describe what you did and how long it was. N/A if no exercise", True, WHITE) 

            # Free Response Answers
            food_response = small_font.render(free_response_answers[0], True, WHITE)
            exercise_response = small_font.render(free_response_answers[1], True, WHITE)

            food_response_rect = food_response.get_rect(center=(infoObject.current_w // 2, height_delta * 6))
            exercise_response_rect = exercise_response.get_rect(center=(infoObject.current_w // 2, height_delta * 10))

            # Question Positioning
            height_delta = infoObject.current_h // 11
            stimulant_rect = stimulant_text.get_rect(center=(infoObject.current_w // 2, height_delta))
            meal_rect = meal_text.get_rect(center=(infoObject.current_w // 2, height_delta * 3))
            food_description_rect = food_description_text.get_rect(center=(infoObject.current_w // 2, height_delta * 5))
            exercise_rect = exercise_text.get_rect(center=(infoObject.current_w // 2, height_delta * 7))
            exercise_type_rect = exercise_type_text.get_rect(center=(infoObject.current_w // 2, height_delta * 9))

            screen.blit(stimulant_text, stimulant_rect)
            screen.blit(meal_text, meal_rect)
            screen.blit(food_description_text, food_description_rect)
            screen.blit(exercise_text, exercise_rect)
            screen.blit(exercise_type_text, exercise_type_rect)

            screen.blit(food_response, food_response_rect)
            screen.blit(exercise_response, exercise_response_rect)

            all_boxes = []
            all_boxes.append(stimulant_boxes)
            all_boxes.append(meal_boxes)
            all_boxes.append(exercise_bool_boxes)
            for box_holder in all_boxes:
                for box in box_holder:
                    box.render_checkbox()
            pygame.display.flip()
            
            # Loop
            # Subject info page handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                        for box_holder in all_boxes:
                            for box in box_holder:
                                box.update_checkbox(event)
                                if box.checked:  # If this box is checked
                                    for b in box_holder:
                                        if b != box:
                                            b.checked = False  # Uncheck other boxes
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    # Change question selection
                    if event.key == pygame.K_DOWN and free_response_index < 1:
                        free_response_index += 1
                    elif event.key == pygame.K_UP and free_response_index > 0:
                        free_response_index -= 1
                    elif event.key == pygame.K_BACKSPACE:
                        free_response_answers[free_response_index] = free_response_answers[free_response_index] [:-1]
                    elif event.key == pygame.K_RETURN:
                        for ind, box_holder in enumerate(all_boxes):
                            for count, box in enumerate(box_holder):
                                if box.checked:
                                    button_answers[ind] = count
                        in_questionaire_physiological = False
                        in_buffer_screen = True
                    else:
                        free_response_answers[free_response_index] += event.unicode


        elif in_buffer_screen:
            if not saved_questionnaire_data:
                #Save results of questionnaire locally
                first_name = identity_answers[0]
                last_name = identity_answers[1]
                eid = identity_answers[2]
                stim = ""
                meal = ""
                describe_meal = free_response_answers[0]
                exercise_yn = ""
                exercise_description = free_response_answers[1]

                #Iterate through checkbox arrays to find checked boxes and store their values
                for box in stimulant_boxes:
                    if box.get_checked():
                        stim = box.get_caption()
                        break

                for box in meal_boxes:
                    if box.get_checked():
                        meal = box.get_caption()
                        break

                for box in exercise_bool_boxes:
                    if box.get_checked():
                        exercise = box.get_caption()

                #Use questionnaire to update metadata and track user
                user_table, metadata = track_user(user_table, first_name, last_name, eid, stim, meal, 
                                                    describe_meal, exercise_yn, exercise_description)
                user_table.to_csv(table_path, index=False)  # Save modifications locally
                session_num = metadata.iloc[0, 13]
                directory = create_user_directory(first_name, last_name, session_num)
                saved_questionnaire_data = True
        
            # Display buffer screen that appears before the trials
            screen.fill(BLACK)
            buffer_screen_title = large_font.render("Ready?", True, WHITE)
            start_trial_text = medium_font.render("Press S to Start Trial", True, GREEN)

            # Positioning Text
            buffer_screen_title_rect = buffer_screen_title.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 4))
            start_trial_text_rect = start_trial_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 + 50))

            # Blit Text to Screen
            screen.blit(buffer_screen_title, buffer_screen_title_rect)
            screen.blit(start_trial_text, start_trial_text_rect)
            pygame.display.flip()

            # Processing Inputs at the Buffer Screen
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if event.key == pygame.K_s:
                        in_buffer_screen = False


        elif in_input:
            # Display Input Menu for Setting Number of Trials
            screen.fill(BLACK)
            prompt_text = medium_font.render("Enter Number of Recordings (Even):", True, WHITE)
            input_display = medium_font.render(input_text, True, GREEN if not input_error else RED)
            instructions_text = small_font.render("Press Enter to Confirm", True, WHITE)

            # Positioning Text
            prompt_rect = prompt_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 3))
            input_rect = input_display.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2))
            instructions_rect = instructions_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 + 100))

            # Blit Text to Screen
            screen.blit(prompt_text, prompt_rect)
            screen.blit(input_display, input_rect)
            screen.blit(instructions_text, instructions_rect)
            pygame.display.flip()

            # Processing Inputs at the Input Menu
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    elif event.key == pygame.K_RETURN:
                        if input_text.isdigit():
                            entered_number = int(input_text)
                            if entered_number > 0:
                                if entered_number % 2 != 0:
                                    entered_number += 1  # Make it even
                                    input_error = True
                                total_trials = entered_number
                                in_input = False
                                in_menu = True
                            else:
                                input_error = True
                        else:
                            input_error = True
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.unicode.isdigit():
                        input_text += event.unicode


        elif in_trial_menu: 
            # Display Trial Menu (Accessible via 'M' during trials)
            screen.fill(BLACK)
            menu_title = medium_font.render("Trial Menu", True, WHITE)
            quit_text = medium_font.render("Press Q to Quit", True, RED)
            resume_text = medium_font.render("Press R to Resume", True, GREEN)

            # Positioning Text
            menu_title_rect = menu_title.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 3))
            quit_rect = quit_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2))
            resume_rect = resume_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 + 100))

            # Blit Text to Screen
            screen.blit(menu_title, menu_title_rect)
            screen.blit(quit_text, quit_rect)
            screen.blit(resume_text, resume_rect)
            pygame.display.flip()

            # Processing Inputs at the Trial Menu 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        in_trial_menu = False


        elif in_after_session_menu:
            # Display After Session Menu
            screen.fill(BLACK)
            question_text = large_font.render("Do you want to continue?", True, WHITE)
            continue_text = medium_font.render("Press Y to continue", True, GREEN)
            quit_text = medium_font.render("Press N to exit", True, RED)

            # Positioning Text
            question_rect = question_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 4))
            continue_rect = continue_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 - 50))
            quit_rect = quit_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 + 50))

            # Blit Text to Screen
            screen.blit(question_text, question_rect)
            screen.blit(continue_text, continue_rect)
            screen.blit(quit_text, quit_rect)
            pygame.display.flip()

            if not uploaded_session:
                # Zip the data and upload it
                zip_path = zip_directory(directory, directory + '.zip')
                client = authenticate()
                file_id = '1679766376012'
                upload_file(client, '289622073398', zip_path) # uploads the zipped directory
                uploaded_session = True

            #Update the table in Box
            update_file(client, file_id, table_path)

            # Processsing Inputs at the After Session Menu
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        in_after_session_menu = False
                        in_menu = True
                        start_enable_time = time.time() + time_between_sessions # Enable the start button 180 seconds after the current time
                    elif event.key == pygame.K_n: 
                        # Not fully sure if all of the following lines are necessary, but they are functional
                        in_after_session_menu = False
                        running = False
                        eeg_processor.board.stop_stream()
                        eeg_processor.board.release_session()
                        pygame.quit()
                        sys.exit()


        else:
            # Display Current Trial Number
            screen.fill(BLACK)
            # Redraw green bars
            pygame.draw.rect(screen, GREEN, (*left_green_bar_pos, green_bar_width, green_bar_height))
            pygame.draw.rect(screen, GREEN, (*right_green_bar_pos, green_bar_width, green_bar_height))

            # Draw Current Trial Info
            trial_info = small_font.render(f"Trial {trial_number}/{total_trials}", True, WHITE)
            trial_info_rect = trial_info.get_rect(topright=(infoObject.current_w - 50, 50))
            screen.blit(trial_info, trial_info_rect)

            # Draw Focus Period '+' sign
            plus_text = large_font.render("+", True, WHITE)
            plus_rect = plus_text.get_rect(center=center_pos)
            screen.blit(plus_text, plus_rect)
            pygame.display.flip()

            # Collect data during focus period
            focus_duration = 3  # seconds
            focus_start_time = time.time()
            while time.time() - focus_start_time < focus_duration:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                clock.tick(60)

            if not running:
                break

            # Show Arrow (Moved Up)
            arrow_length = 100
            arrow_color = WHITE
            arrow_width = 20
            arrow_y_offset = arrow_offset_y  # Move arrow up by arrow_offset_y pixels

            # Clear screen but keep green bars and trial info
            screen.fill(BLACK)
            # Redraw green bars
            pygame.draw.rect(screen, GREEN, (*left_green_bar_pos, green_bar_width, green_bar_height))
            pygame.draw.rect(screen, GREEN, (*right_green_bar_pos, green_bar_width, green_bar_height))
            # Redraw trial info
            screen.blit(trial_info, trial_info_rect)

            # Draw Arrow
            if direction == 'left':
                pygame.draw.polygon(screen, arrow_color, [
                    (center_pos[0] - arrow_length, center_pos[1] - arrow_y_offset),
                    (center_pos[0], center_pos[1] - arrow_y_offset - arrow_width),
                    (center_pos[0], center_pos[1] - arrow_y_offset + arrow_width)
                ])
            else:
                pygame.draw.polygon(screen, arrow_color, [
                    (center_pos[0] + arrow_length, center_pos[1] - arrow_y_offset),
                    (center_pos[0], center_pos[1] - arrow_y_offset - arrow_width),
                    (center_pos[0], center_pos[1] - arrow_y_offset + arrow_width)
                ])
            pygame.display.flip()

            # Wait before starting the loading bar
            pre_loading_duration = 1  # second
            pre_loading_start = time.time()
            while time.time() - pre_loading_start < pre_loading_duration:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                clock.tick(60)

            if not running:
                break

            # Loading Bar
            loading_duration = 14  # seconds

            # Perf counter is a more accurate timer than time.time()
            loading_start_time = time.perf_counter()

            # Initialize loading bar variables
            current_direction = direction
            current_length = 0
            # From center to left green bar
            max_length = center_pos[0] - (left_green_bar_pos[0] + green_bar_width)

            while time.perf_counter() - loading_start_time < loading_duration and current_length < max_length:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                        elif event.key == pygame.K_m:
                            in_trial_menu = True
                            break

                # Calculate loading bar progress
                elapsed_time = time.perf_counter() - loading_start_time
                loading_progress = elapsed_time / loading_duration

                screen.fill(BLACK)
                # Redraw green bars
                pygame.draw.rect(screen, GREEN, (*left_green_bar_pos, green_bar_width, green_bar_height))
                pygame.draw.rect(screen, GREEN, (*right_green_bar_pos, green_bar_width, green_bar_height))
                # Redraw trial info
                screen.blit(trial_info, trial_info_rect)

                # Redraw Arrow
                if direction == 'left':
                    
                    # Adds some noise for the bar to move back and forth
                    if current_direction == 'left':
                        current_length += 3
                        if np.random.random() < 0.03:
                            current_direction = 'right'
                    else:
                        current_length -= 3
                        if np.random.random() < 0.1:
                            current_direction = 'left'

                    pygame.draw.polygon(screen, arrow_color, [
                        (center_pos[0] - arrow_length, center_pos[1] - arrow_y_offset),
                        (center_pos[0], center_pos[1] - arrow_y_offset - arrow_width),
                        (center_pos[0], center_pos[1] - arrow_y_offset + arrow_width)
                    ])                    
                    

                    # Draw loading bar moving left from center
                    pygame.draw.rect(screen, WHITE, (
                        center_pos[0] - current_length,  # Start at center and move left
                        center_pos[1] - loading_bar_thickness // 2,
                        current_length,
                        loading_bar_thickness
                    ))
                else:

                    # Adds some noise for the bar to move back and forth
                    if current_direction == 'right':
                        current_length += 3
                        if np.random.random() < 0.03:
                            current_direction = 'left'
                    else:
                        current_length -= 3
                        if np.random.random() < 0.1:
                            current_direction = 'right'

                    pygame.draw.polygon(screen, arrow_color, [
                        (center_pos[0] + arrow_length, center_pos[1] - arrow_y_offset),
                        (center_pos[0], center_pos[1] - arrow_y_offset - arrow_width),
                        (center_pos[0], center_pos[1] - arrow_y_offset + arrow_width)
                    ])
                    

                    # Draw loading bar moving right from center
                    pygame.draw.rect(screen, WHITE, (
                        center_pos[0],  # Start at center and move right
                        center_pos[1] - loading_bar_thickness // 2,
                        current_length,
                        loading_bar_thickness
                    ))

                pygame.display.flip()
                clock.tick(60)
            
            save_data(eeg_processor, elapsed_time, direction, trial_number, directory)

            if not running:
                break

            # Optional rest period with accessible menu
            rest_duration = 2  # seconds
            rest_start_time = time.time()
            while time.time() - rest_start_time < rest_duration:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            break
                        elif event.key == pygame.K_m:
                            in_trial_menu = True
                            break

                # Display Rest Text with Menu Instruction
                screen.fill(BLACK)
                # Redraw green bars
                pygame.draw.rect(screen, GREEN, (*left_green_bar_pos, green_bar_width, green_bar_height))
                pygame.draw.rect(screen, GREEN, (*right_green_bar_pos, green_bar_width, green_bar_height))
                # Redraw trial info
                screen.blit(trial_info, trial_info_rect)

                rest_text = small_font.render("Rest (Press M for Menu)", True, WHITE)
                rest_rect = rest_text.get_rect(center=center_pos)
                screen.blit(rest_text, rest_rect)
                pygame.display.flip()
                clock.tick(60)

            if not running:
                break

            # Alternate Direction
            if direction == 'left':
                direction = 'right'
            else:
                direction = 'left'

            if direction == 'left':
                trial_number += 1

            # Check if all trials are completed
            if trial_number > total_trials:
                # Display a completion message
                screen.fill(BLACK)
                completion_text = medium_font.render("All Trials Completed!", True, GREEN)
                completion_rect = completion_text.get_rect(center=center_pos)
                screen.blit(completion_text, completion_rect)
                pygame.display.flip()
                # Wait for 3 seconds before returning to menu
                time.sleep(3)
                trial_number = 1  # Reset trial number
                in_after_session_menu = True

        # Handle Trial Menu outside the main loop to avoid missing quit events
        while in_trial_menu and running:
            # Display Trial Menu (Accessible via 'M' during trials)
            screen.fill(BLACK)
            menu_title = medium_font.render("Trial Menu", True, WHITE)
            quit_text = medium_font.render("Press Q to Quit", True, RED)
            resume_text = medium_font.render("Press R to Resume", True, GREEN)

            # Positioning Text
            menu_title_rect = menu_title.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 3))
            quit_rect = quit_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2))
            resume_rect = resume_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 + 100))

            # Blit Text to Screen
            screen.blit(menu_title, menu_title_rect)
            screen.blit(quit_text, quit_rect)
            screen.blit(resume_text, resume_rect)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    in_trial_menu = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                        in_trial_menu = False
                    elif event.key == pygame.K_r:
                        in_trial_menu = False


if __name__ == "__main__":
    main()