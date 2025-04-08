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

def draw_plus_sign(screen, center_pos, plus_length, thickness, color):
    # Draw horizontal line
    pygame.draw.line(screen, color,
                     (center_pos[0] - plus_length // 2, center_pos[1]),
                     (center_pos[0] + plus_length // 2, center_pos[1]),
                     thickness)
    # Draw vertical line
    pygame.draw.line(screen, color,
                     (center_pos[0], center_pos[1] - plus_length // 2),
                     (center_pos[0], center_pos[1] + plus_length // 2),
                     thickness)


def create_user_directory(first_name, last_name, session_num):
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
        self.board_id = BoardIds.CYTON_DAISY_BOARD.value
        self.board = BoardShim(self.board_id, params)
        self.board.prepare_session()
        self.board.start_stream()
        print("BrainFlow streaming started...")

        # Sampling rate and window size
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.window_size_sec = 7  # seconds
        self.window_size_samples = int(self.window_size_sec * self.sampling_rate)

        # we set raw window size to 10 seconds
        self.window_size_raw = int(10 * self.sampling_rate)
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

    def get_recent_data(self):
        """
        Returns the most recent 7 seconds of processed EEG data.

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

        return recent_data
    
#Save last 7 seconds of signal and metadata to its own .pkl file in the session directory
def save_data(eeg_processor, metadata, direction, trial_num, directory):
    sig = eeg_processor.get_recent_data()
    #Establish a filename [direction]_[trial number].pkl
    filename = direction + '_' + str(trial_num) + '.pkl'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    intermediate = os.path.join(script_dir, directory)
    filepath = os.path.join(intermediate, filename)

    #Dump signal and metadata into pickle file - this saves into the folder that we created earlier
    with open(filepath, 'wb') as f:
        pickle.dump((sig, metadata), f)
    

def main():
    session_num = input("Enter the session number: ")
    eeg_processor = EEGProcessor()

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
    total_trials = 20 # Default number of trials
    time_between_sessions = 180 # number of seconds to wait between sessions of data collection
    start_enable_time = time.time() # the time at/after which the start button is enabled
    saved_questionnaire_data = False
    
    # Calculate font sizes based on screen height
    font_size_large = infoObject.current_h // 10
    font_size_medium = infoObject.current_h // 15
    font_size_small = infoObject.current_h // 20

    # Initialize fonts with the calculated sizes
    large_font = pygame.font.SysFont(None, font_size_large)
    medium_font = pygame.font.SysFont(None, font_size_medium)
    small_font = pygame.font.SysFont(None, font_size_small)

    green_bar_width = infoObject.current_w // 60
    green_bar_height = infoObject.current_h // 3
    loading_bar_thickness = infoObject.current_h // 30
    arrow_y_offset = infoObject.current_h // 10

    # Positions for Green Bars
    left_green_bar_pos = (infoObject.current_w // 50, infoObject.current_h // 2 - green_bar_height // 2)
    right_green_bar_pos = (infoObject.current_w - infoObject.current_w // 50 - green_bar_width, infoObject.current_h // 2 - green_bar_height // 2)

    # Center Position
    center_pos = (infoObject.current_w // 2, infoObject.current_h // 2)

    # Plus sign settings
    plus_length = infoObject.current_h // 15  # Adjust as needed


    # Clock
    clock = pygame.time.Clock()

    # Input Variables
    input_text = ""
    input_error = False
    questionnaire_error = False


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

    yes_exercise = Checkbox(screen, width_delta * 5, height_delta * 7, 10, caption='yes', font_color=(255, 255, 255))
    no_exercise = Checkbox(screen, width_delta * 6, height_delta * 7, 11, caption='no', font_color=(255, 255, 255))

    exercise_bool_boxes = []
    exercise_bool_boxes.append(yes_exercise)
    exercise_bool_boxes.append(no_exercise)
    
    direction = 'left'  # Start with 'left' and alternate


    while running:

        if in_menu:
            screen.fill(BLACK)
            title_text = large_font.render("EEG Motor Imagery", True, WHITE)
            start_text = medium_font.render("Press S to Start", True, GREEN)
            set_text = medium_font.render("Press N to Set Number", True, WHITE)
            quit_text = medium_font.render("Press Q to Quit", True, RED)
            trials_text = small_font.render(f"Total Trials: {total_trials}", True, WHITE)

            # Positioning Text
            title_rect = title_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 5))
            start_rect = start_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 - font_size_medium))
            set_rect = set_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2))
            quit_rect = quit_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 + font_size_medium))
            trials_rect = trials_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 2 -  1.1 * font_size_large))

            # Blit Text to Screen
            screen.blit(title_text, title_rect)
            screen.blit(start_text, start_rect)
            screen.blit(set_text, set_rect)
            screen.blit(quit_text, quit_rect)
            screen.blit(trials_text, trials_rect)
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
            screen.fill(BLACK)

            # Positions based on scaled heights
            height_delta = infoObject.current_h // 8

            # Render and position texts
            first_name_text = medium_font.render("Enter first name", True, WHITE)
            first_name_rect = first_name_text.get_rect(center=(infoObject.current_w // 2, height_delta))

            last_name_text = medium_font.render("Enter last name", True, WHITE)
            last_name_rect = last_name_text.get_rect(center=(infoObject.current_w // 2, height_delta * 2.5))

            eid_text = medium_font.render("Enter EID", True, WHITE)
            eid_rect = eid_text.get_rect(center=(infoObject.current_w // 2, height_delta * 4))

            # Render and position responses
            first_name_response = medium_font.render(identity_answers[0], True, WHITE)
            first_name_response_rect = first_name_response.get_rect(center=(infoObject.current_w // 2, height_delta * 1.5))

            last_name_response = medium_font.render(identity_answers[1], True, WHITE)
            last_name_response_rect = last_name_response.get_rect(center=(infoObject.current_w // 2, height_delta * 3))

            eid_response = medium_font.render(identity_answers[2], True, WHITE)
            eid_response_rect = eid_response.get_rect(center=(infoObject.current_w // 2, height_delta * 5))

            # Blit texts to the screen
            screen.blit(first_name_text, first_name_rect)
            screen.blit(first_name_response, first_name_response_rect)
            screen.blit(last_name_text, last_name_rect)
            screen.blit(last_name_response, last_name_response_rect)
            screen.blit(eid_text, eid_rect)
            screen.blit(eid_response, eid_response_rect)

            # Error message if needed
            if input_error:
                error_text = small_font.render("Please fill out all fields before proceeding.", True, RED)
                error_rect = error_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h - font_size_small))
                screen.blit(error_text, error_rect)

            pygame.display.flip()

            # Event handling remains the same


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
                        identity_answers[identity_index] = identity_answers[identity_index][:-1]
                        input_error = False  # Reset error flag when user modifies input
                    elif event.key == pygame.K_RETURN:
                        # Check if all fields are filled
                        if all(answer.strip() != "" for answer in identity_answers):
                            in_questionaire_subject = False 
                            in_questionaire_physiological = True
                            input_error = False  # Reset the error flag
                        else:
                            input_error = True  # Set the error flag to display an error message
                    else:
                        identity_answers[identity_index] += event.unicode
                        input_error = False  # Reset error flag when user modifies input



        elif in_questionaire_physiological:
            # Display the questions about the subject's physiological condition
            screen.fill(BLACK)

            # Multiple Choice Questions
            stimulant_text = small_font.render("How much stimulant (e.g. caffeine) have you consumed in the past 12 hours?", True, WHITE)
            meal_text = small_font.render("Have you consumed a light, medium, or heavy meal in the past 12 hours?", True, WHITE)
            exercise_text = small_font.render("Have you exercised in the past 12 hours?", True, WHITE)

            # Free Response Questions
            food_description_text = small_font.render("Describe what you ate in detail, include portion size if possible", True, WHITE)
            exercise_type_text = small_font.render("If you have exercised, please describe what you did and how long. N/A if no exercise", True, WHITE)

            # Free Response Answers
            food_response = small_font.render(free_response_answers[0], True, WHITE)
            exercise_response = small_font.render(free_response_answers[1], True, WHITE)

            # Positions for texts
            height_delta = infoObject.current_h // 12
            stimulant_rect = stimulant_text.get_rect(center=(infoObject.current_w // 2, height_delta))
            meal_rect = meal_text.get_rect(center=(infoObject.current_w // 2, height_delta * 3))
            food_description_rect = food_description_text.get_rect(center=(infoObject.current_w // 2, height_delta * 5))
            exercise_rect = exercise_text.get_rect(center=(infoObject.current_w // 2, height_delta * 7))
            exercise_type_rect = exercise_type_text.get_rect(center=(infoObject.current_w // 2, height_delta * 9))

            # Render texts
            screen.blit(stimulant_text, stimulant_rect)
            screen.blit(meal_text, meal_rect)
            screen.blit(food_description_text, food_description_rect)
            screen.blit(exercise_text, exercise_rect)
            screen.blit(exercise_type_text, exercise_type_rect)

            # Render free response answers
            food_response_rect = food_response.get_rect(center=(infoObject.current_w // 2, height_delta * 6))
            exercise_response_rect = exercise_response.get_rect(center=(infoObject.current_w // 2, height_delta * 10))
            screen.blit(food_response, food_response_rect)
            screen.blit(exercise_response, exercise_response_rect)

            # Render checkboxes
            all_boxes = []
            all_boxes.append(stimulant_boxes)
            all_boxes.append(meal_boxes)
            all_boxes.append(exercise_bool_boxes)
            for box_holder in all_boxes:
                for box in box_holder:
                    box.render_checkbox()

            # Display error message if any question is unanswered
            if questionnaire_error:
                error_text = small_font.render("Please answer all questions before proceeding.", True, RED)
                error_rect = error_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h - 50))
                screen.blit(error_text, error_rect)

            pygame.display.flip()
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Update checkboxes
                    for box_holder in all_boxes:
                        for box in box_holder:
                            box.update_checkbox(event)
                            if box.checked:
                                for b in box_holder:
                                    if b != box:
                                        b.checked = False  # Uncheck other boxes in the same group
                    questionnaire_error = False  # Reset error flag when user selects an option
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    # Change free response selection
                    if event.key == pygame.K_DOWN and free_response_index < 1:
                        free_response_index += 1
                    elif event.key == pygame.K_UP and free_response_index > 0:
                        free_response_index -= 1
                    elif event.key == pygame.K_BACKSPACE:
                        free_response_answers[free_response_index] = free_response_answers[free_response_index][:-1]
                        questionnaire_error = False  # Reset error flag when user modifies input
                    elif event.key == pygame.K_RETURN:
                        # Check if all questions are answered
                        all_questions_answered = True

                        # Check if an option is selected in each checkbox group
                        for box_holder in all_boxes:
                            if not any(box.checked for box in box_holder):
                                all_questions_answered = False
                                break

                        # Check if free response answers are not empty
                        if any(answer.strip() == "" for answer in free_response_answers):
                            all_questions_answered = False

                        if all_questions_answered:
                            # Save the answers from checkboxes
                            for ind, box_holder in enumerate(all_boxes):
                                for count, box in enumerate(box_holder):
                                    if box.checked:
                                        button_answers[ind] = count

                            in_questionaire_physiological = False
                            in_buffer_screen = True
                            questionnaire_error = False  # Reset the error flag
                        else:
                            questionnaire_error = True  # Set the error flag to display an error message
                    else:
                        free_response_answers[free_response_index] += event.unicode
                        questionnaire_error = False  # Reset error flag when user modifies input


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
                        exercise_yn = box.get_caption()

                #Use questionnaire data to update metadata and create session directory
                directory = create_user_directory(first_name, last_name, session_num)
                metadata = {"First Name"            : first_name,
                            "Last Name"             : last_name,
                            "EID"                   : eid,
                            "Stimulant Use"         : stim,
                            "Meal Size"             : meal,
                            "Meal Description"      : describe_meal,
                            "Exercised"             : exercise_yn,
                            "Exercise Description"  : exercise_description}
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
            draw_plus_sign(screen, center_pos, plus_length, loading_bar_thickness, WHITE)
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
            arrow_color = WHITE
            arrow_length = infoObject.current_w // 10  
            arrow_width = infoObject.current_h // 40
            arrow_y_offset = infoObject.current_h // 10

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

            draw_plus_sign(screen, center_pos, plus_length, loading_bar_thickness, WHITE)

            pygame.display.flip()

            # Wait before starting the loading bar
            pre_loading_duration = 1.2  # second
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
            loading_duration = 7  # seconds
            loading_start_time = time.time()

            while time.time() - loading_start_time < loading_duration:
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
                elapsed_time = time.time() - loading_start_time
                loading_progress = elapsed_time / loading_duration

                screen.fill(BLACK)
                # Redraw green bars
                pygame.draw.rect(screen, GREEN, (*left_green_bar_pos, green_bar_width, green_bar_height))
                pygame.draw.rect(screen, GREEN, (*right_green_bar_pos, green_bar_width, green_bar_height))
                # Redraw trial info
                screen.blit(trial_info, trial_info_rect)

                # Redraw Arrow
                if direction == 'left':
                    pygame.draw.polygon(screen, arrow_color, [
                        (center_pos[0] - arrow_length, center_pos[1] - arrow_y_offset),
                        (center_pos[0], center_pos[1] - arrow_y_offset - arrow_width),
                        (center_pos[0], center_pos[1] - arrow_y_offset + arrow_width)
                    ])
                    # Calculate current length of the loading bar
                    # From center to left green bar
                    max_length = center_pos[0] - (left_green_bar_pos[0] + green_bar_width)
                    current_length = loading_progress * max_length

                    # Draw loading bar moving left from center
                    pygame.draw.rect(screen, WHITE, (
                        center_pos[0] - current_length,  # Start at center and move left
                        center_pos[1] - loading_bar_thickness // 2,
                        current_length,
                        loading_bar_thickness
                    ))
                else:
                    pygame.draw.polygon(screen, arrow_color, [
                        (center_pos[0] + arrow_length, center_pos[1] - arrow_y_offset),
                        (center_pos[0], center_pos[1] - arrow_y_offset - arrow_width),
                        (center_pos[0], center_pos[1] - arrow_y_offset + arrow_width)
                    ])
                    # Calculate current length of the loading bar
                    # From center to right green bar
                    max_length = (right_green_bar_pos[0]) - center_pos[0]
                    current_length = loading_progress * max_length

                    # Draw loading bar moving right from center
                    pygame.draw.rect(screen, WHITE, (
                        center_pos[0],  # Start at center and move right
                        center_pos[1] - loading_bar_thickness // 2,
                        current_length,
                        loading_bar_thickness
                    ))

                draw_plus_sign(screen, center_pos, plus_length, loading_bar_thickness, WHITE)

                pygame.display.flip()
                save_data(eeg_processor, metadata, direction, trial_number, directory)
                clock.tick(60)

            if not running:
                break

            # Optional rest period with accessible menu
            # random rest between 3, 5 seconds
            rest_duration = np.random.uniform(3, 5)
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