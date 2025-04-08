import pygame
import sys
import time
import pickle
import numpy as np
from scipy.signal import butter, lfilter, iirnotch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
# from checkbox import Checkbox # Already removed
import platform
import serial
import serial.tools.list_ports
import datetime
from datetime import timedelta
import pandas as pd
from boxsdk import Client, OAuth2
import zipfile
import os
import questionary
from rich.console import Console
from rich.text import Text
import argparse # Added import
import random # Added import

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

def draw_plus_sign(surface, center_pos, plus_length, thickness, color):
    """
    Draws a plus sign at the specified center position on the given surface.
    """
    # Draw horizontal line
    pygame.draw.line(
        surface, # Draw on the passed surface
        color,
        (center_pos[0] - plus_length // 2, center_pos[1]),
        (center_pos[0] + plus_length // 2, center_pos[1]),
        thickness
    )
    # Draw vertical line
    pygame.draw.line(
        surface, # Draw on the passed surface
        color,
        (center_pos[0], center_pos[1] - plus_length // 2),
        (center_pos[0], center_pos[1] + plus_length // 2),
        thickness
    )

class EEGProcessor:
    """
    Handles EEG data acquisition and basic filtering via BrainFlow.
    """
    def __init__(self, board_id_to_use): # Accept board_id
        # Initialize BrainFlow
        BoardShim.enable_dev_board_logger()
        params = BrainFlowInputParams()
        # TODO: Add serial port detection back if needed for Cyton
        # serial_port = find_serial_port()
        # if serial_port and board_id_to_use == BoardIds.CYTON_BOARD.value:
        #     params.serial_port = serial_port
        # else:
        #     print("Warning: Serial port not automatically found or using synthetic board.")

        self.board_id = board_id_to_use # Use passed board_id
        self.board = BoardShim(self.board_id, params)
        try:
            self.board.prepare_session()
            self.board.start_stream()
            print(f"BrainFlow streaming started using board ID: {self.board_id}...")
        except Exception as e:
             print(f"Error initializing BrainFlow: {e}")
             print("Please ensure the board is connected and drivers are installed, or use --use-synthetic for testing.")
             sys.exit(1)

        # Sampling rate and window size
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.window_size_sec = 7  # seconds
        self.window_size_samples = int(self.window_size_sec * self.sampling_rate)

        # We set raw window size to 10 seconds
        self.window_size_raw = int(10 * self.sampling_rate)
        self.lowcut = 1.0
        self.highcut = 50.0
        self.notch = 60.0

        # EEG channels
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)

        # Initialize buffers
        self.raw_data_buffer = np.empty((len(self.eeg_channels), 0))
        self.processed_data_buffer = np.empty((len(self.eeg_channels), 0))

    def stop(self):
        # Stop data stream and release session
        self.board.stop_stream()
        self.board.release_session()
        print("BrainFlow streaming stopped.")

    def get_recent_data(self):
        """
        Returns the most recent 7 seconds of processed EEG data.
        """
        data = self.board.get_board_data()
        if data.shape[1] == 0:
            return self.processed_data_buffer  # No new data

        # Append new raw data
        eeg_data = data[self.eeg_channels, :]
        self.raw_data_buffer = np.hstack((self.raw_data_buffer, eeg_data))

        # Process new data
        new_processed_data = np.empty(self.raw_data_buffer.shape)
        for i in range(len(self.eeg_channels)):
            # Filter each channel
            channel_data = self.raw_data_buffer[i, :].copy()
            # Bandpass filter
            b, a = butter(2, [self.lowcut, self.highcut], btype='band', fs=self.sampling_rate)
            channel_data = lfilter(b, a, channel_data)
            # Notch filter
            b, a = iirnotch(self.notch, 30, fs=self.sampling_rate)
            channel_data = lfilter(b, a, channel_data)
            new_processed_data[i, :] = channel_data

        self.processed_data_buffer = np.hstack((self.processed_data_buffer, new_processed_data))

        # Trim buffer sizes
        max_buffer_size = self.window_size_samples * 2
        if self.raw_data_buffer.shape[1] > self.window_size_raw:
            self.raw_data_buffer = self.raw_data_buffer[:, -self.window_size_raw:]
        if self.processed_data_buffer.shape[1] > max_buffer_size:
            self.processed_data_buffer = self.processed_data_buffer[:, -max_buffer_size:]

        if self.processed_data_buffer.shape[1] >= self.window_size_samples:
            return self.processed_data_buffer[:, -self.window_size_samples:]
        else:
            return self.processed_data_buffer

def save_data(eeg_processor, metadata, direction, trial_num, session_path): # Changed directory to session_path
    """
    Save the last 7 seconds of EEG data plus metadata into a pickle file within the session_path.
    """
    sig = eeg_processor.get_recent_data()
    filename = f"{direction}_{trial_num}.pkl"
    # Construct filepath using the full session_path
    filepath = os.path.join(session_path, filename)

    try:
        with open(filepath, 'wb') as f:
            pickle.dump((sig, metadata), f)
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="EEG Motor Imagery Data Collection Script")
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Specify the base directory to save session data. Defaults to script location."
    )
    parser.add_argument(
        "-s", "--use-synthetic",
        action="store_true",
        help="Use BrainFlow's synthetic board instead of Cyton board."
    )
    parser.add_argument(
        "--four-bars",
        action="store_true",
        help="Display four bars (up, down, left, right) instead of the default three."
    )
    args = parser.parse_args()

    # Determine board ID based on flag
    board_id = BoardIds.SYNTHETIC_BOARD.value if args.use_synthetic else BoardIds.CYTON_BOARD.value

    # --- Command Line Questionnaire ---
    console = Console()
    console.print(Text("Welcome to the EEG Motor Imagery Data Collection", style="bold blue"))

    session_num = questionary.text(
        "Enter the session number:",
        validate=lambda text: text.isdigit() or "Please enter a valid number"
    ).ask()
    if session_num is None: sys.exit("Operation cancelled.")

    console.print(Text("\nSubject Information:", style="bold green"))
    first_name = questionary.text("Enter first name:", validate=lambda text: len(text.strip()) > 0 or "First name cannot be empty").ask()
    if first_name is None: sys.exit("Operation cancelled.")
    last_name = questionary.text("Enter last name:", validate=lambda text: len(text.strip()) > 0 or "Last name cannot be empty").ask()
    if last_name is None: sys.exit("Operation cancelled.")
    eid = questionary.text("Enter EID:", validate=lambda text: len(text.strip()) > 0 or "EID cannot be empty").ask()
    if eid is None: sys.exit("Operation cancelled.")

    console.print(Text("\nPhysiological Information (Past 12 Hours):", style="bold green"))

    stim_choices = ['0 mg', '1 - 49 mg', '50 - 99 mg', '100 - 150 mg', '> 150 mg']
    stim = questionary.select(
        "How much stimulant (e.g. caffeine) have you consumed?",
        choices=stim_choices
    ).ask()
    if stim is None: sys.exit("Operation cancelled.")

    meal_choices = ['No meal', 'Light meal', 'Medium meal', 'Heavy meal', 'Not sure']
    meal = questionary.select(
        "Have you consumed a meal?",
        choices=meal_choices
    ).ask()
    if meal is None: sys.exit("Operation cancelled.")

    describe_meal = questionary.text(
        "Describe what you ate in detail (include portion size if possible):",
        validate=lambda text: len(text.strip()) > 0 or "Description cannot be empty"
        ).ask()
    if describe_meal is None: sys.exit("Operation cancelled.")

    exercise_yn = questionary.confirm("Have you exercised?").ask()
    if exercise_yn is None: sys.exit("Operation cancelled.")

    exercise_desc = "N/A"
    if exercise_yn:
        exercise_desc = questionary.text(
            "Describe the exercise (type and duration):",
            validate=lambda text: len(text.strip()) > 0 or "Description cannot be empty"
        ).ask()
        if exercise_desc is None: sys.exit("Operation cancelled.")

    # --- Directory and Metadata Setup ---
    # Determine base directory for saving data
    if args.output_dir:
        base_dir = args.output_dir
        console.print(Text(f"\nUsing specified output directory: {base_dir}", style="yellow"))
    else:
        # Default to 'local_sessions' in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "local_sessions")
        console.print(Text(f"\nUsing default output directory: {base_dir}", style="yellow"))

    # Create the session-specific directory path
    session_dir_name = f"{first_name}_{last_name}_Session{session_num}"
    full_session_path = os.path.join(base_dir, session_dir_name)

    # Create the directory if it doesn't exist
    try:
        os.makedirs(full_session_path, exist_ok=True)
        console.print(Text(f"Session data will be saved to: {full_session_path}", style="green"))
    except OSError as e:
        console.print(Text(f"Error creating directory {full_session_path}: {e}", style="bold red"))
        sys.exit(1)

    # Metadata dictionary (remains the same)
    metadata = {
        "First Name": first_name,
        "Last Name": last_name,
        "EID": eid,
        "Stimulant Use": stim,
        "Meal Size": meal,
        "Meal Description": describe_meal,
        "Exercised": "Yes" if exercise_yn else "No",
        "Exercise Description": exercise_desc,
        "Board ID Used": board_id # Add board ID to metadata
    }
    # REMOVED: console.print(Text("\nUser directory created and metadata saved.", style="yellow")) # Handled above

    # --- Initialize EEG --- (Pygame init moved later)
    console.print(Text("Starting EEG Processor...", style="yellow"))
    try:
        eeg_processor = EEGProcessor(board_id_to_use=board_id) # Pass board_id
    except SystemExit: # Catch exit from EEGProcessor init failure
         sys.exit(1)

    # --- Pygame and GUI Setup --- (Done ONCE before loop) ---
    console.print(Text("Initializing Pygame window...", style="yellow"))
    pygame.init()
    infoObject = pygame.display.Info()
    screen = pygame.display.set_mode((infoObject.current_w, infoObject.current_h), pygame.FULLSCREEN)
    pygame.display.set_caption("Motor Imagery Task")

    # Scaling Setup
    gui_scale = 0.7
    virtual_width = int(infoObject.current_w * gui_scale)
    virtual_height = int(infoObject.current_h * gui_scale)
    offset_x = (infoObject.current_w - virtual_width) // 2
    offset_y = (infoObject.current_h - virtual_height) // 2
    virtual_screen = pygame.Surface((virtual_width, virtual_height))

    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    # Control variables
    running = True
    in_menu = True
    in_input = False
    in_trial_menu = False
    in_buffer_screen = False
    in_after_session_menu = False
    trial_number = 1 # Will be reset when session starts
    total_trials = 20 # Default, adjustable via menu
    time_between_sessions = 180
    start_enable_time = time.time()

    # Base directions list
    base_directions = ["up", "down", "left", "right"]
    full_trial_sequence = [] # Will be populated when session starts
    current_direction = None # Will be set when sequence is generated

    # Font sizes (scaled)
    font_size_large = virtual_height // 10
    font_size_medium = virtual_height // 15
    font_size_small = virtual_height // 20
    large_font = pygame.font.SysFont(None, font_size_large)
    medium_font = pygame.font.SysFont(None, font_size_medium)
    small_font = pygame.font.SysFont(None, font_size_small)

    # Geometry (scaled and relative to virtual screen)
    center_x = virtual_width // 2
    center_y = virtual_height // 2
    center_pos = (center_x, center_y)
    bar_offset_x = virtual_width // 4.5
    bar_offset_y = virtual_height // 4.5
    bar_thin_factor = 0.5
    bar_width_vert = int((virtual_width // 30) * bar_thin_factor)
    bar_height_vert = int((virtual_height // 5) * bar_thin_factor)
    bar_width_horiz = int((virtual_width // 5) * bar_thin_factor)
    bar_height_horiz = int((virtual_height // 30) * bar_thin_factor)

    # Rectangles (defined once)
    left_bar_rect = pygame.Rect(center_x - bar_offset_x, center_y - bar_height_vert // 2, bar_width_vert, bar_height_vert)
    right_bar_rect = pygame.Rect(center_x + bar_offset_x - bar_width_vert, center_y - bar_height_vert // 2, bar_width_vert, bar_height_vert)
    top_bar_rect = pygame.Rect(center_x - bar_width_horiz // 2, center_y - bar_offset_y, bar_width_horiz, bar_height_horiz)
    bottom_bar_rect = pygame.Rect(center_x - bar_width_horiz // 2, center_y + bar_offset_y - bar_height_horiz, bar_width_horiz, bar_height_horiz)

    loading_bar_thickness = virtual_height // 30
    clock = pygame.time.Clock()
    input_text = ""
    input_error = False
    plus_length = virtual_height // 15

    # --- Main Loop ---
    while running:

        # -----------------------------
        # 1. MAIN MENU
        # -----------------------------
        if in_menu:
            virtual_screen.fill(BLACK)
            title_text = large_font.render("EEG Motor Imagery", True, WHITE)
            start_text = medium_font.render("Press S to Start Session", True, GREEN)
            # Display current total_trials setting in the prompt
            set_text = medium_font.render(f"Press N to Set Total Trials (Current: {total_trials})", True, WHITE)
            quit_text = medium_font.render("Press Q to Quit", True, RED)

            title_rect = title_text.get_rect(center=(center_x, virtual_height // 5))
            start_rect = start_text.get_rect(center=(center_x, center_y - font_size_medium))
            set_rect = set_text.get_rect(center=(center_x, center_y))
            quit_rect = quit_text.get_rect(center=(center_x, center_y + font_size_medium))

            virtual_screen.blit(title_text, title_rect)
            virtual_screen.blit(start_text, start_rect)
            virtual_screen.blit(set_text, set_rect)
            virtual_screen.blit(quit_text, quit_rect)

            screen.fill(BLACK)
            screen.blit(virtual_screen, (offset_x, offset_y))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        if time.time() >= start_enable_time:
                            in_menu = False
                            # --- Generate trial sequence --- (Moved here)
                            # Default to three bars
                            active_directions = ["up", "left", "right"]
                            if args.four_bars:
                                # If flag is set, use all four directions
                                active_directions = ["up", "down", "left", "right"]
                                console.print(Text("Using 4 bars (up, down, left, right).", style="magenta"))
                            else:
                                # Default case: three bars
                                console.print(Text("Using 3 bars (up, left, right).", style="magenta"))

                            num_directions = len(active_directions)
                            if num_directions == 0:
                                console.print(Text("Error: No directions available!", style="bold red"))
                                running = False # Stop if no directions
                                break # Exit event loop

                            # Adjust total_trials to be divisible by num_directions
                            temp_total_trials = total_trials # Use temp var for check
                            if temp_total_trials % num_directions != 0:
                                old_total = temp_total_trials
                                temp_total_trials = (temp_total_trials // num_directions) * num_directions
                                console.print(Text(f"Adjusted total trials from {old_total} to {temp_total_trials} to be divisible by {num_directions} directions.", style="yellow"))

                            if temp_total_trials == 0:
                                console.print(Text(f"Error: Cannot run with 0 trials (input was < {num_directions}).", style="bold red"))
                                running = False # Stop if adjusted trials is 0
                                break # Exit event loop

                            # Assign potentially adjusted value back to main variable
                            total_trials = temp_total_trials

                            num_reps = total_trials // num_directions
                            full_trial_sequence = active_directions * num_reps
                            random.shuffle(full_trial_sequence)
                            console.print(Text(f"Generated randomized trial sequence with {total_trials} trials.", style="cyan"))
                            # --- Sequence generation end ---

                            trial_number = 1 # Reset trial number for the new sequence
                            current_direction = full_trial_sequence[trial_number - 1]
                            in_buffer_screen = True # Proceed to buffer screen

                    elif event.key == pygame.K_n:
                        in_input = True
                        in_menu = False
                        input_text = ""
                        input_error = False
                    elif event.key == pygame.K_q:
                        running = False
            if not running: break # Exit main loop if running became False

        # -----------------------------
        # 2. SET NUMBER OF TRIALS
        # -----------------------------
        elif in_input:
            virtual_screen.fill(BLACK)
            prompt_text = medium_font.render("Enter Number of Recordings (Even):", True, WHITE)
            input_display = medium_font.render(input_text, True, GREEN if not input_error else RED)
            instructions_text = small_font.render("Press Enter to Confirm", True, WHITE)
            return_text = small_font.render("Press ESC to Return to Menu", True, WHITE)

            prompt_rect = prompt_text.get_rect(center=(center_x, virtual_height // 3))
            input_rect = input_display.get_rect(center=(center_x, center_y))
            instructions_rect = instructions_text.get_rect(center=(center_x, center_y + virtual_height // 10))
            return_rect = return_text.get_rect(center=(center_x, center_y + virtual_height // 7))

            virtual_screen.blit(prompt_text, prompt_rect)
            virtual_screen.blit(input_display, input_rect)
            virtual_screen.blit(instructions_text, instructions_rect)
            virtual_screen.blit(return_text, return_rect)

            screen.fill(BLACK)
            screen.blit(virtual_screen, (offset_x, offset_y))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        in_input = False
                        in_menu = True
                    elif event.key == pygame.K_RETURN:
                        if input_text.isdigit():
                            entered_number = int(input_text)
                            if entered_number > 0:
                                # Update total_trials here directly
                                total_trials = entered_number
                                console.print(Text(f"Total trials set to {total_trials}. Will be adjusted if not divisible by directions.", style="blue"))
                                in_input = False
                                in_menu = True # Return to menu
                            else:
                                input_error = True # Zero trials not allowed
                        else:
                            input_error = True # Not a digit
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                        input_error = False # Clear error on backspace
                    elif event.unicode.isdigit():
                        input_text += event.unicode
                        input_error = False # Clear error on valid input
            if not running: break

        # -----------------------------
        # 5. BUFFER SCREEN
        # -----------------------------
        elif in_buffer_screen:
            virtual_screen.fill(BLACK)
            buffer_screen_title = large_font.render("Ready?", True, WHITE)
            start_trial_text = medium_font.render("Press S to Start Trial", True, GREEN)

            buffer_screen_title_rect = buffer_screen_title.get_rect(center=(center_x, center_y // 2))
            start_trial_text_rect = start_trial_text.get_rect(center=(center_x, center_y))

            virtual_screen.blit(buffer_screen_title, buffer_screen_title_rect)
            virtual_screen.blit(start_trial_text, start_trial_text_rect)

            screen.fill(BLACK)
            screen.blit(virtual_screen, (offset_x, offset_y))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_s:
                        in_buffer_screen = False # Start the main trial loop
            if not running: break

        # -----------------------------
        # 6. TRIAL MENU (accessible via M)
        # -----------------------------
        elif in_trial_menu:
            virtual_screen.fill(BLACK)
            menu_title = medium_font.render("Trial Menu", True, WHITE)
            quit_text = medium_font.render("Press Q to Quit", True, RED)
            resume_text = medium_font.render("Press R to Resume", True, GREEN)

            menu_title_rect = menu_title.get_rect(center=(center_x, center_y // 2))
            quit_rect = quit_text.get_rect(center=(center_x, center_y))
            resume_rect = resume_text.get_rect(center=(center_x, center_y + virtual_height // 10))

            virtual_screen.blit(menu_title, menu_title_rect)
            virtual_screen.blit(quit_text, quit_rect)
            virtual_screen.blit(resume_text, resume_rect)

            screen.fill(BLACK)
            screen.blit(virtual_screen, (offset_x, offset_y))
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
            if not running: break

        # -----------------------------
        # 7. AFTER SESSION MENU
        # -----------------------------
        elif in_after_session_menu:
            virtual_screen.fill(BLACK)
            question_text = large_font.render("Do you want to continue?", True, WHITE)
            continue_text = medium_font.render("Press Y to continue", True, GREEN)
            quit_text = medium_font.render("Press N to exit", True, RED)

            question_rect = question_text.get_rect(center=(center_x, center_y // 2))
            continue_rect = continue_text.get_rect(center=(center_x, center_y))
            quit_rect = quit_text.get_rect(center=(center_x, center_y + virtual_height // 10))

            virtual_screen.blit(question_text, question_rect)
            virtual_screen.blit(continue_text, continue_rect)
            virtual_screen.blit(quit_text, quit_rect)

            screen.fill(BLACK)
            screen.blit(virtual_screen, (offset_x, offset_y))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        in_after_session_menu = False
                        in_menu = True # Go back to menu, allows setting new trial # / starting new seq
                        start_enable_time = time.time() + time_between_sessions
                    elif event.key == pygame.K_n:
                        in_after_session_menu = False
                        running = False
                        # No need to stop eeg/quit here, handled after main loop
            if not running: break

        # -----------------------------
        # 8. MAIN TRIAL LOOP
        # -----------------------------
        else: # This is the actual trial sequence now
            if not current_direction: # Safety check if sequence generation failed but loop continued
                print("Error: Current direction not set. Exiting.")
                running = False
                break

            # 8a. Focus Period
            virtual_screen.fill(BLACK)
            pygame.draw.rect(virtual_screen, GREEN, left_bar_rect)
            pygame.draw.rect(virtual_screen, GREEN, right_bar_rect)
            pygame.draw.rect(virtual_screen, GREEN, top_bar_rect)
            if args.four_bars:
                pygame.draw.rect(virtual_screen, GREEN, bottom_bar_rect)

            trial_info = small_font.render(f"Trial {trial_number}/{total_trials}", True, WHITE)
            trial_info_rect = trial_info.get_rect(topright=(virtual_width - 30, 30))
            virtual_screen.blit(trial_info, trial_info_rect)

            draw_plus_sign(virtual_screen, center_pos, plus_length, loading_bar_thickness, WHITE)

            screen.fill(BLACK)
            screen.blit(virtual_screen, (offset_x, offset_y))
            pygame.display.flip()

            focus_duration = 3
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

            # 8b. Direction Arrow
            virtual_screen.fill(BLACK)
            pygame.draw.rect(virtual_screen, GREEN, left_bar_rect)
            pygame.draw.rect(virtual_screen, GREEN, right_bar_rect)
            pygame.draw.rect(virtual_screen, GREEN, top_bar_rect)
            if args.four_bars:
                pygame.draw.rect(virtual_screen, GREEN, bottom_bar_rect)
            virtual_screen.blit(trial_info, trial_info_rect)

            arrow_color = WHITE
            arrow_length = virtual_width // 15
            arrow_width = virtual_height // 40

            if current_direction == 'left':
                pygame.draw.polygon(
                    virtual_screen, arrow_color,
                    [
                        (center_x - arrow_length, center_y),
                        (center_x, center_y - arrow_width),
                        (center_x, center_y + arrow_width)
                    ]
                )
            elif current_direction == 'right':
                pygame.draw.polygon(
                    virtual_screen, arrow_color,
                    [
                        (center_x + arrow_length, center_y),
                        (center_x, center_y - arrow_width),
                        (center_x, center_y + arrow_width)
                    ]
                )
            elif current_direction == 'up':
                pygame.draw.polygon(
                    virtual_screen, arrow_color,
                    [
                        (center_x, center_y - arrow_length),
                        (center_x - arrow_width, center_y),
                        (center_x + arrow_width, center_y)
                    ]
                )
            else:
                pygame.draw.polygon(
                    virtual_screen, arrow_color,
                    [
                        (center_x, center_y + arrow_length),
                        (center_x - arrow_width, center_y),
                        (center_x + arrow_width, center_y)
                    ]
                )

            draw_plus_sign(virtual_screen, center_pos, plus_length, loading_bar_thickness, WHITE)

            screen.fill(BLACK)
            screen.blit(virtual_screen, (offset_x, offset_y))
            pygame.display.flip()

            pre_loading_duration = 1.7
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

            # 8c. Loading Bar
            loading_duration = 7
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

                elapsed_time = time.time() - loading_start_time
                loading_progress = elapsed_time / loading_duration

                virtual_screen.fill(BLACK)
                pygame.draw.rect(virtual_screen, GREEN, left_bar_rect)
                pygame.draw.rect(virtual_screen, GREEN, right_bar_rect)
                pygame.draw.rect(virtual_screen, GREEN, top_bar_rect)
                if args.four_bars:
                    pygame.draw.rect(virtual_screen, GREEN, bottom_bar_rect)
                virtual_screen.blit(trial_info, trial_info_rect)

                if current_direction == 'left':
                    pygame.draw.polygon(
                        virtual_screen, arrow_color,
                        [
                            (center_x - arrow_length, center_y),
                            (center_x, center_y - arrow_width),
                            (center_x, center_y + arrow_width)
                        ]
                    )
                    max_length = center_x - (left_bar_rect.x + left_bar_rect.width)
                    current_length = loading_progress * max_length
                    pygame.draw.rect(
                        virtual_screen, WHITE,
                        (
                            center_x - current_length,
                            center_y - loading_bar_thickness // 2,
                            current_length,
                            loading_bar_thickness
                        )
                    )

                elif current_direction == 'right':
                    pygame.draw.polygon(
                        virtual_screen, arrow_color,
                        [
                            (center_x + arrow_length, center_y),
                            (center_x, center_y - arrow_width),
                            (center_x, center_y + arrow_width)
                        ]
                    )
                    max_length = (right_bar_rect.x) - center_x
                    current_length = loading_progress * max_length
                    pygame.draw.rect(
                        virtual_screen, WHITE,
                        (
                            center_x,
                            center_y - loading_bar_thickness // 2,
                            current_length,
                            loading_bar_thickness
                        )
                    )

                elif current_direction == 'up':
                    pygame.draw.polygon(
                        virtual_screen, arrow_color,
                        [
                            (center_x, center_y - arrow_length),
                            (center_x - arrow_width, center_y),
                            (center_x + arrow_width, center_y)
                        ]
                    )
                    max_length = center_y - (top_bar_rect.y + top_bar_rect.height)
                    current_length = loading_progress * max_length
                    pygame.draw.rect(
                        virtual_screen, WHITE,
                        (
                            center_x - loading_bar_thickness // 2,
                            center_y - current_length,
                            loading_bar_thickness,
                            current_length
                        )
                    )

                elif current_direction == 'down':
                    pygame.draw.polygon(
                        virtual_screen, arrow_color,
                        [
                            (center_x, center_y + arrow_length),
                            (center_x - arrow_width, center_y),
                            (center_x + arrow_width, center_y)
                        ]
                    )
                    max_length = (bottom_bar_rect.y) - center_y
                    current_length = loading_progress * max_length
                    pygame.draw.rect(
                        virtual_screen, WHITE,
                        (
                            center_x - loading_bar_thickness // 2,
                            center_y,
                            loading_bar_thickness,
                            current_length
                        )
                    )

                draw_plus_sign(virtual_screen, center_pos, plus_length, loading_bar_thickness, WHITE)

                screen.fill(BLACK)
                screen.blit(virtual_screen, (offset_x, offset_y))
                pygame.display.flip()

                save_data(eeg_processor, metadata, current_direction, trial_number, full_session_path)

                clock.tick(60)
                if not running or in_trial_menu:
                    break

            if not running:
                break

            # 8d. Rest Period
            rest_duration = np.random.uniform(3, 5)
            rest_start_time = time.time()
            while time.time() - rest_start_time < rest_duration and not in_trial_menu and running:
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

                virtual_screen.fill(BLACK)
                pygame.draw.rect(virtual_screen, GREEN, left_bar_rect)
                pygame.draw.rect(virtual_screen, GREEN, right_bar_rect)
                pygame.draw.rect(virtual_screen, GREEN, top_bar_rect)
                if args.four_bars:
                    pygame.draw.rect(virtual_screen, GREEN, bottom_bar_rect)
                virtual_screen.blit(trial_info, trial_info_rect)

                rest_text = small_font.render("Rest (Press M for Menu)", True, WHITE)
                rest_rect = rest_text.get_rect(center=center_pos)
                virtual_screen.blit(rest_text, rest_rect)

                screen.fill(BLACK)
                screen.blit(virtual_screen, (offset_x, offset_y))
                pygame.display.flip()
                clock.tick(60)

            if not running:
                break

            # --- Trial Completion / Next Direction --- #
            trial_number += 1
            if trial_number > total_trials:
                virtual_screen.fill(BLACK)
                completion_text = medium_font.render("All Trials Completed!", True, GREEN)
                completion_rect = completion_text.get_rect(center=center_pos)
                virtual_screen.blit(completion_text, completion_rect)

                screen.fill(BLACK)
                screen.blit(virtual_screen, (offset_x, offset_y))
                pygame.display.flip()
                time.sleep(3)
                in_after_session_menu = True # Go to after session menu
                # Don't reset trial_number here, it indicates completion
            elif running: # Check running flag before accessing sequence
                # Get next direction from the pre-shuffled sequence
                current_direction = full_trial_sequence[trial_number - 1]

        # If we triggered the trial menu (check outside the main else block)
        while in_trial_menu and running:
            # ... (Trial menu drawing/logic as before, inside its own loop) ...
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False; in_trial_menu = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: running = False; in_trial_menu = False
                    elif event.key == pygame.K_r: in_trial_menu = False
            if not running: break # Break from this inner while loop
        if not running: break # Break from outer main loop if quit from trial menu

    # Final cleanup
    # Check if eeg_processor was successfully initialized before stopping
    if 'eeg_processor' in locals() and eeg_processor:
        eeg_processor.stop()
    pygame.quit()
    console.print(Text("Application finished.", style="bold blue"))
    sys.exit()

if __name__ == "__main__":
    main()