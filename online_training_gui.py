import pygame
import sys
import time
import numpy as np
from eeg_processor import EEGProcessor, partition_trial_data
from Noise_Model import noise_model
import display_functions
import torch
from Model import PCNN_3Branch
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split as tts
from tqdm import tqdm
from torch.utils.data import DataLoader as DL
from torch.utils.data import TensorDataset as TData
from model_prep import preprocess, balancing
from path import Path
import random
import zipfile
import pickle
import io

def get_box_drive_path():
    base_path = Path.home() / "Box"
    if base_path.exists():
        return base_path
    else:
        raise FileNotFoundError("Box Drive folder not found.")

def access_folder(folder_name="LHNT EEG"):
    box_drive_path = get_box_drive_path()
    folder_path = box_drive_path / folder_name
    if folder_path.exists():
        return list(folder_path.iterdir())  # Returns a list of files and folders
    else:
        raise FileNotFoundError(f"Folder '{folder_name}' not found in Box Drive.")

def name_match(n, file_list):
    matched_files = [x for x in file_list if n in x.name.lower()]
    return matched_files if matched_files else None

def all_sessions_match(file_list):
    all_files = "session"
    matched_files = [x for x in file_list if all_files in x.name.lower()]
    return matched_files if matched_files else None

def main():
    eeg_processor = EEGProcessor()
    # load model being used
    model = PCNN_3Branch()
    checkpoint = torch.load("matt_pcnn.pth", map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint.state_dict())
    model = model.float()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    in_after_session_menu = False
    trial_number = 1
    total_trials = 3 # Default number of trials - changed to 2 on 1/25/2025
    batch_size = 3 #default number of trails in a single batch - added 1/25/2025
    time_between_sessions = 180 # number of seconds to wait between sessions of data collection
    start_enable_time = time.time() # the time at/after which the start button is enabled

    # Batch data buffer
    batch_data = []


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

    #Stored name
    in_name_input = False
    name = ""
    
    direction = 'left'  # Start with 'left' and alternate

    config = {
            'BLACK': BLACK,
            'WHITE': WHITE,
            'GREEN': GREEN,
            'RED': RED,
            'large_font': large_font,
            'medium_font': medium_font,
            'small_font': small_font,
            'total_trials': total_trials,
            'infoObject': infoObject,
            'start_enable_time': start_enable_time
            }
        

    while running:
        # Update within loop for input screen to properly update menu screen
        config['total_trials'] = total_trials
        config['start_enable_time'] = start_enable_time
        
        if in_menu:
            display_functions.display_menu(screen, config)
            # Processing Input at the Main Menu
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s: 
                        if time.time() >= start_enable_time:
                            in_menu = False
                            in_name_input = True
                    elif event.key == pygame.K_n:
                        in_input = True
                        in_menu = False
                        input_text = ""
                        input_error = False
                    elif event.key == pygame.K_q:
                        running = False


        elif in_name_input:
            screen.fill(BLACK)

            # Render and center the prompt text
            prompt_text = medium_font.render("Enter First and Last Name:", True, WHITE)
            prompt_rect = prompt_text.get_rect(center=(infoObject.current_w // 2, infoObject.current_h // 3))
            screen.blit(prompt_text, prompt_rect)

            # Base dimensions for the input box
            box_padding = 20  # Extra space around text
            min_box_width = 500  # Minimum width of the box
            box_height = 70

            # Get the width of the rendered input text
            input_display = small_font.render(name, True, GREEN)
            text_width, text_height = input_display.get_size()

            # Adjust box width dynamically
            box_width = max(min_box_width, text_width + box_padding * 2)
            box_x = (infoObject.current_w // 2) - (box_width // 2)
            box_y = (infoObject.current_h // 2) - (box_height // 2)

            # Draw the input box
            pygame.draw.rect(screen, GREEN, (box_x, box_y, box_width, box_height), width=3)

            # Center input text inside the box
            input_rect = input_display.get_rect(midleft=(box_x + box_padding, infoObject.current_h // 2))
            screen.blit(input_display, input_rect)

            # Update the display
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN and " " in name.strip(" "):
                        name = name.replace(" ","_").lower()  # Space to underscore, upper to lower
                        in_name_input = False

                        try:
                            files = access_folder()
                            matched_files = name_match(name, files) #check if name found in offline database

                            if matched_files: #if name found
                                selected_file = random.choice(matched_files)

                                # Display success message with the selected file
                                success_text = small_font.render(f"Using {selected_file.name} for testing", True, GREEN)
                                success_rect = success_text.get_rect(
                                    center=(infoObject.current_w // 2, infoObject.current_h // 1.5))
                                screen.blit(success_text, success_rect)
                                pygame.display.flip()
                                time.sleep(2)  # Pause for 2 seconds before proceeding

                            else: #if name not found --> use random session
                                selected_file = random.choice(all_sessions_match(files))  # Select a random file from any of the sessions

                                # Display error message
                                error_text = small_font.render(f"No data found for this user, using random session: {selected_file.name}", True, RED)
                                error_rect = error_text.get_rect(
                                    center=(infoObject.current_w // 2, infoObject.current_h // 1.5))
                                screen.blit(error_text, error_rect)
                                pygame.display.flip()
                                time.sleep(2)  # Pause before retrying

                            in_name_input = False
                            in_questionaire_subject = True  # Proceed to the next screen

                            #selecting the random pickles to use and filling offline_sigs with the offline data
                            with zipfile.ZipFile(selected_file, "r") as zip_ref:
                                #print(zip_ref.namelist())
                                left_pkl_files = [f for f in zip_ref.namelist() if os.path.basename(f).startswith("left")]
                                selected_left_pkl = random.choice(left_pkl_files)  # Select a random left pkl
                                right_pkl_files = [f for f in zip_ref.namelist() if os.path.basename(f).startswith("right")]
                                selected_right_pkl = random.choice(right_pkl_files)  # Select a random right pkl
                                #print(selected_left_pkl)
                                #print(selected_right_pkl)

                                offline_left_sigs = []
                                offline_right_sigs = []

                                with zip_ref.open(selected_left_pkl) as file:
                                    left_pkl_data = pickle.load(io.BytesIO(file.read()))
                                    offline_left_sigs.append(left_pkl_data[0])

                                with zip_ref.open(selected_right_pkl) as file:
                                    right_pkl_data = pickle.load(io.BytesIO(file.read()))
                                    offline_right_sigs.append(right_pkl_data[0])

                                #print(offline_left_sigs)
                                #print(offline_right_sigs)

                        except FileNotFoundError as e:
                            error_text = small_font.render(str(e), True, RED)
                            error_rect = error_text.get_rect(
                                center=(infoObject.current_w // 2, infoObject.current_h // 1.5))
                            screen.blit(error_text, error_rect)
                            pygame.display.flip()
                            time.sleep(2)

                    elif event.key == pygame.K_BACKSPACE:
                        name = name[:-1]
                    elif event.key == pygame.K_SPACE and " " not in name:
                        name += " "  #
                    elif event.unicode.isalpha():
                        name += event.unicode  # Allow only letters

        elif in_input:
            # Display Input Menu for Setting Number of Trials
            display_functions.display_input_menu(screen, config, input_text, input_error)
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
            display_functions.display_in_trial_menu(screen, config)
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
            display_functions.display_after_session_menu(screen, config)
            # Processing Inputs at the After Session Menu 
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
            pre_loading_duration = 1.5  # second
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
            loading_duration = 14  # Max trial time in seconds

            # Perf counter is a more accurate timer
            loading_start_time = time.perf_counter()

            # Initialize loading bar variables
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


                screen.fill(BLACK)
                # Redraw green bars
                pygame.draw.rect(screen, GREEN, (*left_green_bar_pos, green_bar_width, green_bar_height))
                pygame.draw.rect(screen, GREEN, (*right_green_bar_pos, green_bar_width, green_bar_height))
                # Redraw trial info
                screen.blit(trial_info, trial_info_rect)

                model_input = eeg_processor.get_recent_data(.5) # Change to 1 with real board

                current_direction = model(model_input)[0]            

                if current_direction[0] > current_direction[1]:
                    current_direction = "left"
                else:
                    current_direction = "right"

                # Calling noise model
                # current_direction = noise_model(direction, current_direction)

                # Redraw Arrow
                if direction == 'left':
                    
                    # Move bar to reflect input from model
                    if current_direction == 'left':
                        current_length += 3
                        
                    else:
                        current_length -= 3
                        

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

                    # Move bar to reflect input from model
                    if current_direction == 'right':
                        current_length += 3
                        
                    else:
                        current_length -= 3
                        
                    model_input = eeg_processor.get_recent_data(.5) # Change to 1 with real board

                    current_direction = model(model_input)[0]

                    if current_direction[0] > current_direction[1]:
                        current_direction = "left"
                    else:
                        current_direction = "right"

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
                # save_data(eeg_processor, metadata, direction, trial_number, directory)
                clock.tick(60)

            trial_length = time.perf_counter() - loading_start_time
            trial_data = eeg_processor.get_recent_data(trial_length)
            trial_beginning, test_window, trial_end = partition_trial_data(trial_data)
            # Print shapes
            print("trial_beginning size =", trial_beginning.shape)
            print("test_window size =", test_window.shape)
            print("trial_end size =", trial_end.shape)


            batch_data.append(trial_data)

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

            # Screen for uploading previous batch of trials to model
            # also activates after final trial is completed (when batch size =/= 2)
            if direction == 'right' and ((trial_number % batch_size == 0) or trial_number == total_trials):
                # TODO: Call function to train model (full batches) input parameter is np array training_data
                # train model with batch_data list
                epochs = 10
                trials = 6
                criterion = torch.nn.CrossEntropyLoss()
                optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                train_losses = []
                val_losses = []
                accs = []

                # preprocess left hand and right hand signals
                segments_left = preprocess(offline_left_sigs)
                segments_right = preprocess(offline_right_sigs)
                for i in range(epochs):
                    # randomly sample p fraction of dataset, ensuring balanced left and right classes
                    offline_left, offline_right = balancing(segments_left, segments_right, p=0.25)
                    labels_left = [0] * len(online_left)
                    labels_right = [1] * len(online_right)

                    ## collect real time signals
                    left_signals = [sig for i, sig in enumerate(batch_data) if i % 2 == 0]
                    right_signals = [sig for i, sig in enumerate(batch_data) if i % 2 == 1]

                    # preprocess left and right hand signals
                    left_segments = preprocess(left_signals)
                    right_segments = preprocess(right_signals)

                    # balance classes for online data
                    online_left, online_right = balancing(left_segments, right_segments)
                    left_labels = [0] * len(online_left)
                    right_labels = [1] * len(online_right)

                    # combine data and create data loaders
                    all_sigs = offline_left + offline_right + online_left + online_right
                    all_labels = labels_left + labels_right + left_labels + right_labels

                    train_sigs, val_sigs, train_labs, val_labs = tts(all_sigs, all_labels, test_size = 0.25)
                    train_ds = TData(torch.from_numpy(np.stack(train_sigs)), torch.tensor(train_labs))
                    val_ds = TData(torch.from_numpy(np.stack(val_sigs)), torch.tensor(val_labs))

                    train_dl = DL(train_ds, batch_size = 64, shuffle = True)
                    valid_dl = DL(val_ds, batch_size = 64, shuffle = True)

                    ## train and evaluate the model on the new data
                    model.train()
                    total_train_loss = 0.0
                    pbar = tqdm(total=len(train_dl))
                    for j, (batch, label) in enumerate(train_dl):
                        batch = batch.to(device).to(torch.float32)
                        label = label.to(device)
                        optim.zero_grad()
                        y_pred = model(batch)
                        loss = criterion(y_pred, label)
                        loss.backward()
                        optim.step()
                        total_train_loss += loss.item()
                        pbar.set_description(f"Epoch {i + 1}    loss={total_train_loss / (j + 1):0.4f}")
                        pbar.update(1)
                    pbar.close()
                    train_losses.append(total_train_loss/len(train_dl))

                    model.eval()
                    total_val_loss = 0.0
                    total_accuracy = 0.0
                    with torch.no_grad():
                        p_bar = tqdm(total=len(valid_dl))
                        for j, (batch, label) in enumerate(valid_dl):
                            batch = batch.to(device).to(torch.float32)
                            label = label.to(device)
                            y_pred = model(batch)
                            loss = criterion(y_pred, label)
                            prob_pred = torch.nn.functional.softmax(y_pred, -1)
                            acc = (prob_pred.argmax(-1) == label).float().mean()
                            total_val_loss += loss.item()
                            total_accuracy += acc.item()
                            p_bar.set_description(f"val_loss={total_val_loss / (j + 1):.4f}  val_acc={total_accuracy / (j + 1):.4f}")
                            p_bar.update(1)
                        p_bar.close()
                    val_losses.append(total_val_loss/len(valid_dl))
                    accs.append(total_accuracy/len(valid_dl))

                    torch.save(model.state_dict(), f"saved_models/{person}_model{i}.pt")
                print(batch_data)
                batch_data = []
                clock.tick(60)
                batch_load_example_time = 3 #seconds
                batch_load_start_time = time.time()
                while time.time() - batch_load_start_time < batch_load_example_time:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                                break

                    # Display Batch Text 
                    screen.fill(BLACK)
                    batch_load_text = medium_font.render("Sending previous batch to model!", True, WHITE)
                    batch_load_rect = batch_load_text.get_rect(center=center_pos)
                    screen.blit(batch_load_text, batch_load_rect)

                    batch_size_text = small_font.render(f"Batch size currently set to {batch_size} trials.", True, WHITE)
                    batch_size_rect = batch_size_text.get_rect(center=(infoObject.current_w / 2, infoObject.current_h / 2 + 50))
                    screen.blit(batch_size_text, batch_size_rect)
                    pygame.display.flip()

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