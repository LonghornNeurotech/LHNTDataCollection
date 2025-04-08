# Instructions for EEG Motor Imagery Data Collection (Rover)

This document provides instructions on how to set up and run the `data_collection_rover.py` script for collecting EEG data during motor imagery tasks using an OpenBCI Cyton board.

## Prerequisites

*   **Python:** Python 3.8 or higher installed.
*   **Git:** Git installed for cloning the repository.
*   **EEG Hardware:** OpenBCI Cyton board and USB Dongle.

## Setup

1.  **Clone the Repository:**
    Open your terminal or command prompt and clone the repository. Replace `<repository_directory>` with the desired location name.
    ```bash
    git clone -b rd_3df_rover https://github.com/LonghornNeurotech/LHNTDataCollection.git
    cd LHNTDataCollection # Or your chosen <repository_directory> name
    ```

2.  **Install Miniconda (Environment Management):**
    We recommend using Conda for managing Python environments and dependencies.
    *   Download and install **Miniconda** for your operating system: [https://www.anaconda.com/download/success](https://www.anaconda.com/download/success)
    *   Follow the installation instructions, ensuring you add Conda to your system's PATH or initialize it for your shell (the installer usually handles this).

3.  **Create and Activate Conda Environment:**
    Create a dedicated environment for this project.
    ```bash
    conda create --name lhnt python=3.9 # Or choose a specific Python 3.8+ version
    conda activate lhnt
    ```
    *(Note: You'll need to run `conda activate lhnt` in your terminal each time you want to work on this project).*

4.  **Install Python Dependencies:**
    Once the Conda environment is active, install the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Verify Hardware Connection (Optional but Recommended):**
    *   Connect the Cyton board to the battery and plug the USB Dongle into your computer.
    *   Turn on the Cyton board (switch to "PC").
    *   Download and run the [OpenBCI GUI](https://openbci.com/downloads).
    *   In the GUI, select "Live (from Cyton)" -> "Serial (from Dongle)". It should automatically find the Dongle's serial port. Make sure to select **8 channels**.
    *   Start the session and check if data streams correctly (you should see signals appearing). This confirms the board, dongle, and drivers are working. **Close the GUI before running the Python script, as only one application can connect to the board at a time.**

## Running the Script

1.  **Navigate to the Script Directory:**
    Open a terminal or command prompt where you cloned the repository.
    ```bash
    cd path/to/LHNTDataCollection # Or your repository directory
    ```

2.  **Activate Conda Environment:**
    If not already active, activate the environment:
    ```bash
    conda activate lhnt
    ```

3.  **Execute the Script:**
    Run the script using Python.
    ```bash
    python data_collection_rover.py [OPTIONS]
    ```

4.  **Command-Line Options:**
    *   `-o <path>`, `--output-dir <path>`: Specify a directory to save the session data. If omitted, data will be saved in a `local_sessions` directory created within the script's directory. Example: `python data_collection_rover.py -o /path/to/my/data`
    *   `-s`, `--use-synthetic`: Use BrainFlow's synthetic board for testing without actual EEG hardware. Useful for development or debugging the GUI. Example: `python data_collection_rover.py -s`
    *   `--four-bars`: Display four direction cues (Up, Down, Left, Right). By default, only three cues (Up, Left, Right) are shown. Example: `python data_collection_rover.py --four-bars`

5.  **Initial Questionnaire:**
    When the script starts, you will be prompted in the terminal to enter:
    *   Session Number
    *   Subject's First Name, Last Name, and EID
    *   Information about recent stimulant intake, meals, and exercise. Answer honestly and accurately.

6.  **GUI Interaction:**
    After the questionnaire, a full-screen Pygame window will appear. Follow the on-screen prompts:
    *   **Main Menu:**
        *   Press `S` to start the data collection session.
        *   Press `N` to set the total number of trials for the session (default is 20). The script will adjust this number if necessary to ensure it's divisible by the number of active directions (3 by default, 4 if `--four-bars` is used).
        *   Press `Q` to quit the application.
    *   **Trial Sequence:**
        *   **Ready Screen:** Press `S` to begin the trial sequence or the next trial.
        *   **Focus:** A '+' sign appears for 3 seconds. Keep your eyes focused on it.
        *   **Cue:** An arrow (Up, Left, Right, and potentially Down if using `--four-bars`) appears for ~1.7 seconds. Note the direction.
        *   **Imagery & Loading Bar:** The arrow remains. Imagine performing the cued movement (e.g., clenching left hand for 'left' cue) for 7 seconds as the loading bar fills. Data is saved during this period. Try to minimize actual muscle movement and eye movement.
        *   **Rest:** A "Rest" message appears for a random duration (3-5 seconds). Relax and prepare for the next trial.
        *   **During Trial/Rest:**
            *   Press `M` to open a pause menu (Resume with `R`, Quit with `Q`).
            *   Press `ESC` to quit the application immediately.
    *   **After Session:** Once all trials are complete, you'll be asked if you want to continue to another session (`Y`) or exit (`N`). Continuing takes you back to the main menu after a short break.

## Output Data

*   Data for each trial is saved as a `.pkl` file within a session-specific folder (e.g., `local_sessions/FirstName_LastName_Session1/` or `<output-dir>/FirstName_LastName_Session1/`).
*   The filename indicates the direction and trial number (e.g., `left_1.pkl`, `up_2.pkl`).
*   Each `.pkl` file contains a Python tuple: `(eeg_data, metadata)`.
    *   `eeg_data`: A NumPy array representing the filtered EEG signals collected during the 7-second imagery period for that trial. The shape is typically `(num_eeg_channels, num_samples)`.
    *   `metadata`: A Python dictionary containing the subject/session information entered during the questionnaire and the board ID used.
*   The default `local_sessions` directory is included in the `.gitignore` file and will not be tracked by Git. If you use a custom `--output-dir`, ensure you manage that data appropriately (e.g., add it to `.gitignore` if you don't want to commit large data files). 