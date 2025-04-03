
from path import Path
import pickle
import os

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

def access_test_folder():
    box_drive_path = get_box_drive_path()
    folder_path = box_drive_path / "LHNT EEG" / "GUI Test Uploads"
    if folder_path.exists():
        return list(folder_path.iterdir())  # Returns a list of files and folders
    else:
        raise FileNotFoundError(f"Folder '{folder_path}' not found in Box Drive.")


def save_batch_data(user_name, file_name, data, type):
    try:
        files = access_test_folder()
        matches = name_match(user_name, files)

        box_drive_path = get_box_drive_path()
        base_folder = os.path.join(box_drive_path, "LHNT EEG", "GUI Test Uploads")

        # Ensure matches is a list (handle None case)
        if not matches:  # This checks if matches is None or an empty list
            matches = []

        # User has no existing directory
        if len(matches) == 0:
            # Create a new directory for the user
            user_folder = os.path.join(base_folder, user_name)
            os.makedirs(user_folder, exist_ok=True)
            print(f"Created new directory: {user_folder}")
        else:
            # Use the existing user folder
            user_folder = os.path.join(base_folder, matches[0])

        # Ensure 'test_segments' directory exists inside the user's folder
        batch_data_folder = os.path.join(user_folder, type + "_" + "batch_data")
        os.makedirs(batch_data_folder, exist_ok=True)
        print(f"Ensured 'test_segments' directory exists at: {batch_data_folder}")

        # Define the file path for pickling
        num_files = len([f for f in os.listdir(batch_data_folder) if os.path.isfile(os.path.join(batch_data_folder, f))])

        
        file_path = os.path.join(batch_data_folder, f"{file_name + str(num_files+1)}.pkl")

        # Save data using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Data saved at: {file_path}")

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except Exception as e:
        print(f"Error: {e}")
    return

def save_test_data(user_name, file_name, data):
    try:
        files = access_test_folder()
        matches = name_match(user_name, files)

        box_drive_path = get_box_drive_path()
        base_folder = os.path.join(box_drive_path, "LHNT EEG", "GUI Test Uploads")

        # Ensure matches is a list (handle None case)
        if not matches:  # This checks if matches is None or an empty list
            matches = []

        # User has no existing directory
        if len(matches) == 0:
            # Create a new directory for the user
            user_folder = os.path.join(base_folder, user_name)
            os.makedirs(user_folder, exist_ok=True)
            print(f"Created new directory: {user_folder}")
        else:
            # Use the existing user folder
            user_folder = os.path.join(base_folder, matches[0])

        # Ensure 'test_segments' directory exists inside the user's folder
        test_segments_folder = os.path.join(user_folder, "test_segments")
        os.makedirs(test_segments_folder, exist_ok=True)
        print(f"Ensured 'test_segments' directory exists at: {test_segments_folder}")

        # Define the file path for pickling
        num_files = len([f for f in os.listdir(test_segments_folder) if os.path.isfile(os.path.join(test_segments_folder, f))])

        
        file_path = os.path.join(test_segments_folder, f"{file_name + str(num_files+1)}.pkl")

        # Save data using pickle
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Data saved at: {file_path}")

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except Exception as e:
        print(f"Error: {e}")