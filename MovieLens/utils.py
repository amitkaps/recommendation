import os
import time, sys
from IPython.display import clear_output


def create_directory(directory_path):
    """
    Checks whether a directory exists in the current path, and if not creates it.
    
    directory_path: path string for the folder (relative to current working directory)
    """
    
    # Get current path
    current_path = os.getcwd()
    current_path
    
    # define the name of the directory to be created
    new_dir_path= current_path + directory_path
    
    # Check if feature dir exists
    if os.path.exists(new_dir_path):
        print("Directory already exists %s" % new_dir_path)
    else:     
        try:
            os.mkdir(new_dir_path)
        except OSError:
            print("Creation of the directory %s failed" % new_dir_path)
        else:
            print("Successfully created directory %s" % new_dir_path)

            

def update_progress(progress):
    bar_length = 40
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)