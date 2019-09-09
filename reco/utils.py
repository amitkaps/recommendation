import os, time, sys, math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from IPython.display import clear_output, SVG, display, HTML
import base64


def svg_fixed(svg, width="100%"):
    _html_template='<img width="{}" src="data:image/svg+xml;base64,{}" >'
    text = _html_template.format(width, base64.b64encode(svg))
    return HTML(text)


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
    
    
def random_split (df, ratios):
    
    """Function to split pandas DataFrame into train, validation and test
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
    
    Returns: 
        list: List of pd.DataFrame split by the given specifications.
    """
    seed = 42                  # Set random seed
    df = df.sample(frac=1)     # Shuffle the data
    samples = df.shape[0]      # Number of samples
    
    # Converts [0.7, 0.2, 0.1] to [0.7, 0.9]
    split_ratio = np.cumsum(ratios).tolist()[:-1] # Get split index
    
    # Get the rounded integer split index
    split_index = [round(x * samples) for x in split_ratio]
    
    # split the data
    splits = np.split(df, split_index)

    return splits


def encode_user_item(df, user_col, item_col):
    """Function to encode users and items
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be used.
        user_col (string): Name of the user column.
        item_col (string): Name of the item column.
    
    Returns: 
        transform_df (pd.DataFrame): Modifed dataframe with the users and items index columns
        n_users (int): number of users
        n_items (int): number of items
    """
    
    encoded_df = df.copy()
    
    user_encoder = LabelEncoder()
    user_encoder.fit(encoded_df[user_col].values)
    n_users = len(user_encoder.classes_)
    
    item_encoder = LabelEncoder()
    item_encoder.fit(encoded_df[item_col].values)
    n_items = len(item_encoder.classes_)

    encoded_df["user_index"] = user_encoder.transform(encoded_df[user_col])
    encoded_df["item_index"] = item_encoder.transform(encoded_df[item_col])
    
    print("Number of users: ", n_users)
    print("Number of items: ", n_items)
    
    return encoded_df, n_users, n_items