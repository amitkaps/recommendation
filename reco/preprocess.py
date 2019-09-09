import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_user_item(df, user_col, item_col):
    """Function to encode users and items
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be used.
        user_col (string): Name of the user column.
        item_col (string): Name of the item column.
    
    Returns: 
        transform_df (pd.DataFrame): Modifed dataframe with the users and items index
        user_encoder: sklearn Label Encoder for users
        item_encoder: sklearn Label Encoder for users
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
    
    return encoded_df, user_encoder, item_encoder


def random_split (df, ratios, shuffle=False):
    
    """Function to split pandas DataFrame into train, validation and test
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
    
    Returns: 
        list: List of pd.DataFrame split by the given specifications.
    """
    seed = 42                  # Set random seed
    if shuffle == True:
        df = df.sample(frac=1)     # Shuffle the data
    samples = df.shape[0]      # Number of samples
    
    # Converts [0.7, 0.2, 0.1] to [0.7, 0.9]
    split_ratio = np.cumsum(ratios).tolist()[:-1] # Get split index
    
    # Get the rounded integer split index
    split_index = [round(x * samples) for x in split_ratio]
    
    # split the data
    splits = np.split(df, split_index)
    
    # Add split index (this makes splitting by group more efficient).
    for i in range(len(ratios)):
        splits[i]["split_index"] = i

    return splits


def user_split (df, col_time, col_user, ratios, chrono=False):
    
    """Function to split pandas DataFrame into train, validation and test (by user in chronological order)
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be split.
        col_time (string): column name for timestamp
        col_user (string): column name for user
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
        chrono (boolean): whether to sort in chronological order or not
    
    Returns: 
        list: List of pd.DataFrame split by the given specifications.
    """
    seed = 42                  # Set random seed
    samples = df.shape[0]      # Number of samples
    
    # Split by each group and aggregate splits together.
    splits = []

    # Sort in chronological order, the split by users
    if chrono == True:
        df_grouped = df.sort_values(col_time).groupby(col_user)
    else:
        df_grouped = df.groupby(col_user)

        
    
    for name, group in df_grouped:
        group_splits = random_split(df_grouped.get_group(name), ratios, shuffle=False)
        
        # Concatenate the list of split dataframes.
        concat_group_splits = pd.concat(group_splits)
        splits.append(concat_group_splits)
    
    # Concatenate splits for all the groups together.
    splits_all = pd.concat(splits)

    # Take split by split_index
    splits_list = [ splits_all[splits_all["split_index"] == x] for x in range(len(ratios))]

    return splits_list


def sample_data():

    data = pd.DataFrame({
        "user_index": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        "item_index": [1, 1, 2, 2, 2, 1, 2, 1, 2, 3, 3, 3, 3, 3, 1],
        "rating": [4, 4, 3, 3, 3, 4, 5, 4, 5, 5, 5, 5, 5, 5, 4],
        "timestamp": [
            '2000-01-01', '2000-01-01', '2000-01-02', '2000-01-02', '2000-01-02',
            '2000-01-01', '2000-01-01', '2000-01-03', '2000-01-03', '2000-01-03',
            '2000-01-01', '2000-01-03', '2000-01-03', '2000-01-03', '2000-01-04'
        ]
    })
    
    return data