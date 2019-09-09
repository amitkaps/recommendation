import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from reco.evaluate import user_item_crossjoin, filter_by


def encode_user_item(df, user_col, item_col, rating_col, time_col):
    """Function to encode users and items
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be used.
        user_col (string): Name of the user column.
        item_col (string): Name of the item column.
        rating_col (string): Name of the rating column.
        timestamp_col (string): Name of the timestamp column.
    
    Returns: 
        encoded_df (pd.DataFrame): Modifed dataframe with the users and items index
    """
    
    encoded_df = df.copy()
    
    user_encoder = LabelEncoder()
    user_encoder.fit(encoded_df[user_col].values)
    n_users = len(user_encoder.classes_)
    
    item_encoder = LabelEncoder()
    item_encoder.fit(encoded_df[item_col].values)
    n_items = len(item_encoder.classes_)

    encoded_df["USER"] = user_encoder.transform(encoded_df[user_col])
    encoded_df["ITEM"] = item_encoder.transform(encoded_df[item_col])
    
    encoded_df.rename({rating_col: "RATING", time_col: "TIMESTAMP"}, axis=1, inplace=True)
    
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


def user_split (df, ratios, chrono=False):
    
    """Function to split pandas DataFrame into train, validation and test (by user in chronological order)
    
    Params:     
        df (pd.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
        chrono (boolean): whether to sort in chronological order or not
    
    Returns: 
        list: List of pd.DataFrame split by the given specifications.
    """
    seed = 42                  # Set random seed
    samples = df.shape[0]      # Number of samples
    col_time = "TIMESTAMP"
    col_user = "USER"
    
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

def neg_feedback_samples(
    df,
    rating_threshold, 
    ratio_neg_per_user=1
):
    """ function to sample negative feedback from user-item interaction dataset.

    This negative sampling function will take the user-item interaction data to create 
    binarized feedback, i.e., 1 and 0 indicate positive and negative feedback, 
    respectively. 

    Args:
        df (pandas.DataFrame): input data that contains user-item tuples.
        rating_threshold (int): value below which feedback is set to 0 and above which feedback is set to 1
        ratio_neg_per_user (int): ratio of negative feedback w.r.t to the number of positive feedback for each user. 

    Returns:
        pandas.DataFrame: data with negative feedback 
    """
    
    #df.rename({"user_id":"USER", "movie_id":"ITEM", "rating":"RATING"}, inplace=True)
    #print(df.columns)
    #print(df.columns)
    df.columns = ["USER", "ITEM", "RATING", "unix_timestamp"]
    #print(df.columns)
    
    seed = 42
    
    df_pos = df.copy()
    df_pos["RATING"] = df_pos["RATING"].apply(lambda x: 1 if x >= rating_threshold else 0)
    df_pos = df_pos[df_pos.RATING>0]


    # Create a dataframe for all user-item pairs 
    df_neg = user_item_crossjoin(df)

    #remove positive samples from the cross-join dataframe
    df_neg = filter_by(df_neg, df_pos, ["USER", "ITEM"])    

    #Add a column for rating - setting it to 0
    df_neg["RATING"] = 0
   
    # Combine positive and negative samples into a single dataframe
    df_all = pd.concat([df_pos, df_neg], ignore_index=True, sort=True)
    df_all = df_all[["USER", "ITEM", "RATING"]]
    
    
    # Sample negative feedback from the combined dataframe.
    df_sample = (
        df_all.groupby("USER")
        .apply(
            lambda x: pd.concat(
                [
                    x[x["RATING"] == 1],
                    x[x["RATING"] == 0].sample(
                        min(
                            max(
                                round(len(x[x["RATING"] == 1]) * ratio_neg_per_user), 1
                            ),
                            len(x[x["RATING"] == 0]),
                        ),
                        random_state=seed,
                        replace=False,
                    )
                    if len(x[x["RATING"] == 0] > 0)
                    else pd.DataFrame({}, columns=["USER", "ITEM", "RATING"]),
                ],
                ignore_index=True,
                sort=True,
            )
        )
        .reset_index(drop=True)
        .sort_values("USER")
    )

#     print("####")
#     print(df_sample.columns)
#     print(df.columns)
#     df_sample_w_ts = pd.merge(df_sample, df, on=["USER", "ITEM"], how="left")
#     print(df_sample.columns)
    df_sample.columns = ["movie_id", "rating", "user_id"]
    return df_sample[["user_id", "movie_id", "rating"]]
#    return df_sample


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