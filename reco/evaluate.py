import numpy as np
import pandas as pd

def get_embedding(model, name):
    embedding = model.get_layer(name = name).get_weights()[0]
    return embedding

def get_predictions(model, data):
    """
    Get predictions for all user-item combinations
    
    Params:
        data (pandas.DataFrame): DataFrame of entire rating data
        model (Keras.model): Trained keras model
        
    Returns:
        pd.DataFrame: DataFrame of rating predictions for each user and each item
        
    """
    # Create the crossjoin for user-item
    user_item = user_item_crossjoin(data)
    
    # Score for every user-item combination
    user_item["RATING_PRED"] = model.predict([user_item.USER, user_item.ITEM])
    
    return user_item


def user_item_crossjoin(df):
    """
    Get cross-join of all users and items
    
    Args:
        df (pd.DataFrame): Source dataframe.

    Returns:
        pd.DataFrame: Dataframe with crossjoins
    
    """
    
    crossjoin_list = []
    for user in df.USER.unique():
        for item in df.ITEM.unique():
            crossjoin_list.append([user, item])

    cross_join_df = pd.DataFrame(data=crossjoin_list, columns=["USER", "ITEM"])
    
    return cross_join_df
    

def filter_by(df, filter_by_df, filter_by_cols):
    """From the input DataFrame (df), remove the records whose target column (filter_by_cols) values are
    exist in the filter-by DataFrame (filter_by_df)

    Args:
        df (pd.DataFrame): Source dataframe.
        filter_by_df (pd.DataFrame): Filter dataframe.
        filter_by_cols (iterable of str): Filter columns.

    Returns:
        pd.DataFrame: Dataframe filtered by filter_by_df on filter_by_cols
    """

    return df.loc[
        ~df.set_index(filter_by_cols).index.isin(
            filter_by_df.set_index(filter_by_cols).index
        )
    ]


def get_top_k_items(df, col_user, col_rating, k=10):
    """Get the top k items for each user.

    Params:
        dataframe (pandas.DataFrame): DataFrame of rating data
        col_user (str): column name for user
        col_rating (str): column name for rating
        k (int): number of items for each user

    Returns:
        pd.DataFrame: DataFrame of top k items for each user, sorted by `col_user` and `rank`
    """
    # Sort dataframe by col_user and (top k) col_rating
    top_k_items = (
        df.groupby(col_user, as_index=False)
        .apply(lambda x: x.nlargest(k, col_rating))
        .reset_index(drop=True)
    )
    # Add ranks
    top_k_items["rank"] = top_k_items.groupby(col_user, sort=False).cumcount() + 1
    return top_k_items


def recommend_topk(model, data, train, k=5):
    
    """
    Params:
        data (pandas.DataFrame): DataFrame of entire rating data
        train (pandas.DataFrame): DataFrame of train rating data
        k (int): number of items for each user

    Returns:
        pd.DataFrame: DataFrame of top k items for each user, sorted by `col_user` and `rank`
    
    """
    
    # Get predictions for all user-item combination
    all_predictions = get_predictions(model, data)
    
    # Handle Missing Values
    all_predictions.fillna(0, inplace=True)
    
    # Filter already seen items
    all_predictions_unseen = filter_by(all_predictions, train, ["USER", "ITEM"])
    
    recommend_topk_df = get_top_k_items(all_predictions_unseen, "USER", "RATING_PRED", k=5)
    
    return recommend_topk_df



def get_hit_df(rating_true, rating_pred, k):
    
    # Make sure the prediction and true data frames have the same set of users
    common_users = set(rating_true["USER"]).intersection(set(rating_pred["USER"]))
    rating_true_common = rating_true[rating_true["USER"].isin(common_users)]
    rating_pred_common = rating_pred[rating_pred["USER"].isin(common_users)]
    n_users = len(common_users)

    df_hit = get_top_k_items(rating_pred_common, "USER", "RATING_PRED", k)
    df_hit = pd.merge(df_hit, rating_true_common, on=["USER", "ITEM"])[
        ["USER", "ITEM", "rank"]
    ]

    # count the number of hits vs actual relevant items per user
    df_hit_count = pd.merge(
        df_hit.groupby("USER", as_index=False)["USER"].agg({"hit": "count"}),
        rating_true_common.groupby("USER", as_index=False)["USER"].agg(
            {"actual": "count"}
        ),
        on="USER",
    )
    
    return df_hit, df_hit_count, n_users


def precision_at_k(rating_true, rating_pred, k):
    
    df_hit, df_hit_count, n_users = get_hit_df(rating_true, rating_pred, k)
    
    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / k).sum() / n_users


def recall_at_k(rating_true, rating_pred, k):

    df_hit, df_hit_count, n_users = get_hit_df(rating_true, rating_pred, k)

    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / df_hit_count["actual"]).sum() / n_users



def ndcg_at_k(rating_true, rating_pred, k):

    df_hit, df_hit_count, n_users = get_hit_df(rating_true, rating_pred, k)
    
    if df_hit.shape[0] == 0:
        return 0.0

    # calculate discounted gain for hit items
    df_dcg = df_hit.copy()
    # relevance in this case is always 1
    df_dcg["dcg"] = 1 / np.log1p(df_dcg["rank"])
    # sum up discount gained to get discount cumulative gain
    df_dcg = df_dcg.groupby("USER", as_index=False, sort=False).agg({"dcg": "sum"})
    # calculate ideal discounted cumulative gain
    df_ndcg = pd.merge(df_dcg, df_hit_count, on=["USER"])
    df_ndcg["idcg"] = df_ndcg["actual"].apply(
        lambda x: sum(1 / np.log1p(range(1, min(x, k) + 1)))
    )

    # DCG over IDCG is the normalized DCG
    return (df_ndcg["dcg"] / df_ndcg["idcg"]).sum() / n_users