
def split_sets_random(features, target, test_ratio=0.2):
    """Split sets randomly

    Parameters
    ----------
    features : pd.DataFrame
        Input dataframe
    target : pd.Series
        Target column
    test_ratio : float
        Ratio used for the validation and testing sets (default: 0.2)

    Returns
    -------
    Numpy Array
        Features for the training set
    Numpy Array
        Target for the training set
    Numpy Array
        Features for the validation set
    Numpy Array
        Target for the validation set
    Numpy Array
        Features for the testing set
    Numpy Array
        Target for the testing set
    """
    from sklearn.model_selection import train_test_split


    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=test_ratio, random_state=8)

    return X_train, X_val, y_train, y_val

