import os
import torch
import pandas as pd


def load_ams(X_y=False, device='cpu'):
    """ Loads and returns the dataset for the NEL Project (regression)

    This data comes from a farm in North Italy, where the production
    is entirely based on Automatic Milking Systems (AMS).
    It has been already preprocessed and cleaned.   
    The file is located in NEL_project/dt_cleaned.csv

    Basic information:
    - Number of data instances: 177;
    - Number of input features: 14;
    - Target's (lactose_percent) range: [4.707 - 5.059].

    Parameters
    ----------
    X_y : bool (default=False)
        Return data as two objects of type torch.Tensor, otherwise as a
        pandas.DataFrame.

    Returns
    -------
    pandas.DataFrame
        An object of type pandas.DataFrame which holds the data. The
        target is the last column.
    """
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "dt_cleaned.csv"))
    if X_y:
        return torch.from_numpy(df.values[:, :-1]).float().to(device), torch.from_numpy(df.values[:, -1]).float().to(device)
    else:
        return df
