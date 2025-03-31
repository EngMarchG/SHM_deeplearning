import os
import torch
import warnings
import numpy as np
import pandas as pd


def format_array(arr, decimals=2):
    """
    Pretty print a 1D array with a specified number of decimals.

    Args:
        arr (_type_): Any 1D array or list.
        decimals (int, optional): Number of decimals to print. Defaults to 2.

    Returns:
        _type_: Formatted string representation of the array.
    """
    return '[' + ',\n '.join(', '.join(f"{val:.{decimals}f}" for val in arr[i:i+10]) 
                            for i in range(0, len(arr), 10)) + ']'


def load_and_combine_data(folder_path, include, exclude=None, max_files=None):
    """
    Load and combine data from multiple CSV files in a specified folder.

    Args:
        folder_path (str): Path to the folder containing the CSV files.
        include (str): String to include in the file names.
        exclude (str, optional): String to exclude from the file names. Defaults to None.
        max_files (int, optional): Maximum number of files to combine. Defaults to None, which means all files.

    Returns:
        pandas.DataFrame: Combined data from all the matching CSV files.
    """
    # Get a list of all the csv files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and include in f]
    
    # Sort the files to ensure consistent behavior across different platforms
    files.sort()
    
    # Filter the files based on the exclude parameter
    if exclude is not None:
        files = [f for f in files if exclude not in f]
    
    # Limit the number of files based on the max_files parameter
    if max_files is not None and max_files < len(files):
        files = files[:max_files]
    
    # Load and combine the data from the files
    dataframes = [pd.read_csv(os.path.join(folder_path, f), header=None) for f in files]
    
    # Reset the column names
    for df in dataframes:
        df.columns = range(df.shape[1])
    
    # Combine the data from the dataframes
    data = pd.concat(dataframes, axis=0)
    
    return data


def normalize_data(data):
    """
    Normalize a pandas DataFrame using the mean and standard deviation.

    Args:
        data (pandas.DataFrame): Data to normalize.

    Returns:
        pandas.DataFrame: Normalized data.
        float: Mean of the original data.
        float: Standard deviation of the original data.
    """
    # Check if the data is a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        mean = data.values.mean()
        std = data.values.std()
    elif isinstance(data, torch.Tensor):
        mean = data.mean().item()
        std = data.std().item()
    else:
        mean = np.mean(data)
        std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std


def min_max_scale(data):
    """
    Scale a pandas DataFrame or a PyTorch tensor to the range [0, 1] using the minimum and maximum values.

    Args:
        data (pandas.DataFrame or torch.Tensor): Data to scale.

    Returns:
        pandas.DataFrame or torch.Tensor: Scaled data.
        float: Minimum of the original data.
        float: Maximum of the original data.
    """
    if isinstance(data, pd.DataFrame):
        min_val = data.values.min()
        max_val = data.values.max()
    elif isinstance(data, torch.Tensor):
        min_val = data.min().item()
        max_val = data.max().item()
    elif isinstance(data, np.ndarray) or isinstance(data, list):
        min_val = np.min(data)
        max_val = np.max(data)
    else:
        raise TypeError("Data should be a pandas DataFrame, a PyTorch tensor, a numpy array, or a list.")
    
    if min_val == max_val:
        warnings.warn("Minimum value is equal to maximum value in the data. All elements in the scaled data will be 1.")
        return np.ones(data.shape) if isinstance(data, np.ndarray) else torch.ones(data.shape), min_val, max_val

    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data, min_val, max_val



def load_and_normalize_data_tensor(folder_path, include, exclude=None, max_files=None, scaling_type='min_max'):
    """
    Load and normalize data from multiple CSV files in a specified folder.
    Depends on the load_and_combine_data and normalize_data functions.

    Args:
        folder_path (path): Path to the folder containing the CSV files.
        include (str): String to include in the file names.
        exclude (str): String to exclude from the file names. Defaults to None.
        max_files (int, optional): Maximum number of files to combine. Defaults to None, which means all files.

    Returns:
        tensor: Normalized data from all the matching CSV files.
        float: Mean of the original data.
        float: Standard deviation of the original data.
    """
    data = load_and_combine_data(folder_path, include, exclude, max_files)
    if scaling_type == 'min_max':
        data_normalized, data_return, data_return2 = min_max_scale(data) # This gives min and max
    else:
        data_normalized, data_return, data_return2 = normalize_data(data) # This gives mean and std
    data_tensor = torch.tensor(data_normalized.values, dtype=torch.float32)
    return data_tensor, data_return, data_return2


def denormalize_data(data, data_return, data_return2, scaling_type='min_max'):
    """
    Denormalize a tensor using the mean and standard deviation or min and max.

    Args:
        data (torch.Tensor): Data to denormalize.
        data_return (float): Mean or min of the original data.
        data_return2 (float): Standard deviation or max of the original data.
        scaling_type (str): Type of scaling applied to the data. Defaults to 'min_max'.

    Returns:
        torch.Tensor: Denormalized data.
    """
    if scaling_type == 'minmax':
        return data * (data_return2 - data_return) + data_return
    else:
        return data * data_return2 + data_return