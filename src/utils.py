import os
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to convert NumPy types to native Python types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_json(data, file_path):
    """
    Save data to a JSON file with support for NumPy types.

    Parameters:
    - data (dict): Data to save.
    - file_path (str): Path to the JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)


def flatten_state(state):
    """
    Flatten the state dictionary into a single-dimensional list.
    
    Parameters:
    - state (dict): State dictionary from the environment.
    
    Returns:
    - list: Flattened state representation.
    """
    return [
        state["pool_entropy"],  # Numeric value
        state["attempts_remaining"],  # Numeric value
        len(state["feedback_history"]),  # Derived numeric value
    ]

def load_json(file_path):
    """
    Load a JSON file if it exists, otherwise return an empty dictionary.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: Loaded JSON data or an empty dictionary.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

def save_json(data, file_path):
    """
    Save a dictionary to a JSON file.

    Parameters:
    - data (dict): Data to save.
    - file_path (str): Path to the JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)

def load_alpha_beta_mapping(file_path):
    """
    Load the JSON file and convert string keys back to tuples.
    """
    return load_json(file_path)

def save_alpha_beta_mapping(alpha_beta_mapping, file_path):
    """
    Convert tuple keys to strings and save the dictionary to a JSON file.
    """
    string_key_mapping = {
        key: value for key, value in alpha_beta_mapping.items()
    }
    save_json(string_key_mapping, file_path)

