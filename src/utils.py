import os
import json

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
        json.dump(data, f, indent=4)