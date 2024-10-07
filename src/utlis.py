import pickle
import os
import dill



def save_data_to_pickle(data, file_path):
    """
    Save the given data to a pickle file.

    Parameters:
    data (any): The data to be saved.
    filename (str): The name of the file where data will be saved.
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as file:
        dill.dump(data, file)

# Example usage:
# data = {'key': 'value'}
# save_data_to_pickle(data, 'data.pkl')