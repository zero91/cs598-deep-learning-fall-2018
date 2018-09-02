import numpy as np
import h5py

def load_data(path):
    """
    Read data from HDF5 files.
    Returns:
        x_train (60000, 784)
        y_train (60000,)
        x_test  (10000, 784)
        y_test  (10000,)
    """
    data_hf = h5py.File(path, 'r')
    x_train = np.array(data_hf['x_train'])
    y_train = np.array(data_hf['y_train'][:, 0])
    x_test = np.array(data_hf['x_test'])
    y_test = np.array(data_hf['y_test'][:, 0])

    data_hf.close()
    return x_train, y_train, x_test, y_test


