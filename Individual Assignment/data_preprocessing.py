# data_preparation.py

import scipy.io
import pickle

# Load the Traffic Flow Forecasting Dataset from .mat File
mat = scipy.io.loadmat('traffic_dataset.mat')

# Extract Data from the .mat File
tra_X_tr = mat['tra_X_tr']
tra_Y_tr = mat['tra_Y_tr']
tra_X_te = mat['tra_X_te']
tra_Y_te = mat['tra_Y_te']
tra_adj_mat = mat['tra_adj_mat']

# Save the data using pickle
with open('tra_X_tr.pkl', 'wb') as f:
    pickle.dump(tra_X_tr, f)

with open('tra_Y_tr.pkl', 'wb') as f:
    pickle.dump(tra_Y_tr, f)

    # Save the data using pickle
with open('tra_X_te.pkl', 'wb') as f:
    pickle.dump(tra_X_te, f)

with open('tra_Y_te.pkl', 'wb') as f:
    pickle.dump(tra_Y_te, f)

print("Data preparation complete. Data saved as pickle files.")
