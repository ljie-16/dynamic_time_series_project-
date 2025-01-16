import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the dataset.
    """
    data = pd.read_csv(file_path)
    data = data[['lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10', 'lag_11', 'lag_12', 'lag_13', 
                 'lag_14', 'lag_15', 'lag_16', 'lag_17', 'lag_18', 'lag_19', 'lag_20', 
                 'lag_21', 'lag_22', 'lag_23', 'lag_24', 'y']]
    
    X = data.drop(columns=['y'])
    y = data['y']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def augment_and_resample(X, y):
    """
    Augments and resamples the data using SMOTE and under-sampling.
    """
    # Augment using SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Under-sample to balance classes
    undersampler = RandomUnderSampler()
    X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
    
    return X_resampled, y_resampled
