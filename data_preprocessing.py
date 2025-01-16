import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from bootstrapping import BootstrappingWrapper
from smoothers import SpectralSmoother  # Assuming these are part of your custom implementation

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

def augment_with_mbb(X, y, block_length=24, n_samples=100, seed=33):
    """
    Augments the data using Moving Block Bootstrapping (MBB).
    
    Parameters:
    - X: Feature data (scaled).
    - y: Target labels.
    - block_length: Length of the blocks for bootstrapping.
    - n_samples: Number of bootstrap samples to generate.
    - seed: Random seed for reproducibility.
    
    Returns:
    - X_augmented: Augmented feature data.
    - y_augmented: Corresponding augmented labels.
    """
    np.random.seed(seed)
    X_augmented = []
    y_augmented = []
    
    # Loop through each column for augmentation
    for i in range(X.shape[1]):
        smoother = SpectralSmoother(smooth_fraction=0.18, pad_len=12)
        bts = BootstrappingWrapper(smoother, bootstrap_type='mbb', block_length=block_length)
        bts_samples = bts.sample(X[:, i], n_samples=n_samples)
        
        # Flatten and append bootstrap samples
        for sample in bts_samples:
            X_augmented.append(sample)
            y_augmented.append(y)
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    return X_augmented, y_augmented

def preprocess_and_augment(file_path):
    """
    Full pipeline for preprocessing and augmenting the data.
    """
    # Load and preprocess
    X, y = load_and_preprocess_data(file_path)
    
    # Augment using MBB
    X_augmented, y_augmented = augment_with_mbb(X, y)
    
    return X_augmented, y_augmented
