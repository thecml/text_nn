from torch.utils.data import Dataset
import torch
import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import make_classification
import numpy as np

class SyntheticDataset(Dataset):
    """Synthetic dataset."""

    def __init__(self):
        # Make data
        n_features = 10
        X, y = make_classification(n_samples=10000, n_features=2, n_informative=2,
                                   n_redundant=0, n_repeated=0, n_classes=2,
                                   n_clusters_per_class=2, class_sep=2, flip_y=0,
                                   weights=[0.5, 0.5], random_state=0)
        
        # Add noise
        num_additional_features = 8
        additional_features = np.random.randn(10000, num_additional_features)
        X = np.concatenate((X, additional_features), axis=1)
        
        # Convert to float32
        X = X.astype('float32')

        # Create a DataFrame from the features
        feature_columns = [f'x{i}' for i in range(1, n_features+1)]
        df = pd.DataFrame(X, columns=feature_columns)
        df['target'] = y
        self.data = df
        
        # Save features and target
        self.n_features = n_features
        self.X = df.drop('target', axis=1)
        self.y = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]