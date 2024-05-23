from torch.utils.data import Dataset
import torch
import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

class IncomeDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, file_name):
        df = pd.read_csv(file_name)
        df = df.sample(1000).reset_index(drop=True) # TODO
        
        self.target = 'income'
        self.n_features = df.shape[1] - 1

        # Do some label encoding
        categorical_columns = []
        categorical_dims =  {}
        for col in df.columns[df.dtypes == object]:
            print(col, df[col].nunique())
            l_enc = LabelEncoder()
            df[col] = df[col].fillna("VV_likely")
            df[col] = l_enc.fit_transform(df[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        df.dropna()
        
        # One-hot encoding of categorical variables
        self.data = df.astype('float32')
        
        # Normalize features
        X_features = self.data.drop(self.target, axis=1)
        normalized_X_features = pd.DataFrame(
            StandardScaler().fit_transform(X_features),
            columns = X_features.columns)
        
        # Save features and target
        self.X = normalized_X_features
        self.y = self.data[self.target]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]