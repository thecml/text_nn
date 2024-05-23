from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
import argparse

import rationale_net.datasets.factory as dataset_factory
import rationale_net.datasets.income_dataset as income_dataset
import rationale_net.utils.embedding as embedding
import rationale_net.utils.model as model_factory
import rationale_net.utils.generic as generic
import rationale_net.learn.train as train
import os
import torch
import datetime
import pickle
import pdb
from pathlib import Path
import wget
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

if __name__ == '__main__':
    args = generic.parse_args()
    
    # Load dataset
    file_name = Path(os.getcwd()+'/data/adult.csv')
    dataset = income_dataset.IncomeDataset(file_name)

    # Split dataset
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    n_features = dataset.n_features
    
    # update args and print
    results_path_stem = args.results_path.split('/')[-1].split('.')[0]
    #args.model_path = '{}.pt'.format(os.path.join(args.save_dir, results_path_stem))
    args.model_path = Path(os.getcwd() + f'/{args.save_dir}/model.pt')

    # model
    gen, model = model_factory.get_model(args, n_features)

    # train
    if args.train:
        epoch_stats, model, gen = train.train_model(train_loader, valid_loader, model, gen, args)
        print(epoch_stats)

    # test
    if args.test:
        test_stats = train.test_model(test_loader, model, gen, args)
        print(test_stats)
