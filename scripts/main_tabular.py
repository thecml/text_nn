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
from sklearn.preprocessing import LabelEncoder
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

    # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    # Dataloaders
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    
    n_features = 14
    
    # update args and print
    #settings = dotdict()
    #settings['save_dir'] = "snapshot"

    #embeddings, word_to_indx = embedding.get_embedding_tensor(args)

    #train_data, dev_data, test_data = dataset_factory.get_dataset(args, word_to_indx)

    results_path_stem = args.results_path.split('/')[-1].split('.')[0]
    #args.model_path = '{}.pt'.format(os.path.join(args.save_dir, results_path_stem))
    args.model_path = Path(os.getcwd() + f'/{args.save_dir}/model.pt')

    # model
    gen, model = model_factory.get_model(args, n_features)

    print()
    # train
    if args.train:
        epoch_stats, model, gen = train.train_model(train_loader, test_loader, model, gen, args)
        args.epoch_stats = epoch_stats
        save_path = args.results_path
        print("Save train/dev results to", save_path)
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path,'wb') )

    # test
    if args.test:
        test_stats = train.test_model(test_data, model, gen, args)
        args.test_stats = test_stats
        args.train_data = train_data
        args.test_data = test_data

        save_path = args.results_path
        print("Save test results to", save_path)
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path,'wb'))
