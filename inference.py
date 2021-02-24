import os
import gzip
import json
import math
import random
import pickle
import pprint
import argparse
from preprocess import convert_unique_idx, split_train_test, create_pair
from util_origin import load_model, get_args, get_device, set_env

import numpy as np
import pandas as pd
import train
import torch

from race_dataset_origin import Dataset

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class DatasetLoader(object):
    def load(self):
        """Minimum condition for dataset:
          * All users must have at least one item record.
          * All items must have at least one user record.
        """
        raise NotImplementedError


class RaceDataset(DatasetLoader):
    def __init__(self, data_path):
        self.file_path = data_path

    def load(self):
        # Load data
        names = ['user', 'sequence']
        train_df = pd.read_csv(self.file_path, delimiter=':', names=names)
        return train_df


def get_prepared_data(data_path, test_size=0.2):
    dataset = RaceDataset(data_path)
    df = dataset.load()
    df = df.sort_values(by=['user'], ascending=True).reset_index(drop=True)

    # users = df['user'].unique()
    record_array = []

    for index, row in df.iterrows():
        item_list = row['sequence'].split(',')
        unique_item_list = np.unique(item_list)
        for item in unique_item_list:
            # record_df = record_df.append(pd.DataFrame([[row['user'], item]], columns=['user', 'item']))
            record_array.append([row['user'], item])

    record_df = pd.DataFrame(record_array, columns=['user', 'item'], dtype=int)

    record_df['time'] = 0
    record_df = record_df.reset_index(drop=True)

    record_df, user_mapping = convert_unique_idx(record_df, 'user')
    record_df, item_mapping = convert_unique_idx(record_df, 'item')

    print('Complete assigning unique index to user and item')

    user_size = len(record_df['user'].unique())
    item_size = len(record_df['item'].unique())
    train_user_list, test_user_list = split_train_test(record_df,
                                                       user_size,
                                                       )
    print('Complete spliting items for training and testing')

    train_pair = create_pair(train_user_list)
    print('Complete creating pair')

    dataset = {'user_size': user_size, 'item_size': item_size,
               'user_mapping': user_mapping, 'item_mapping': item_mapping,
               'train_user_list': train_user_list, 'test_user_list': test_user_list,
               'train_pair': train_pair}
    print(len(train_pair))
    print(len(user_mapping))
    print(len(item_mapping))
    print(len(train_user_list))
    print(len(test_user_list))
    print(user_size)
    print(item_size)
    return dataset


def load_bpr_model(train_args, dataset):
    user_size, item_size = dataset['user_size'], dataset['item_size']
    train_user_list, test_user_list = dataset['train_user_list'], dataset['test_user_list']
    train_pair = dataset['train_pair']
    print('test_model.user_size: ' + str(user_size))
    print('test_model.item_size: ' + str(item_size))
    test_model = train.BPR(user_size, item_size, train_args.dim, train_args.weight_decay).to(DEVICE)
    test_model.load_state_dict(torch.load(train_args.model))
    return test_model


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir',
    #                     type=str,
    #                     default='./data-zf/train_data.txt',
    #                     help="File path for raw data")
    # parser.add_argument('--test_size',
    #                     type=float,
    #                     default=0.2,
    #                     help="Proportion for training and testing split")
    # args = parser.parse_args()
    # # Print arguments
    # for k, v in sorted(vars(args).items()):
    #     print(k, '=', v)

    args = set_env(kind='zf')  # kind=['ml' or 'zf']
    data_dir = os.environ['SM_CHANNEL_EVAL']
    output_dir = os.environ['SM_OUTPUT_DATA_DIR']
    test_data_path = 'data-zf/train_data.txt'
    # test_data_path = os.path.join(data_dir, 'test_seq_data.txt')
    output_path = os.path.join(output_dir, 'output.csv')
    # model_dir = os.environ['SM_CHANNEL_MODEL']
    model_dir = './output/bpr.pt'

    prepared_data = get_prepared_data(test_data_path)
    train_args = train.get_train_args()

    model = train.train(train_args, prepared_data)
    # model = load_bpr_model(train_args, prepared_data)

    dataset = Dataset(test_data_path, max_len=args.sequence_length)
    # max_item_count = 3706 #for data_ml
    # max_item_count = 65427  # for data_zf
    tr_dl = torch.utils.data.DataLoader(dataset, 1)

    f = open(output_path, 'w')

    model = model.to(DEVICE)
    model.eval()
    print('--------------begin testing!--------------')
    i = 0
    for batch, (user_id, sequence) in enumerate(tr_dl):
        # print(batch)
        sequence = sequence[:, 1:].to(DEVICE)

        user_id -= torch.tensor([1])
        user_id.to(DEVICE)
        y_pred = model.recommend(user_id)
        # y = int(torch.argmax(y_pred).data)
        # f.write('%s\n ' % y)
        topk = torch.topk(y_pred, 10)[1].data[0].tolist()
        f.write('%s\n' % topk)

        i += 1
        # if i > 3 : break

    f.close()
