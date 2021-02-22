import os
import gzip
import json
import math
import random
import pickle
import pprint
import argparse
from preprocess import convert_unique_idx, split_train_test, create_pair

import numpy as np
import pandas as pd


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


def main(args):
    dataset = RaceDataset(args.data_dir)
    df = dataset.load()
    df = df.sort_values(by=['user'], ascending=True).reset_index(drop=True)

    # users = df['user'].unique()
    record_df = pd.DataFrame(columns=['user', 'item'])
    record_array = []

    for index, row in df.iterrows():
        item_list = row['sequence'].split(',')
        unique_item_list = np.unique(item_list)
        for item in unique_item_list:
            # record_df = record_df.append(pd.DataFrame([[row['user'], item]], columns=['user', 'item']))
            record_array.append([row['user'], item])
        if index % 1000 == 0:
            print(index)
        # if index % 10000 == 0:
        #     np.save('preprocessed/' + str(index / 10000 + 1) + '.npy', record_array)
        #     record_array = []

    # record_array = np.load('preprocessed/1.0.npy')
    # print(record_array.shape)
    # for i in range(1, 6):
    #     record_array = np.append(record_array, np.load('preprocessed/' + str(i + 1) + '.0.npy'), axis=0)

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
                                                       test_size=args.test_size,
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

    # write to files.pickle
    dirname = os.path.dirname(os.path.abspath(args.output_data))
    os.makedirs(dirname, exist_ok=True)
    with open(args.output_data, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default='data-zf/train_data.txt',
                        help="File path for raw data")
    parser.add_argument('--output_data',
                        type=str,
                        default=os.path.join('preprocessed', 'race.pickle'),
                        help="File path for preprocessed data")
    parser.add_argument('--test_size',
                        type=float,
                        default=0.2,
                        help="Proportion for training and testing split")
    args = parser.parse_args()
    # Print arguments
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    main(args)
