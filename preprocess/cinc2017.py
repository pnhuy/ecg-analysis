import argparse
import os

import biosppy
import numpy as np
import pandas as pd
import pyhrv
from matplotlib import pyplot as plt
import pyhrv.tools as tools
from scipy import io as sio
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from utils import *


# try:
#     from preprocess.ptbxl import poincare
# except ImportError:
#     import sys 
#     sys.path.append(os.path.dirname(os.path.realpath(__file__)))
#     from ptbxl import poincare

LABELS = ['N', 'A', 'O', '~'] # , '~'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/cinc2017/raw/training')
    parser.add_argument('--output_dir', type=str, default='dataset/cinc2017/processed')
    parser.add_argument('--sampling_rate', type=int, default=300)
    parser.add_argument('--mode', type=str, default='poincare')
    args = parser.parse_args()
    return args

def read_cinc2017_mat_file(fp):
    data = sio.loadmat(fp)['val'].squeeze()
    data = np.nan_to_num(data)
    return data

def poincare(nni=None,
             rpeaks=None, marker='o',
             figsize=None, fp=None):
    # Check input values
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Prepare Poincaré data
    x1 = np.asarray(nn[:-1])
    x2 = np.asarray(nn[1:])

    if figsize is None:
        figsize = (6, 6)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlim([100, 400])
    ax.set_ylim([100, 400])
    # ax.plot(x1, x2, markersize=2)
    ax.plot(x1, x2, '%s' % marker, markersize=15, alpha=0.5, zorder=1, color='black')
    plt.axis('off')
    if fp: 
        plt.savefig(fp)
    plt.close()

def preprocess_cinc2017(data_dir, data_label, output_image_dir='dataset/cinc2017/processed', output_label='dataset/cinc2017/processed/y_{}.csv', sampling_rate=300):
    labels = pd.read_csv(data_label, header=None, names=['path', 'label'])
    # import ipdb; ipdb.set_trace()
    labels = labels[labels.label.isin(LABELS)]
    train, test = train_test_split(labels, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.25, random_state=42)
    values = []
    
    
    for dataset, label_df in zip(('train', 'val', 'test'), (train, val, test)):
        output_image_dir_ = output_image_dir #.format(dataset)
        output_label_ = output_label.format(dataset)

        rows = []
        # os.makedirs(os.path.join(output_image_dir_, dataset), exist_ok=True)
        # Process images
        for _, row in tqdm(label_df.iterrows(), total=len(label_df), desc=dataset):
            # filenames = os.path.join(dataset, row.path)
            filenames = row.path
            fp = os.path.join(output_image_dir_, f'{row.path}')
            os.makedirs(
                os.path.dirname(fp), exist_ok=True
            )
            
            lbl = [0] * len(LABELS)
            lbl_idx = LABELS.index(row.label)
            lbl[lbl_idx] = 1

            r = [filenames] + lbl
            rows.append(r)

            if os.path.isfile(fp):
                continue

            data = read_cinc2017_mat_file(os.path.join(data_dir, f"{row.path}.mat"))
            _, rpeaks = biosppy.signals.ecg.ecg(data, sampling_rate=sampling_rate, show=False)[1:3]
            nni = tools.nn_intervals(rpeaks)
            values.append(nni)

            poincare(fp=fp, nni=nni)
        
        # Process label
        label_df = pd.DataFrame(rows, columns=['idx'] + LABELS)
        label_df.to_csv(output_label_, index=False)

def process_cinc2017_timeseries(args):
    train_df = pd.read_csv(os.path.join(args.output_dir, 'y_train.csv'))
    val_df = pd.read_csv(os.path.join(args.output_dir, 'y_val.csv'))
    test_df = pd.read_csv(os.path.join(args.output_dir, 'y_test.csv'))

    train_files = [os.path.join(args.data_path, i) for i in train_df['idx'].to_list()]
    val_files = [os.path.join(args.data_path, i) for i in val_df['idx'].to_list()]
    test_files = [os.path.join(args.data_path, i) for i in test_df['idx'].to_list()]

    train_features = extract_ts_features(train_files, verbose=True)
    train_features.to_csv(os.path.join(args.output_dir, 'train_features.csv'), index=False)

    val_features = extract_ts_features(val_files, verbose=True)
    val_features.to_csv(os.path.join(args.output_dir, 'val_features.csv'), index=False)

    test_features = extract_ts_features(test_files, verbose=True)
    test_features.to_csv(os.path.join(args.output_dir, 'test_features.csv'), index=False)


def main():
    args = parse_args()

    # Process train data
    if args.mode == 'poincare':
        data_dir = os.path.join(args.data_path, 'training')
        data_label = os.path.join(args.data_path, 'training/REFERENCE.csv')
        output_image_dir='dataset/cinc2017/processed'
        output_label='dataset/cinc2017/processed/y_{}.csv'
        preprocess_cinc2017(data_dir, data_label, output_image_dir, output_label)

    elif args.mode == 'timeseries':
        process_cinc2017_timeseries(args)
    
if __name__ == '__main__':
    main()