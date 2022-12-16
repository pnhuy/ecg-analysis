import argparse
import ast
import os
from multiprocessing import Pool, cpu_count
from functools import partial

import biosppy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhrv
import pyhrv.tools as tools
import torch
import wfdb
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import tqdm
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--sampling_rate', type=int, default=500)
    args = parser.parse_args()
    return args


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def aggregate_diagnostic(agg_df, y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


def poincare(nnis=None,
             rpeaks=None, marker='o',
             figsize=None, fp=None):
    if figsize is None:
        figsize = (6, 6)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlim([180, 700])
    ax.set_ylim([180, 700])

    # Check input values
    for nni in nnis:
        nn = pyhrv.utils.check_input(nni, rpeaks)

        # Prepare Poincaré data
        x1 = np.asarray(nn[:-1])
        x2 = np.asarray(nn[1:])

        # ax.plot(x1, x2, markersize=2)
        ax.plot(x1, x2, '%s' % marker, markersize=15, alpha=0.5, zorder=1, color='black')
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    
    # plt.axis('off')
    if fp: 
        plt.savefig(fp, bbox_inches='tight')
    
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

# def process_row(row, dataset, sampling_rate, pbar):
#     idx, sample = row
#     # sample = data[idx]
#     fp = os.path.join('dataset', 'ptb-xl', 'processed', dataset, f'{idx}.png')
#     # if os.path.isfile(fp):
#     #     continue
#     nnis = []
#     for i in range(sample.shape[1]):
#         try:
#             _, rpeaks = biosppy.signals.ecg.ecg(sample[:, i], sampling_rate=sampling_rate, show=False)[1:3]
#             nni = tools.nn_intervals(rpeaks)
#             nnis.append(nni)
#         except ValueError:
#             pass
#     # import pdb; pdb.set_trace()
#     poincare(fp=fp, nnis=nnis)
#     pbar.update(1)
#     return 0

def preprocessed(args):
    path = args.data_path
    sampling_rate = args.sampling_rate

    print('Load the raw data...')

    # load and convert annotation data
    Y = pd.read_csv(os.path.join(path, 'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    X = load_raw_data(Y, sampling_rate, path)

    print('Processing...')

    agg_df = pd.read_csv(os.path.join(path, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(agg_df, x))

    # Split data into train and test
    val_fold = 9
    test_fold = 10
    
    # Train
    X_train = X[np.where(~Y.strat_fold.isin([val_fold, test_fold]))]
    y_train = Y[(~Y.strat_fold.isin([val_fold, test_fold]))].diagnostic_superclass
    print('Train:', X_train.shape, y_train.shape)
    
    # Val
    X_val = X[np.where(Y.strat_fold == val_fold)]
    y_val = Y[Y.strat_fold == val_fold].diagnostic_superclass
    print('Val:', X_val.shape, y_val.shape)

    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
    print('Test:', X_test.shape, y_test.shape)

    for dataset, data in zip(('train', 'val', 'test'), (X_train, X_val, X_test)):
        os.makedirs(os.path.join('dataset', 'ptb-xl', 'processed', dataset), exist_ok=True)
        for idx, sample in tqdm(enumerate(data), desc=dataset, total=data.shape[0]):
            fp = os.path.join('dataset', 'ptb-xl', 'processed', dataset, f'{idx}.png')
            if os.path.isfile(fp):
                continue
            nnis = []
            for i in range(sample.shape[1]):
                try:
                    _, rpeaks = biosppy.signals.ecg.ecg(sample[:, i], sampling_rate=sampling_rate, show=False)[1:3]
                    nni = tools.nn_intervals(rpeaks)
                    nnis.append(nni)
                except ValueError:
                    pass
            # import pdb; pdb.set_trace()
            poincare(fp=fp, nnis=nnis)

    # Process labels
    mlb = MultiLabelBinarizer().fit(y_train)

    y_train = mlb.transform(y_train)
    y_val = mlb.transform(y_val)
    y_test = mlb.transform(y_test)

    y_train = pd.DataFrame(y_train, columns=mlb.classes_)
    y_val = pd.DataFrame(y_val, columns=mlb.classes_)
    y_test = pd.DataFrame(y_test, columns=mlb.classes_)

    y_train.to_csv(
        os.path.join('dataset', 'ptb-xl', 'processed', 'y_train.csv'),
        index_label='idx'
    )

    y_val.to_csv(
        os.path.join('dataset', 'ptb-xl', 'processed', 'y_val.csv'),
        index_label='idx'
    )

    y_test.to_csv(
        os.path.join('dataset', 'ptb-xl', 'processed', 'y_test.csv'),
        index_label='idx'
    )


if __name__ == '__main__':
    args = parse_args()
    preprocessed(args)

