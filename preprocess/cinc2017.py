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

# try:
#     from preprocess.ptbxl import poincare
# except ImportError:
#     import sys 
#     sys.path.append(os.path.dirname(os.path.realpath(__file__)))
#     from ptbxl import poincare

LABELS = ['N', 'A', 'O'] # , '~'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/cinc2017/raw')
    parser.add_argument('--output_dir', type=str, default='dataset/cinc2017/processed')
    parser.add_argument('--sampling_rate', type=int, default=300)
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

def preprocess_cinc2017(data_dir, data_label, output_image_dir='dataset/cinc2017/processed/{}', output_label='dataset/cinc2017/processed/y_{}.csv', sampling_rate=300):
    labels = pd.read_csv(data_label, header=None, names=['path', 'label'])
    # import ipdb; ipdb.set_trace()
    labels = labels[labels.label.isin(LABELS)]
    train, val = train_test_split(labels, test_size=0.2, random_state=42)
    values = []
    
    
    for dataset, label_df in zip(('train', 'val'), (train, val)):
        output_image_dir_ = output_image_dir.format(dataset)
        output_label_ = output_label.format(dataset)

        rows = []
        os.makedirs(output_image_dir_, exist_ok=True)
        # Process images
        for _, row in tqdm(label_df.iterrows(), total=len(label_df), desc=dataset):
            filenames = row.path.replace('/', '_')
            fp = os.path.join(output_image_dir_, f'{filenames}.png')
            
            lbl = [0] * len(LABELS)
            lbl_idx = LABELS.index(row.label)
            lbl[lbl_idx] = 1

            r = [filenames] + lbl
            rows.append(r)

            # if os.path.isfile(fp):
            #     continue

            data = read_cinc2017_mat_file(os.path.join(data_dir, f"{row.path}.mat"))
            _, rpeaks = biosppy.signals.ecg.ecg(data, sampling_rate=sampling_rate, show=False)[1:3]
            nni = tools.nn_intervals(rpeaks)
            values.append(nni)

            poincare(fp=fp, nni=nni)
        
        # Process label
        label_df = pd.DataFrame(rows, columns=['idx'] + LABELS)
        label_df.to_csv(output_label_, index=False)


def main():
    args = parse_args()

    # Process train data
    data_dir = os.path.join(args.data_path, 'training')
    data_label = os.path.join(args.data_path, 'training/REFERENCE.csv')
    output_image_dir='dataset/cinc2017/processed/{}'
    output_label='dataset/cinc2017/processed/y_{}.csv'

    preprocess_cinc2017(data_dir, data_label, output_image_dir, output_label)

if __name__ == '__main__':
    main()