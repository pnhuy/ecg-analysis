import argparse
import numpy as np
import pandas as pd
import os
import glob
from scipy import io as sio
from tqdm.auto import tqdm
from matplotlib import pyplot as plt 
import pyhrv 
import io
import PIL
from utils import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


SNOMED_CT_MAPPINGS = {
    "164889003": "AF",
    "426627000": "Brady",
    "164909002": "LBBB",
    "10370003": "PR",
    "284470004": "PAC",
    "63593006": "PAC",
    "427393009": "SA",
    "426177001": "SB",
    "426783006": "SNR",
    "427084000": "STach",

    "39732003": "LAD",
    "270492004": "IAVB",
    "164890007": "AFL",
    "713427006": "CRBBB",
    "59118001": "CRBBB",
    "713426002": "IRBBB",
    "445118002": "LAnFB",
    "251146004": "LQRSV",
    "698252002": "NSIVCB",
    "427172004": "PVC",
    "17338001": "PVC",
    "164947007": "LPR",
    "111975006": "LQT",
    "164917005": "QAb",
    "47665007": "RAD",
    "59118001": "RBBB",
    "63593006": "SVPB",
    "164934002": "TAb",
    "59931005": "TInv",
    "17338001": "VPB",
}

# Load challenge data.
def load_header_file(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    header = [line.strip('\n') for line in header]
    return header

def load_challenge_data(header_file):
    header = load_header_file(header_file)
    mat_file = header_file.replace('.hea', '.mat')
    x = sio.loadmat(mat_file)
    recording = np.asarray(x['val'], dtype=np.float64)
    return recording, header

def get_dx_from_header(headers):
    dx = []
    for line in headers:
        if 'Dx' in line:
            tmp = line.split(': ')[1].split(',')
            for d in tmp:
                dx.append(SNOMED_CT_MAPPINGS.get(d, 'Other'))
    return sorted(list(set(dx)))

def poincare(nnis=None,
             rpeaks=None, marker='o',
             figsize=None, fp=None):
    # Clear the last figure
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    if figsize is None:
        figsize = (6, 6)
    fig = plt.figure(figsize=figsize)
    plt.axis('off')
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
        ax.plot(x1, x2, '%s' % marker, markersize=5, alpha=0.5, zorder=1, color='black')
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    
    
    if fp is not None: 
        plt.savefig(fp, bbox_inches='tight')
        output = None
    else:
        buf = io.BytesIO()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.savefig(buf, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        output = PIL.Image.open(buf)
    
    return output

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/cinc2020/raw')
    parser.add_argument('--out_dir', type=str, default='dataset/cinc2020/processed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='signal')
    args = parser.parse_args()
    return args


def process_image(in_fp, out_fp):
    header_file = in_fp.replace('.mat', '.hea')
    mat_file = in_fp.replace('.hea', '.mat')

    sample = load_mat(mat_file)
    headers = load_header_file(header_file)
    sampling_rate = int(headers[0].split(' ')[2])

    nnis = signal_to_nnis(sample, sampling_rate)
    poincare(nnis, fp=out_fp)


def process_cinc2020_poincare_diagram(args):
    os.makedirs(args.out_dir, exist_ok=True)
    header_files = glob.glob(
        os.path.join(args.data_path, 'training/**/*.hea'),
        recursive=True
    )
    
    # Collect labels
    labels = []
    pbar = tqdm(header_files, desc='Processing')
    for in_fp in pbar:
        filename = in_fp.replace(args.data_path, '').strip('/')
        out_fp = os.path.join(args.out_dir, filename).replace('.hea', '.png')
        out_dir = os.path.dirname(out_fp)

        header = load_header_file(in_fp)
        lbl = get_dx_from_header(header)
        labels.append(lbl)        

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        if not os.path.isfile(out_fp):
            process_image(in_fp, out_fp)
        

    idx = [i.replace(args.data_path, '').strip('/') for i in header_files]

    mlb = MultiLabelBinarizer().fit(labels)
    enc = mlb.transform(labels)
    label_df = pd.DataFrame(enc, columns=mlb.classes_, index=idx)

    train_df, test_df = train_test_split(label_df, test_size=0.2, random_state=args.seed)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=args.seed)

    train_df.to_csv(os.path.join(args.out_dir, 'y_train.csv'), index_label='idx')
    val_df.to_csv(os.path.join(args.out_dir, 'y_val.csv'), index_label='idx')
    test_df.to_csv(os.path.join(args.out_dir, 'y_test.csv'), index_label='idx')

def process_cinc2020_timeseries(args):
    train_df = pd.read_csv(os.path.join(args.out_dir, 'y_train.csv'))
    val_df = pd.read_csv(os.path.join(args.out_dir, 'y_val.csv'))
    test_df = pd.read_csv(os.path.join(args.out_dir, 'y_test.csv'))

    train_files = [os.path.join(args.data_path, i) for i in train_df['idx'].to_list()]
    val_files = [os.path.join(args.data_path, i) for i in val_df['idx'].to_list()]
    test_files = [os.path.join(args.data_path, i) for i in test_df['idx'].to_list()]

    train_features = extract_ts_features(train_files, verbose=True)
    train_features.to_csv(os.path.join(args.out_dir, 'train_features.csv'), index=False)

    val_features = extract_ts_features(val_files, verbose=True)
    val_features.to_csv(os.path.join(args.out_dir, 'val_features.csv'), index=False)

    test_features = extract_ts_features(test_files, verbose=True)
    test_features.to_csv(os.path.join(args.out_dir, 'test_features.csv'), index=False)


if __name__ == '__main__':
    args = parse_arg()
    if args.mode == 'poincare':
        process_cinc2020_poincare_diagram(args)
    elif args.mode == 'timeseries':
        process_cinc2020_timeseries(args)
