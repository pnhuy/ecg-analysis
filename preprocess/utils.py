
import io

import biosppy
import numpy as np
import pandas as pd
import PIL
import pyhrv
from matplotlib import pyplot as plt
from scipy import io as sio
from tqdm.auto import tqdm
import wfdb
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters
from tsfresh.feature_extraction.extraction import extract_features

EXCLUDING_SETTINGS = ['approximate_entropy', 'sample_entropy', 'matrix_profile', 'number_cwt_peaks',
                      'partial_autocorrelation', 'agg_linear_trend', 'augmented_dickey_fuller']


def _read_data(fp):
    fp = fp.replace('.hea', '').replace('.mat', '')
    signals, fields = wfdb.rdsamp(fp, channels=[0])

    signal = signals[:, 0]
    signal = np.nan_to_num(signal)
    return signal, fields

def _signal_to_dataframe(signal, id):
    n = len(signal)
    ids = [id] * n
    time = np.arange(n)
    df = pd.DataFrame(dict(id=ids, time=time, x=signal))
    return df

def extract_ts_features(file_names, settings=EfficientFCParameters(), verbose=False):
    dfs = []
    settings = {k:v for k, v in settings.items() if k not in EXCLUDING_SETTINGS}
    pbar = tqdm(file_names, disable=(not verbose))
    for fp in pbar:
        pbar.set_postfix_str(fp)
        signal, fileds = _read_data(fp)
        signal_df = _signal_to_dataframe(signal, fp)

        ft_df = extract_features(
            signal_df,
            default_fc_parameters=settings,
            column_id='id',
            column_sort="time",
            disable_progressbar=True,
            n_jobs=8
        )

        dfs.append(ft_df)
    dfs = pd.concat(dfs, axis=0, ignore_index=True)
    return dfs


def load_mat(fp):
    sample = sio.loadmat(fp)['val']
    return sample

def load_header_file(header_file):
    with open(header_file, 'r') as f:
        header = f.readlines()
    header = [line.strip('\n') for line in header]
    return header

def signal_to_nnis(sample, sampling_rate):
    nnis = []
    for i in range(sample.shape[0]):
        try:
            _, rpeaks = biosppy.signals.ecg.ecg(sample[i, :], sampling_rate=sampling_rate, show=False)[1:3]
            nni = pyhrv.tools.nn_intervals(rpeaks)
            nnis.append(nni)
        except ValueError:
            pass
    return nnis

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
        output = PIL.Image.open(buf).convert('RGB')
    
    return output